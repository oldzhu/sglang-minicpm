# 会话记录 — 第 9 轮：baseline 与 W8A8 FP8 流的代码级深度对比（2026-04-28 11:00）

W4A8 深度会话延续。前序：
- 1–6：`CHAT_w4a8-deepdive_20260427_1500.{en,zh}.md`
- 7：`CHAT_w4a8-fp8-reuse-int8_20260427_1730.{en,zh}.md`
- 8：`CHAT_w4a8-clarifications-baseline_20260428_0930.{en,zh}.md`

## 第 9 轮 — 自顶向下的代码/指令/硬件对比

### 用户提问（精简自原文）
1. W8A8 加载为什么做 `INT4 → BF16 → FP8` 而不是像 baseline 那样保留 INT4？
2. baseline 中 "fp16/bf16 group scales kept" 是什么意思？占寄存器/内存吗？
3. "BF16 epilogue" 是把 FP32 转成 BF16？
4. Marlin INT4→BF16 inline 反量化 vs W8A8 的 BF16→FP8 per-token 激活量化——哪个更高效，为什么？算力、资源、所有方面。
5. "FP8×FP8 → BF16，没有额外的 FP32→BF16，对吗？"
6. KV 方面，由于我们做 FP8×FP8 GEMM，FlashAttention 内是否跳过 FP8→BF16 转换？
7. "请结合源代码行号、函数、计算指令、寄存器、内存使用解释。"

### Agent 回答 / 操作

用 `runSubagent` (Explore) 彻底追踪两条路径后，写了一份综合双语文档。

**关键回答**：
- **Q1**：现有 FP8 GEMM kernel 要求两个输入都是 FP8 格式。SM120 上没有稠密 kernel 直接接 INT4 权重 + FP8 激活。加载时转换是变通做法，破坏了 W4 打包优势。
- **Q2**：group scales 是常驻 HBM 张量（每层 ~700 KB，可忽略），不是寄存器。它们是 INT4 映射回原始实数范围的关键。GEMM 期间寄存器成本：每线程每 K-tile 少量值。
- **Q3**：是的。张量核心 MMA 始终在 FP32 中累加（PTX 没有针对 `mma.sync.aligned.*.bf16.bf16.f32` 的 BF16 累加器选项）。Epilogue 把 FP32→BF16（`cvt.rn.bf16.f32`）再写 HBM。
- **Q4**：Marlin 的反量化高效得多——融合在 K 循环内、跑寄存器、与 MMA 重叠、免费搭乘权重 HBM 流量。Path B 的激活量化是**独立 Triton kernel**，有 kernel-launch 开销、激活张量的 HBM round-trip、每组 reduce-max。Path B 还多一个 2× 权重 HBM 带宽惩罚。
- **Q5**：错——FP8×FP8 MMA 也在 FP32 中累加（`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`）。FP32→BF16 cast 在 epilogue 中仍然发生，与 BF16 路径完全一样。
- **Q6**：两点都错：(a) FP8×FP8 GEMM 是 **linear 层**，不是注意力；注意力是另一个 kernel，做 Q·K^T 和 softmax-times-V。(b) 我们的注意力后端不论 KV 存储类型都跑 BF16，所以 FP8 KV 在 FlashAttention 内被反量化到 BF16。要跳过需要 FP8 FlashAttention kernel——我们没有。

**创建文档**：
- [EXPLAIN_baseline_vs_w8a8fp8_flow_20260428_1100.{en,zh}.md](../EXPLAIN_baseline_vs_w8a8fp8_flow_20260428_1100.zh.md)：完整双语深度解析，含对 gptq.py、utils_w4a8_fp8.py、fp8_kernel.py、marlin_template.h、model_runner.py、minicpm_backend.py 的文件:行引用；实际 MMA 指令；指令级时间线；HBM/SMEM/寄存器占用对比。

### 结果
- 6 个用户问题都用代码级证据回答。
- 修正两个重要误解（Q5 关于 epilogue、Q6 关于 KV 路径）。
- 用户现有完整参考文档，可在批准 NVFP4 KV / W4-FP8 spike 前查阅。

### 待办（与第 7/8 轮相同）
- 等用户选择：NVFP4 KV 提案 P1 调研 和/或 W4-FP8 spike。

## 交叉引用
- [EXPLAIN_baseline_vs_w8a8fp8_flow_20260428_1100.zh.md](../EXPLAIN_baseline_vs_w8a8fp8_flow_20260428_1100.zh.md)
- [ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md](../ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md) —— 高层成本分析
- [PROPOSAL_W4A8_REAL_001.zh.md](../PROPOSAL_W4A8_REAL_001.zh.md) —— 真实 W4A8 kernel 的样子
