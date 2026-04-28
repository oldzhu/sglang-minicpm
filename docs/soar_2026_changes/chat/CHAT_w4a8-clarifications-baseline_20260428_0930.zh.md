# 会话记录 — 第 8 轮：澄清 W8A8 FP8 vs INT8、Phase 0 vs Proposal B、baseline 流程（2026-04-28）

W4A8 深度会话延续。前序：
- 1–6 轮：`CHAT_w4a8-deepdive_20260427_1500.{en,zh}.md`
- 第 7 轮：`CHAT_w4a8-fp8-reuse-int8_20260427_1730.{en,zh}.md`

## 第 8 轮 — 澄清 + baseline 流程

### 用户提问
1. "之前我们测的 w8a8 是对 4-bit 量化模型把 int4 权重转成 int8 再把激活从 bf16 转 int8，精度还行但跑分变差，对吗？我们没测过 w8a8 fp8 吧？"
2. "Proposal B 那个，我记得 Phase 0 已经测过 FP8、BF16、INT8，跟 Proposal B 不是同一个测试吗？"
3. "现在两个都做。我现在理解 w8a8 fp8 流程是：(1) 加载时把 INT4 解到 BF16 再转 FP8 e4m3 存储，跑 CUTLASS FP8×FP8 BF16-out GEMM；(2) 激活：每次 forward 把 BF16 转 FP8 e4m3 per-token。当前 baseline 流程是怎样？是 w4a16 (bf16) 对吧？"

### Agent 回答 / 操作

**W8A8 历史的修正**：
- 早先 "W4A8 #1"（commit `7ce21c3f5`）实际是 **W8A8 FP8**，不是 INT8。用户直觉对（在 W4 模型上做 A8 量化导致回退），但激活/权重类型是 FP8 e4m3，不是 INT8。
- 我们从未端到端测过 W8A8 INT8。INT8 只作为 Phase 0 微基准存在（136 TF，已终止）。

**Phase 0 vs Proposal B 的区别**：
- Phase 0 = 通过 `torch._scaled_mm` 测稠密 FP8×FP8→BF16 GEMM。两个输入在寄存器里已是 FP8。测的是硬件上限（281 TF）。
- Proposal B = W4-权重 × FP8-激活 GEMM，INT4 在内核中反量化为 FP8 再喂 MMA。测的是 Phase 0 故意不包含的**反量化税**。通过阈值（≥250 / 180–249 / <180 TF）告诉我们 281 TF 上限有多少在 W4→FP8 反量化后存活。

**Baseline 流程（v18 W4A16 BF16）**：
1. 加载：HBM 中 INT4 打包，fp16/bf16 group scales，无权重膨胀。
2. Forward：激活 BF16（不量化）。Marlin K 循环：INT4 → 寄存器内 BF16 反量化 → BF16×BF16 MMA 148 TF → BF16 epilogue。
3. KV：HBM 中 FP8 e5m2，FlashAttention 内反量化回 BF16。

与 W4A8 #1 误标签对比：

| | v18 baseline | W4A8 #1 (W8A8 FP8) | 真实 W4A8 FP8（留档） |
|---|---|---|---|
| 权重 HBM | INT4 打包 | FP8（2× 膨胀） | INT4 打包 |
| 激活 | BF16 | FP8 | FP8 |
| MMA | BF16 148 TF | FP8 281 TF | FP8 281 TF |
| 权重字节/参数 | 0.5 | 1.0 | 0.5 |

W4A8 #1 为什么回退：权重字节翻倍对 decode（权重带宽受限）的伤害大于 FP8 MMA 峰值的收益。

**文档补丁**：
- `ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.{en,zh}.md` — 加 § 7 "2026-04-28 补充澄清"，含三方对比表（baseline / W4A8 #1 / 真实 W4A8）和 Phase 0 vs Proposal B 对比。
- `PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.{en,zh}.md` — 加 § 1.1 "与 Phase 0（已完成）的区别"，含并排对比表。

### 结果
- 三个开放歧义全部书面解决。
- 用户现已建立一致的 baseline 流程理解，可对比未来迭代。

### 待办
- 同第 7 轮：等用户在（a）NVFP4 KV 提案 P1 调研、（b）W4-FP8 spike、（c）两者顺序进行 之间选择，再开始任何代码工作。

## 交叉引用
- [ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md § 7](../ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md)
- [PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.zh.md § 1.1](../PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.zh.md)
- [PHASE0_INT8_vs_FP8_SM120_20260427_1630.zh.md](../PHASE0_INT8_vs_FP8_SM120_20260427_1630.zh.md)
- [CHANGE_W4A8_001_iteration_002.zh.md](../CHANGE_W4A8_001_iteration_002.zh.md) — 误标签说明
