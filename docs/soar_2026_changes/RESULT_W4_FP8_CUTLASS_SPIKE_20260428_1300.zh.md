# 结果 —— SM120 上 W4-FP8 CUTLASS spike

**日期**：2026-04-28 13:00
**状态**：在 fcloud 执行完成（RTX 6000D sm_120, CUDA 12.8, torch 2.9.1+cu128）。**判定：RED。**
**前序**：[PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.zh.md](PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.zh.md)
**代码**：[spike_w4fp8/bench_w4fp8_sm120.cu](spike_w4fp8/bench_w4fp8_sm120.cu) + [run_w4fp8_spike.sh](spike_w4fp8/run_w4fp8_spike.sh)

## 测量内容

一个手写的 CUDA kernel，在内层 K 循环中执行 W4 → FP8 反量化链（解包 int4 → 减零点 → 乘 bf16 group scale → 通过 PTX cvt 把 bf16x2 转 FP8 e4m3x2），然后发射 `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`。这隔离出真实 W4-FP8 kernel 在密集 FP8 上限（Phase 0 测得 281 TF）之上需付的**反量化税**。

kernel 故意做到最简——无 TMA、无 swizzle、无软件流水——所以测得的 TFLOPS 是调优实现可达性能的**下界**。如果连这个 scaffold 都达 ≥250 TF，真实生产 kernel 可行；如果连 180 TF 都过不了，反量化链是瓶颈，应放弃 W4-FP8 路径。

## 构建步骤

| 步 | 命令 |
|---|---|
| 上传 | scp `bench_w4fp8_sm120.cu` 和 `run_w4fp8_spike.sh` 到 fcloud `/root/` |
| 构建+运行 | `bash /root/run_w4fp8_spike.sh` |

预期编译参数：`nvcc -arch=sm_120 -O3 -std=c++17 -Xptxas -v`。需 nvcc 支持 `sm_120`——需要 CUDA toolkit 12.8 或更新。

## 通过/失败标准（来自提案 §6）

| 在 M=16384, N=14336, K=4096 测得的 TFLOPS | 判定 |
|---|---|
| **≥ 250 TF**（密集 FP8 281 TF 的 ≥ 89%）| **GREEN** —— 推进完整 W4-FP8 kernel |
| **180–249 TF** | **YELLOW** —— 边际，暂列在 NVFP4 KV 后 |
| **< 180 TF** | **RED** —— 永久放弃 W4-FP8 密集 |

## 结果表

| 形状 | 平均 ms/iter | TFLOPS | 占 281 TF FP8 上限 % | 判定 |
|---|---|---|---|---|
| M=16384 N=14336 K=4096（提案默认）| 12.304 | **156.4** | **55.7%** | **RED** |
| M=2048 N=4096 K=4096（小）| 0.441 | 155.8 | 55.4% | RED |
| M=1 N=14336 K=4096（decode 形状）| 0.004 | 32.6 | 11.6% | RED（带宽受限，符合预期）|
| M=8192 N=8192 K=8192（方形中等）| 7.032 | 156.4 | 55.6% | RED |

**Build 信息**：23 个寄存器，0 spill，1 个 barrier（ptxas -v）。所有大形状收敛在 ~156 TF —— 即受制于带反量化的计算峰值，并非启动开销也并非带宽瓶颈（除 decode 形状外）。

## 为何 RED —— 根因归因

Scaffold 内层 K 循环执行：

1. 从 smem 加载 32 个打包 int4 权重（每 8 个权重 1 条 LDS.32）
2. 解包 8 int4 → 8 fp32（手动 shift + sub-zp + fmul scale，8 IADD + 8 FMUL）
3. 通过 4 条 `cvt.rn.satfinite.e4m3x2.f32` PTX 指令把 8 fp32 打包为 4 个 e4m3 对
4. 一条 `mma.sync.aligned.m16n8k32`（每 warp 每 iter 4096 FMA）

若仅 MMA 单独按 281 TF 跑，16384·14336·4096 这一规模需 **~7 ms**。实测 12.3 ms，因此反量化开销 **~5.3 ms = 总周期的 43%**。

## 注意事项 —— 这是下界

1. **未融合反量化**：生产 kernel 用 `lop3.b32` 在 **3 条 PTX 指令**内完成 8 int4 解包+偏移+cast 到 fp16（vs 我们的 ~16 条）。Marlin 类融合解包能把反量化代价降低 ~3-4 倍，恢复大部分 43% gap。
2. **无 TMA / 无 async copy**：smem 加载在关键路径上；真实 kernel 用 TMA + 2-3 级软件流水把它隐藏在 MMA 之下。
3. **合成 smem（lane 索引复用）**：这种做法**最大化** MMA 利用率，所以 156 TF 实际上就是 "带反量化的计算峰值" —— 更现实的 kernel 还得付 L2/HBM 带宽税，但调优实现能隐藏掉。
4. **PTX cvt 路径**：使用 `cvt.rn.satfinite.e4m3x2.f32`（2 fp32 → 1 fp8x2）。SM120 上不存在 bf16-direct 路径，所以这是支持的路线。代价 ~每 2 值 1 条指令。

**SM120 上调优生产级 W4-FP8 kernel 的现实上限**：~200-240 TFLOPS（281 TF 的 70-85%），参考 Marlin-W4A16 在 Hopper/Ada 上通常达到 BF16 峰值的 75-85%。

## 注意事项

1. **合成数据**：kernel 从 gmem 读垃圾累加垃圾。MMA 吞吐是真的，但结果与任何东西都不相关。本 spike 无法验证正确性——仅时序。
2. **无 TMA / 无 swizzle**：调优 kernel 大约能在此 scaffold 与 FP8 峰值之间再缩 10–20%。
3. **每 warp 单 tile**：scaffold 在共享 smem 上跑固定 K_LOOP_ITERS=64 内层循环，grid 大小让总 FMA ≈ M·N·K。这是真实 GEMM 的合理代价代理，但不模拟全 GEMM 的 SMEM 带宽或 L2 带宽压力。
4. **反量化链**：这是**最贵**版本——bf16 乘后再 bf16→FP8 cast。更激进实现可把零点吸收进 scale（去掉减），或用 lop3.b32 技巧并行 int4 解包。所以若 spike 测到 YELLOW，调优 kernel 可能跨入 GREEN。

## 决策（结果出炉后）

**判定：RED → 无限期搁置 W4-FP8 密集 kernel 工作。**

依据：
- 下界 spike 在 FP8 上限的 55.7%，意味着调优 kernel 大致能达 200-240 TF（~75-85%），相对当前 W4A16 Marlin（~140-170 TF-equiv）**仅在 prefill GEMM 上 +15-40%**。
- Decode 阶段带宽受限（32 TF 反映 HBM，不是计算）—— **W4-FP8 在 decode 上几乎零增益**，而 decode 占 S₁ 和 S₈ 的大部分。
- 端到端预估收益：**仅在 prefill-heavy 路径上 ~5-12%**，代价是 **3-4 周 CUTLASS 级 kernel 工程** 加精度校准风险。
- 对比 NVFP4 KV cache（下一优先级）：~3 天 plumbing（survey 显示树内已有 80%），节省 ~44% KV 内存，在长上下文时直接帮助 S∞（更大 batch）。ROI 高得多。

**行动项**：
1. ✅ 更新 [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md) —— 标记 W4-FP8 dense 为 RED/搁置。
2. ✅ 推进 NVFP4 KV cache P2 plumbing（参考 [SURVEY_NVFP4_KV_P1_20260428_1130.zh.md](SURVEY_NVFP4_KV_P1_20260428_1130.zh.md)，先 `--force-dense-minicpm` smoke）。
3. 仅当 NVIDIA / CUTLASS 上游发布可直接 drop-in 的 SM120 调优 W4-FP8 kernel（零工程成本）时再回看。

## 交叉引用

- [PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.zh.md](PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.zh.md)
- [PHASE0_INT8_vs_FP8_SM120_20260427_1630.zh.md](PHASE0_INT8_vs_FP8_SM120_20260427_1630.zh.md) —— 281 TF FP8 上限参考
- [ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md](ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md) —— 完整 kernel 成本分析（3–4 周）
