# 结果 —— SM120 上 W4-FP8 CUTLASS spike（模板，fcloud 跑完后填写）

**日期**：TBD
**状态**：等待 fcloud 执行。
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

## 结果表（运行后填写）

| 形状 | 平均 ms/iter | TFLOPS | 占 281 TF FP8 上限 % | 备注 |
|---|---|---|---|---|
| M=16384 N=14336 K=4096（提案默认）| TBD | TBD | TBD% | 主判定 |
| M=2048 N=4096 K=4096（小）| TBD | TBD | TBD% | tile 较小，可能显出 launch 开销 |
| M=1 N=14336 K=4096（decode 形状）| TBD | TBD | N/A | 带宽受限；FLOPS 无意义 |
| M=8192 N=8192 K=8192（方形中等）| TBD | TBD | TBD% | 检查 K-scaling |

## 注意事项

1. **合成数据**：kernel 从 gmem 读垃圾累加垃圾。MMA 吞吐是真的，但结果与任何东西都不相关。本 spike 无法验证正确性——仅时序。
2. **无 TMA / 无 swizzle**：调优 kernel 大约能在此 scaffold 与 FP8 峰值之间再缩 10–20%。
3. **每 warp 单 tile**：scaffold 在共享 smem 上跑固定 K_LOOP_ITERS=64 内层循环，grid 大小让总 FMA ≈ M·N·K。这是真实 GEMM 的合理代价代理，但不模拟全 GEMM 的 SMEM 带宽或 L2 带宽压力。
4. **反量化链**：这是**最贵**版本——bf16 乘后再 bf16→FP8 cast。更激进实现可把零点吸收进 scale（去掉减），或用 lop3.b32 技巧并行 int4 解包。所以若 spike 测到 YELLOW，调优 kernel 可能跨入 GREEN。

## 下一步决策树

- **GREEN** → 在 NVFP4 KV cache 上线后排上 W4-FP8 完整 kernel 迭代。
- **YELLOW** → 暂时无限期推迟 W4-FP8；仅在 NVFP4 KV 没达预期时再复审。在 [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md) 中记录 YELLOW 原因。
- **RED** → 从优化路线图永久移除 W4-FP8 密集。更新 catalog。

## 交叉引用

- [PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.zh.md](PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.zh.md)
- [PHASE0_INT8_vs_FP8_SM120_20260427_1630.zh.md](PHASE0_INT8_vs_FP8_SM120_20260427_1630.zh.md) —— 281 TF FP8 上限参考
- [ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md](ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md) —— 完整 kernel 成本分析（3–4 周）
