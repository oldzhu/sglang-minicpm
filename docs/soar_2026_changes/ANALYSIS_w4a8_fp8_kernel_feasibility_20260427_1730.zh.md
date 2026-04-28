# 分析 — 现有 QServe W4A8 INT8 kernel 能否复用为 W4A8 FP8？

**日期**：2026-04-27 17:30
**背景**：用户问 `sgl-kernel/csrc/gemm/qserve_w4a8_per_*_gemm.cu` 现有 W4A8 INT8 kernel 能否简单地把 INT8 替换成 FP8。
**状态**：仅分析。无代码改动。

## TL;DR

**不能简单替换。** MMA 指令一行可改，但 W4 反量化内循环——kernel 主要开销所在——必须端到端重写，因为 QServe 刻意选择整数算术让反量化结果直接进 INT8 IMMA 而不离开整数流水。预计工作量：**约 3–4 周** CUDA 工作（单个有经验工程师，乐观估计）。

## 1. 现有 INT8 kernel 在做什么

[sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu](sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu) — QServe 风格稠密 W4 权重 × A8 激活 GEMM，SM80+：

- MMA：`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`（第 136 行）
- 累加器：`int32_t`
- W4 反量化：4 个打包 int4 → 掩码 `0x0F0F0F0F` / `0xF0F0F0F0>>4` → 减 int8 零点 → 乘 int8 group scale → 结果保持 int8（280–285 行）
- group scales / zeros：`int8_t scales_i8`、`int8_t zeros`
- Epilogue：`int32 → fp16`，乘 per-token + per-channel float scale

关键设计：**整个计算流水保持整数算术**，反量化后的权重直接喂给 INT8 IMMA。这是 QServe 的"两级缩放"——group scale 选取使每组最大值在减零点后仍在 int8 范围内，剩余部分由 epilogue 中的 per-channel float scale 吸收。

## 2. 改 FP8 e4m3 路径需要改什么

| 阶段 | INT8（当前） | FP8（需要） |
|---|---|---|
| MMA 指令 | `s32.s8.s8.s32` | `f32.e4m3.e4m3.f32` |
| 累加器 | `int32_t` | `float` |
| **W4 反量化内循环** | int4 → int8 通过 `(w − z) * s` 整数运算 | int4 → fp16 → 乘 fp16 group scale → **编码 FP8 e4m3 比特模式**（符号 + 4 位指数偏置 7 + 3 位尾数，含饱和/钳位） |
| group scale 存储 | int8 | fp16 / bf16 |
| Epilogue | int32 → fp16 乘 scale | fp32 → bf16 乘 scale（基本兼容） |

MMA 行简单。**反量化重写**复杂因为：
- 整数减+乘合成 2 op；FP8 编码需要按位操作构造 e4m3 字段（指数偏置 + 饱和）。
- 寄存器压力变化（fp16 中间值，不再是 int8 寄存器打包）。
- group scales 变 fp16——加载/广播路径不同。

## 3. 其他必要改动

1. **架构目标**：当前 `__CUDA_ARCH__ >= 800`（Ampere）。FP8 QMMA `m16n8k32.f32.e4m3.e4m3.f32` 需 **sm_89 / 90+ / 120** 和 CUDA 12.0+。需要单独文件（或 `#if` 保护）——不能复用同一份源码。
2. **SM120 tile 重新调优**：CTA/WARP/STAGES 是为 A100/H100 INT8 IMMA 调的。FP8 寄存器压力不同。需要在 RTX 6000D 上重新扫描。
3. **激活 FP8 e4m3 per-token quantizer**：`sgl_per_token_quant_fp8` 已存在 sgl-kernel。直接用。
4. **加载路径**：GPTQ checkpoint 本身就有 fp16 group scales——比合成 int8 scales 更容易。
5. **绑定 + Python 接线**：新 op 注册到 [sgl-kernel/csrc/common_extension.cc](sgl-kernel/csrc/common_extension.cc)，Python 包装，sglang 量化 linear 方法（参考 `gptq_marlin`）。
6. **数值验证**：FP8 e4m3 尾数 = 3 位。outliers 重要。需要 `perf_public_set.jsonl` 上扫描精度，保持 > 99% 归一化以拿 C=1.0。

## 4. 实施工作分解

| # | 任务 | 天 |
|---|---|---|
| 1 | fork `qserve_w4a8_per_group_gemm.cu` → 新 `qserve_w4a8fp8_per_group_gemm.cu` | 0.5 |
| 2 | 重写 W4-int4 → FP8 e4m3 反量化内循环（难点） | 4–6 |
| 3 | 替换 MMA 指令 + fp32 累加器 + epilogue | 1 |
| 4 | 接入激活 FP8 quantizer | 0.5 |
| 5 | 加载器：保留 INT4 权重 + fp16 group scales + FP8 路径 | 1 |
| 6 | sgl-kernel binding + Python op 注册 | 0.5 |
| 7 | sglang linear 方法（参考 `gptq_marlin`） | 1 |
| 8 | SM120 tile/stage 调优扫描 | 2–3 |
| 9 | 对比 W4A16 baseline 数值验证（≥ 99% 归一化） | 2 |
| 10 | S1/S8/Smax 基准 + 迭代 | 2 |
| | **乐观总计** | **14–18** |

含调试迭代的现实估计：**3–4 周**。

## 5. 预期收益

- BF16 = 142 TF 实测；FP8 = 281 TF 实测（Phase 0）→ **~2× 算力峰值**
- Decode **权重带宽受限**（W4 打包存储与 W4A16 一致）。S1 仅 **+5–15%**。
- Prefill / 大 batch **算力受限** → 高并发/长上下文 **+30–60%**。
- 官方分数估计（S1 40% + S8 30% + S∞ 30%）：精度保持时 **+8–15%**。

## 6. 与备选项的对比

| 维度 | W4-FP8 GEMM（本分析） | NVFP4 KV 缓存 | 目录其他项 |
|---|---|---|---|
| 工作量 | 3–4 周新 CUDA kernel | ~1 周（存储 + 量化；复用路径） | 视项 |
| 新颖性风险 | SM120 上未验证；无参考实现 | 冠军组合在用；参考实现存在 | 视项 |
| 精度风险 | 高（FP8 e4m3 3 位尾数） | 中（KV 敏感性） | 视项 |
| 算力杠杆 | +2× FP8 TFLOPS 进 MMA | 无——KV 反量化回 BF16 | 视项 |
| 内存杠杆 | 激活 2× 缩小（vs BF16）——decode 收益小 | KV 2× 缩小 vs FP8（4× vs BF16）——**长上下文大收益** | 视项 |
| 冠军组合证据 | 无 | 有 | n/a |

## 7. 2026-04-28 补充澄清

### 7.1 “W4A8 #1” 到底是什么 vs v18 baseline

之前标为“W4A8 #1”的测试（commit `7ce21c3f5`）实际是 **W8A8 FP8**，不是 INT8 也不是真正的 W4A8。流程对比：

| 阶段 | v18 baseline (W4A16 BF16) | W4A8 #1 误标签 (W8A8 FP8) | 真实 W4A8 FP8（选项 A 留档） |
|---|---|---|---|
| 权重 HBM 类型 | INT4 打包 | **FP8 e4m3（2× 肨胀）** | INT4 打包 |
| K 循环反量化 | INT4 → BF16 | 无 | INT4 → FP8 e4m3 |
| 激活 | BF16（不量化） | BF16 → FP8 e4m3 per-token | BF16 → FP8 e4m3 per-token |
| MMA | BF16×BF16 → FP32, 148 TF | FP8×FP8 → FP32, 281 TF 峰 | FP8×FP8 → FP32, 281 TF 峰 |
| 权重字节/参数 | 0.5 B | 1.0 B | 0.5 B |
| 结果 | 参考 | **+118%/+56%/+30% 回退**（decode 是权重带宽受限；权重字节翻倍） | 假设：两个世界的最优 |

“W4A8 #1” 的回退与“在带宽受限 decode 上权重 HBM 字节翻倍”一致。不能否定真实 W4A8。

我们从未在模型上端到端测试 **W8A8 INT8**。INT8 仅作为 Phase 0 合成微基准出现（136 TF，在任何模型级测试前已终止）。

### 7.2 Phase 0 vs CUTLASS spike（Proposal B）

| 测试 | MMA 输入 | 衡量 | 数字 |
|---|---|---|---|
| Phase 0（已） | FP8 × FP8（K 循环无反量化） | 硬件 FP8 上限 | 281 TF |
| Proposal B（拟） | INT4 打包 → 内核反量化 → FP8 × FP8 | W4→FP8 反量化后的 FP8 上限 | TBD |

Proposal B 衡量的是 Phase 0 故意不包含的反量化税。

## 8. 建议

1. **W4-FP8 稠密 GEMM 留档为选项 A**。NVFP4 KV 落地后再考虑。
2. **下一轮迭代选 NVFP4 KV 缓存**——见 `PROPOSAL_NVFP4_KV_CACHE_20260427_1730.{en,zh}.md`。
3. 可选：跑一次 **廉价 CUTLASS W4-FP8 spike** 在投入 3–4 周前先验证 FP8 上限——见 `PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.{en,zh}.md`。

## 交叉引用

- [PHASE0_INT8_vs_FP8_SM120_20260427_1630.zh.md](PHASE0_INT8_vs_FP8_SM120_20260427_1630.zh.md)
- [RESEARCH_w4a8_kernel_landscape_20260427_1530.zh.md](RESEARCH_w4a8_kernel_landscape_20260427_1530.zh.md)
- [PROPOSAL_W4A8_REAL_001.zh.md](PROPOSAL_W4A8_REAL_001.zh.md)
- [SM120_RTX_PRO_HARDWARE.md](SM120_RTX_PRO_HARDWARE.md)
- [sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu](../../sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu)
- [sgl-kernel/csrc/gemm/qserve_w4a8_per_chn_gemm.cu](../../sgl-kernel/csrc/gemm/qserve_w4a8_per_chn_gemm.cu)
