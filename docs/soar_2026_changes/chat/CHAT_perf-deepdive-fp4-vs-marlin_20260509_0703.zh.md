# CHAT — 性能深挖：NVFP4‑FOS vs GPTQ Marlin（精度 + S1/S8 速度差）

时间戳：20260509_0703  •  分支：`mixed_minicpm_cudagraph` @ `d48793563`

## 1. 用户问题（原文）

> 在转回 gptq 目录前，请先深入挖掘以下两点：
> 1. 为什么 NVFP4‑FOS 权重精度比 GPTQ INT4 低？
> 2. 为什么 NVFP4‑FOS 在 S1/S8 速度基准上比 GPTQ INT4（FP8 KV cache）Marlin 慢？看起来反直觉，因为 NVFP4 张量峰值应该是 FP8 的 2 倍。
> 对两条执行路径（NVFP4 与 Marlin 内核）做代码审查可能有帮助；或者其他 profiling/方式都行。
> 这两个问题的回答不仅能改进 NVFP4+FOS，也能在我们决定转回 GPTQ_FP8_DENSE 时帮到那条路径。
> 最好包含 prefill/decode、算力/带宽瓶颈、Fused op 的分析报告。

## 2. Agent 行动

- 纯只读研究，无代码改动，无 fcloud 测试（实例 iter‑7 后仍处暂停）。
- 并行派发 2 个 `Explore` 子 agent（thoroughness=medium）：
  - **A** 梳理 NVFP4 / `modelopt_fp4` 路径（loader、`apply()`、`fp4_gemm`、CUTLASS SM120 dispatch）。
  - **B** 梳理 GPTQ Marlin 路径（loader、`process_weights_after_loading`、`apply_gptq_marlin_linear`、`gptq_marlin_gemm`、MMA、构建目标、混合精度）。
- 交叉核对一处不一致（B 称 SM120 不在构建目标；A 确认在 CUDA ≥ 12.8 时启用 `compute_120a/sm_120a`）。采用后者，已对照 `sgl-kernel/CMakeLists.txt`。
- 产出双语分析报告：
  - [docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_007.en.md](../CHANGE_0151_phase_b_four_over_six_007.en.md)
  - [docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_007.zh.md](../CHANGE_0151_phase_b_four_over_six_007.zh.md)

## 3. 核心结论（一页摘要）

### Q1 精度差（88.73 % vs 99.11 %）
1. **NVFP4 是 W4A4，GPTQ 是 W4A16。** modelopt FP4 在 `apply` 中每次 forward 都用 `fp4_quantize` 把激活量化到 FP4 —— 主导因素。
2. **校准**：GPTQ 是 Hessian‑aware OBQ + 列间误差补偿；modelopt FP4 + FOS 是静态 per‑block max‑abs + 二选一 scale 搜索（M=4 vs M=6），无 Hessian、无激活感知。
3. **混合精度**：GPTQ 基线开 `sparse_qkv_w8`（24 条稀疏注意力 QKV 跑 W8 g=128）；NVFP4 路径全 W4，无敏感层兜底。
4. **码本**：FP4 E2M1 电平 {0, 0.5, 1, 1.5, 2, 3, 4, 6} 非均匀；均匀 INT4 更贴合 RMSNorm 归一化后的权重分布。

FOS 提升 ~3.6 pp（iter‑6 ablation 67.33 % vs iter‑5 70.98 %），有效但相对 ~10 pp 总差仍小。

### Q2 S1/S8 速度差（FP4 峰值更高却更慢）
1. **S1/S8 是带宽 + launch 工况，不是算力工况**。4096×4096 单层下 NVFP4 权重比 Marlin 重约 9 %：9.00 MB vs 8.25 MB（FP8 e4m3 g=16 块 scale 占 1 MB，对比 FP16 g=128 group scale 仅 0.25 MB）。
2. **FP4 GEMM 没有小 M tile 特化。** SM120 dispatch（`cutlass_fp4_bf16_gemm_dispatch_sm120`）M ≤ 256 用 128×128×128 —— S1 decode M=1 时 M 维约 128× 过度配置。Marlin 内置 5 套小批量配置 + 运行时 scorer，专为 M=1–8 调好。
3. **NVFP4 每条线性、每步多一个 `fp4_quantize` kernel**；Marlin 没有（BF16 激活直接入 GEMM）。
4. **算力峰值**：Marlin 的 `mma.sync.aligned.m16n8k16` BF16 路径在 SM120 上锁在 148 TFLOPS；NVFP4 的 `OpClassBlockScaledTensorOp` 可达 296 TFLOPS（FP8）/ 593 TFLOPS（FP4）—— 但要 M 足够大才能填满 128×M tile。
5. 与实测翻转完全一致：S1 +43 % 慢、S8 +4.4 % 慢、**Smax −13.4 % 快**。

### Roofline 总览

| 阶段 | M | 瓶颈 | 胜方 | 原因 |
|---|---|---|---|---|
| Decode | 1 | 带宽 + tile 利用率 + 额外激活量化 kernel | **Marlin** | 权重小、小 M tile picker |
| Prefill 块（S1 长上下文） | 至 65 536/step | 混合 | 小块持平，大块 **NVFP4** | FP4 峰值仅大 M 兑现 |
| Decode（S8） | 8 | 带宽 + 小算力 | **Marlin** | 差距收窄 |
| Prefill（Smax / S∞） | 千级 | 算力 | **NVFP4** | FP4 峰值兑现 |

### Fused op

- Marlin：反量化 + bias + 累加在 kernel 内融合；forward 之外无额外开销。
- NVFP4：反量化 + alpha 融合；每条线性、每步多一个 `fp4_quantize` kernel。
- server 级融合（`--enable-fused-qk-norm-rope`、`--enable-mixed-chunk`）两路径同样可用。

## 4. 结果

- 决策：**搁置 NVFP4‑FOS**，按计划转回 GPTQ_FP8_DENSE 目录。
- 报告 §6 建议的目录优先级调整：
  - **降权** “重写 Marlin 内核”（其 Ampere 风格 MMA 把 SM120 锁在 148 TFLOPS BF16，但重写工程量极大；权重已近最优 4.13 bit/元素）。
  - **升权** scheduler / KV / fused‑pre‑attn / mcq runaway‑think 缓解。
- 若未来重启 NVFP4 的可选后续：试 W4A16 NVFP4（关激活量化）、加 FP4 路径的 `sparse_qkv_w8` 等价物、给 `cutlass_fp4_bf16_gemm_dispatch_sm120` 加小 M tile 配置、把 `fp4_quantize` 融到上一个 RMSNorm kernel、在 FOS 之上叠加 OBQ 风格校准。

## 5. 交叉引用

- 配套实测：CHANGE_0151_phase_b_four_over_six_006（iter‑7）、TEST_RESULTS_TRACKING 行 NVFP4‑FOS‑7
- 硬件：SM120_RTX_PRO_HARDWARE.md
- 下轮要细化的目录：OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md

## 6. 创建/修改的文件

- 新建：`docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_007.en.md`
- 新建：`docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_007.zh.md`
- 新建：本聊天日志 + 英文双胞胎

## 7. 待解决问题 / 下轮

- 验证 Q1：试 **NVFP4 W4A16**（modelopt 提供关闭激活量化的选项），可隔离 W4A4 vs W4A16 的贡献量（按假设是主因）。
- 验证 Q2：在 S1 与 Smax 上跑 `nsys` + `ncu`（命令见报告 §7）。不阻塞 GPTQ 目录推进。
