# CHANGE_0151 Phase B（FourOverSix）— iter‑7 后深入分析：为何 GPTQ‑Marlin 在精度与 S1/S8 速度上都优于 NVFP4‑FOS

> 与 CHANGE_0151_phase_b_four_over_six_006（iter‑7 实测）配套文档。
> 状态：**研究 / 不改代码**。纯只读代码审查 + roofline 分析。
> 分支：`mixed_minicpm_cudagraph` @ `d48793563`。硬件：SM120（RTX PRO 6000 Blackwell, 96 SM）。
> 硬件权威参考：[docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md](SM120_RTX_PRO_HARDWARE.md)。

## 1. 背景与诉求

iter‑7 后实测：

| 配置 | 归一化精度 | S1 (s) | S8 (s) | Smax (s) |
|---|---|---|---|---|
| GPTQ + sparse_qkv_w8 + FP8 KV（Test 12 基线） | **99.11 %** | **121.71** | **44.09** | 35.86 |
| NVFP4‑FOS（iter‑7，复现 iter‑5 量化） | 88.73 % | 173.83（+43 %） | 46.05（+4.4 %） | **31.07（−13.4 %）** |

在转回 GPTQ_FP8_DENSE 优化目录之前，用户希望先弄清楚：

1. 为什么 **NVFP4‑FOS 精度低于** GPTQ INT4（Marlin）？
2. 为什么 **NVFP4‑FOS 在 S1/S8 慢于** GPTQ Marlin？直观上 FP4 的张量核峰值约为 FP8 的 2×、BF16 的 4×，看起来反直觉。
3. 提供两条 GEMM 执行路径的代码审查 + roofline 分析，覆盖 **prefill vs decode**、**算力 vs 带宽瓶颈**、**fused op**。

下文结论既适用于 NVFP4‑FOS（如果未来重启），也适用于 GPTQ_FP8_DENSE 优化目录（用于明确 Marlin 路径已经做得很好的部分，以及仍然没有用满 SM120 的部分）。

## 2. 方法

对仓库中两条路径做只读代码考古：

- **NVFP4 / `--quantization modelopt_fp4`**：注册 → loader → `apply()` → `fp4_gemm()` → CUTLASS SM120 kernel。
- **GPTQ Marlin / `--quantization gptq_marlin`**：注册 → loader → `process_weights_after_loading` → `apply_gptq_marlin_linear` → `gptq_marlin_gemm` → Marlin kernel。

之后基于 SM120 硬件参数做 per‑byte / per‑FLOP 的 roofline 推算。

## 3. 问题 1 — 精度差距（88.73 % vs 99.11 %）

精度差距由 **四个累加根因** 组成，FOS scale 搜索均不能修复：

### 3.1 激活精度：NVFP4 是 **W4A4**，GPTQ Marlin 是 **W4A16**

这是最大单一因素，前几轮被忽视。

NVFP4 路径（[python/sglang/srt/layers/quantization/modelopt_quant.py](../../python/sglang/srt/layers/quantization/modelopt_quant.py#L1168), `ModelOptFp4LinearMethod.apply`）：

```python
# 每次 forward 都把激活动态量化到 FP4：
x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)
assert x_fp4.dtype == torch.uint8
out = fp4_gemm(x_fp4, w, x_scale_interleaved, w_scale_interleaved, alpha, output_dtype, w_n)
```

`x_fp4` 是 FP4（E2M1，每符号 8 个量化电平）。RMSNorm 之后、Attention 之后的激活带有显著离群值且分布长尾，先经过 per‑tensor 又经过 16 元素 per‑block 缩放后压到 4 位尾数，会在每一层永久损失信号，错误**沿 80+ 层累积**。

Marlin 路径（[python/sglang/srt/layers/quantization/marlin_utils.py](../../python/sglang/srt/layers/quantization/marlin_utils.py#L494), `apply_gptq_marlin_linear`）：

```python
output = gptq_marlin_gemm(
    reshaped_x,            # BF16 / FP16，未量化
    weight,                # 重排后的 INT4
    weight_scale,          # 重排后的 FP16 scale
    ...,
    is_k_full=is_k_full,
    use_fp32_reduce=use_fp32_reduce,
)
```

激活全程保持 BF16；kernel 内部把 INT4 权重就地反量化到 BF16，运行 `mma.sync.aligned.m16n8k16` 并在 **FP32 累加**，仅权重有损。

仅这一项就能解释大半精度差距。NVFP4 W4A4 比名字暗示的更激进；modelopt 主推它是为推理算力，并非 W4A16 的等价替代。

### 3.2 校准算法：Hessian‑aware OBQ vs 静态 max‑abs + FOS

GPTQ（gptqmodel）实现 OBQ：每层先做一次校准前向收集激活，构造 Hessian `H = X^T X`，再逐列量化权重，并把已量化列引入的舍入误差**补偿到剩余列**。这是**激活感知**的层级重建误差最小化，即便只有 32 条 sequential 校准样本，也能让 W4 权重对实际数据分布做一定追踪。

本仓的 modelopt NVFP4 + FOS（`benchmark/soar/demo_sala/preprocess_model.py::run_nvfp4_quantization`）做的是：

1. per‑block（16）**max‑abs** 推 FP8 块 scale；
2. per‑tensor amax 推 FP32 weight_scale_2 / input_scale；
3. **FOS** = 每个 block 在 `M ∈ {4, 6}` 中选**仅基于权重**的局部 MSE 最小者（43.14 % 块选 M=4）。

完全 **没有 Hessian、没有逐元素补偿、没有激活感知目标**。FOS 只是从两种静态缩放选项里挑更优的那个。校准数据只参与 per‑tensor `input_scale`，不会改变 per‑block scale 或量化电平。

结论：哪怕校准样本无穷多、FOS 完美，NVFP4‑FOS 也只能逼近“两选一的静态 RTN”，无法达到 Hessian 补偿的 GPTQ 精度。

### 3.3 混合精度不对称：`sparse_qkv_w8` vs 全 W4

GPTQ 基线开了 `SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8` —— 8 个稀疏注意力层的 Q/K/V（24 条线性，可选含 O 共 32 条）升级为 **W8 g=128**。这正是 SALA 架构标记的敏感层（稀疏注意力分发 logits 反馈到路由），同时是输出通道动态范围最高的层。

iter‑5 / iter‑7 的 NVFP4‑FOS 对所有线性都用**统一 W4**，没有 W8 兜底。最敏感的 sparse‑attn 投影与其它层同精度量化。modelopt 本身支持 per‑module 量化规则，但本仓还未实现 FP4 路径上的 `sparse_qkv_w8` 等价物。

### 3.4 码本不匹配：FP4 E2M1 电平 vs LLM 权重分布

INT4（Marlin）电平**均匀**：`{−8,…,+7}` × per‑group scale。GPTQ 校准后的 LLM 权重在 RMSNorm 归一化输入下近似高斯，均匀电平 + 适中 group（g=128）匹配较好。

FP4 E2M1 电平（符号 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}）**不均匀**，零附近密、远端疏。这适合对数正态数据，但**在权重密度低的远端浪费码点**。FOS 通过切换 per‑block 最大可表示值（M=4 vs M=6）部分弥补，但不能改变 block 内部电平位置。

### 3.5 为什么 FOS 本身看起来收益有限

iter‑6 ablation（`SOAR_NVFP4_FOS_ENABLE=0` 早门 67.33 %）与 iter‑5/iter‑7（FOS 开，ori‑acc 70.98 %–71.24 %）差距约 3.6 pp，确实有效但相对到 GPTQ 的 ~9 pp 差仍小。这与上述分析一致：FOS 优化的是更深问题中的小切片（block 内静态缩放二选一），不触及主因（W4A4 + 无 Hessian 补偿 + 敏感层无 W8 兜底）。

### 3.6 精度归因表

| 因素 | GPTQ Marlin | NVFP4‑FOS | 估计影响 |
|---|---|---|---|
| 激活 dtype | BF16（W4A16） | FP4（W4A4） | **主导**，多 pp |
| 校准 | OBQ + Hessian + 误差补偿 | 静态 max‑abs + FOS scale 搜索 | 数 pp |
| 敏感层兜底 | sparse_qkv_w8（24 条 W8） | 全 W4 | 1–2 pp |
| 码本电平 | 均匀 INT4 | 非均匀 FP4 E2M1 | 0.5–1 pp |
| 块大小 | 128（FP16 scale） | 16（FP8 scale） | NVFP4 略占优 |

实测归一化精度差 ~10 pp，可由 1–4 项完全解释；FOS 单兵作战不能填平。

## 4. 问题 2 — 在 S1/S8 上的速度差距（FP4 峰值更高却更慢）

关键认知：**S1/S8 不在 FP4 593 TFLOPS 峰值发挥的工况**。它们由 decode 与小 M prefill 块主导，瓶颈是**带宽与 launch**，不是算力。Smax 才暴露长上下文的算力瓶颈，FP4 在那里确实赢（−13.4 %）。

### 4.1 单层显存占用（4096 × 4096 线性）

| 组件 | GPTQ Marlin（W4A16, g=128） | NVFP4（W4A4, block=16） |
|---|---|---|
| 权重打包 | (K/8, N) int32 = **8.00 MB** | (N, K/2) uint8 = **8.00 MB** |
| per‑group / per‑block scale | (K/128, N) FP16 = **0.25 MB** | (N, K/16) FP8 e4m3 = **1.00 MB** |
| zero‑point | 0（对称量化） | 0 |
| g_idx | 0（无 desc_act） | 0 |
| per‑tensor 额外项 | — | input_scale + weight_scale_2 + alpha（可忽略） |
| **每层合计** | **~8.25 MB** | **~9.00 MB** |
| **每权重等效位数** | **4.13 bit** | **4.50 bit** |

**NVFP4 权重比 GPTQ Marlin 多约 9 %**。S1 decode 受 HBM → SM 的权重流速限制，这 9 % 几乎直接等于 token/s 的差距。S8 因 8× kv‑cache + 激活流量稀释，差距收窄至 +4.4 %。

### 4.2 tile 形状随 M 的适配

NVFP4 SM120 派发（[sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu](../../sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu#L483), `cutlass_fp4_bf16_gemm_dispatch_sm120`）：

```
M ≤ 256:  MmaTileShape = 128 × 128 × 128, ClusterShape 1×1×1
M  > 256: MmaTileShape = 256 × 128 × 128, ClusterShape 1×1×1
```

S1 decode `M=1` 时 kernel 运行 128×128×128 tile 去算 1×N 行 —— M 维约 **128× 过度配置**，CTA 占用与 SM 利用率崩塌。当前 sgl‑kernel 对 FP4 没有更小 M 的特化。

Marlin 反之内置 **5 套小批量配置**（[sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu](../../sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#L163)）：

```
(thread_k=128, thread_n=256, threads=256)
(thread_k=64,  thread_n=256, threads=256)
(thread_k=128, thread_n=128, threads=256)
(thread_k=64,  thread_n=128, threads=128)
(thread_k=128, thread_n=64,  threads=128)
```

并通过运行时 **scorer**（310–350 行）按 M、N、K、occupancy、smem 适配、波次综合打分挑选最佳。M=1–8 时 Marlin 落到 thread_m_blocks=1（M 维 16）+ 宽 N 的形状，更贴合“行 × 矩阵”的实际工况。Marlin 在结构上**为 decode 工况调好**，目前的 FP4 kernel 没有。

### 4.3 NVFP4 独有的每次 forward 开销

每次 NVFP4 forward 还要：

- `fp4_quantize(x, input_scale_inv)` —— 量化激活、生成交错 FP8 块 scale。kernel 不大，但**每条线性、每个 token‑step 都启动一次**。
- `cutlass_scaled_fp4_mm_sm100a_sm120a` host 侧运行时 SM 检查（`getSMVersion()`），首次入队后会缓存，但仍有 Python 侧开销。

Marlin **没有**每次 forward 的激活变换；激活直接以 BF16 进 kernel。

S1 长跑里，数千 decode step × 80+ 层，每条线性几个 µs 的差就会累积成数秒。

### 4.4 MMA 指令与 SM120 利用

| 路径 | MMA 指令 | 张量核类别 | SM120 理论峰值 |
|---|---|---|---|
| GPTQ Marlin（反量化后 BF16） | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | warp 级 Ampere 风格 | **148 TFLOPS BF16** |
| NVFP4（FP4 输入 + FP8 块 scale） | CUTLASS `OpClassBlockScaledTensorOp`，SM120 原生 | warp 级 block‑scaled | **296 TFLOPS FP8 / 593 TFLOPS FP4** |

NVFP4 路径**确实**原生面向 SM120（`ArchTag = cutlass::arch::Sm120`，构建标志 `compute_120a/sm_120a`，`-DENABLE_NVFP4=1`），所以在足够大的 M（S∞ / Smax）下能兑现 FP4 峰值。小 M 时不行 —— 128×M 固定 tile 在 MMA 峰值前就把利用率打爆。

Marlin 在 SM120 上结构性地**被锁在 148 TFLOPS BF16**，因其 kernel 围绕 `m16n8k16` BF16 MMA 构建，无法访问 QMMA 或块缩放 FP4。这正是 **Smax 翻盘** 的原因：M~千级时 tile 充实，FP4 4× 峰值兑现；S1 不行。

### 4.5 按阶段的 roofline（SM120, BW = 1398 GB/s）

|  | M | 瓶颈 | NVFP4‑FOS | GPTQ Marlin | 胜方 |
|---|---|---|---|---|---|
| Decode | 1 | 权重带宽 + tile 利用率 | 9 MB/层 + 128 tile 浪费 + per‑forward 激活量化 | 8.25 MB/层 + 小 M tile picker | **Marlin** |
| Prefill 块（S1 长上下文） | 至 65 536/step | 混合：小块带宽，大块算力 | 大 M 时 tile 适配 | BF16 峰值上限 | 小块持平，大块 **NVFP4** |
| Decode（S8） | 8 | 带宽 + 小算力 | 同 S1，但 8× 激活摊薄权重 | 小 M tile picker | **Marlin**（差距收窄） |
| Prefill（Smax / S∞） | 千级 | 算力 | FP4 峰值 593 TFLOPS，tile 充实 | BF16 峰值 148 TFLOPS | **NVFP4**（与实测 −13.4 % 一致） |

恰好对应 S1 → S8 → Smax 的速度差符号翻转。

### 4.6 fused op

| Fused op | NVFP4 路径 | Marlin 路径 |
|---|---|---|
| 权重反量化进 MMA | 有（block‑scaled tensor op） | 有（LUT + 移位内联） |
| per‑tensor `alpha = input_scale × weight_scale_2` | 有（epilogue） | 不需要（无激活 scale） |
| bias add | 不在 kernel 内，host 侧后处理 | 有（epilogue） |
| 激活量化（per forward） | **GEMM 前单独 kernel** | 无 |
| QK‑norm + RoPE | server 端 `--enable-fused-qk-norm-rope`，两路径同样可用 | 同 |
| Mixed‑chunk 调度 | server 端 `--enable-mixed-chunk`，两路径同样可用 | 同 |

净帐：NVFP4 每条线性、每步多一个 GEMM 前 kernel；Marlin 没有。两者都把主反量化融进 GEMM。server 级融合（qk‑norm‑rope、mixed‑chunk）相同。

## 5. 建议的 profiling（确认而非阻塞）

1. **`nsys profile`** S1 与 Smax，按 kernel 名分组：
   - S1：top kernel 应是 `gptq_marlin_gemm`（GPTQ）或 `cutlass_scaled_fp4_mm_*` + `fp4_quantize`（NVFP4）；后者的 `fp4_quantize` 应占 NVFP4 时长的几个百分点。
   - Smax：top kernel 是大 M FP4 GEMM；SM 利用率应 > 60 %。
2. **`ncu`** + `--section MemoryWorkloadAnalysis,LaunchStats,Occupancy,SchedulerStats,WarpStateStats` 在 M ∈ {1, 8, 64, 512, 4096} 固定形状下跑 FP4 GEMM，确认小 M tile 利用率不足。
3. **`cuobjdump --dump-sass`** 检查安装的 `sgl_kernel*.so`，确认 `cutlass_fp4_bf16_gemm_dispatch_sm120` 实际产 SM120 SASS，以及 Marlin 的回落目标（compute_90 还是 compute_120a）。
4. 单独写微基准对比 `gptq_marlin_gemm` vs `fp4_gemm`，覆盖 {bs=1, 8, 64, 512, 4k} × {seq=1k, 8k, 32k}，使用模型实际形状。

## 6. 启示

### 对 NVFP4‑FOS（暂搁，未来可能重启）

如重启则需：

- **改为 W4A16 NVFP4**（FP4 权重 + BF16 激活）而非 W4A4。modelopt 支持关闭激活量化的配置。预期回收 3.1 大部分精度。
- **加 FP4 路径上的 `sparse_qkv_w8` 等价物**（或 per‑module 量化配置），让稀疏注意力 QKV 留在更高精度。预期回收 3.3。
- **给 `cutlass_fp4_bf16_gemm_dispatch_sm120` 加小 M tile 配置**（如 16×128×128 或 32×128×128）+ 类似 Marlin 的运行时 scorer。预期 S1 大幅收益。
- **把 `fp4_quantize` 与上一个 op（多半是 residual + RMSNorm）融合**，去掉每条线性的 GEMM 前 kernel。
- **在 FOS 之上叠加 OBQ 风格 / Hessian 校准**。

### 对 GPTQ_FP8_DENSE 目录（下一轮）

审计确认 Marlin 路径在本仓**结构上已接近峰值**；明显的提升项是：

- **它在 SM120 上被 BF16 峰值（148 TFLOPS）封顶**。要打破天花板需要 W8 sparse‑attn QKV 走 FP8 W8A8 路径（cutlass `OpClassBlockScaledTensorOp` FP8 模式），或迁移到 Hopper/Blackwell FP8 GEMM —— 是大工程。
- **Marlin 没有 TMA**，超长上下文 prefill 阶段会留带宽。给 Marlin 加 TMA bulk load 不简单但有具体收益。
- **每次 forward 开销已近零**，不必再追。
- **显存占用已近最优**（4.13 bit/权重），无简单进一步压缩。
- 现实可做的目录目标因此都在 GEMM **之上的层**：调度 / scheduler / KV / attention / fused‑pre‑attn，而非 Marlin 内核内部。

这进一步细化既有目录 [docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md) 的优先级：**降权** “重写 Marlin”，**升权** “scheduler / KV / fused‑pre‑attn / mcq runaway‑think 缓解”。

## 7. 验证命令（如需做实测）

```bash
# 启动实例
python3 scripts/fcloud/fcloud_workflow.py start-instance
python3 scripts/fcloud/fcloud_workflow.py sync

# nsys S1（GPTQ 基线）
python3 scripts/fcloud/fcloud_exec.py exec '
cd /root/submission_sim &&
source prepare_env.sh &&
nsys profile -o /root/nsys_gptq_s1 -t cuda,nvtx --force-overwrite=true \
  python3 /root/data/eval_model_001.py \
    --data_path /root/data/perf_public_set.jsonl --max-concurrent 1 \
    --output_dir /root/outputs_nsys
'

# NVFP4 同理（先在 prepare_env 设 SOAR_QUANT_PROFILE=nvfp4_fos，再量化、跑）
```

## 8. 结论摘要

| 问题 | 结论 |
|---|---|
| Q1 精度差距 | 主要由 **W4A4 vs W4A16 激活精度**（NVFP4 每次 forward 把激活量化到 FP4）主导，其次是 **OBQ Hessian 校准 vs 静态 FOS 校准**，再次是 **统一 W4 vs sparse_qkv_w8 混合精度**。FOS scale 搜索单兵填不平。 |
| Q2 S1/S8 速度差距 | **是带宽与 launch 工况，不是算力工况**。NVFP4 权重重 ~9 %（FP8 块 scale，g=16）；FP4 GEMM 没有小 M tile 特化（128×M tile vs Marlin 5 套小批量 + scorer）；每次 forward 多一个 `fp4_quantize` kernel。FP4 4× 峰值只在 Smax / S∞ 的大 M 工况兑现，与实测（Smax −13.4 %）完全一致。 |
| Q2 fused op / roofline | Marlin 已融合反量化 + bias + 累加；NVFP4 融合反量化 + alpha，但每次 forward 多付一个激活量化 kernel。server 级融合（qk‑norm‑rope、mixed‑chunk）相等。Marlin 在 SM120 上被 BF16 峰值（148 TFLOPS）封顶；NVFP4 在大 M 时可吃到 FP4 593 TFLOPS。 |

## 9. 回滚

本文档无代码改动，无需回滚。

## 10. 下一步

按 §6 的优先级调整，转回 GPTQ_FP8_DENSE 目录。

## 11. 交叉引用

- iter‑5 计划：CHANGE_0151_phase_b_four_over_six_004
- iter‑5 参考量化：TEST_RESULTS_TRACKING 的 NVFP4‑FOS‑5
- iter‑6 FOS=0 ablation：CHANGE_0151_phase_b_four_over_six_005，行 NVFP4‑FOS‑6
- iter‑7 复现 + 速度：CHANGE_0151_phase_b_four_over_six_006，行 NVFP4‑FOS‑7
- 硬件参考：SM120_RTX_PRO_HARDWARE.md
- 下轮要更新的目录：OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md

## 12. 关键调用点引用

- NVFP4 loader：[python/sglang/srt/layers/quantization/modelopt_quant.py](../../python/sglang/srt/layers/quantization/modelopt_quant.py#L861)（`ModelOptFp4Config`）、[1068 行](../../python/sglang/srt/layers/quantization/modelopt_quant.py#L1068)（`ModelOptFp4LinearMethod`）、[1168 行](../../python/sglang/srt/layers/quantization/modelopt_quant.py#L1168)（`apply()` 含 `fp4_quantize` + `fp4_gemm`）。
- NVFP4 SM120 内核：[sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu](../../sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu#L483)（`cutlass_fp4_bf16_gemm_dispatch_sm120`，tile 配置 123–137，dispatch 657）。
- Marlin loader：[python/sglang/srt/layers/quantization/gptq.py](../../python/sglang/srt/layers/quantization/gptq.py#L219)（`GPTQMarlinConfig`）、[563 行](../../python/sglang/srt/layers/quantization/gptq.py#L563)（`GPTQMarlinLinearMethod`）、[893 行](../../python/sglang/srt/layers/quantization/gptq.py#L893)（`apply`）。
- Marlin 派发与 scorer：[python/sglang/srt/layers/quantization/marlin_utils.py](../../python/sglang/srt/layers/quantization/marlin_utils.py#L464)（`apply_gptq_marlin_linear`）；[sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu](../../sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#L163)（小批量配置，scorer 310–350）。
- Marlin MMA：[sgl-kernel/csrc/gemm/marlin/marlin_template.h](../../sgl-kernel/csrc/gemm/marlin/marlin_template.h#L77)（FP16/BF16 的 `mma.sync.aligned.m16n8k16`）。
- 构建目标：[sgl-kernel/CMakeLists.txt](../../sgl-kernel/CMakeLists.txt#L231)（CUDA ≥ 12.8 时 `compute_100a/120a` 与 `-DENABLE_NVFP4=1`）。
