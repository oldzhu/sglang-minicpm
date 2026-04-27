# CHANGE_W4A8_001 — 第 002 轮迭代：验证结果与决策

**状态：本具体路径放弃。真正的 W4A8 尚未测试。**

> **关键命名错误更正（2026-04-27 事后补充）：**
>
> commit `7ce21c3f5` 实际实现并测试的是 **W8A8 FP8 blockwise**，**不是** W4A8。
> 代码在加载期将 GPTQ INT4 权重反量化到 BF16，然后再量化为 **FP8（8 位）存储**，因此运行时 GEMM 是 FP8 权重 × FP8 激活。原 GPTQ INT4 权重的 4 位带宽优势在加载期就被丢掉了。
>
> v18 基线是 **W4A16**（Marlin：4 位 INT4 权重存储，寄存器内反量化到 BF16，BF16 张量核 148 TF）。真正的 W4A8 应当**同时保留 INT4 存储并使用 FP8 张量核 296 TF**，从而同时拿到 2× 带宽优势（vs FP8 权重）和 2× 算力优势（vs BF16 MMA）。
>
> 下文记录的 S1/S8/Smax +118%/+56%/+30% 回退，是在内存带宽受限的工作负载（decode bs=1）下把权重内存占用翻倍的**必然结果**。它**并不**否定真正 W4A8 的假设。
>
> **真正的 W4A8（如 QQQ 风格的 W4-INT8 Marlin，或 W4A8-Machete）是一个独立的优化方向，应当单独评估** — 见提案 `PROPOSAL_W4A8_REAL_001.zh.md`。

本文与 `CHANGE_W4A8_001_iteration_001.zh.md`（实现说明）配套。本文记录"加载期 GPTQ INT4 → FP8 块缩放 GEMM"路径（被误标为 W4A8）的 fcloud 验证结果及放弃该具体实现的决定。

## 测试设置

- **commit**：`7ce21c3f5` — "W4A8 #1: load-time GPTQ INT4 -> FP8 blockwise GEMM (env-gated)"
- **fcloud 实例**：223.167.85.181（SM120 RTX PRO 6000 Blackwell）
- **配置**：GPTQ `sparse_qkv_w8` + FP8 e5m2 KV + dense + torch.compile (`max-bs=8`) + Test 20 服务器参数（`chunk=32K, prefill-max-req=1, running=24, sched-cons=1.0, mixed-chunk`）
- **变量**：`SOAR_W4A8_FP8_GEMM=1`（基线 `=0`）
- **覆盖范围**：标准 attention 的 `qkv_proj`/`o_proj`（8 层）+ MLP `gate_up_proj`/`down_proj`（32 层）；lightning attention 不动；INT8 sparse_qkv 通过位宽守卫自动跳过
- **基线参考**：v18 Test 12 — S1=121.71s, S8=44.09s, Smax=35.86s, ori_accuracy=79.29%

## 验证步骤（全部在 fcloud 上）

1. ✅ CPU 单元测试（`test/srt/quantization/test_utils_w4a8_fp8.py`）— 3/3 通过；FP8 e4m3 round-trip 阈值从 2e-2 放宽到 5e-2（3 位尾数 FP8 的合理预期）。
2. ✅ 启用 `SOAR_W4A8_FP8_GEMM=1` 重启服务器 — 214s Ready；通过 `/proc/<pid>/environ` 确认环境变量传递正确。
3. ✅ 冒烟测试（Paris 续写）— 通过。
4. ✅ 精度评估（150 样本，concurrency=32）。
5. ✅ 速度 S1 / S8 / Smax — 三档全部测完。

## 精度结果

| 指标 | W4A8 #1（本轮） | v18 Test 12 基线 | Δ |
|---|---:|---:|---:|
| 平均 | **79.20%** | 79.29% | −0.09 pt |
| 归一化 | 99.00% | 99.11% | — |
| C | **1.0** | 1.0 | 不变 |
| mcq | 53.33 | 63.33 | −10（噪声） |
| cwe | 82.67 | 72.00 | +10.67 |
| fwe | 100.00 | 97.78 | +2.22 |
| niah | 96.67 | 100.00 | −3.33 |
| qa | 63.33 | 63.33 | 0 |

**精度判定：中性**。整体平均在 ±0.1pt 内；分任务漂移落在 150 样本本地噪声范围内。两侧归一化 ≥ 99% → C=1.0。mcq/cwe 的此消彼长与 Tests 29/34a/v18-revert 中观察到的局部噪声同性质。

## 速度结果 — 净回退

| 档位 | W4A8 #1（本轮） | v18 Test 12 基线 | Δ |
|---|---:|---:|---:|
| S1 | **265.32 s** | 121.71 s | **+118%** |
| S8 | **68.88 s** | 44.09 s | **+56%** |
| Smax | **46.44 s** | 35.86 s | **+30%** |

服务端指标：

| 档位 | 平均 TTFT | 平均 TPOT | 输出吞吐 |
|---|---:|---:|---:|
| S1 | 120.78 ms | 15.64 ms | 62.70 tok/s |
| S8 | 193.45 ms | 18.35 ms | 378.93 tok/s |
| Smax | 11097.76 ms | 26.76 ms | n/a |

**速度判定：明确变差**。按 SOAR 公式 `S1×0.4 + S8×0.3 + Smax×0.3` 估算分数影响：

- 速度分数（本轮）≈ `(121.71/265.32)×40 + (44.09/68.88)×30 + (35.86/46.44)×30 ≈ 18.3 + 19.2 + 23.2 = 60.7`
- 速度分数（基线）= 100
- **最终分数：60.7 × 1.0 = 60.7（基线 100）**，损失约 **−39%**。

即便达到 FP8/BF16 的理论 2× 比（296 TF / 148 TF），S1 仍会回退，因为 decode bs=1 路径是**对权重的内存带宽受限**，而非算力受限。Marlin INT4 权重比 FP8 小 2 倍，在带宽受限路径上 INT4 必胜。

## 根因分析

为什么 cutlass `fp8_blockwise_scaled_mm` 在每一档都输给 Marlin INT4：

1. **S1（decode bs=1, M=1）**：kernel 完全受权重内存带宽限制。INT4（0.5 字节/元素）读取的字节数是 FP8（1 字节/元素）的一半。Marlin 的 INT4 反量化-融合 GEMM 在 1398 GB/s HBM 上搬运的数据更少。
2. **S8 / Smax（中小 M）**：cutlass FP8 blockwise 有固定的每次调用开销（CUTLASS scheduler、块缩放广播、K-tile 调度）。在我们的 hidden 尺寸（≤ 7168）和 M ≤ 256 的场景，这些开销相对于真正 MMA 工作量的占比很高。Marlin 紧凑、手调的 INT4 GEMM 反而胜出。
3. **296 TF FP8 QMMA 峰值在此工作负载下无法兑现**。要打满 QMMA，需要持续的大 M 致密 GEMM（每个 tile ≥ 1024×1024×1024）。decode 和 S8 prefill 远低于这个门槛。
4. **张量核心并不是瓶颈** — 真正的瓶颈是带宽和 kernel 启动开销。把权重格式换成"精度更高但更大"的版本，在非算力受限的情况下毫无帮助。

这与 SM120 硬件参考（`docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md`）一致：FP8 只有在 M 大到算术强度跨过带宽 roofline 拐点之后才会赢。

## 决策：放弃本具体路径（经 cutlass 的 W8A8 FP8 blockwise），但**不**反驳 W4A8 假设

- 保持 `SOAR_W4A8_FP8_GEMM=0`（仓库默认值）。
- **不**在任何提交包中携带此路径。
- env flag 与代码保留，供反量化/量化工具函数复用。
- **本实验并未反驳 W4A8 假设** — 我们测的是 W8A8 FP8，不是 W4A8。真正的 W4A8（INT4 存储 + FP8 MMA）需要不同的 kernel（QQQ 风格的 W4-INT8 Marlin、W4A8-Machete 或自定义 CUTLASS 混合输入 GEMM），应在独立的迭代中评估。

## 树中保留的内容

实现本身**保留**（env 门控，默认关），原因有二：

1. **正确性已验证** — 加载期 GPTQ INT4 → FP8 转换 + cutlass blockwise GEMM 输出准确。若未来出现确实需要 FP8 的场景（极不同的工作负载、M=1k+ 的稠密 prefill），代码即开即用。
2. **可复用积木** — `python/sglang/srt/layers/quantization/utils_w4a8_fp8.py` 提供的 `gptq_int4_dequantize()` 与 `fp8_blockwise_quantize()/dequantize()` 工具函数，可能对未来的量化实验有用（如 `_003` 探索 NVFP4-带回退或混合 INT4-MMA 路径）。

单元测试保持绿色，env flag 保持默认 0。

## 验证命令（备查）

```bash
# 同步 + 启用 W4A8 重启
sed -i 's/SOAR_W4A8_FP8_GEMM:-0/SOAR_W4A8_FP8_GEMM:-1/' /root/submission_sim/prepare_env.sh
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server

# 测试
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax
```

## 回滚 / 关闭说明

W4A8 已默认关闭。若要在 fcloud 上显式关闭：

```bash
sed -i 's/SOAR_W4A8_FP8_GEMM:-1/SOAR_W4A8_FP8_GEMM:-0/' /root/submission_sim/prepare_env.sh
python3 scripts/fcloud/fcloud_workflow.py restart-server
```

若要彻底删除代码（不推荐，保留可选性）：

```bash
git revert 7ce21c3f5
```

## 下一步

1. **真正 W4A8 评估** — 见 `PROPOSAL_W4A8_REAL_001.zh.md`。三个候选 kernel：
   - **QQQ 风格 W4-INT8 Marlin**：INT4 权重 + INT8 激活 + INT8 张量核（SM120 上 296 TF）。成熟开源 kernel。
   - **W4A8-Machete（vllm/compressed-tensors）**：优先 Hopper；SM120 回移工作量未知。
   - **自定义 CUTLASS 混合输入 GEMM**：最高成本，全量 bring-up。
2. **同时推进其他方向**（股宝受限或 kernel 融合受限的向量，在 decode 上有收益）：
   - Marlin INT4 SM120 tile 重调优（降低每次调用开销）
   - QKV / O 与 attention 输出投影融合
   - `fused_qk_norm_rope` 变体
   - 推测解码（小 M 但每步贡献更多有效 token）
3. 英文配套文档 `CHANGE_W4A8_001_iteration_002.en.md`。

## 本轮触及文件

无（结果文档）。实现自第 001 轮（commit `7ce21c3f5`）未变。
