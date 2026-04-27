# 变更 迭代 W4A8 #1 — 续篇 001（实施记录）

> 本文档是 `PROPOSAL_iteration_W4A8_001.zh.md` 的续篇。原始提案中的 §5
> 计划要求在 `preprocess_model.py` 中离线生成 `weight_fp8` 张量并写入
> safetensors。Step 2 实施过程中我们改用了一个等效但更简单的方案——
> **加载期** 转换，且不修改任何磁盘格式。本文档记录最终落地实现。

## 1. 设计变更对比

| 项目 | 原方案 (§5) | 最终方案 |
|---|---|---|
| FP8 权重生成 | `preprocess_model.py` 离线后处理，写入 safetensors 的 `weight_fp8` / `weight_fp8_scale` | **加载期** 在 `gptq.py` 的 `process_weights_after_loading` 中即时反量化 + FP8 量化，作为非持久 buffer 挂在 layer 上 |
| `model.safetensors.index.json` | 需要重写以登记新张量 | 保持不动 |
| 资格标记 | 在预处理脚本里按 key 解析 | 在 `minicpm.py` 中给对应 linear 模块设置 `_soar_w4a8_eligible = True` |
| 磁盘体积 | 模型包额外 +~0.5 GB | 与基线完全一致 |
| 运行时显存 | 持平 | FP8 buffer 与 Marlin buffer 同时驻留，约 +0.5 GB |
| 提交包 | 需要重新做量化 | 复用现有 GPTQ 产物 |

**为什么这样更安全**：完全规避了磁盘改写带来的索引/分片错乱风险；
`prepare_model.sh` 不需要任何改动；通过 `SOAR_W4A8_FP8_GEMM=1` 即可整体开关。

## 2. 实际改动文件

1. **`python/sglang/srt/layers/quantization/utils_w4a8_fp8.py`**（新增）
   - `gptq_int4_dequantize(qweight, qzeros, scales, group_size=128)`：
     向量化的 PyTorch 解包，覆盖 `gptqmodel` 的 4-bit `desc_act=False`
     格式，返回 `(K, N)` BF16。
   - `fp8_blockwise_quantize(w, block_size=128)`：返回
     `cutlass_w8a8_block_fp8_linear_with_fallback` 期望的布局
     `(N, K)` `float8_e4m3fn` + `(N//128, K//128)` fp32。
   - `fp8_blockwise_dequantize(...)`：单元测试用的反向工具。

2. **`python/sglang/srt/layers/quantization/gptq.py`**
   - 新增 `import os`。
   - `GPTQMarlinLinearMethod.process_weights_after_loading` 在调用现有
     Marlin 重排逻辑 **之前**，先调用新增的 `_soar_maybe_setup_w4a8_fp8`
     辅助函数。该函数：
     - 仅当 `SOAR_W4A8_FP8_GEMM=1`、`_soar_w4a8_eligible == True`、
       4-bit / `group_size=128` / `desc_act=False` 且两个分区维度都是
       128 的整数倍时才生效。
     - 从 layer 读取 `qweight` / `qzeros` / `scales`，反量化到
       `(K, N)` BF16，转置到 `(N, K)`，按 128×128 切块量化为 FP8
       e4m3，将 `weight_fp8` 与 `weight_fp8_scale` 注册为非持久 buffer。
     - 成功时设置 `layer._soar_w4a8_active = True`。
   - `GPTQMarlinLinearMethod.apply` 在 `_soar_w4a8_active` 时直接走
     `cutlass_w8a8_block_fp8_linear_with_fallback`；任何异常都会被
     捕获、记录、清标志，再回落到原 Marlin 路径。

3. **`python/sglang/srt/models/minicpm.py`**
   - `MiniCPMMLP.__init__` 给 `gate_up_proj`、`down_proj` 打上
     `_soar_w4a8_eligible = True`。
   - `MiniCPMAttention.__init__`（标准 attention，`mixer_type == "minicpm4"`）
     给 `qkv_proj`、`o_proj` 打标。
   - `MiniCPMLightningMixer` 故意 **不打标**，保持 BF16 Marlin（与原提案
     修订 #3 一致）。

4. **`benchmark/soar/demo_sala/prepare_env.sh`**
   - 在其他 lightning 相关环境变量旁添加
     `export SOAR_W4A8_FP8_GEMM="${SOAR_W4A8_FP8_GEMM:-0}"`。
   - 在脚本末尾的诊断 echo 区追加对应的输出行。

5. **`test/srt/quantization/test_utils_w4a8_fp8.py`**（新增）
   - 三个 CPU-only 烟雾测试：
     - `test_gptq_int4_dequantize_synthetic`：手工按 gptqmodel 布局打包
       已知 INT4 权重与零点，验证函数还原结果与 `(q - z) * s` 公式的
       相对 Frobenius 误差 < 1e-6。
     - `test_fp8_blockwise_quantize_roundtrip`：随机 BF16 `(512, 384)`
       张量量化-反量化往返，相对 Frobenius 误差 < 2e-2。
     - `test_fp8_blockwise_handles_zero_block`：避免全零块产生 NaN。

## 3. 不改动的文件

- `benchmark/soar/demo_sala/preprocess_model.py`
- `benchmark/soar/demo_sala/prepare_model.sh`
- `model.safetensors.index.json` 与各权重分片

## 4. 验证流程（fcloud）

1. `python3 scripts/fcloud/fcloud_workflow.py sync`
2. `python3 scripts/fcloud/fcloud_workflow.py restart-server`
   - 默认 `SOAR_W4A8_FP8_GEMM=0`，期望与基线 S1 / S8 / Smax 一致。
3. 在 fcloud 跑单元测试：
   ```bash
   cd /root/submission_sim
   python3 sglang/test/srt/quantization/test_utils_w4a8_fp8.py
   ```
4. 修改 fcloud 上的 `/root/submission_sim/prepare_env.sh` 把
   `SOAR_W4A8_FP8_GEMM=1`（或在 `restart-server` 前 export）。
5. `restart-server` → `wait-server` → `accuracy` → `speed --variant all`。
6. 与 v18 基线（Test 12：S1=121.71s, S8=44.09s, Smax=35.86s, accuracy=79.29%）对比。
7. 服务端日志应包含
   `[SOAR W4A8] enabled FP8 blockwise GEMM for layer prefix=...` 行
   （每个标准 attention / MLP linear 一条，lightning 层应缺席）。

## 5. 回滚

- 把 `SOAR_W4A8_FP8_GEMM=0`（或 unset）后重启服务即可。
- 模型产物不变，不需要重新量化。

## 6. 下一步

- 在 fcloud 跑通后录入 Test 13。
- 若 FP8 路径在某层抛错，逐层告警 + 自动回落能保证服务不挂；据此日志
  逐层定位问题。
- 后续迭代：在拿到数据后将 `_soar_w4a8_eligible` 白名单扩展到 lightning
  Q/K/V/O。
