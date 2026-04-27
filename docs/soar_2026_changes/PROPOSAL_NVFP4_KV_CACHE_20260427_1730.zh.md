# 提案 — NVFP4 KV 缓存作为下一轮优化迭代

**日期**：2026-04-27 17:30
**状态**：仅提案。**尚无代码改动。** 按工作流规则等待用户批准。
**前置**：`PHASE0_INT8_vs_FP8_SM120_20260427_1630.{en,zh}.md` 终止 W4-INT8，搁置 W4-FP8。

## 1. 目标与预期收益

把当前 FP8 KV 缓存换成 **NVFP4 量化的 KV 缓存**（对 K/V 张量在 HBM 上做仅存储压缩）。

- KV 缓存内存：**−50%** vs FP8 KV（vs BF16 KV 为 4× 缩小）。
- KV 带宽（decode 是 KV 读带宽受限）：**−50%** vs FP8 KV。
- 端到端官方分数预期：**+8–15%**（在长上下文/高并发 KV 流量主导时收益最大）。
- **算力不变。** K/V 在注意力矩阵乘前反量化回 BF16；矩阵乘仍在 BF16 张量核心 148 TF 上。**不会**用到 593 TF FP4 张量核心（那要 GEMM 两侧输入都是 FP4）。

## 2. 规则合规检查

| 规则 | 状态 |
|---|---|
| 提交包 ≤ 2 GB | ✅ 无影响（运行时量化） |
| 现场量化（不可提交预量化权重） | ✅ KV 是运行时 |
| 总运行 ≤ 5 h | ✅ 增加 μs/token KV 量化开销 |
| 禁止技巧（评测期间私下开 prefix cache） | ✅ 无 |
| 并发标志尊重 | ✅ 正交 |
| 精度 ≥ 97% 归一化以 C ≠ 0 | ⚠️ 必须验证；KV 量化有已知敏感性 |

## 3. 精度/稳定性风险

- **主要风险**：NVFP4 KV 在 sub-7B 级模型上若 K（尤其旋转编码后的高幅 key）outliers 被钳，可能掉 1–3 个归一化精度点。缓解措施：
  - 每块 scale（16 元素 block-fp4 + bf16 scale）——已是标准 NVFP4 layout。
  - 可选：保留前/后 N 个 token 或第一层注意力为 FP8 KV（混合）——匹配最新一周冠军描述的"混合 NVFP4 + FP8 KV"。
  - **任何官方提交前必须用 `perf_public_set.jsonl` 验证。** 目标归一化 > 99%（C=1.0）。若降到 97–99%，再看速度收益是否值得 ×0.92 / ×0.96。
- **稳定性**：KV 缓存 layout 变化局限于注意力后端；如后端支持干净则无 cudagraph 影响。

## 4. 实施计划（改动前）

### 4.1 可参考的实现
- vllm `csrc/quantization/fp4/`（离线 NVFP4 权重量化；不直接适用但共享 e2m1 + block-scale 数据 layout）。
- `sglang.srt.layers.attention.flashattention` → KV dtype 配置：今支持 `auto / fp8_e5m2 / fp8_e4m3 / nvfp4`。验证 NVFP4 路径是否端到端打通（可能不完整）。
- 查 `sgl-kernel/csrc/attention/` 是否已有 NVFP4 KV 反量化（本提案 Phase 0）。

### 4.2 阶段划分
| 阶段 | 目标 | 工作量 |
|---|---|---|
| **P1 — 调研** | 读 sglang FlashAttention KV-dtype 路径；确认 NVFP4 是否已接通或找到缺口 | 1 d |
| **P2 — 接线** | 端到端把 `--kv-cache-dtype nvfp4` 接通：sglang 启动器 → 注意力后端 → cudagraph 捕获 | 2–3 d |
| **P3 — 精度验证** | 跑精度评测；迭代 outlier 处理（per-block scale dtype、可选第一层 FP8 回退） | 2–3 d |
| **P4 — 速度验证** | fcloud 上跑 S1/S8/Smax；确认收益 | 1 d |
| **P5 — 混合（可选）** | 若纯 NVFP4 不到 99%：第一层 / sink-token FP8 + 其余 NVFP4 | 2 d |
| | **总计** | **6–10 天** |

### 4.3 可能涉及的文件
- `python/sglang/srt/configs/model_config.py` — `kv_cache_dtype` 校验加 `nvfp4`
- `python/sglang/srt/layers/attention/flashattention_backend.py` — KV 读时反量化
- `python/sglang/srt/layers/radix_attention.py` — KV 写时量化
- `sgl-kernel/csrc/attention/` — 可能扩展 FlashAttention kernel 以支持 NVFP4 K/V 反量化
- `benchmark/soar/demo_sala/prepare_env.sh` — 把 `--kv-cache-dtype fp8_e5m2` 改成 `nvfp4`

## 5. 验证命令

### 精度
```bash
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
```
**通过条件**：归一化精度 > 99%（C=1.0）。97–99% 仅当速度收益足以补偿 ×0.92/×0.96 时可接受。

### 速度
```bash
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```
**通过条件**（vs v18 baseline S1=121.71 / S8=44.09 / Smax=35.86）：S8/Smax 应改善；S1 小幅或持平。

## 6. 结果汇总表（模板——测试后填入）

| 运行 | 配置 | S1 (s) | S8 (s) | Smax (s) | ori_acc | norm_acc | C |
|---|---|---|---|---|---|---|---|
| Baseline (v18) | W4A16 + FP8 KV | 121.71 | 44.09 | 35.86 | 79.29 | 99.11% | 1.0 |
| NVFP4 KV (P2) | W4A16 + NVFP4 KV | TBD | TBD | TBD | TBD | TBD | TBD |
| 混合 (P5) | W4A16 + NVFP4 KV + 第一层 FP8 | TBD | TBD | TBD | TBD | TBD | TBD |

## 7. 回滚步骤

```bash
# 还原 prepare_env.sh
git checkout HEAD -- benchmark/soar/demo_sala/prepare_env.sh
# 还原 sglang 源码（若改动）
git checkout HEAD -- python/sglang/srt/layers/attention/ python/sglang/srt/configs/
# 同步 fcloud
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
```

## 8. 下一步建议（本迭代之后）

若 NVFP4 KV 成功（归一化 > 99%、+8–15% 速度）：
- 重启搁置的 **W4-FP8 稠密 GEMM**（`PROPOSAL_W4A8_REAL_001` 选项 A）。
- 考虑稀疏注意力 pattern 调优（`OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` 下一项）。

若 NVFP4 KV 精度失败（< 97%）：
- 放弃前先试混合（P5）。
- 混合仍 < 97%：回退 FP8 KV，转向 W4-FP8 spike（`PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730`）。

## 9. 交叉引用

- [ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md](ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md)
- [PHASE0_INT8_vs_FP8_SM120_20260427_1630.zh.md](PHASE0_INT8_vs_FP8_SM120_20260427_1630.zh.md)
- [ANALYSIS_nvfp4_offline_quant_20260427_0857.zh.md](ANALYSIS_nvfp4_offline_quant_20260427_0857.zh.md) — 注意：那次分析是关于 NVFP4 *权重* 量化（精度崩溃）。KV 量化是不同、更温和的应用。
- [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)
- 微信参考：最新一周冠军组合 = "W4A16 GPTQ + 混合 NVFP4/FP8 KV 缓存 + 其他"。
