# CHANGE_0151 — Phase B FourOverSix NVFP4（首次端到端运行）

## 背景与动机
- 延续 `PROPOSAL_phase_b_four_over_six_nvfp4_20260505.zh.md` 与 `CHANGE_0150_phase_a_nvfp4_baseline.zh.md`。
- 目标：在 NVFP4 仅权重量化基础上加入 FourOverSix（FOS）逐 16 元素块的尺度选择补丁（同时尝试 `M=4` 与 `M=6`，按 MSE 选优），在与原 NVFP4 相同的速度预算下恢复精度。
- 本次迭代覆盖 fcloud RTX PRO 实例上的**第一次端到端跑通**：量化 → 加载 → smoke → 精度。S1/S8/Smax 速度**有意未跑**，因为精度低于淘汰阈值（见结果章节）。

## 规则合规声明
- 量化由 `prepare_model.sh` 调用 `preprocess_model.py` **现场完成**，符合"不得提交预量化权重"的规则。
- 所有权重/尺度沿用标准 NVFP4 排布（uint8 紧凑半字 + `weight_scale` fp8 + `weight_scale_2` + `input_scale`）；FOS 补丁仅改变每个 16 元素块的尺度选值。落盘格式可由 sglang 原生 `modelopt_fp4` 加载器加载（服务端日志 `Detected nvfp4 checkpoint`）。
- 不引入新依赖，复用已 pin 的 `nvidia-modelopt 0.43.0`。

## 实施计划（变更前）
- 复用 Phase A 的 NVFP4 校准路径（`mtq.quantize` + `NVFP4_DEFAULT_CFG`，校准源 `perf_public_set.jsonl`）。
- 在导出阶段**之前**激活 `_install_four_over_six_patch`，使每次 `NVFP4QTensor.get_weights_scaling_factor` 调用都同时评估 M=4 与 M=6。
- 使用**手动流式导出**替换 modelopt 的 `export_hf_checkpoint(...)`，原因：
  1. modelopt 0.43.0 的 `export_hf_checkpoint` 在每个 Linear 上隐藏一份 fp16 引用（Plan-A 诊断观察到 ~148 MiB / Linear 增长），在 84 GiB 的 SM120 GPU 上跑到 ~307/560 层即 OOM；
  2. fcloud 容器存在 **64 GiB CPU cgroup 上限**（`/sys/fs/cgroup/memory.max=68719476736`），"先搬到 CPU"的方案同样会被 SIGKILL。
- 手动导出按 Linear 流式：量化一个 Linear → 把 4 个张量 + bias 拷到 CPU → 释放该 Linear 的 GPU 权重，再处理下一个；每 64 个 Linear 触发一次 `gc.collect()`+`empty_cache()`。

## 实际代码变更（变更后）

文件（均在 `mixed_minicpm_cudagraph` 分支）：
- [benchmark/soar/demo_sala/preprocess_model.py](benchmark/soar/demo_sala/preprocess_model.py)
  - `run_nvfp4_quantization` 中以新的手动流式导出替换 `export_hf_checkpoint(...)`。
  - 从 modelopt 内部导入 `requantize_resmooth_fused_llm_layers`、`is_quantlinear`、`QUANTIZATION_NVFP4`、`NVFP4QTensor`、`to_quantized_weight`、`get_quant_config`、`convert_hf_quant_config_format`、`_patch_revert_weight_conversion`/`_unpatch_revert_weight_conversion`。
  - 逐 Linear 循环：
    - 跳过非量化模块（`QUANTIZATION_NONE`）；
    - 通过 `is_quantlinear(_sub)` 跳过非 Linear 模块（commit `f14c3f3e8`，修掉外层 `MiniCPMSALAForCausalLM` 因子模块为 NVFP4 而被误判的 `NotImplementedError`）；
    - 计算 `weight_scale_2` → `weight_scale`（FOS 补丁在此触发）→ 紧凑 `weight`；
    - 4 个张量 + bias 拷 CPU 后将 `_sub._parameters["weight"] = None` 释放 GPU；
    - 删除 `_amax`/`_scale` 等量化器缓冲。
  - 之后遍历 `named_parameters()` 与 `named_buffers()`，**按各模块 `_non_persistent_buffers_set` 过滤非持久 buffer**（commit `829128503`，否则 rotary `cos_cached`+`sin_cached` 会写入 ~16 GiB fp32 数据到 ckpt）。
  - 通过 `_patch_revert_weight_conversion()` 后调用 `model.save_pretrained(state_dict=cpu_state_dict, save_modelopt_state=False)`；最后把 `quantization_config` 合并回 `config.json`。
- [benchmark/soar/demo_sala/prepare_env.sh](benchmark/soar/demo_sala/prepare_env.sh)
  - `nvfp4_fos` profile 设置 `SOAR_NVFP4_FOUR_OVER_SIX=1`（已存在）。
  - `nvfp4` 与 `nvfp4_fos` 均使用 `--quantization modelopt_fp4`（已存在）。

本次提交：
- `a2cbedd65` preprocess(nvfp4): replace export_hf_checkpoint with manual streaming export
- `f14c3f3e8` preprocess(nvfp4): skip non-Linear modules in manual streaming loop
- `829128503` preprocess(nvfp4): skip non-persistent buffers in manual export

## 验证命令

量化：
```bash
cd /root/submission_sim && \
  export SOAR_QUANT_PROFILE=nvfp4_fos SOAR_NVFP4_FOUR_OVER_SIX=1 SOAR_QUANT_FORCE=1 && \
  source prepare_env.sh && \
  python3 -u preprocess_model.py \
    --input  /root/models/openbmb/MiniCPM-SALA \
    --output /root/models/MiniCPM-SALA-NVFP4-FOS \
    --mode   nvfp4
```

Smoke：
```bash
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50,"temperature":0.0}'
```

精度：
```bash
cd /root/data && python3 eval_model_001.py \
  --api_base http://127.0.0.1:30000 \
  --model_path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --data_path /root/data/perf_public_set.jsonl --concurrency 32
```

## 结果汇总

| 指标 | NVFP4-FOS（本次） | GPTQ baseline（test 12 参考） |
|------|-------------------|-------------------------------|
| ckpt 大小 | 6.5 GiB（2 shards） | 8.0 GiB（sparse_qkv_w8） |
| 量化 GPU 峰值 | 34.20 GiB | n/a |
| FOS 选 M=4 比例 | 43.14%（224 个 Linear / 521.1M 块） | n/a |
| 服务端加载内存 | 7.31 GiB | ~12 GiB |
| Smoke（`2+2`） | 输出连贯 `<think>` | OK |
| 本地精度（avg） | **75.98%** | 79.29% |
| → cwe | 74.33% | n/a |
| → fwe | 92.22% | n/a |
| → mcq | 63.33% | （baseline 较高） |
| → niah | 93.33% | n/a |
| → qa | 56.67% | （baseline 较高） |
| → len_0_4k | 63.33% | n/a |
| → len_4k_32k | 70.25% | n/a |
| → len_32k_128k | 83.58% | n/a |

**结论：精度未通过门槛。** 75.98% < 77% → 标准化精度 ≤ 97% → C = 0（淘汰）。
精度损失主要集中在 `mcq`(63.33%) 与 `qa`(56.67%)；长上下文检索 `niah`(93.33%) 与计数 `fwe`(92.22%) 几乎不掉。

S1/S8/Smax 速度**本次未跑** —— 在 C=0 情形下无意义；待精度恢复后再跑。

## 回滚

回滚 3 个提交即可移除 FOS 导出分支：
```bash
git revert --no-edit 829128503 f14c3f3e8 a2cbedd65
git push minicpm-src mixed_minicpm_cudagraph
```
`nvfp4` profile（无 FOS）与 GPTQ baseline 不受影响，仍为现役提交配置。

## 诊断：3 pp 精度损失可能来自哪里

1. **每块 FOS 目标与下游注意力/MLP 的真实损失不一致。** 当前补丁按局部 MSE 在 `M ∈ {4,6}` 间二选一；这与"GEMM + 激活 + softmax 后的输出误差"不同。在长上下文 QA/MCQ 中，模型需要在 100k 上下文中关注一个 token，微小尺度误差会沿 Q/K/V/O 投影与 32 层叠加放大。
2. **缺乏按层/按投影的跳过名单。** GPTQ baseline 之所以保留 Q/K/V w8（`sparse_qkv_w8` preset）正是因为注意力投影对精度敏感。FOS-NVFP4 一刀切 fp4，Q-proj 与 K-proj 大概率是主要精度水池。
3. **`mcq`/`qa` 依赖紧凑数值推理与短答案** —— 恰是对决策边界附近 logit 精度最敏感的任务。`niah`/`fwe` 对尺度噪声更鲁棒。
4. **校准目标未与 FOS 对齐。** `_install_four_over_six_patch` 仅在导出时挂钩；校准阶段收集的 `_amax` 用的是未打补丁的尺度公式，存在校准-导出错配。

## 下一步建议

按推荐优先级排列：

A. **按层跳过 FOS（最便宜，速度无损）。** Q/K/V 投影（或前后各 4 层）退回到普通 NVFP4，不开 FOS。预期可恢复 ~1.5–2 pp 的 mcq/qa。

B. **QKV 混精度回退。** MLP（`gate/up/down_proj`）保持 fp4；attention QKV 退回 w8。需在 modelopt 加载器路径上承担不同的落盘排布，工作量较大但收益更高。

C. **跑 Phase A baseline（无 FOS）以隔离 FOS 影响。** 若 plain NVFP4 也只有 ~76%，问题就是"fp4 太粗对该模型"，而不是 FOS 选错 M；同一代码路径只需把 `SOAR_NVFP4_FOUR_OVER_SIX=0`，成本低。

D. **在校准前向过程中也激活 FOS 感知尺度公式**，而非只在导出时；可改善尺度选择统计。

E. **暂搁 Phase B，重排优先级。** 官方分数 = 性能 × 正确性，C=0 时无任何分数。若 A–D 在小迭代内无法把精度抬到 ≥ 78%（带安全裕度），回到 GPTQ+FP8 baseline，按 `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` 推进速度优化。

## 交叉引用

- `PROPOSAL_phase_b_four_over_six_nvfp4_20260505.zh.md` — FOS 设计
- `CHANGE_0150_phase_a_nvfp4_baseline.zh.md` — 纯 NVFP4 上一迭代
- `PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.zh.md` — 原始组合提案
- `SM120_RTX_PRO_HARDWARE.md` — GPU 约束（84 GiB，NVFP4 = 593 TFLOPS）
- TEST_RESULTS_TRACKING.md — 本迭代精度行已更新
