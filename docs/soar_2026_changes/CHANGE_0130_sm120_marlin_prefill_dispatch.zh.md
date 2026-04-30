# CHANGE_0130_sm120_marlin_prefill_dispatch

**日期**: 2026-04-21
**状态**: 已在 fcloud 完成测试且准确率失败；本地已准备回退
**特性范围**: 面向 GPTQ + FP8 KV + dense 基线的 SM120 Marlin 优化

## 1）背景与动机
- 问题描述：
  当前安全基线的 profiling 显示，GEMM 是主要瓶颈，约占 prefill 时间的 85.3%，占 decode 时间的 63.5%。当前 Marlin GPTQ 路径在 [sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu] 中仍然走的是 SM80 风格的 `mma.sync.aligned.m16n8k16` 逻辑。此前 CHANGE_0125 的 tile 扩展虽然成功编译，但没有带来可测收益，因为 scorer 对 MiniCPM-SALA 的权重形状正确地持续选择了窄 tile。
- 为什么该改动预期会提速：
  当前 SM120 路径仍然使用通用评分启发式。一个受控的一阶段迭代，可以在不改变模型语义和量化格式的前提下，为 MiniCPM-SALA 的长上下文 prefill 场景定制 Marlin 的 SM120 exec-config 选择逻辑，从而改善中大 `M` 情况下的真实核函数选择和运行行为。
- 目标阶段：
  以 prefill 为主，decode 只有在调度变化同时改善中等 batch GEMM 时才可能受益。

## 2）SOAR 规则合规性检查
- 允许原因：
  该改动属于官方 MiniCPM-SALA 模型上的纯推理内核与分发优化，完全在比赛允许的优化范围内。
- 不触碰限制（prefix cache/固定并发/可复现性）：
  不使用 prefix cache 技巧，不修改固定并发规则，不替换基座模型，也不提交预量化模型文件。活跃路径仍然是文档中定义的 GPTQ + FP8 KV + dense 基线。
- 对正确性系数 C 的预期影响：
  准确率风险较低。本次迭代不改模型架构、不改 prompt、不改评测逻辑、不改权重格式，目标是保持 `C = 1.0`。

## 3）改动前实施计划
- 计划修改的文件/函数：
  - `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`
    - `get_thread_config_list(...)`
    - `score_sm120_candidate(...)`
    - `get_exec_config(...)` 附近的 SM120 自动选择路径
    - `log_sm120_exec_config_once(...)` 附近的诊断日志
  - `sgl-kernel/csrc/gemm/marlin/marlin_template.h`
    - 第一阶段以检查为主；只有在可以保持最小改动的前提下，才考虑非常局部的 staging/load-path 变更
  - `python/sglang/srt/layers/quantization/marlin_utils.py`
    - 如果内核侧 dispatch 变化需要 Python 侧协调，则检查 workspace 大小和调用路径假设
  - `python/sglang/srt/layers/quantization/gptq.py`
    - 检查 Python 侧是否存在额外的 shape/path 限制，阻碍新的 dispatch 行为
  - `benchmark/soar/demo_sala/prepare_env.sh`
    - 仅当需要新增 server arg 或环境变量用于受控 A/B 验证时才改动
- 最小化 diff 策略：
  第一阶段只做 dispatch/scoring 专项化，不尝试完整的 SM120 原生 MMA 重写。除非需要一个受控调试开关，否则保持基线量化路径和 server args 不变。
- 回滚方案：
  将所有改动限制在 Marlin 选择/调试逻辑范围内，确保可以通过一个小补丁整体回退。一旦出现回归，立即撤回 CHANGE_0130，恢复当前最佳基线配置。

## 4）实际代码改动
- 补丁摘要：
  已在 Marlin 自动选型路径中实现一个受控的 SM120 prefill 感知评分策略。改动刻意保持很小，并且只限制在 `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`。
- 最终修改文件：
  - `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`
- 核心逻辑变化：
  - 新增 `sm120_prefill_policy_enabled()`，通过环境变量 `SGLANG_MARLIN_SM120_PREFILL_POLICY` 控制开关。
  - 新增 `is_sm120_prefill_shape(prob_m)`，当前阈值为 `M >= 64`。
  - 新增 `get_sm120_wave_cap(prob_m)`，让 prefill-heavy 形状比默认路径更早截断 wave-ratio 的加分。
  - 新增受限的 `prefill_tile_bonus`，只有在候选配置已经较好填满 GPU 且至少达到一轮有效 wave 时才生效，避免在明显欠填充的形状上盲目偏向大 tile。
  - 扩展 `log_sm120_exec_config_once(...)`，额外打印 `policy=prefill|default`，便于在 fcloud 服务器日志中确认当前走的是哪条策略。
  - 没有修改模型代码、server args、量化格式，也没有改变当前基线服务路径。

## 5）验证命令
### 正确性
```bash
cd /home/oldzhu/sglang
python3 scripts/fcloud/fcloud_workflow.py accuracy
```

### 速度
```bash
cd /home/oldzhu/sglang
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

### 在 fcloud 上重建 sgl-kernel
```bash
cd /root/sglang-minicpm/sgl-kernel
export CXX=g++ CC=gcc
export CCACHE_DIR=/root/.ccache CCACHE_MAXSIZE=10G
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
cp dist/sgl_kernel-*.whl /root/submission_sim/
cd /root/submission_sim
source prepare_env.sh
```

## 6）结果汇总
| 指标 | 基线 | 新方案 | 变化 |
|---|---:|---:|---:|
| Accuracy / overall_accuracy | 99.11% normalized（Test 12 基线族） | 95.50% normalized | -3.61 pts |
| Accuracy / ori_accuracy | 79.29%（Test 12 基线族） | 76.40% | -2.89 pts |
| S1 benchmark_duration (s) | 110.54s（Test 25B 调优基线） | 未测试 | — |
| S8 benchmark_duration (s) | 40.54s（Test 25B 调优基线） | 未测试 | — |
| S∞ benchmark_duration (s) | 33.59s（Test 25B 调优基线） | 未测试 | — |

### 6.1）fcloud 验证结果（2026-04-22）
- 远端流程已完成：`sgl-kernel` 增量重建、wheel 复制到 `/root/submission_sim`、执行 `prepare_env.sh`、重启 server、健康检查、完整准确率评测。
- 准确率结果：`ori_accuracy=76.40%`、`overall_accuracy=95.50%`，低于 97% 生存线，官方系数会变成 `C=0`。
- 失败特征：
  - `mcq=50.00%`
  - `qa=50.00%`
  - `mcq` 的平均输出长度膨胀到 `11143.1` tokens
- 结论：该变体不再继续跑速度测试。回退 CHANGE_0130，并将该启发式视为准确率不安全。

## 7）风险评估
- 准确率风险：
  低。基于同一 GPTQ 路径的 kernel dispatch 调优理论上不应显著改变输出，但任何数值路径变化仍需通过完整正确性测试验证。
- 稳定性风险：
  中。Marlin 路径是性能关键的 CUDA 代码，如果 dispatch 选择错误，可能导致 occupancy 下降或触发潜在 shape 问题。
- 可复现风险：
  低。若本次迭代只限于确定性的选择启发式与少量日志增强，则可复现性风险较低。

## 7.5）当前验证状态
- 编辑器诊断：`gptq_marlin.cu` 补丁后无错误。
- 本地主机构建：不再作为 `sgl-kernel` 的权威验证方式；按照仓库规则，CUDA wheel 的验证必须在 fcloud 上完成。
- fcloud 验证已完成：
  - 增量重建 wheel：完成
  - 用新 wheel 重启 server：完成
  - 正确性测试：失败（`76.40% / 95.50% normalized`，`C=0`）
  - 速度测试：由于不再满足提交安全性，已主动跳过
- 本地恢复状态：
  - CHANGE_0130 的 scoring heuristic 已在本地回退
  - 下一次远端动作应当是在用户重新启动 fcloud 后，对回退后的基线重新验证

## 8）回滚说明
1. 回退 `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu` 中的 CHANGE_0130 补丁，以及任何对应的 Python 侧协调改动。
2. 保持 `benchmark/soar/demo_sala/prepare_env.sh` 继续使用当前 GPTQ + FP8 KV + dense 的已验证调优参数，恢复现有基线。

## 9）下一步建议
- 不再将这一版 prefill-aware scoring heuristic 纳入提交路径。
- 在 fcloud 重新启动后，先验证回退后的基线是否回到已知安全准确率区间。
- 如果后续继续推进 SM120 GEMM，优先转向更明确的 kernel-path 调查，而不是继续做启发式 tile 偏置。
