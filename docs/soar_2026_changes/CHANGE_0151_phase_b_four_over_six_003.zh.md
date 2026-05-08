# CHANGE 0151 — Phase B FourOverSix，续篇 003

承接 [CHANGE_0151_phase_b_four_over_six_002.zh.md](CHANGE_0151_phase_b_four_over_six_002.zh.md)。

iter-4 在保持标定 *内容*（stratified 90 qa,mcq,cwe，`FOS=1`，seed=20260320）
和 Tier1 调度不变的前提下，单独隔离 **标定序列长度** 这一变量。

## 背景

| 迭代 | 标定内容 | calib_seq_len | 调度 | 平均 ori | Δ vs iter-1 |
|------|----------|---------------|------|----------|-------------|
| 1    | sequential 32 (混合) | **4096** | Tier1 | ~73.13% | — |
| 2    | stratified 90 qa,mcq,cwe FOS=1 | **16384** | 保守 | ~62.02% | −11pt |
| 3    | stratified 90 qa,mcq,cwe FOS=1 | **16384** | Tier1 | 68.20% | −5pt |
| 4    | stratified 90 qa,mcq,cwe FOS=1 | **4096** | Tier1 | **66.00%** | **−7.13pt** |

iter-3 显示仅切回 Tier1 调度可恢复约 6pt 差距；剩余 ~5pt 当时被归因为
"标定内容和/或序列长度"。iter-4 固定内容、把序列长度对齐到 iter-1（4096），
单独验证序列长度的贡献。

## 实施

本次迭代不修改源代码，仅调用层环境变量覆盖。

fcloud 上重新量化：

```bash
python3 scripts/fcloud/fcloud_exec.py exec \
  'cd /root/submission_sim && \
   SOAR_QUANT_PROFILE=nvfp4_fos \
   SOAR_NVFP4_FOUR_OVER_SIX=1 \
   SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096 \
   SOAR_GPTQ_CALIBRATION_SAMPLES=90 \
   SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,mcq,cwe \
   SOAR_GPTQ_CALIBRATION_SAMPLING=stratified \
   SOAR_GPTQ_CALIBRATION_SEED=20260320 \
   python3 -u preprocess_model.py \
     --input /root/models/openbmb/MiniCPM-SALA \
     --output /root/models/MiniCPM-SALA-NVFP4-FOS \
     --mode nvfp4'
```

本次同时验证了新加入的 **`_init_rope` 上游补丁**（CHANGE_0152）：

```
[preprocess][init-rope-patch] mode=nvfp4 dst:
patched /root/models/MiniCPM-SALA-NVFP4-FOS/modeling_minicpm_sala.py
(replaced 2 _init_rope headers, 2 else-branches)
```

`grep -c "transformers>=4.43 standardizes rope_scaling" .../modeling_minicpm_sala.py`
返回 **2**，符合预期。

iter-4 服务端启动（与 iter-3 完全一致）：

```bash
python3 scripts/fcloud/fcloud_workflow.py restart-server \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --env SOAR_QUANT_PROFILE=nvfp4_fos \
  --env SOAR_NVFP4_FOUR_OVER_SIX=1 \
  --env SOAR_TIER1_LONG_CONTEXT=1 \
  --env SOAR_TORCH_COMPILE_MAX_BS=24
```

`/get_server_info` 在线确认 iter-3 系列调度配置（chunk=65536，
prefill_max_req=4，sched_cons=0.8，max_run=24，modelopt_fp4，KV fp8_e5m2）。

## 预设中止门

承接 iter-3 的用户指令：若 run-1 ori_accuracy < 70%，则跳过 run-2 与速度
基准、暂停实例、记录、决策下一步方向。

## 结果 — iter-4 run-1

`outputs/20260508_043812/predictions.jsonl`

| 指标 | 数值 |
|------|------|
| ori_accuracy（Average Score） | **66.00%** |
| cwe | 66.67% |
| fwe | 80.00% |
| mcq | 46.67% |
| niah | 93.33% |
| qa | 43.33% |
| Total Duration | 3084.38 s |
| Output TPS | 348.97 |
| FOS pct_m4 | 43.14%（与 iter-2/3 一致 — 仅取决于权重） |

**触发中止门。** run-2 与速度基准跳过；实例已暂停。

## 对比

| 指标 | iter-1（均值） | iter-2（均值） | iter-3 run-1 | **iter-4 run-1** |
|------|---------------|---------------|--------------|------------------|
| ori_accuracy | ~73.13% | ~62.02% | 68.20% | **66.00%** |
| Δ vs iter-1 | — | −11pt | −5pt | **−7.13pt** |
| cwe | (n/a) | ~67.3 | 77.67 | 66.67 |
| fwe | (n/a) | ~76.1 | 76.67 | 80.00 |
| mcq | (n/a) | ~51.7 | 56.67 | **46.67** |
| niah | (n/a) | ~71.7 | 80.00 | **93.33** |
| qa | (n/a) | ~43.3 | 50.00 | 43.33 |

把 calib_seq_len 从 16384 改回 4096，并保持 iter-2 的 stratified 内容，**未能**
恢复 iter-1 的精度，反而比 iter-3 还低 2.20pt。

## 结论

iter-{2,3,4} 与 iter-1 之间剩余约 5–7pt 的差距，**不能** 归因于标定序列长度。
在 Tier1 调度固定、seqlen 与 iter-1 对齐的条件下，平均仍比 iter-1 低 ~7pt。

把 iter-1（73%）与 iter-{2,3,4}（62–68%）区分开的主导变量，是 **标定内容**：
- iter-1：`SOAR_GPTQ_CALIBRATION_SAMPLES=32` 顺序混合（默认任务分布，5 类全在）
- iter-{2,3,4}：stratified 90，`TASK_INCLUDE=qa,mcq,cwe`，`FOS=1`（仅 3/5 类，
  且按 four-over-six 评分挑选样本）

为何 iter-2/3/4 标定更差，几个候选假设：

1. **任务失衡**：把 `niah` 与 `fwe` 从标定中剔除，会让这两类任务的激活分布在
   量化时缺乏代表样本。iter-4 印证：niah=93.33%（在 Tier1 下表现极佳）
   但 mcq/qa 大幅下滑。
2. **FOS 样本偏倚**：`FOUR_OVER_SIX=1` 倾向选 token 统计利于 M=4 尾数刻度的
   标定样本，这可能让长尾激活（mcq/qa）覆盖不足。
3. **序列长度分布差异**：stratified 90 的长度桶分布即使在同一 seqlen 上限下，
   仍可能与 sequential 32 显著不同。

## 下一步决策

FOS 暂停。前向有三条候选路径：

| 选项 | 说明 | 成本 | 预期结果 |
|------|------|------|----------|
| A | 重量化：`SAMPLES=32` sequential（不带 `TASK_INCLUDE`、不开 `FOS_SCORE`）— 完全复刻 iter-1 标定 | 1 轮 fcloud（~30 分钟量化 + 1× 精度 ~50 分钟） | 若恢复 iter-1 精度 → 印证假设；否则有其它隐藏变量 |
| B | 重量化：`SAMPLES=90` stratified 跨 **5 类全任务**（去掉 `TASK_INCLUDE` 滤镜，关 `FOS=0`） | 1 轮 fcloud | 检验任务失衡单变量是否就是回归源 |
| C | 完全放弃 FOS → 切回纯 NVFP4（无 four-over-six 补丁）+ 走 Phase A 的 W4A8 路线 | 较大重构 | 切换到另一条优化轨道 |

建议优先做 **选项 A**：成本最低，能直接验证 "内容假设"。若 A 能恢复
iter-1（~73%），再做选项 B（5 类任务全覆盖）通常会进一步增益。

## 验证命令

```bash
# 确认 _init_rope 补丁已落到重量化模型目录
grep -c "transformers>=4.43 standardizes rope_scaling" \
  /root/models/MiniCPM-SALA-NVFP4-FOS/modeling_minicpm_sala.py
# 期望：2

# 重跑精度
python3 scripts/fcloud/fcloud_workflow.py accuracy \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS
```

## 回滚

本迭代未改源代码。若要回滚模型产物，去掉
`SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096`（默认 16384）即可恢复 iter-3 行为。

## 交叉引用

- 续篇 002：[CHANGE_0151_phase_b_four_over_six_002.zh.md](CHANGE_0151_phase_b_four_over_six_002.zh.md)
- `_init_rope` 补丁：[CHANGE_0152_init_rope_transformers5_compat.zh.md](CHANGE_0152_init_rope_transformers5_compat.zh.md)
- 测试行：TEST_RESULTS_TRACKING.md → NVFP4-FOS-4
- 对话记录：chat/CHAT_phase-b-fos-iter4_20260508_1230.zh.md
