# 对话记录 — Phase B FOS iter-4（2026-05-08）

## 第 1 轮 — 用户要求先检查"已打补丁"情况

**用户**："pls also check before try to do pacth as some environment may altready pacthed."

**Agent**：检查 `benchmark/soar/demo_sala/preprocess_model.py` 中
`_patch_modeling_init_rope_inplace`，已经具备 3 层幂等保护：
1. 标记位检查：文件包含 `"transformers>=4.43 standardizes rope_scaling"` → 跳过；
2. 模式计数保护：`header_count==0` 或 `else_count==0` → 跳过；
3. 无操作保护：替换后文本 == 原文 → 跳过。

由于标记位与上游注释字节一致，"自己打过的"和"上游已打过的"两种情况都能命中。
无需改代码。

## 第 2 轮 — 用户批准 iter-4

**用户**："ok pls go"

**Agent 执行**：

1. 检查 console JWT，有效期剩 5.74 天。✅
2. `start-instance`（HTTP 200）；约 3 分钟后 JupyterLab 起来。✅
3. `sync` — fcloud 上 `/root/sglang-minicpm` 不是 git 仓库，diff 失败走
   force-copy 回退；上传后 `/root/submission_sim/preprocess_model.py` 中
   标记位数 = **0**（旧 tar 解出来的树覆盖了我们的版本）。❌
4. 手动用 `base64 | base64 -d` 通过 `fcloud_exec` 上传补丁后的
   `preprocess_model.py` 与 `prepare_env.sh`；标记位数 = **2**。✅
5. 用以下参数重新量化 NVFP4-FOS：
   - `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096`
   - stratified 90 qa,mcq,cwe + FOS=1 + seed=20260320
   - 直接调 `preprocess_model.py --mode nvfp4`
   进程 PID 978，<3 分钟完成。标定 90/90，streaming 导出 224 个 Linear，
   FOS pct_m4=43.14%。`_init_rope` 补丁触发（markers=2）。✅
6. `restart-server` 带 iter-1 Tier1 环境（`SOAR_TIER1_LONG_CONTEXT=1`，
   `SOAR_TORCH_COMPILE_MAX_BS=24`）。`wait-server` 第一次 300 s 超时
   （torch.compile autotune），第二次 59 s 通过。✅
7. `accuracy` run-1：3084.38 s 完成。

   **结果**：ori_accuracy = **66.00%** — 低于 70% 中止门。

   - cwe: 66.67（vs iter-3 77.67）
   - fwe: 80.00（vs iter-3 76.67）
   - mcq: **46.67**（vs iter-3 56.67）— −10pt
   - niah: 93.33（vs iter-3 80.00）
   - qa: 43.33（vs iter-3 50.00）

8. **触发中止门。** 跳过 run-2 与速度基准。
9. `pause-instance` 第一次 504，重试 200 OK，实例已暂停。✅

## 结论

把标定 seqlen 从 16384 改回 4096、保持 iter-2 stratified 内容，**未能** 恢复
iter-1 的精度。iter-4（66.00%）相比 iter-3（68.20%）反而下降 2.20pt，
比 iter-1（~73.13%）低 7.13pt。

**判定**：相比 iter-1 的回归，主导变量是标定 *内容*（qa,mcq,cwe stratified 90
+ FOS=1），**不是** seqlen。FOS 暂停。

## 交叉引用

- 新增变更文档：[CHANGE_0151_phase_b_four_over_six_003.en.md](../CHANGE_0151_phase_b_four_over_six_003.en.md)，[CHANGE_0151_phase_b_four_over_six_003.zh.md](../CHANGE_0151_phase_b_four_over_six_003.zh.md)
- 测试追踪行：NVFP4-FOS-4
- 预测输出：`/root/submission_sim/outputs/20260508_043812/predictions.jsonl`
- 量化日志：`/root/logs/iter4_quant.log`（在 fcloud）

## 建议下一步（待用户批准）

**选项 A** — 用 `SAMPLES=32` sequential（不带 `TASK_INCLUDE`、关 `FOS_SCORE`）
重量化，复刻 iter-1 标定。这是验证"标定内容假设"成本最低的一步。
