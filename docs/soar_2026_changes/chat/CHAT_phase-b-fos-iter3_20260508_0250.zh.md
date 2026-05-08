# 对话记录 — Phase B FOS 迭代 3（在迭代 2 ckpt 上做调度 A/B）
日期：2026-05-08 02:50

## 用户问题（本轮）

1. "let us retest NVFP4-FOS iter 1 2 times again for accuracy and speed
   benchmark before make decision" —— 推翻"放弃 FOS"的早期结论。
2. 进一步收紧："for `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096`, I would like to keep
   it the same as iter 2"（即保持 16384）—— 复用迭代 2 的 ckpt，只回退调度。
3. 设置中止门限："if 1st time accuracy is lower than 70%, then let us stop the
   instance and set `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096` as iter1 and then
   requant to test."

## Agent 回答 / 操作

- 修改 `benchmark/soar/demo_sala/prepare_env.sh` 第 61 行：把硬编码的
  `export SOAR_TIER1_LONG_CONTEXT=0` 改为
  `export SOAR_TIER1_LONG_CONTEXT="${SOAR_TIER1_LONG_CONTEXT:-0}"`，保持
  迭代 2 默认值（0）不变，但允许调用方覆盖。
- 提交 `fb6ee34d8` 并推送 `minicpm-src/mixed_minicpm_cudagraph`。
- 校验控制台 JWT（剩余 5.82 天），恢复实例。
- 发现 fcloud 上 `/root/sglang-minicpm` 不是 git clone → `sync` 走的是
  从陈旧快照树强制覆盖的回退路径，补丁未上去。改为直接用 base64 经
  `fcloud_exec` 把补丁后的 `prepare_env.sh` 写入 `/root/submission_sim/`。
- 用迭代 3 环境变量重启服务端：
  `SOAR_QUANT_PROFILE=nvfp4_fos SOAR_NVFP4_FOUR_OVER_SIX=1`
  `SOAR_TIER1_LONG_CONTEXT=1 SOAR_TORCH_COMPILE_MAX_BS=24`。
  实测 `/get_server_info`：chunk=65536，prefill_max_req=4，sched_cons=0.8，
  max_run=24，quant=modelopt_fp4，kv=fp8_e5m2。
- 跑精度第 1 轮（迭代 2 ckpt + 迭代 1 调度）：ori = **68.20%**
  （cwe 77.67 / fwe 76.67 / mcq 56.67 / niah 80.00 / qa 50.00）。
- 触发中止门限（<70%）。跳过第 2 轮和速度测试。
- 经控制台 API 暂停实例。

## 产出

- 仅恢复 Tier1 调度即可挽回迭代 1→迭代 2 回归中 ~6pt（60.73% → 68.20%），
  但不足以单独解决问题 —— 距离迭代 1 仍差 5pt。
- 校准内容/seqlen 与调度策略都会显著影响 FOS 精度。
- 下一步（按用户预设方案）：迭代 4 = 用
  `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096` 重量化，保持迭代 2 的"分层抽样
  90 条 qa,mcq,cwe"；以迭代 3 的服务端配置测试。需用户确认后启动
  实例 + 约 25 分钟量化。

## 关联引用

- 补丁提交：`fb6ee34d8`。
- 文档：[CHANGE_0151_phase_b_four_over_six_002.zh.md](../CHANGE_0151_phase_b_four_over_six_002.zh.md)
- 测试登记：TEST_RESULTS_TRACKING.md → NVFP4-FOS-3。
- 预测结果：fcloud `/root/data/outputs/20260508_025542/predictions.jsonl`。
- 记忆：`/memories/session/phase_b_fos_result.md`（已追加迭代 3 章节）。
