# CHAT — Phase B FOS iter-6（FOS=0 消融 + tokenizer 修复）

开始：2026-05-08 16:10 本地时间。结束：2026-05-08 16:55 本地时间。

## 第 1 轮 — 用户："go"

用户触发
[CHANGE_0151_phase_b_four_over_six_004.zh.md](../CHANGE_0151_phase_b_four_over_six_004.zh.md)
的下一步计划：方案 C（NVFP4 导出持久化 tokenizer）+ 方案 A2（iter-6 = 关闭
FOS）。用户中途离线 —— 按 copilot-instructions 自主执行。

## 代理动作

1. 在
   [benchmark/soar/demo_sala/preprocess_model.py](../../../benchmark/soar/demo_sala/preprocess_model.py)
   中实现 **方案 C**：流式导出后调用
   `tokenizer.save_pretrained(str(dst))`。第一版（提交 `39c0045c5`）复用了
   函数内的 `tokenizer` 局部变量 —— 但该变量在约 1587 行已被 `del`，导致
   iter-6 量化抛出 `UnboundLocalError`。改为从 `src` 重新加载 tokenizer
   后再保存（提交 `83921b207`）修复。
2. 启动 fcloud，sync（拉到 `83921b207`）。
3. 以 FOS=0 重新量化：`SAMPLES=32 sequential MAX_CALIB_SEQ_LEN=4096
   SOAR_QUANT_PROFILE=nvfp4_fos SOAR_NVFP4_FOUR_OVER_SIX=0`。量化成功；
   日志出现 `tokenizer.save_pretrained complete`；chat-template 与
   init-rope 补丁均一次成功，无需任何手动 `cp`。
4. 以 `--quant-mode gptq` + Tier1 long-ctx 环境重启 server；wait-server
   在第二次扫描就绪（torch.compile 预热约累计 6 分钟）。
5. 跑 accuracy：**67.33%**，耗时 3163.71 s。低于 70% abort gate → 跳过
   run-2 与 speed。
6. 立刻 pause 实例（首次 504，重试成功）。

## 结果

各任务：cwe 70.00 / fwe 90.00 / mcq 43.33 / niah 90.00 / qa 43.33。
**对比 iter-5（71.24%）下滑 −3.91pt**，损失集中在 mcq（−10pt）与 qa
（−6.67pt）。FOS 在短题 / 思考-回答失控任务上具有保护作用；它**不是** iter-5
与 iter-1 之间剩余 ~1.89pt 差距的来源。

结论：**FOS 保持默认开启**；iter-5 作为当前 NVFP4-FOS 最佳配置；停止围绕
FOS flag 的进一步实验。

## 交叉引用

- [CHANGE_0151_phase_b_four_over_six_005.en.md](../CHANGE_0151_phase_b_four_over_six_005.en.md)
- [CHANGE_0151_phase_b_four_over_six_005.zh.md](../CHANGE_0151_phase_b_four_over_six_005.zh.md)
- [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md) 行 `NVFP4-FOS-6`
- 提交（`minicpm-src/mixed_minicpm_cudagraph`）：
  - `39c0045c5` preprocess(nvfp4): persist tokenizer files（首版，bug）
  - `83921b207` preprocess(nvfp4): reload tokenizer from src（修复 UnboundLocalError）
  - （本次提交）docs(0151): iter-6 NVFP4-FOS FOS=0 消融结果

## 未决项 / 后续

1. 用 run-2 + speed bench 验证 iter-5 可复现性。
2. 用 S1/S8/Smax 数据决定提交路线（iter-5 NVFP4-FOS vs 当前 GPTQ 基线）。
3. 通过 server 端 `generation_config` 缓解 mcq 失控（不修改评测脚本）。
