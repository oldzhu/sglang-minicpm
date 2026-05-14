# 聊天记录 —— CHANGE_0165 第 4 轮最终评估（灾难性失败）

**会话**：2026-05-14 约 21:24
**主题**：CHANGE_0165 第 4 轮（preflight diff=0 修复）的最终 accuracy + speed 评估。MEDUSA Stage 3a K=1 passthrough + cuda-graph + torch-compile。

## 第 1 轮 —— 用户 "go"

**用户**："go"，按 CHANGE_0165 §5.6 授权最终评估管线。

**Agent 行动**：
1. `fcloud_workflow.py start-instance` → HTTP 200 "任务已启动"。
2. 轮询 `wait-server`（12×10s）→ READY。
3. `fcloud_workflow.py full` → sync（0e4634c..757ea5a 快进）+ restart-server（带 MEDUSA + cuda-graph + torch-compile + force-dense + dense-as-sparse 参数）+ accuracy eval。
4. ~8 分钟后 server health 200（cuda-graph 捕获约 6 分钟，16 个 buckets bs=[1..24]）。
5. 评估运行约 2 小时 55 分钟，完成。
6. 读取 `/root/data/outputs/20260514_101850/{summary.json,summary.txt}` → **灾难**：
   - ori_accuracy=**13.16%**（vs 79.29% 基线）→ C=0（淘汰）
   - normalized=**16.44%**（vs 99.11%）
   - duration=10488.91s（约 2 小时 55 分，相对基线 4244s 增加 147%）
   - mcq=0%（0/30），avg_out_tokens=55,816 → 失控
   - niah=6.67%、cwe=8%、fwe=27.78%、qa=23.33%
7. 检查 3 条 mcq 预测 → 重复垃圾：`</think>\n</think>\n...`、`_ _ _ _ ...`、`$\n$\n...`。
8. Decode 阶段服务器日志每个 batch 都打印 `cuda graph: False`（虽然启动时 16 个 bucket 都捕获成功），且 `accept_len=1.03–1.19, accept_rate=0.51–0.60`（kernel “以为”推测在工作）。
9. 跳过 speed 测试（明显慢于基线；C=0 无意义）。
10. `pause-instance` 第一次返回 openresty 502 HTML；**重试**成功，HTTP 200 "任务已暂停"。fcloud 已暂停。

## 产出与制品

- `TEST_RESULTS_TRACKING.md` —— 新增行 `Stage3a-preflight0-CATASTROPHIC`（commit 0e4634c1d + 757ea5a6d）。
- `CHANGE_0165_medusa_preflight_diff.en.md` / `.zh.md` —— 追加 §5.7 "Iteration 4 final eval — CATASTROPHIC FAILURE"，含完整数字、3 条样例 mcq 字符串、4 项根因假设（H1–H4）、4 项后续选项。
- fcloud 产物（pause 不会清除）：
  - `/root/data/outputs/20260514_101850/{predictions.jsonl,summary.json,summary.txt}`
  - `/tmp/dump_ngram.pkl`（52937B）、`/tmp/dump_medusa.pkl`（52938B）、`/tmp/iter4_diff.txt`（TOTAL=0）—— 预检证据

## 关键发现

**预检在 4 个边界 × 25 个字段上 0-diff 是推测解码正确性的必要而**非**充分条件。** 预检只覆盖了第 1 次 verify 迭代。多迭代漂移（GLA 状态漂移、eager-mode decode 路径差异、kernel 内 logits 索引差异）对单次迭代边界快照不可见。

## 待用户决策的开放问题

1. **回滚** commit `0e4634c1d`（iter-4 修复）并重测，以定位回归源？
2. **加强监测** —— 多迭代预检（迭代 1、2、5、10、25）和/或 kernel 内 verify tracer？
3. **彻底放弃 Stage 3a Medusa**，以 v23 = **Stage 2-cgraph**（`46553947b`，S1=118.28s，80.11%，C=1.0）作为最终提交？

推荐：先低成本尝试 (1)；若仍不通则锁定 (3) 作为提交。

## 交叉引用

- CHANGE_0165 §5.5（预检 0-diff 达成）/ §5.6（最终评估计划）/ §5.7（本次失败）。
- 最佳已知 Medusa 基线：Stage 2-cgraph（TEST_RESULTS_TRACKING 行，commit `46553947b`）。
- 整体最佳：Test 12 / v18-revert（`8d1e4d12b`，S1=110.51s，79.29%，C=1.0）。
