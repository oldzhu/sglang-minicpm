# CHAT — Medusa Stage 3b ndt=2 验证结果（验证后）

日期：2026-05-13 ~10:00 UTC
主题：在 fcloud 上验证 CHANGE_0164（ndt=2 重构）
分支：`mixed_minicpm_cudagraph`
相关文档：
- [CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.zh.md](../CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.zh.md)
- [PROPOSAL_medusa_stage3b_k1_draft_token_num_2.zh.md](../PROPOSAL_medusa_stage3b_k1_draft_token_num_2.zh.md)
- [CHAT_medusa-stage3b-distribution-mismatch_20260513_0337.zh.md](CHAT_medusa-stage3b-distribution-mismatch_20260513_0337.zh.md)
- [TEST_RESULTS_TRACKING 中 `Stage3b-ndt2-CATASTROPHIC` 行](../TEST_RESULTS_TRACKING.md)

## 用户问题（原文/精简）
1. "pls go" —— 执行 ndt=2 的 fcloud 验证流程。
2. （隐含）汇报结果；决定下一步。

## 代理动作
1. 把 commit `4b442f421`（CHANGE_0164 ndt=2 重构 + 文档）推到 `minicpm-src`。
2. `start-instance`（恢复挂起的 fcloud）。
3. `sync` —— 快进 `c7da86d4..4b442f4`。
4. `restart-server`；`wait-server` 217 s 就绪（CUDA graph capture 826 s）。
5. 服务器日志确认 `MedusaWorker Stage 3b: loaded trained heads from /root/medusa_head_k1_gptq.pt`，`trained=True, K=1, num_heads=1, hidden=4096`。
6. 07:43 UTC 启动 accuracy eval；09:55 UTC 完成（7911 s）。
7. `pause-instance`（先一次 504，再次重试成功）。

## 结果
**灾难性精度回退** —— 见 CHANGE_0164 Result 段。
- ori_accuracy = **15.13 %**，normalized = **18.92 %**，**C = 0（淘汰）**。
- mcq = **0.00 %**，平均输出 64460 tokens（吃满 max_out_len）—— 每条 MCQ 都失控。
- niah=3.33 %、cwe=15.67 %、fwe=20.0 %、qa=36.67 %。
- 评测时长 7911 s（约为基线 2.6×）。
- kernel 一路报告 `accept_len=1.46, accept_rate=0.73` —— 推测结构正常，但提交的 token 是错的。

比之前的 v23 ndt=1 静默错误（78.71 %）还要糟糕，是所有 Medusa 尝试中最差的一次。bonus 位置输出的 token 本身已被破坏（MCQ 永远输出不到 stop token）。

## 决策
**回退 CHANGE_0164。** 把 Medusa worker 从 commit `3a15a6de3`（Stage3a-force-dense 基线：78.40 % acc，S1=204.86 s）恢复回 Stage 3a（`ndt=1`），作为目前已知最稳的 Medusa 状态。立项后续调查，在重试 Stage 3b 之前先二分定位 ndt=2 verify 布局 bug。

## 待解问题 / 后续
- 在前向计算用到我们外部提供的 `positions` 之前，`NgramVerifyInput.prepare_for_verify` 是否会修改 `batch.seq_lens`？需要插桩，和能跑的 NGRAM 单步比对。
- 在 NGRAM 约定下，bonus token 的 KV 是写在 `seq_lens` 还是 `seq_lens-1`？kernel 结构上 accept 了但 commit 的是垃圾，强烈暗示 positions 或 `req_to_token` 存在 off-by-one。
- 是否应该完全放弃 ndt=2 重构，转而调查 v23 ndt=1 为什么会静默错误（–0.58 pt）？那个回退幅度小得多，可能更便宜修。

## 交叉引用
- commit `4b442f421`（ndt=2 重构）—— 待回退。
- commit `3a15a6de3`（Stage3a-force-dense）—— 回退目标。
- TEST_RESULTS_TRACKING 中 `Stage3b-ndt2-CATASTROPHIC` 行。
