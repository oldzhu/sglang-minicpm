# 对话 — Medusa Stage 3b：训练头已加载但 draft 永远不被接受

日期：2026-05-13 03:37 UTC
主题：CHANGE_0164 Medusa Stage 3b — 重训后 `accept_len=1.00` 的根因定位

## 用户提问
- "go" / "pls go" — 在 fcloud 上跑完 Medusa Stage 3b 全流程（采集 GPTQ 隐状态，重训头，重启服务，测速），期望 `accept_len > 1.0`。

## 智能体动作 / 结论

### 1. 三轮完整流水线
| 轮次 | Head 版本            | 训练精度 | S1 (s) | S8 (s) | Smax (s) | 平均 accept_len |
|------|----------------------|----------|--------|--------|----------|-----------------|
| v0   | Stage 3a（零初始化） | n/a      | 202.70 | 61.60  | 43.29    | 1.00            |
| v1   | 训练但标签错（当前 token） | 99.76% | 215.13 | 64.43 | 45.21 | 1.00 |
| v2   | 训练标签正确（next-token，shift 1） | 99.89% | 216.82 | 64.59 | 45.26 | 1.00 |

### 2. 离线验证 v2 head 本身是正确的
`verify_head.py` 在 dump 的隐状态上用 checkpoint 验证：
- `W1 norm = 24.125`（非零真权重）
- `match_shifted (y_pred[i] == y_next[i+1]) = 99.91%` — 头确实预测下一个 token
- `match_same (y_pred[i] == y_next[i]) = 2.27%` — 证明标签是 next-token 而非 current-token

也通过 `MedusaHeads` 模块 API 复现：
- `load_trained_weights()` 在同样的 fcloud 环境里成功，`W1 norm` 从 0 → 24.125。
- `heads(h)` 前向给出 99.91% 的 next-token 匹配。

### 3. 验证服务端加载
重训后的服务启动日志：
```
MedusaHeads: loaded 1 head(s) from /root/medusa_head_k1_gptq.pt
MedusaWorker Stage 3b: loaded trained heads from /root/medusa_head_k1_gptq.pt
MedusaWorker ready: K=1, num_heads=1, hidden=4096, dtype=torch.bfloat16, device=cuda:0, trained=True, approx_weight_MiB=32.0
```
说明运行时 `_use_trained_heads=True`，权重确实加载。

### 4. 那为什么推理时 accept rate=0？
阅读 `python/sglang/srt/speculative/ngram_info.py` 中 `NgramVerifyInput.prepare_for_verify`、`_greedy_verify`、`verify_tree_greedy`：

我们的 K=1 worker 设 `draft_token_num=1`：
- verify 前向每个请求只处理 **1 个位置**，`input_ids = draft_token`（只有 draft）。
- `target_predict = argmax(next_token_logits)` 在该单一位置 — 这是模型对 **draft 之后** token 的预测（即 p+1 位置，而非 p）。
- `candidates[0] = draft_token` 与 `target_predict[0]` 比较来决定是否接受。

这相当于在问：**"模型是否认为 draft 就是 draft 之后那个 token？"** — 这在结构上根本无法校验 draft。要判断 draft 是否正确，应该用 **bonus 位置**（消费 draft 之前的状态）的 `argmax(lm_head)` 与 draft 比对，而非消费 draft 之后的状态。

这与 **Stage 3a 在 fcloud 上也 accept_len=1.00** 的观测相符（medusa K=1 路径从来就没有产生过任何接受，即使用 n-gram draft = `output_ids[-1]`）。

### 5. 假设（下一轮验证）
- 我们 NgramVerifyInput 的 `draft_token_num` 应该是 **2**（1 个 bonus + 1 个 draft），每个请求的 `batch.input_ids = [last_accepted_token, draft]`。bonus 位置是模型从上一个已接受状态预测下一 token 的位置 — 这个预测才是用来比对 draft 的。
- 或者：K=1 medusa 路径需要一个自定义 verify，把 `argmax(post_draft_logits) == draft` 改成 `argmax(pre_draft_logits) == draft`。在 sglang spec 设计里 "pre-draft logits" 只能来自 bonus 位置的输出。

### 6. 节省成本
fcloud 实例已于 03:37 UTC 暂停 (`pause-instance` 返回 `{"status":"success"}`).

## 交叉引用
- 源码：`python/sglang/srt/speculative/medusa_worker.py`（`_forward_verify_k1` 行 283-450）
- 源码：`python/sglang/srt/speculative/ngram_info.py`（`prepare_for_verify`、`_greedy_verify`、`verify`）
- 训练器：`benchmark/soar/demo_sala/train_medusa_head.py`（`_train_from_gptq_dump`，label-shift 修复）
- 训练头：fcloud 上 `/root/medusa_head_k1_gptq.pt`（v2，32 MiB，next-token 标签，训练 top-1 = 99.89%）
- 验证脚本：本地 `/tmp/verify_head.py`、`/tmp/test_load.py` — 在 fcloud 上复现，均确认 head 正确
- 提交：`minicpm-src/mixed_minicpm_cudagraph` 上的 `c7da86d42`（CHANGE_0164 dump mode + trainer）
- 应加入 TEST_RESULTS_TRACKING.md 的一行：Test #N，2026-05-13，c7da86d42，GPTQ+FP8 KV dense Tier1 + MEDUSA K=1 + 训练头，S1=216.82s S8=64.59s Smax=45.26s accept=1.00 → 没有加速

## 结论 / 下一步
头训练流水线（dump → train → load）端到端是对的。剩余 bug 在 `MedusaWorker._forward_verify_k1` 的 **verify 接受设置**：`draft_token_num=1` 单一 draft 位置不给 verify 比对的位置。修复需要把 verify 输入改成每请求 2 个位置（bonus + draft），并让 verify 把 draft 和 `argmax(logits[bonus 位置])` 比对。这是 NgramVerifyInput 布局问题，不是 head 问题。

**下一轮建议**：
1. 研究 sglang 自带 EAGLE/Medusa verify 的 bonus + draft 位置布局（那边能跑通，模式可借鉴）。
2. 重构 `_forward_verify_k1`：每请求 `draft_token = [last_accepted, head_predicted_draft]`，`draft_token_num=2`，retrive 结构标 0 为 bonus、1 为 0 的 draft 子节点。
3. 仅在 bonus 位置 capture hidden（下一步的 draft 应基于此预测）。
4. 重新 dump → 用 bonus 位置的隐状态重训 head → 重新测试。

本轮没有得到 accept_len > 1.0 的有效结果。下一轮必须先修 verify 布局，再谈加速。
