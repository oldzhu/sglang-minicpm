# 提案 — Medusa Stage 3b 重构：切换到 draft_token_num=2（bonus + draft 布局）

日期：2026-05-13
作者：Copilot agent（SOAR 2026 / MiniCPM-SALA）
状态：已提案 — 实施前需用户确认

## 1. 背景

CHANGE_0164 加入了 GPTQ 对齐的 head 训练和 Medusa K=1 Stage 3b 推理。在 fcloud 上跑了三轮：

| 轮次 | Head 版本     | 训练 top-1 | S1 (s) | S8 (s) | Smax (s) | Accept |
|------|---------------|------------|--------|--------|----------|--------|
| 3a   | 零初始化       | —          | 202.70 | 61.60  | 43.29    | 1.00   |
| 3b-v1| 标签错（当前 token） | 99.76% | 215.13 | 64.43 | 45.21 | 1.00 |
| 3b-v2| 标签对（next-token） | 99.89% | 216.82 | 64.59 | 45.26 | 1.00 |

离线验证证明训练头本身是正确的：在 dump 的隐状态上跑推理侧前向（`SiLU(W1·h) + h` → `F.linear(., lm_head.weight)`），重现 99.91% next-token 精度。bug 不在 head。

## 2. 根因（来自 sgl-kernel/csrc/speculative/eagle_utils.cu::VerifyTreeGreedy 的代码验证）

`MedusaWorker._forward_verify_k1` 的 K=1 verify 路径使用 `NgramVerifyInput(draft_token_num=1)`。kernel 里：

```cpp
last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];     // root
accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx; // root 总是 accept
for (j = 1; j < num_speculative_tokens; ++j) { ... 比较子节点 ... }
// num_speculative_tokens == draft_token_num，ndt=1 时循环 0 次
predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx];
```

kernel 把 `retrive_index[0]` 视为 **root**（上一步已接受的 token，从不验证）。子节点（位置 1..ndt-1）和 `target_predict[root_position]` 比对。`draft_token_num=1` 时没有子节点 → 永远没法 accept 任何 draft。

更糟的是，当前我们传入 `input_ids = [head_predicted_draft]`（位置 seq_len 处单个 token）。模型产出 `logits[draft_pos]` = "draft 之后的 token"，被写入 `predicts[root_pos]` 并 append 到 `req.output_ids`。结果：
- **正确性 bug**：append 的是 "一个未验证 draft 之后的 token"，而非 "上一步真实输出之后的 token"。draft 错了的时候就把序列污染了。
- **速度 bug**：无论 head 多好都是 0 acceptance。

**Stage 3a 和 Stage 3b 都受此影响** — 它们共享同一个 `_forward_verify_k1` 布局。

## 3. 提议修复

### 3a. Verify 布局 — draft_token_num = 2

每个请求每步占 2 个 verify 位置：
- 位置 0（root）：`last_output` = `req.output_ids[-1]`（已知正确的 token）
- 位置 1（child / draft）：训练头预测的 `req._medusa_draft_token`（首步没有缓存时退回 `last_output`）

```python
# 每个请求贡献 2 个 token
for req in batch.reqs:
    last_out = req.output_ids[-1]
    cached = getattr(req, "_medusa_draft_token", None)
    draft = cached if (cached is not None and self._use_trained_heads) else last_out
    draft_token_list.extend([last_out, draft])
draft_tokens = torch.tensor(draft_token_list, dtype=torch.int64, device=self.device)  # (bs*2,)

# Tree: 2 节点线性链
#   retrive_index[bx]        = [0, 1]
#   retrive_next_token[bx]   = [1, -1]   (root 的第一个 child 是 idx 1；child 没有 child)
#   retrive_next_sibling[bx] = [-1, -1]
retrive_index = torch.arange(bs * 2, device=self.device, dtype=torch.int64).reshape(bs, 2)
retrive_next_token = torch.tensor([[1, -1]] * bs, ...)
retrive_next_sibling = torch.tensor([[-1, -1]] * bs, ...)

# positions: 每请求 [seq_len, seq_len+1]
positions = torch.repeat_interleave(batch.seq_lens, 2) + torch.tile(torch.arange(2), (bs,))

# tree_mask: 每个新位置 attend 该请求的全部 prefix；position 1 还 attend position 0

spec_info = NgramVerifyInput(
    draft_token=draft_tokens,           # (bs*2,)
    tree_mask=...,                       # 详见下
    positions=positions,                 # (bs*2,)
    retrive_index=retrive_index,         # (bs, 2)
    retrive_next_token=retrive_next_token,
    retrive_next_sibling=retrive_next_sibling,
    draft_token_num=2,
)
spec_info.capture_hidden_mode = CaptureHiddenMode.FULL  # 抓所有 bs*2 个位置
```

### 3b. Hidden state 捕获 — 位置 0（root），而非 last

下一步的 draft 需要从 `h(本步新加 token 的位置)` 来预测。两种情况：
- Draft 被接受：append 的 token 是 `[predicts[root]=draft, predicts[draft]=new_bonus]`。下一步 last_output = `new_bonus`，需要 `h(new_bonus 位置) = h(位置 1)`。
- Draft 被拒：append 的 token 是 `predicts[root] = 修正`（只 1 个 token）。下一步 last_output = 修正，但修正插入在 root_position，我们没它的 hidden。下一次 forward 时位置 0 的 hidden 自然就是 `h(修正)`。

更简洁的设计：**始终捕获位置 0 的 hidden** = `h(last_output)` — 模型从这个状态预测下一个 token。训练头干的事正是：给定 `h(last_output)` 预测 draft。

实现：
- 用 `CaptureHiddenMode.FULL`（捕获 `bs*2` 个位置）。
- 后处理时 `hidden_states.view(bs, 2, hidden_size)[:, 0, :]` 取每个请求的 root 位置 hidden。
- 这个 hidden 经训练头产出 **下一步** 的 draft。

也和训练一致：我们训的是 `h → next-token`，h 是想预测其下一 token 的那个位置的 hidden。

### 3c. 训练数据 — 修布局后是否要重新 dump

`/root/gptq_hidden_collect.pt` 是在老 `draft_token_num=1` 布局下采的，`input_ids = [last_output]`。那个单一位置的 hidden 也代表 h(last_output)，但 KV/attention 上下文略有不同：
- 旧：req 看到 prefix + 1 个新 token (= last_output)。`h_old(last_output)` attend 自己 + prefix。
- 新：req 看到 prefix + 2 个新 token (= [last_output, draft])。位置 0 的 `h_new(last_output)` 只 attend 自己 + prefix（与旧完全相同的上下文）。位置 1 的 hidden attend 位置 0 + prefix。

所以 `h_new[:, 0, :]` 应当 **逐 byte 等于** `h_old[:]`！训练数据无需重 dump，**只要捕获位置 0**。

廉价验证计划：修完后在小 batch 上同时跑老（ndt=1）和新（ndt=2 位置 0）的捕获，比 norm / 余弦相似度。如果一致就直接复用 head v2。

### 3d. Dump 模式调整

`_dump_hidden_buffer` 应存 FULL 捕获的位置 0，而非 LAST。在 `_forward_verify_k1` 改一行：

```python
# CaptureHiddenMode.FULL 给出 (bs*2, hidden_size)。reshape 后取位置 0 给 head/dump 用。
raw_per_req = raw_hidden_states.view(bs, 2, -1)
root_hidden = raw_per_req[:, 0, :]   # (bs, hidden_size)
```

## 4. 风险 / 影响分析

### 正确性风险
- Stage 3a（当前提交的行为）**也受 §2 描述的正确性 bug 影响**。当前提交在 Medusa 启用时可能会 append 错误 token。需要看精度影响：之前的 fcloud accuracy 跑 MEDUSA on 得到 77.87% — 在 baseline 79.29% 的 noise 范围内 — 暗示该 bug 触发频率不高（可能因为 `draft = output_ids[-1]` 让 `h(last_output)` 的 argmax 恰好不是 last_output，也可能短生成长度掩盖了失败模式）。
- 修复后 verify 路径与标准 2-token 投机一致，结构上不会污染序列。

### 速度风险
- 2-token verify 让 verify forward 处理 2 倍 token（仍 eager，无 cuda-graph）。每步 decode 多了一点常数开销。要 accept rate > 0 才回本。
- 期望 accept rate 从离线 head 99.91% 落到在线 → 大致 1.99 tokens/step → S1 从 ~200s 降到 ~100-130s（理论 2x；考虑开销后实际 1.4-1.7x）。

### 提交大小风险
- 除 `medusa_worker.py`（已经在提交里）外没有额外打包变更。Head 文件 32 MiB；只有 `prepare_env.sh` 里 SOAR_MEDUSA_HEAD_PATH 设了才会加载。竞赛提交时把 head 加入 tarball（远低于 2GB 限制）。

### 回滚
- `prepare_env.sh` 里 `SOAR_SPEC_MEDUSA=0` 关掉整条路径，与 v22 baseline byte-equivalent。

## 5. 实施计划（一个 feature iteration）

1. 重构 `python/sglang/srt/speculative/medusa_worker.py` 和 `benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py` 两份的 `_forward_verify_k1`：
   - 构造 `draft_token` 为 `(bs*2,)` 展平的 `[last_output, draft]` 序列
   - 为 2 节点线性链构造 `retrive_index`、`retrive_next_token`、`retrive_next_sibling`
   - 构造 `tree_mask`，每请求多 2 个新 token（大小 `sum(seq_len_i+2)`）
   - 构造 `positions` 为 `[seq_len_i, seq_len_i+1]`
   - 设 `draft_token_num=2`
   - 把 `capture_hidden_mode` 改为 `CaptureHiddenMode.FULL`
   - forward 后 reshape 隐状态为 `(bs, 2, hidden)` 并取 `[:, 0, :]` 给 head 和 dump 用
2. 验证训练头 v2 是否仍然适用（既然位置 0 的 hidden 等于旧布局单位置的 hidden）。若相似度 < 99% 再重 dump、重训。
3. fcloud 端到端跑：重启 server 带 v2 head → S1 小测 → 若 accept_len > 1.0，再跑 S8/Smax + accuracy。
4. 写 `CHANGE_0164_medusa_stage3b_k1_001.{en,zh}.md`（CHANGE_0164 的延续）。
5. 更新 `TEST_RESULTS_TRACKING.md`。

## 6. 估计 fcloud 成本

- 1 次 server 重启：~15 min（cuda graph capture）
- 1 次小烟测（S1 前 5 个样本）：~2 min
- 若 work，全 S1/S8/Smax + accuracy：~10 min
- 如需重训 head 再重启 1 次：~15 min
- 合计 ~30-45 min 算力

## 7. 决策请求

是否批准 §3-§5 的重构？具体：
- [ ] 批准把 `_forward_verify_k1` 的 `draft_token_num` 从 1 改为 2
- [ ] 批准把 `capture_hidden_mode` 从 `LAST` 改为 `FULL`，使用位置 0
- [ ] 批准若位置 0 与旧 dump 的相似度 ≥ 99% 则复用现有 head v2
- [ ] 批准本轮 ~30-45 min 的 fcloud 成本
