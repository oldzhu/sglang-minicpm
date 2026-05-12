# 聊天记录：Medusa KV 过度释放调试 —— 陈旧 K1/K2 幽灵释放根因与修复

**文件**：`CHAT_medusa-kv-overfree-debug_20260512_0800.zh.md`  
**会话日期**：2026-05-12  
**主题**：调试 CHANGE_0158/0159/0160 后持续存在的 `available_size = max + 14` 崩溃；识别陈旧 K1/K2 行根因；应用 CHANGE_0161；完成 S1/S8/Smax 速度测试。

---

## 1. 用户问题 / 需求

*（来自前一会话的绑定指令，此处转录以供追踪）：*

> "pls break the rule a little bit, I would like you working on medusa feature until it can be tested in fcloud, just keep documents detail and clear so we can review, reference and tracking the medusa implementation."

本会话是该指令的自主延续：
- CHANGE_0160 已在前一轮提交并同步；会话摘要记录服务器仍以 `available_size = max + 14` 崩溃。
- Agent 任务：找到真正根因，修复它，运行速度测试，更新文档，暂停 fcloud。

---

## 2. Agent 操作

### 2.1 诊断 —— CHANGE_0160 无效

CHANGE_0160 在 verify 后向 `req_to_token[req_pool_idx, bonus_pos]` 写零，
由 `accept_length >= 1` 保护。前一会话的服务器日志表明此保护**永远不触发**：
K=1 零初始化 Medusa head 的 `accept_length` 始终为 0。

证据：
- 指标日志：`spec_accept_length = 1.00` = `accept_length + 1 = 1` → `accept_length = 0`
- `spec_accept_rate = 0.50` = 1/2 草稿词元：根词元（始终接受）+ 1 个零初始化草稿（始终拒绝）

### 2.2 根因调查

阅读 `chunk_cache.py` 中的 `cache_finished_req`（第 51–93 行）和 `common.py` 中的
`alloc_for_decode`（第 496–560 行），并追踪：
- `prepare_for_decode()`（schedule_batch.py:1939）在 `spec_algorithm != NONE` 时提前返回
- 因此 `alloc_for_decode()` 在 MEDUSA DECODE 步骤中**从未被调用**
- `kv_committed_len` 仍然递增（通过 `scheduler_output_processor_mixin.py:306`，每步 +1）
- `cache_finished_req` 计算 `k1_total = (kv_committed_len - kernel_size) // kernel_stride + 1`
- 随着解码推进，`k1_total` 超过 prefill 阶段实际分配的 K1 插槽数

关键发现：**`req_to_sparse_k1_token` 是复用的池表**，在请求间不会清零。当 `req_pool_idx`
被新请求（prefill 比前一个请求更短）复用时，位置 `[k1_new_prefill .. k1_old_total-1]`
保留了旧请求的非零插槽 ID。CHANGE_0158 的零过滤器（`k1_indices.ne(0)`）
让这些陈旧非零 ID 通过 → 它们被释放到分配器 → `available_size` 每个陈旧 ID +1
→ 跨已完成请求的累计 +14 触发健康检查崩溃。

### 2.3 应用 CHANGE_0161

**修复**：在 `cache_finished_req` 中读取并释放 K1/K2 索引后，清零整个 K1/K2 行：

```python
self.req_to_token_pool.req_to_sparse_k1_token[req.req_pool_idx, :k1_total] = 0
# ... K2 相同
self.req_to_token_pool.req_to_sparse_k2_token[req.req_pool_idx, :k2_total] = 0
```

修改的文件（主版本和提交副本均已更新）：
- `python/sglang/srt/mem_cache/chunk_cache.py`
- `benchmark/soar/demo_sala/sglang/python/sglang/srt/mem_cache/chunk_cache.py`

提交：`5dfa9ce05` —— 已推送到 `minicpm-src`。

### 2.4 fcloud 同步与服务器重启

```
python3 scripts/fcloud/fcloud_workflow.py sync   # → 更新至 5dfa9ce05
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server  # → 175s 后就绪
```

### 2.5 速度测试 —— 全部通过

| 变体 | 时长 | 请求数 | 崩溃次数 |
|------|------|--------|---------|
| S1 | **202.96s** | 48/48 | 0 |
| S8 | **61.65s** | 72/72 | 0 |
| Smax | **43.40s** | 96/96 | 0 |

与 Stage 2 基线（提交 `46553947b`）对比：
- S1：202.96 vs 118.28（+72% 更慢）
- S8：61.65 vs 43.87（+41% 更慢）
- Smax：43.40 vs 35.75（+21% 更慢）

速度回退是**预期的**：K=1 零初始化 Medusa 增加了 verify 开销（分配 + extend + 释放），
但 `accept_length=0` 始终 → 推测解码零吞吐收益。

---

## 3. 结果

### 崩溃完全解决

所有四种崩溃模式现已修复：

| 变更 | 根因 | 由谁修复 |
|------|------|---------|
| CHANGE_0158 | 释放了从未分配的 K1/K2 插槽（MEDUSA 跳过 `alloc_for_decode`） | 零过滤 `ne(0)` |
| CHANGE_0159 | `accept_length=1` 时释放主 KV bonus 插槽 0 哨兵 | 零过滤 `ne(0)` |
| CHANGE_0160 | bonus 位置写零（**无效** —— 保护永不触发） | 已提交但无操作 |
| **CHANGE_0161** | **前一个更长请求的陈旧非零 K1/K2 ID 在池复用中存活** | **释放后清零行** |

### Stage 3a K=1 完全稳定

K=1 零初始化 heads 的 MEDUSA Stage 3a 在以下场景中稳定运行：
- 精度评估（前一轮确认 76.04%）
- 完整 S1/S8/Smax 速度基准测试套件

### Stage 3a 不是提交候选

速度比基线慢 21–72%。要从 Medusa 获益：
- **Stage 3b**：加载训练好的 MedusaHead 权重，在 prefill 后目标模型隐藏状态上运行 head 前向，预测 `num_heads` 个实际草稿词元
- 使用训练好的 heads 和 K>1，接受率 > 0 → 每个 verify 步骤接受多个词元 → 真实加速

---

## 4. 待解问题

1. **CHANGE_0160 清理**：是否应将现在无效的 `accept_length >= 1` 写零块从 `medusa_worker.py` 中移除以提高可读性？（低优先级 —— 它是无害的死代码，仅针对 K=1。）

2. **池表释放时清零约定**：是否应让 `req_to_token_pool.free()` 自动清零 K1/K2 行？目前需要调用者手动执行（CHANGE_0161 模式）。系统性修复将防止其他代码路径出现类似 bug。

3. **MEDUSA DECODE 中的 K1/K2 分配**：如果 Stage 3b（K>1）在 verify 期间需要 K1/K2 插槽，当前跳过 `alloc_for_decode` 会破坏吗？需要审计 verify 前向路径是否向 K1/K2 位置写入。

---

## 5. 交叉引用

### 创建/修改的文档
- [CHANGE_0161_medusa_stale_k1k2_zero_out.en.md](CHANGE_0161_medusa_stale_k1k2_zero_out.en.md) — 英文变更文档
- [CHANGE_0161_medusa_stale_k1k2_zero_out.zh.md](CHANGE_0161_medusa_stale_k1k2_zero_out.zh.md) — 中文变更文档
- [TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md) — 添加了 "Stage3a-cgraph" 和 "Stage3a-stable (CHANGE_0161)" 行

### 提交
- `5dfa9ce05` — CHANGE_0161：在 `cache_finished_req` 中清零陈旧 K1/K2 行

### 新增的 TEST_RESULTS_TRACKING 行
- **Stage3a-cgraph**（CHANGE_0157 精度行，76.04%，从 PLANNED 更新）
- **Stage3a-stable (CHANGE_0161)**（仅速度，S1=202.96s / S8=61.65s / Smax=43.40s）

### 相关前期变更
- CHANGE_0158：[CHANGE_0158_medusa_sparse_kv_overfree_fix.en.md](CHANGE_0158_medusa_sparse_kv_overfree_fix.en.md)
- CHANGE_0159：[CHANGE_0159_medusa_main_kv_bonus_zero_fix.en.md](CHANGE_0159_medusa_main_kv_bonus_zero_fix.en.md)
- CHANGE_0160：[CHANGE_0160_medusa_bonus_pos_garbage_fix.en.md](CHANGE_0160_medusa_bonus_pos_garbage_fix.en.md)
- CHANGE_0157：[CHANGE_0157_medusa_cuda_graph_verify_fix.en.md](CHANGE_0157_medusa_cuda_graph_verify_fix.en.md)
