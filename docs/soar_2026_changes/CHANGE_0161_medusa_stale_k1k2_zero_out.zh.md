# CHANGE_0161：在 `cache_finished_req` 中清零陈旧 K1/K2 行，修复 `+14` 幽灵释放崩溃

**变更 ID**：CHANGE_0161  
**日期**：2026-05-12  
**提交**：`5dfa9ce05`  
**分支**：`mixed_minicpm_cudagraph`  
**作者**：AI Agent（SOAR 2026 Medusa 调试）  
**修改文件**：  
- `python/sglang/srt/mem_cache/chunk_cache.py`  
- `benchmark/soar/demo_sala/sglang/python/sglang/srt/mem_cache/chunk_cache.py`  

---

## 背景与动机

在 CHANGE_0158（K1/K2 稀疏零过滤）和 CHANGE_0159（主 KV 零过滤）之后，服务器
仍然持续崩溃，错误信息为 `available_size = max_total_num_tokens + 14`：

```
ValueError: token_to_kv_pool_allocator memory leak detected!
self.max_total_num_tokens=15563319, available_size=15563333, evictable_size=0, protected_size=0
```

差值始终精确为 **+14**，在热身 + 3 次速度测试请求（输入 1918、2760、2304 词元）后出现，
在第 4 次请求（1196 输入词元）时崩溃。

CHANGE_0160 试图在 verify 后对 bonus 位置执行零写操作，但
**无效**，因为 K=1 零初始化 Medusa head 的 `accept_length` 始终为 0，
guard `accept_length >= 1` 永远不会触发。

本变更找出并修复了 +14 幽灵释放的**真正根因**。

---

## 合规声明

- 纯防御性记账操作：读取后清零插槽 ID 数组，不影响任何模型输出、量化或注意力计算。
- 在关键解码路径上不增加新的分配或张量运算。
- 零写操作时间复杂度为 O(k1_total)，每次请求完成时执行，开销可忽略不计（k1_total < seq_len/stride）。
- 不修改 SOAR 提交约束排除的任何文件。

---

## 根因分析

### 关键事实

1. **MiniCPM-SALA 的 page_size = 1**（`prepare_env.sh` 中无 `--page-size` 参数，默认为 1）。
   使用 `TokenToKVPoolAllocator`，其中 `available_size() = len(free_pages)`。
   `free()` 直接追加，无去重、无范围检查。

2. **MEDUSA DECODE 跳过 `prepare_for_decode`**  
   `prepare_for_decode()`（schedule_batch.py）在 `spec_algorithm != NONE` 时提前返回。
   因此 `alloc_for_decode()` 永远不会被调用 → 在解码步骤中不会向
   `req_to_sparse_k1_token[pool_idx, :]` 写入新的 K1/K2 插槽。

3. **`kv_committed_len` 仍然每步递增**  
   `scheduler_output_processor_mixin.py:306` 执行 `kv_committed_len += accept_lens[i]`
   （K=1 MEDUSA 每步 = 1，因为 accept_length=0 → accept_length+1=1 → 1 个新提交词元）。

4. **`cache_finished_req` 使用 `kv_committed_len` 计算 `k1_total`**  
   ```python
   k1_total = (kv_committed_len - kernel_size) // kernel_stride + 1
              if kv_committed_len >= kernel_size else 0
   ```
   随着解码推进，`kv_committed_len` 增大 → `k1_total` 超过 prefill 阶段实际分配的 K1 插槽数。

5. **K1 行仍持有来自**前一个请求**的非零陈旧插槽 ID**  
   `req_to_sparse_k1_token` 是复用的池表。当 `req_pool_idx` 被新请求复用，
   而该新请求的 prefill 比前一个请求更短时，位置
   `[k1_prev_prefill .. k1_prev_total-1]` 保留了旧请求的非零插槽 ID。

6. **CHANGE_0158 的零过滤被绕过**  
   CHANGE_0158 过滤 `k1_indices[k1_indices.ne(0)]`。这些来自前一个请求的
   陈旧非零 ID 通过了过滤器，被释放到分配器中。
   每个落入 `free_pages` 的陈旧非零 ID 使 `available_size` 增加 +1。

### 为何恰好 +14？

+14 在 4 次请求模式中累积：
- 崩溃前有 4 次请求（热身 + 3 次速度测试请求）
- 每次请求完成后调用 `cache_finished_req`
- 根据每次新请求复用哪个 `pool_idx`，以及该 `pool_idx` 上前一个请求的 prefill 有多长，
  会遇到 1 个到若干个陈旧 K1 位置
- 健康检查触发时，所有已完成请求的累计总计为 +14

---

## 实现方案（设计）

修复极其精简：**在 `cache_finished_req` 中读取并释放 K1/K2 索引后，清零整个 K1（和 K2）行**，
这样复用同一 `pool_idx` 的未来请求，在其自身 `k1_prefill` 之外的位置看到的都是零，
使 CHANGE_0158 的零过滤器再次生效。

这类似于 `req_to_token_pool.free(req_pool_idx)` 在语义上将请求插槽标记为空闲——
我们对稀疏 K1/K2 数组需要同样的"释放时清零"语义。

---

## 实际代码变更

### 变更前（仅 CHANGE_0158，仍然崩溃）

```python
k1_total = (kv_committed_len - kernel_size) // kernel_stride + 1 if kv_committed_len >= kernel_size else 0
if k1_total > 0:
    k1_indices = self.req_to_token_pool.req_to_sparse_k1_token[req.req_pool_idx, :k1_total]
    k1_indices_valid = k1_indices[k1_indices.ne(0)].to(torch.int64)
    if k1_indices_valid.numel() > 0:
        self.token_to_kv_pool_allocator.free(k1_indices_valid)
    # [k1_prefill..k1_total-1] 处的陈旧值存活 → 下次复用时幽灵释放！

k2_total = ...
if k2_total > 0:
    k2_indices = self.req_to_token_pool.req_to_sparse_k2_token[req.req_pool_idx, :k2_total]
    k2_indices_valid = k2_indices[k2_indices.ne(0)].to(torch.int64)
    if k2_indices_valid.numel() > 0:
        self.token_to_kv_pool_allocator.free(k2_indices_valid)
    # K2 存在相同的陈旧问题
```

### 变更后（CHANGE_0161）

```python
k1_total = (kv_committed_len - kernel_size) // kernel_stride + 1 if kv_committed_len >= kernel_size else 0
if k1_total > 0:
    k1_indices = self.req_to_token_pool.req_to_sparse_k1_token[req.req_pool_idx, :k1_total]
    k1_indices_valid = k1_indices[k1_indices.ne(0)].to(torch.int64)
    if k1_indices_valid.numel() > 0:
        self.token_to_kv_pool_allocator.free(k1_indices_valid)
    # CHANGE_0161: 清零 K1 行，使该请求的陈旧非零值不会在
    # 使用更小 k1_prefill 的未来请求复用同一 pool_idx 时被幽灵释放。
    self.req_to_token_pool.req_to_sparse_k1_token[req.req_pool_idx, :k1_total] = 0

k2_total = ...
if k2_total > 0:
    k2_indices = self.req_to_token_pool.req_to_sparse_k2_token[req.req_pool_idx, :k2_total]
    k2_indices_valid = k2_indices[k2_indices.ne(0)].to(torch.int64)
    if k2_indices_valid.numel() > 0:
        self.token_to_kv_pool_allocator.free(k2_indices_valid)
    # CHANGE_0161: K2 相同的陈旧行清零。
    self.req_to_token_pool.req_to_sparse_k2_token[req.req_pool_idx, :k2_total] = 0
```

---

## 验证命令

```bash
# 同步到 fcloud 并重启服务器
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server

# 运行速度测试 — 验证无崩溃（所有请求完成）
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax
```

成功标准：
- 所有 48 / 72 / 96 次请求完成（无 `memory leak detected` 崩溃）
- 服务器不触发 `ValueError: available_size > max_total_num_tokens`

---

## 结果汇总

| 指标 | CHANGE_0161 前 | CHANGE_0161 后 |
|------|---------------|---------------|
| S1 在第 4 次请求后崩溃 | 是（available_size=max+14） | 否 |
| S8 崩溃 | 是 | 否 |
| Smax 崩溃 | 是 | 否 |
| S1 时长 | N/A（已崩溃） | **202.96s** |
| S8 时长 | N/A（已崩溃） | **61.65s** |
| Smax 时长 | N/A（已崩溃） | **43.40s** |
| S1 对比 Stage 2 基线（118.28s） | — | **+72% 更慢** |
| S8 对比 Stage 2 基线（43.87s） | — | **+41% 更慢** |
| Smax 对比 Stage 2 基线（35.75s） | — | **+21% 更慢** |

**速度回退说明**：K=1 零初始化 heads 的 Stage 3a 比 Stage 2 透传基线**更慢**。
这是预期的：每个解码步骤现在都为草稿和 verify 词元分配插槽，运行 `forward_extend()` 进行验证，
然后释放被拒绝的插槽。由于 `accept_length=0` 始终（零初始化 heads 不接受任何根词元以外的词元），
所有开销都被付出但没有任何吞吐收益。
需要 Stage 3b（训练好的 heads，K>1，真实接受率 > 0）才能超过基线速度。

---

## 与前几次修复的关系

| 变更 | 解决的根因 | 状态 |
|------|----------|------|
| CHANGE_0158 | `cache_finished_req` 释放的 K1/K2 稀疏插槽从未被分配（MEDUSA 跳过 `alloc_for_decode`） | 已修复（零过滤） |
| CHANGE_0159 | 当 `accept_length=1` 使 `kv_committed_len` 超过 prefill 时，主 KV bonus 词元位置（插槽 0 哨兵）被释放 | 已修复（零过滤） |
| CHANGE_0160 | verify 后对 bonus 位置写零（**无效**—— accept_length 始终为 0） | 已提交但无操作 |
| **CHANGE_0161** | **前一个更长请求的陈旧非零 K1/K2 插槽 ID 在池复用中存活** | **已修复（清零行）** |

---

## 回滚说明

```bash
# 回滚两个文件
git diff HEAD~1 python/sglang/srt/mem_cache/chunk_cache.py
git diff HEAD~1 benchmark/soar/demo_sala/sglang/python/sglang/srt/mem_cache/chunk_cache.py
git revert 5dfa9ce05
git push minicpm-src mixed_minicpm_cudagraph
```

---

## 后续建议

1. **Stage 3b**：实现真正的 MEDUSA head 前向传播，预测 `num_heads` 个草稿词元。
   用实际 head 前向调用替换 `_forward_generate_k1` 中的零初始化草稿。
   需要加载训练好的 MedusaHead 权重，并在 prefill 后目标模型的隐藏状态上运行它们。

2. **MEDUSA DECODE 中的 K1/K2 分配**：考虑 MEDUSA DECODE 是否应该实际
   为 K1/K2 插槽调用 `alloc_for_decode`（与常规解码一样）。目前因为
   `prepare_for_decode` 提前返回而跳过了它。如果 K>1 verify 需要 K1/K2 插槽，
   则需要重新启用或单独路由。

3. **池表释放时清零约定**：考虑在 `req_to_token_pool.free()` 中添加 `clear_on_free` 标志，
   自动清零稀疏 K1/K2 行，以防止其他代码路径出现类似 bug。
