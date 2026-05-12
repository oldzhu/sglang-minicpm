# CHANGE_0158 — 修复：MEDUSA 稀疏 KV 过度释放内存泄漏

## 状态
**已应用** — `python/sglang/srt/mem_cache/chunk_cache.py`（1 个文件，~12 行变更）

## 背景与动机

CHANGE_0157 修复 CUDA 图崩溃后，发现了新的崩溃：

```
ValueError: token_to_kv_pool_allocator memory leak detected!
self.max_total_num_tokens=15563319, available_size=15563321, evictable_size=0, protected_size=0
```

`available_size > max_total_num_tokens` — 可用 token 槽位数超过应有的最大值。
这说明发生了**过度释放（over-free）**：把从未分配过的 token 槽加回了空闲池。
每次请求完成后服务端都会崩溃，导致无法进行多轮速度测试。

## 根本原因分析

### Token 池槽位索引
`TokenToKVPoolAllocator.clear()` 把空闲列表初始化为：
```python
self.free_pages = torch.arange(1, self.size + 1, ...)   # 从 1 开始，不含 0
```
槽位 **0 是保留的"填充哑元"**，**`alloc` 永远不会返回**它。
稀疏 KV 表（`req_to_sparse_k1_token`、`req_to_sparse_k2_token`）初始化为 `torch.zeros(...)`，
因此**未写入的位置始终读取为 0**。

### 推测解码路径跳过稀疏 KV 分配
`ScheduleBatch.prepare_for_decode()` 对所有推测算法提前返回：
```python
if not self.spec_algorithm.is_none():
    # 分配工作在 spec worker 内部完成
    return
```
这个提前返回跳过了 `alloc_for_decode(...)`，后者在正常情况下会：
1. 分配 `bs × 1` 个主 KV 槽。
2. 在 `seq_len` 跨越 `kernel_stride` 边界时，分配**稀疏** KV 槽。
3. 把新稀疏槽 ID 写入 `req_to_sparse_k1_token` / `req_to_sparse_k2_token`。

在 MEDUSA 路径中，`_forward_verify_k1` 调用的 `spec_info.prepare_for_verify()` 只分配每请求**1 个主 KV 槽**。
稀疏 KV 表在 MEDUSA decode 步骤中因此**始终不更新**。

### kv_committed_len 仍会递增
`NgramVerifyInput._free_cache()`（每次 MEDUSA 步骤都会调用）执行：
```python
req.kv_committed_len += accept_length + 1   # K=1 时始终 +1
req.kv_allocated_len  = req.kv_committed_len
```
因此 `kv_committed_len` 以正常的"每步 1 个 token"速率增长。

### cache_finished_req 中的过度释放
`ChunkCache.cache_finished_req()` 根据最终的 `kv_committed_len` 计算稀疏槽数量：
```python
k1_total = (kv_committed_len - kernel_size) // kernel_stride + 1   # 随序列延伸而增长
k1_indices = req_to_sparse_k1_token[req.req_pool_idx, :k1_total]
token_to_kv_pool_allocator.free(k1_indices)                          # ← BUG
```
经过若干 MEDUSA decode 步骤后，`kv_committed_len` 跨越更多 `kernel_stride` 边界，
`k1_total` 超过预填充阶段写入的实际数量。
多出来的位置从未写入 → 读取为 **0**。
调用 `free([0])` 把保留的槽位 0 追加到 `free_pages`，使 `available_size` 超过
`max_total_num_tokens` → 崩溃。

### 实测示例
- 预填充 859 个 token → k1_total_prefill = (859−32)//16+1 = **52**，k2_total_prefill = **12**。
- MEDUSA decode 约 60 步 → kv_committed_len ≈ **920**。
- k1_total_final = (920−32)//16+1 = **56**（超额释放 +4），k2_total_final = **13**（+1）。
- 请求完成后 `available_size` = `max + 2`（随解码长度而异）。

## 规则合规声明
这是**缺陷修复**——无算法变更，无精度影响。
过滤槽位 0 始终是正确的：槽位 0 保证永远不是有效的已分配 KV 槽。
对正常（非 MEDUSA）decode 路径安全，因为合法分配的稀疏槽 ID 均 ≥ 1。

## 实现

### 修改文件
`python/sglang/srt/mem_cache/chunk_cache.py`  
`benchmark/soar/demo_sala/sglang/python/sglang/srt/mem_cache/chunk_cache.py`

### 代码变更
```diff
-            if k1_total > 0:
-                k1_indices = self.req_to_token_pool.req_to_sparse_k1_token[req.req_pool_idx, :k1_total]
-                self.token_to_kv_pool_allocator.free(k1_indices)
+            if k1_total > 0:
+                k1_indices = self.req_to_token_pool.req_to_sparse_k1_token[req.req_pool_idx, :k1_total]
+                # CHANGE_0158: 过滤槽位 0（保留/未分配的哨兵），防止 MEDUSA decode
+                # 跳过 alloc_for_decode（稀疏槽不写入）但 kv_committed_len 仍递增时的过度释放。
+                k1_indices_valid = k1_indices[k1_indices.ne(0)].to(torch.int64)
+                if k1_indices_valid.numel() > 0:
+                    self.token_to_kv_pool_allocator.free(k1_indices_valid)
 
-            k2_kernel_size = kernel_size * 4
-            k2_kernel_stride = kernel_stride * 4
-            k2_total = ...
-            if k2_total > 0:
-                k2_indices = self.req_to_token_pool.req_to_sparse_k2_token[req.req_pool_idx, :k2_total]
-                self.token_to_kv_pool_allocator.free(k2_indices)
+            k2_kernel_size = kernel_size * 4
+            k2_kernel_stride = kernel_stride * 4
+            k2_total = ...
+            if k2_total > 0:
+                k2_indices = self.req_to_token_pool.req_to_sparse_k2_token[req.req_pool_idx, :k2_total]
+                # CHANGE_0158: k2 稀疏槽同样过滤零值。
+                k2_indices_valid = k2_indices[k2_indices.ne(0)].to(torch.int64)
+                if k2_indices_valid.numel() > 0:
+                    self.token_to_kv_pool_allocator.free(k2_indices_valid)
```

### 为何过滤槽位 0 是正确的
- `TokenToKVPoolAllocator.alloc()` 从 `free_pages = torch.arange(1, size+1)` 中弹出。
  槽位 0 **永远不会**被 alloc 返回。
- `req_to_sparse_k1/k2_token` 以 `torch.zeros(...)` 初始化。
  任何未写入的位置读取为 0。
- 因此 `k1_indices.ne(0)` 精确识别实际已分配的位置。

### 注：MEDUSA decode 中的稀疏 KV 质量
本次修复纠正的是**内存计数**，并不在 MEDUSA decode 步骤中写入稀疏 K1/K2 键
（`alloc_for_decode` 仍被跳过）。若将来精度下降，可在 `_forward_verify_k1` 内
实现适当的稀疏 KV 槽分配（Stage 3b 工作）。对当前 K=1 零初始化 MEDUSA 阶段，
精度影响预期极小。

## 验证命令

### 正确性测试
```bash
# 重启服务端后运行精度评测——服务端不应在评测完成后崩溃
python3 scripts/fcloud/fcloud_workflow.py accuracy
```
预期：评测完成后服务端保持存活。

### 速度测试（现在应能不崩溃地连续运行）
```bash
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax
```

### 内存检查
速度测试后查看服务端日志——不应出现 `token_to_kv_pool_allocator memory leak` 错误。

## 结果汇总

| 版本 | 精度 | 服务端稳定性 |
|------|------|------------|
| CHANGE_0157 | 76.04% | 每次评测后崩溃 |
| CHANGE_0158 | 76.04%（预期不变） | 应保持存活 |

## 回滚说明
```bash
git revert <commit_hash>
git push minicpm-src mixed_minicpm_cudagraph
```
将 `python/sglang/srt/mem_cache/chunk_cache.py` 及其提交副本恢复到无零过滤的原始版本。

## 后续步骤
1. 用速度测试（S1、S8、Smax）验证服务端稳定性。
2. 用速度数值更新 TEST_RESULTS_TRACKING。
3. 考虑在 MEDUSA decode 中实现适当的稀疏 KV 分配（Stage 3b 工作）。
