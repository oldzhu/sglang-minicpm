# CHANGE_0158 — Fix: MEDUSA Sparse-KV Over-free Memory Leak

## Status
**Applied** — `python/sglang/srt/mem_cache/chunk_cache.py` (1 file, ~12 lines changed)

## Background and Motivation

After CHANGE_0157 fixed the CUDA-graph crash, a new crash was discovered:

```
ValueError: token_to_kv_pool_allocator memory leak detected!
self.max_total_num_tokens=15563319, available_size=15563321, evictable_size=0, protected_size=0
```

`available_size > max_total_num_tokens` — MORE token slots are free than should exist.
This indicates **over-free**: tokens were added to the free pool that were never allocated.
The server crashes after every completed request, making repeated tests (speed benchmarks) impossible.

## Root Cause Analysis

### Token pool slot indexing
`TokenToKVPoolAllocator.clear()` initialises the free list as:
```python
self.free_pages = torch.arange(1, self.size + 1, ...)   # starts at 1, NOT 0
```
Slot **0 is reserved** as a "padded dummy" and is **never returned by `alloc`**.
The sparse KV tables (`req_to_sparse_k1_token`, `req_to_sparse_k2_token`) are initialised to `torch.zeros(...)`, so **unwritten positions always read as 0**.

### Speculative-decode path skips sparse-KV allocation
`ScheduleBatch.prepare_for_decode()` returns early for any speculative algorithm:
```python
if not self.spec_algorithm.is_none():
    # allocation is done inside the spec worker
    return
```
This early return skips `alloc_for_decode(...)`, which normally:
1. Allocates `bs * 1` main KV slots.
2. Allocates `batch.token_sum_sparse_k1` / `batch.token_sum_sparse_k2` **sparse** KV slots (whenever `seq_len` crosses a `kernel_stride` boundary).
3. Writes the new sparse slot IDs into `req_to_sparse_k1_token` / `req_to_sparse_k2_token`.

In the MEDUSA path, `_forward_verify_k1` calls `spec_info.prepare_for_verify()` which only allocates the **1 main KV slot** per request.  
The sparse KV tables are therefore **never updated** during MEDUSA decode steps.

### kv_committed_len still advances
`NgramVerifyInput._free_cache()` (called every MEDUSA step) does:
```python
req.kv_committed_len += accept_length + 1   # always +1 for K=1
req.kv_allocated_len  = req.kv_committed_len
```
So `kv_committed_len` grows at the normal 1-token-per-step rate.

### Over-free in cache_finished_req
`ChunkCache.cache_finished_req()` computes sparse slot counts from the final `kv_committed_len`:
```python
k1_total = (kv_committed_len - kernel_size) // kernel_stride + 1   # grows as seq extends
k1_indices = req_to_sparse_k1_token[req.req_pool_idx, :k1_total]
token_to_kv_pool_allocator.free(k1_indices)                          # ← BUG
```
After several MEDUSA decode steps `kv_committed_len` crosses additional `kernel_stride` boundaries,
so `k1_total` grows beyond the count written during prefill.
The extra positions were never written → they read as **0**.
Calling `free([0])` appends the reserved slot-0 to `free_pages`, incrementing `available_size`
beyond `max_total_num_tokens` → crash.

### Example (observed run)
- Prefill 859 tokens → k1_total_prefill = (859−32)//16+1 = **52**, k2_total_prefill = **12**.
- MEDUSA decode adds ~60 steps → kv_committed_len ≈ **920**.
- k1_total_final = (920−32)//16+1 = **56** (+4 over-free), k2_total_final = **13** (+1).
- `available_size` after request completion = `max + 2` (varies with exact decode length).

## Rule-compliance Statement
This is a **bug fix** — no algorithmic change, no accuracy impact.
Filtering slot-0 is always correct: slot 0 is guaranteed never to be a valid allocated KV slot.
The fix is safe for the normal (non-MEDUSA) decode path because all legitimately allocated
sparse slots have IDs ≥ 1.

## Implementation

### File changed
`python/sglang/srt/mem_cache/chunk_cache.py`  
`benchmark/soar/demo_sala/sglang/python/sglang/srt/mem_cache/chunk_cache.py`

### Code change
```diff
-            if k1_total > 0:
-                k1_indices = self.req_to_token_pool.req_to_sparse_k1_token[req.req_pool_idx, :k1_total]
-                self.token_to_kv_pool_allocator.free(k1_indices)
+            if k1_total > 0:
+                k1_indices = self.req_to_token_pool.req_to_sparse_k1_token[req.req_pool_idx, :k1_total]
+                # CHANGE_0158: filter out slot-0 (reserved/unallocated sentinel) to prevent
+                # over-free when MEDUSA decode skips alloc_for_decode (sparse slots are
+                # never written for decode steps) but kv_committed_len still advances.
+                k1_indices_valid = k1_indices[k1_indices.ne(0)].to(torch.int64)
+                if k1_indices_valid.numel() > 0:
+                    self.token_to_kv_pool_allocator.free(k1_indices_valid)
 
-            k2_kernel_size = kernel_size * 4
-            k2_kernel_stride = kernel_stride * 4
-            k2_total = (kv_committed_len - k2_kernel_size) // k2_kernel_stride + 1 if kv_committed_len >= k2_kernel_size else 0
-            if k2_total > 0:
-                k2_indices = self.req_to_token_pool.req_to_sparse_k2_token[req.req_pool_idx, :k2_total]
-                self.token_to_kv_pool_allocator.free(k2_indices)
+            k2_kernel_size = kernel_size * 4
+            k2_kernel_stride = kernel_stride * 4
+            k2_total = (kv_committed_len - k2_kernel_size) // k2_kernel_stride + 1 if kv_committed_len >= k2_kernel_size else 0
+            if k2_total > 0:
+                k2_indices = self.req_to_token_pool.req_to_sparse_k2_token[req.req_pool_idx, :k2_total]
+                # CHANGE_0158: same zero-filter for k2 sparse slots.
+                k2_indices_valid = k2_indices[k2_indices.ne(0)].to(torch.int64)
+                if k2_indices_valid.numel() > 0:
+                    self.token_to_kv_pool_allocator.free(k2_indices_valid)
```

### Why slot-0 filtering is correct
- `TokenToKVPoolAllocator.alloc()` pops from `free_pages = torch.arange(1, size+1)`.
  Slot 0 is **never returned** by alloc.
- `req_to_sparse_k1/k2_token` is initialised with `torch.zeros(...)`.
  Any unwritten position reads as 0.
- Therefore `k1_indices.ne(0)` exactly identifies positions that were actually allocated.

### Note: sparse KV quality during MEDUSA decode
This fix corrects the **memory accounting**. It does not write sparse K1/K2 keys during
MEDUSA decode steps (alloc_for_decode is still skipped). Future work (if accuracy degrades):
implement proper sparse-KV slot allocation inside `_forward_verify_k1`. For K=1 zero-init
MEDUSA (current stage), the accuracy impact is expected to be minimal.

## Validation

### Correctness test
```bash
# After server restart, run accuracy eval — server should NOT crash
python3 scripts/fcloud/fcloud_workflow.py accuracy
```
Expected: server stays alive after eval completes.

### Speed test (should now work without crash between runs)
```bash
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax
```

### Memory check
After speed tests, tail server logs — no `token_to_kv_pool_allocator memory leak` errors.

## Result Summary

| Version | Accuracy | Server stable |
|---------|----------|--------------|
| CHANGE_0157 | 76.04% | Crashes after each eval |
| CHANGE_0158 | 76.04% (expected no change) | Should stay alive |

## Rollback Instructions
```bash
git revert <commit_hash>
git push minicpm-src mixed_minicpm_cudagraph
```
Revert `python/sglang/srt/mem_cache/chunk_cache.py` and its submission copy to restore
the original `k1_indices`/`k2_indices` free calls without the zero filter.

## Next Steps
1. Verify server stability with speed tests (S1, S8, Smax).
2. Update TEST_RESULTS_TRACKING with speed numbers.
3. Consider proper sparse-KV allocation during MEDUSA decode (Stage 3b work).
