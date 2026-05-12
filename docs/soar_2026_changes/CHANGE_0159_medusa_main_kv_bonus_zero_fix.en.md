# CHANGE_0159 — Fix: MEDUSA Main-KV Zero-Free Crash (Bonus Token Slot Missing)

## Status
**Applied** — `python/sglang/srt/mem_cache/chunk_cache.py` (1 file, ~9 lines changed)

## Background

CHANGE_0158 fixed over-free for sparse K1/K2 slots in `cache_finished_req`.  
A second crash survived: after CHANGE_0158 was deployed, the server still crashed with:

```
ValueError: token_to_kv_pool_allocator memory leak detected!
self.max_total_num_tokens=15563319, available_size=15582175, evictable_size=0, protected_size=0
```

`available_size - max = 18856` — far larger than the 2 observed in CHANGE_0158 testing.
The crash happened during the 28th S1 request (a long ~52K-token generation).
CHANGE_0158's K1/K2 filter was confirmed present; the issue was in the **main KV** free path.

## Root Cause Analysis

### TokenToKVPoolAllocator.free() blindly appends
`TokenToKVPoolAllocator.free(tensor)` appends ALL entries in `tensor` to `free_pages`:
```python
def free(self, free_index: torch.Tensor):
    if free_index.numel() == 0:
        return
    self.free_pages = torch.cat((self.free_pages, free_index))
```
No zero-check, no duplicate-check. `available_size = len(free_pages)`.

### cache_finished_req frees main KV indices without zero-filter
```python
kv_indices = req_to_token_pool.req_to_token[req.req_pool_idx, :kv_committed_len]
req_to_token_pool.free(req.req_pool_idx)
token_to_kv_pool_allocator.free(kv_indices)   # ← no zero-filter!
```
`req_to_token` is initialised with `torch.zeros(...)`. Any unwritten position reads as **0**.

### MEDUSA bonus token position is never written to req_to_token

For MEDUSA `draft_token_num=1`, `--speculative-num-draft-tokens 2`:

**Each verify step (page_size=1 case)**:
1. `prepare_for_verify` allocates 1 slot `s1` for the draft token:
   - `assign_req_to_token_pool(start=seq_lens, end=seq_lens+1, out_cache_loc=[s1])`
   - Writes `s1` to `req_to_token[idx, seq_lens]` ✓

2. Forward pass on `[draft_token]` → KV at `seq_lens` stored in `s1`.

3. Verify result: if draft accepted (`accept_length=1`):
   - `_free_cache` tries to write to positions `seq_lens` and `seq_lens+1` from `out_cache_loc=[s1]`
   - `assign_req_to_token_pool(start=seq_lens, end=seq_lens+2, out_cache_loc=[s1])`
   - The triton kernel reads `out_cache_loc[0]` for position `seq_lens` → `s1` ✓
   - The triton kernel reads `out_cache_loc[1]` for position `seq_lens+1` — **OUT OF BOUNDS READ** (only 1 entry)
   - `req_to_token[idx, seq_lens+1]` gets the value at `out_cache_loc + 1` in GPU memory, which may be 0 (padding) or garbage
   - `kv_committed_len += accept_length + 1 = 2`

4. In subsequent steps: `req_to_token[idx, seq_lens+1]` = 0 is never overwritten.

### Cumulative over-free

For a long request with ~50K tokens and ~50% draft-accept rate:
- ~25K steps with `accept_length=1` → ~25K positions with `req_to_token=0`
- At `cache_finished_req`: `kv_indices` includes ~25K zeros
- `free(zeros)` → `available_size += 25K` per full request
- The observed `available_size = max + 18856` reflects cumulative zeros freed

(The exact number 18856 < 25K because short initial prefill requests have fewer accept_length=1 steps.)

### Why slot-0 is safe to filter
`TokenToKVPoolAllocator.clear()`:
```python
self.free_pages = torch.arange(1, self.size + 1, ...)  # starts at 1
```
Slot **0 is never allocated**; it is the reserved "padded dummy" slot. Any 0 in `kv_indices` is a position that was never written a real KV slot ID.

## Rule-compliance Statement
Pure bug fix. No algorithmic change, no accuracy change. Filtering zeros from the
main KV free path is safe for all paths because legitimately allocated KV slots always
have ID ≥ 1.

## Implementation

### File changed
`python/sglang/srt/mem_cache/chunk_cache.py`  
`benchmark/soar/demo_sala/sglang/python/sglang/srt/mem_cache/chunk_cache.py`

### Code change
```diff
-        self.req_to_token_pool.free(req.req_pool_idx)
-        self.token_to_kv_pool_allocator.free(kv_indices)
+        self.req_to_token_pool.free(req.req_pool_idx)
+        # CHANGE_0159: filter out slot-0 (reserved sentinel) from main KV indices.
+        # In MEDUSA with draft_token_num=1, when accept_length=1 the bonus token
+        # position in req_to_token is never written (stays 0) but kv_committed_len
+        # is incremented by accept_length+1=2. free([0]) adds the sentinel back
+        # to free_pages causing available_size > max_total_num_tokens crash.
+        kv_indices_valid = kv_indices[kv_indices.ne(0)].to(torch.int64)
+        if kv_indices_valid.numel() > 0:
+            self.token_to_kv_pool_allocator.free(kv_indices_valid)
```

### Memory leak implications
With the zero-filter:
- Zeros are not freed → `available_size` stays at `max`
- The underlying issue (bonus token KV slot not allocated) is masked, not fixed
- This is intentional: proper fix requires restructuring the MEDUSA verify loop
  to also allocate and write a KV slot for the bonus token (Stage 3b work)

### Correctness note
The bonus token's missing KV at `seq_lens+1` means attention in subsequent verify
steps reads from slot-0 (dummy KV data). This may explain some of the accuracy gap
(76.04% vs 80.11% baseline). Fixing this properly is Stage 3b work.

## Validation

### Crash test
```bash
# After restart, run speed S1 — server should NOT crash between requests
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
```
Expected: 48 requests complete without `ConnectionResetError`.

### Full speed suite
```bash
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax
```

### Memory log check
```bash
python3 scripts/fcloud/fcloud_workflow.py server-logs --lines 50
```
No `token_to_kv_pool_allocator memory leak detected` errors.

## Result Summary

| Version | Crash | Cause |
|---------|-------|-------|
| CHANGE_0157 | After each request | MEDUSA K1/K2 sparse over-free |
| CHANGE_0158 | After long requests | Main KV bonus-token zero-free |
| CHANGE_0159 | None (expected) | Zero-filter on main KV path |

## Rollback Instructions
```bash
git revert <commit_hash>
git push minicpm-src mixed_minicpm_cudagraph
```

## Next Steps (Stage 3b)
1. Properly allocate a KV slot for the bonus token in `_forward_verify_k1` or
   `NgramVerifyInput.prepare_for_verify` to fix the corrupted attention issue.
2. Alternatively: set `kv_committed_len += accept_length` (no bonus) and handle
   the bonus token as the first token of the NEXT step's draft.
3. Track whether fixing the bonus KV improves accuracy above 76%.
