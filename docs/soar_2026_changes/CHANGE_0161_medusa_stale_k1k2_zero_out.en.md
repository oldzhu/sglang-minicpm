# CHANGE_0161: Zero-out Stale K1/K2 Rows in `cache_finished_req` to Fix `+14` Phantom-Free Crash

**Change ID**: CHANGE_0161  
**Date**: 2026-05-12  
**Commit**: `5dfa9ce05`  
**Branch**: `mixed_minicpm_cudagraph`  
**Author**: AI agent (SOAR 2026 Medusa debugging)  
**Files modified**:  
- `python/sglang/srt/mem_cache/chunk_cache.py`  
- `benchmark/soar/demo_sala/sglang/python/sglang/srt/mem_cache/chunk_cache.py`  

---

## Background and Motivation

After CHANGE_0158 (K1/K2 sparse zero-filter) and CHANGE_0159 (main KV zero-filter), the server
continued crashing with a consistent `available_size = max_total_num_tokens + 14` error:

```
ValueError: token_to_kv_pool_allocator memory leak detected!
self.max_total_num_tokens=15563319, available_size=15563333, evictable_size=0, protected_size=0
```

The delta was always exactly **+14**, appearing consistently after warmup + 3 speed-test requests
(1918, 2760, 2304 input tokens), crashing on the 4th request (1196 tokens).

CHANGE_0160 attempted to zero out the bonus position in `req_to_token` after verify, but was
**ineffective** because the guard `accept_length >= 1` never fires for K=1 zero-init Medusa heads
(`accept_length` is always 0, since the zero-init head always accepts only the root token).

This change identifies and fixes the **actual root cause** of the +14 phantom-free.

---

## Rule-Compliance Statement

- Pure defensive bookkeeping: zeroing out a slot-ID array after reading it does not change
  any model output, quantization, or attention computation.
- No new allocations or tensor operations on the critical decode path.
- The zero-out is O(k1_total) per request completion — negligible (k1_total < seq_len/stride).
- Does not modify any files excluded by SOAR submission constraints.

---

## Root Cause Analysis

### Key facts

1. **page_size = 1** for MiniCPM-SALA (no `--page-size` in `prepare_env.sh`, defaults to 1).
   Uses `TokenToKVPoolAllocator` where `available_size() = len(free_pages)`.
   `free()` blindly appends without deduplication or range-checks.

2. **MEDUSA DECODE skips `prepare_for_decode`**  
   `prepare_for_decode()` (schedule_batch.py) returns early when `spec_algorithm != NONE`.
   Therefore `alloc_for_decode()` is never called → no new K1/K2 slots written to
   `req_to_sparse_k1_token[pool_idx, :]` during decode steps.

3. **`kv_committed_len` grows each decode step anyway**  
   `scheduler_output_processor_mixin.py:306` increments `kv_committed_len += accept_lens[i]`
   (= 1 per MEDUSA step because accept_length=0 → accept_length+1=1 → 1 new committed token).

4. **`cache_finished_req` uses `kv_committed_len` to compute `k1_total`**  
   ```python
   k1_total = (kv_committed_len - kernel_size) // kernel_stride + 1
              if kv_committed_len >= kernel_size else 0
   ```
   As decode progresses, `kv_committed_len` grows → `k1_total` grows past the number of
   K1 slots that were actually allocated during prefill.

5. **The K1 row still holds non-zero stale slot IDs from the PREVIOUS request**  
   `req_to_sparse_k1_token` is a reused pool table. When `req_pool_idx` is reused for a
   new request whose prefill is SHORTER than the previous request's prefill, positions
   `[k1_prev_prefill .. k1_prev_total-1]` retain the non-zero slot IDs from the old request.

6. **CHANGE_0158's zero-filter is bypassed**  
   CHANGE_0158 filters `k1_indices[k1_indices.ne(0)]`. These stale non-zero IDs from the
   prior request pass the filter and are freed into the allocator.
   Each stale non-zero ID that lands in `free_pages` increments `available_size` by +1.

### Why exactly +14?

The +14 accumulates over the 4-request pattern observed:
- 4 requests before crash (warmup + 3 speed-test requests)
- Each request finishes and calls `cache_finished_req`
- Depending on which pool_idx each new request reuses and how much larger the previous
  request at that pool_idx was, between 1 and several stale K1 positions are encountered
- The cumulative total across all completed requests when the health check fires is +14

---

## Implementation Plan (Design)

The fix is minimal: **after reading and freeing K1/K2 indices in `cache_finished_req`, zero
out the entire K1 (and K2) row** so that any future request reusing the same `pool_idx`
sees zeros in positions beyond its own `k1_prefill`, making CHANGE_0158's zero-filter
effective again.

This is analogous to how `req_to_token_pool.free(req_pool_idx)` conceptually marks the
request slot as free — we need the same "clear on free" semantic for the sparse K1/K2 arrays.

---

## Actual Code Changes

### Before (CHANGE_0158 only, still crashes)

```python
k1_total = (kv_committed_len - kernel_size) // kernel_stride + 1 if kv_committed_len >= kernel_size else 0
if k1_total > 0:
    k1_indices = self.req_to_token_pool.req_to_sparse_k1_token[req.req_pool_idx, :k1_total]
    k1_indices_valid = k1_indices[k1_indices.ne(0)].to(torch.int64)
    if k1_indices_valid.numel() > 0:
        self.token_to_kv_pool_allocator.free(k1_indices_valid)
    # stale values at [k1_prefill..k1_total-1] survive → phantom frees on next reuse!

k2_total = ...
if k2_total > 0:
    k2_indices = self.req_to_token_pool.req_to_sparse_k2_token[req.req_pool_idx, :k2_total]
    k2_indices_valid = k2_indices[k2_indices.ne(0)].to(torch.int64)
    if k2_indices_valid.numel() > 0:
        self.token_to_kv_pool_allocator.free(k2_indices_valid)
    # same stale problem for K2
```

### After (CHANGE_0161)

```python
k1_total = (kv_committed_len - kernel_size) // kernel_stride + 1 if kv_committed_len >= kernel_size else 0
if k1_total > 0:
    k1_indices = self.req_to_token_pool.req_to_sparse_k1_token[req.req_pool_idx, :k1_total]
    k1_indices_valid = k1_indices[k1_indices.ne(0)].to(torch.int64)
    if k1_indices_valid.numel() > 0:
        self.token_to_kv_pool_allocator.free(k1_indices_valid)
    # CHANGE_0161: zero out the K1 row so stale non-zero values from this
    # request do NOT survive to be phantom-freed by a future request that
    # reuses the same pool_idx with a smaller k1_prefill.
    self.req_to_token_pool.req_to_sparse_k1_token[req.req_pool_idx, :k1_total] = 0

k2_total = ...
if k2_total > 0:
    k2_indices = self.req_to_token_pool.req_to_sparse_k2_token[req.req_pool_idx, :k2_total]
    k2_indices_valid = k2_indices[k2_indices.ne(0)].to(torch.int64)
    if k2_indices_valid.numel() > 0:
        self.token_to_kv_pool_allocator.free(k2_indices_valid)
    # CHANGE_0161: same stale-row zero-out for K2.
    self.req_to_token_pool.req_to_sparse_k2_token[req.req_pool_idx, :k2_total] = 0
```

---

## Validation Commands

```bash
# Sync to fcloud and restart server
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server

# Run speed tests — verify NO crash (all requests complete)
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax
```

Success criteria:
- All 48 / 72 / 96 requests complete (no `memory leak detected` crash)
- Server does NOT raise `ValueError: available_size > max_total_num_tokens`

---

## Result Summary

| Metric | Before CHANGE_0161 | After CHANGE_0161 |
|--------|-------------------|-------------------|
| S1 crash after 4 reqs | YES (available_size=max+14) | NO |
| S8 crash | YES | NO |
| Smax crash | YES | NO |
| S1 duration | N/A (crashed) | **202.96s** |
| S8 duration | N/A (crashed) | **61.65s** |
| Smax duration | N/A (crashed) | **43.40s** |
| S1 vs Stage 2 baseline (118.28s) | — | **+72% slower** |
| S8 vs Stage 2 baseline (43.87s) | — | **+41% slower** |
| Smax vs Stage 2 baseline (35.75s) | — | **+21% slower** |

**Note on speed regression**: Stage 3a K=1 with zero-init heads is SLOWER than the Stage 2
pass-through baseline. This is expected: every decode step now allocates slots for both
draft and verify tokens, runs `forward_extend()` for verify, then frees the rejected slots.
With `accept_length=0` always (zero-init heads accept nothing beyond the root), all the
overhead is paid with zero throughput benefit. Stage 3b (trained heads, K>1, genuine
acceptance rate) is required to recoup and exceed the baseline speed.

---

## Relationship to Previous Fixes

| Change | Root cause addressed | Status |
|--------|---------------------|--------|
| CHANGE_0158 | K1/K2 sparse slots freed by `cache_finished_req` were NEVER allocated (MEDUSA skips `alloc_for_decode`) | Fixed (zero-filter) |
| CHANGE_0159 | Main KV bonus-token position (slot 0 sentinel) freed when `accept_length=1` increments `kv_committed_len` past prefill | Fixed (zero-filter) |
| CHANGE_0160 | Zero-write the bonus position after verify (INEFFECTIVE — accept_length always 0) | Committed but no-op |
| **CHANGE_0161** | **Stale non-zero K1/K2 slot IDs from previous longer request survive pool reuse** | **Fixed (zero-out row)** |

---

## Rollback Instructions

```bash
# Revert both files
git diff HEAD~1 python/sglang/srt/mem_cache/chunk_cache.py
git diff HEAD~1 benchmark/soar/demo_sala/sglang/python/sglang/srt/mem_cache/chunk_cache.py
git revert 5dfa9ce05
git push minicpm-src mixed_minicpm_cudagraph
```

---

## Next-Step Suggestions

1. **Stage 3b**: Implement real MEDUSA head forward to predict `num_heads` draft tokens.
   Replace the zero-init draft in `_forward_generate_k1` with an actual head forward call.
   Requires loading trained MedusaHead weights and running them on the target model's
   hidden state after prefill.

2. **K1/K2 allocation in MEDUSA DECODE**: Consider whether MEDUSA DECODE should actually
   call `alloc_for_decode` for K1/K2 slots (like regular decode). Currently it skips it
   because `prepare_for_decode` returns early. If K>1 verify needs K1/K2 slots, this will
   need to be re-enabled or routed separately.

3. **Pool table clear-on-free convention**: Consider adding a `clear_on_free` flag to
   `req_to_token_pool.free()` that zeros the sparse K1/K2 rows automatically, to prevent
   similar bugs in other code paths.
