# CHAT: Medusa KV Over-Free Debug — Stale K1/K2 Phantom-Free Root Cause & Fix

**File**: `CHAT_medusa-kv-overfree-debug_20260512_0800.en.md`  
**Session date**: 2026-05-12  
**Topic**: Debugging the persistent `available_size = max + 14` crash after CHANGE_0158/0159/0160; identifying the stale-K1/K2-row root cause; applying CHANGE_0161; completing S1/S8/Smax speed tests.

---

## 1. User Questions / Asks

*(Binding directive from prior session, reproduced here for tracking):*

> "pls break the rule a little bit, I would like you working on medusa feature until it can be tested in fcloud, just keep documents detail and clear so we can review, reference and tracking the medusa implementation."

This session was autonomous continuation of that directive:
- CHANGE_0160 was committed and synced in the prior turn; the session summary recorded the server still crashing with `available_size = max + 14`.
- Agent task: find the real root cause, fix it, run speed tests, update docs, pause fcloud.

---

## 2. Agent Actions

### 2.1 Diagnosis — CHANGE_0160 proved ineffective

CHANGE_0160 wrote zero to `req_to_token[req_pool_idx, bonus_pos]` after verify, guarded by `accept_length >= 1`. Server logs from prior session showed this guard **never fires**: `accept_length = 0` always for K=1 zero-init Medusa heads.

Evidence:
- Metrics log: `spec_accept_length = 1.00` = `accept_length + 1 = 1` → `accept_length = 0`
- `spec_accept_rate = 0.50` = 1/2 draft tokens: root (always accepted) + 1 zero-init draft (always rejected)

### 2.2 Root cause investigation

Reading `cache_finished_req` in `chunk_cache.py` (lines 51–93) and `alloc_for_decode` in `common.py` (lines 496–560), plus tracing:
- `prepare_for_decode()` (schedule_batch.py:1939) returns early when `spec_algorithm != NONE`
- Therefore `alloc_for_decode()` is **never called** during MEDUSA DECODE steps
- `kv_committed_len` still increments (+1 per step via `scheduler_output_processor_mixin.py:306`)
- `cache_finished_req` computes `k1_total = (kv_committed_len - kernel_size) // kernel_stride + 1`
- As decode progresses, `k1_total` grows beyond the number of K1 slots actually allocated during prefill

Critical discovery: **`req_to_sparse_k1_token` is a reused pool table**, not zeroed between requests. When `req_pool_idx` is reused for a new request with shorter prefill than the previous request, positions `[k1_new_prefill .. k1_old_total-1]` retain non-zero slot IDs from the old request. CHANGE_0158's zero-filter (`k1_indices.ne(0)`) passes these stale non-zero IDs → they are freed into the allocator → `available_size` inflated by +1 per stale ID → cumulative +14 across completed requests triggers the health check crash.

### 2.3 CHANGE_0161 applied

**Fix**: After reading and freeing K1/K2 indices in `cache_finished_req`, zero out the entire K1/K2 row:

```python
self.req_to_token_pool.req_to_sparse_k1_token[req.req_pool_idx, :k1_total] = 0
# ... same for K2
self.req_to_token_pool.req_to_sparse_k2_token[req.req_pool_idx, :k2_total] = 0
```

Files modified (both main and submission copy):
- `python/sglang/srt/mem_cache/chunk_cache.py`
- `benchmark/soar/demo_sala/sglang/python/sglang/srt/mem_cache/chunk_cache.py`

Commit: `5dfa9ce05` — pushed to `minicpm-src`.

### 2.4 fcloud sync and server restart

```
python3 scripts/fcloud/fcloud_workflow.py sync   # → updated to 5dfa9ce05
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server  # → Ready after 175s
```

### 2.5 Speed tests — all passing

| Variant | Duration | Requests | Crashes |
|---------|----------|----------|---------|
| S1 | **202.96s** | 48/48 | 0 |
| S8 | **61.65s** | 72/72 | 0 |
| Smax | **43.40s** | 96/96 | 0 |

Comparison with Stage 2 baseline (commit `46553947b`):
- S1: 202.96 vs 118.28 (+72% slower)
- S8: 61.65 vs 43.87 (+41% slower)
- Smax: 43.40 vs 35.75 (+21% slower)

Speed regression is **expected**: K=1 zero-init Medusa adds verify overhead (alloc + extend + free) with accept_length=0 always → zero throughput benefit from speculative decode.

---

## 3. Outcomes

### Crash completely resolved

All four crash patterns are now fixed:

| CHANGE | Root cause | Fixed by |
|--------|-----------|---------|
| CHANGE_0158 | K1/K2 slots freed that were never allocated (MEDUSA skips `alloc_for_decode`) | zero-filter `ne(0)` |
| CHANGE_0159 | Main KV bonus slot-0 sentinel freed when `accept_length=1` | zero-filter `ne(0)` |
| CHANGE_0160 | Bonus-pos zero-write (INEFFECTIVE — guard never fires) | Committed but no-op |
| **CHANGE_0161** | **Stale non-zero K1/K2 IDs from prior longer request survive pool reuse** | **zero-out row after free** |

### Stage 3a K=1 is fully stable

MEDUSA Stage 3a with K=1 zero-init heads runs stably for:
- Accuracy evaluation (confirmed 76.04% in prior turn)
- Full S1/S8/Smax speed benchmark suites

### Stage 3a is NOT a submission candidate

Speed is 21–72% slower than baseline. To benefit from Medusa:
- **Stage 3b**: Load trained MedusaHead weights, run head forward on target model hidden state after prefill, predict `num_heads` actual draft tokens
- With trained heads and K>1, acceptance rate > 0 → multiple tokens accepted per verify step → real speedup

---

## 4. Open Questions

1. **CHANGE_0160 cleanup**: Should the now-ineffective `accept_length >= 1` zero-write block be removed from `medusa_worker.py` for clarity? (Low priority — it is harmless dead code for K=1.)

2. **Pool table clear-on-free convention**: Should `req_to_token_pool.free()` automatically zero the K1/K2 rows? Currently requires callers to do it manually (CHANGE_0161 pattern). A systemic fix would prevent future bugs in other code paths.

3. **K1/K2 alloc in MEDUSA DECODE**: If Stage 3b (K>1) requires K1/K2 slots during verify, will the current skip of `alloc_for_decode` break? Need to audit whether the verify forward path writes to K1/K2 positions.

---

## 5. Cross-References

### Documents created/modified
- [CHANGE_0161_medusa_stale_k1k2_zero_out.en.md](CHANGE_0161_medusa_stale_k1k2_zero_out.en.md) — English change doc
- [CHANGE_0161_medusa_stale_k1k2_zero_out.zh.md](CHANGE_0161_medusa_stale_k1k2_zero_out.zh.md) — Chinese change doc
- [TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md) — Added "Stage3a-cgraph" and "Stage3a-stable (CHANGE_0161)" rows

### Commits
- `5dfa9ce05` — CHANGE_0161: zero-out stale K1/K2 rows in `cache_finished_req`

### TEST_RESULTS_TRACKING rows added
- **Stage3a-cgraph** (CHANGE_0157 accuracy row, 76.04%, updated from PLANNED)
- **Stage3a-stable (CHANGE_0161)** (speed-only, S1=202.96s / S8=61.65s / Smax=43.40s)

### Related prior changes
- CHANGE_0158: [CHANGE_0158_medusa_sparse_kv_overfree_fix.en.md](CHANGE_0158_medusa_sparse_kv_overfree_fix.en.md)
- CHANGE_0159: [CHANGE_0159_medusa_main_kv_bonus_zero_fix.en.md](CHANGE_0159_medusa_main_kv_bonus_zero_fix.en.md)
- CHANGE_0160: [CHANGE_0160_medusa_bonus_pos_garbage_fix.en.md](CHANGE_0160_medusa_bonus_pos_garbage_fix.en.md)
- CHANGE_0157: [CHANGE_0157_medusa_cuda_graph_verify_fix.en.md](CHANGE_0157_medusa_cuda_graph_verify_fix.en.md)
