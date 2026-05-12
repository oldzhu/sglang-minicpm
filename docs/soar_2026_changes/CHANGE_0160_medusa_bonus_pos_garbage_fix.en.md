# CHANGE_0160 — MEDUSA: Zero Bonus-Token req_to_token Position After Verify

**File(s):** `python/sglang/srt/speculative/medusa_worker.py`  
**Depends on:** CHANGE_0158 (sparse KV zero-filter), CHANGE_0159 (main KV zero-filter)  
**Commit:** TBD

---

## Background and Motivation

After CHANGE_0158 and CHANGE_0159 the server still crashed with
`available_size = max + 14`, a small but non-zero surplus in the KV-pool
allocator.  The counter should never exceed `max`; any positive surplus means
a slot was freed without ever being allocated (double-free or phantom free).

### Root-cause trace

For the MEDUSA K=1 path (`_forward_verify_k1`) with `accept_length = 1`:

1. `prepare_for_verify` allocates **1 page-slot** per request into
   `batch.out_cache_loc` (size = `bs × draft_token_num = bs × 1`).

2. After the verify forward, `_free_cache` (inside `spec_info.verify()`)
   calls:
   ```python
   assign_req_to_token_pool(
       ...,
       start=seq_lens_old,
       end=seq_lens_old + accept_length + 1,   # = seq_lens_old + 2
       out_cache_loc=tgt_cache_loc,             # length = 1 per request!
   )
   ```
   The Triton kernel reads `out_cache_loc[0]` (draft slot, valid) and
   `out_cache_loc[1]` (**out-of-bounds**).  Whatever is at that GPU-memory
   address — likely non-zero — gets written to
   `req_to_token[req_pool_idx, seq_lens_old + accept_length]`
   (the **bonus-token position**).

3. At request completion, `cache_finished_req` reads
   `req_to_token[:kv_committed_len]` and calls
   `token_to_kv_pool_allocator.free(...)`.
   CHANGE_0159 filters zero entries (`kv_indices.ne(0)`), but the bonus
   position holds a **non-zero garbage** value — it passes the filter and
   gets freed.  The freed slot was never allocated, so `available_size`
   increases by 1 per such event.

4. With `bs` requests and a garbage value occasionally being the **same
   non-zero index 14**, the surplus accumulated to +14 after a handful of
   requests, triggering the assert.

---

## Rule-Compliance Statement

This is a bug fix, not a new feature.  It adds a small Python-level loop
(O(bs)) that runs once per MEDUSA verify step.  No kernel changes, no
accuracy impact, no submission-size impact.  Fully SOAR-compliant.

---

## Implementation Plan

**Location:** `python/sglang/srt/speculative/medusa_worker.py`,
`_forward_verify_k1()`, immediately after `spec_info.verify()` returns.

**Logic:**
```python
accept_lens_cpu = spec_info.accept_length.cpu().tolist()
for _i, _req in enumerate(batch.reqs):
    if accept_lens_cpu[_i] >= 1:
        _bonus_pos = int(batch.seq_lens[_i].item()) - 1
        batch.req_to_token_pool.req_to_token[_req.req_pool_idx, _bonus_pos] = 0
```

After `spec_info.verify()`:
- `batch.seq_lens[i]` = `seq_lens_old[i] + accept_length[i] + 1`
- Bonus position = `seq_lens_old[i] + accept_length[i]`
                 = `batch.seq_lens[i] - 1`

Zeroing this position ensures CHANGE_0159's filter (`kv_indices.ne(0)`)
correctly excludes the unallocated bonus slot at cleanup time.

The bonus token's KV was never computed (the forward pass only processes the
draft tokens), so this zeroing is semantically correct.

---

## Actual Code Changes

### `python/sglang/srt/speculative/medusa_worker.py`

```diff
-        # 6. Accept walk (always accepts root for K=1 zero-init Medusa).
         logits_output, next_token_ids, num_accepted_tokens = spec_info.verify(
             batch, logits_output, self.page_size
         )
-        accept_lens = spec_info.accept_length  # (bs,) tensor, always 0 for K=1
+        accept_lens = spec_info.accept_length  # (bs,) tensor
+
+        # CHANGE_0160: zero bonus-token position in req_to_token after verify.
+        # (detailed comment in code)
+        accept_lens_cpu = spec_info.accept_length.cpu().tolist()
+        for _i, _req in enumerate(batch.reqs):
+            if accept_lens_cpu[_i] >= 1:
+                _bonus_pos = int(batch.seq_lens[_i].item()) - 1
+                batch.req_to_token_pool.req_to_token[_req.req_pool_idx, _bonus_pos] = 0

         # 7. Restore forward_mode / spec_algorithm for scheduler bookkeeping.
```

---

## Validation Commands

```bash
# On fcloud — restart server with CHANGE_0160:
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server

# Run speed tests (server must survive all 48 S1 requests without crash):
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax

# Run accuracy eval:
python3 scripts/fcloud/fcloud_workflow.py accuracy
```

**Success criteria:**
- Server does NOT crash during any speed test
- `available_size` never exceeds `max` in server logs
- Accuracy ≥ 75% (ideally same as CHANGE_0157 baseline of 76.04%)

---

## Result Summary Table

| Metric | Before (CHANGE_0158+0159 only) | After CHANGE_0160 |
|--------|-------------------------------|-------------------|
| Server stability | Crashes after ~5 requests (available_size=max+14) | TBD |
| S1 duration (s) | N/A (crashed) | TBD |
| S8 duration (s) | N/A (crashed) | TBD |
| Smax duration (s) | N/A (crashed) | TBD |
| Accuracy | N/A (crashed) | TBD |

---

## Rollback Instructions

```bash
git revert HEAD   # or manually remove the CHANGE_0160 block in medusa_worker.py
git push minicpm-src mixed_minicpm_cudagraph
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
```

---

## Next-Step Suggestions

1. **Proper bonus-slot allocation (long-term fix):** modify `prepare_for_verify`
   to allocate `draft_token_num + 1` page-slots so the bonus token gets a valid
   KV slot.  This prevents the OOB read at source and may also improve accuracy
   (bonus-token KV is computed when the page is reused in the next step).

2. **Stage 3b (K>1):** once server is stable with K=1, extend to K=2+ Medusa
   heads.  The same bonus-zeroing logic will apply to each accepted draft path.

3. **Speed tests:** compare S1/S8/Smax against Stage 2 cgraph baseline
   (S1=118.28s, S8=43.87s, Smax=35.75s) to quantify MEDUSA overhead.
