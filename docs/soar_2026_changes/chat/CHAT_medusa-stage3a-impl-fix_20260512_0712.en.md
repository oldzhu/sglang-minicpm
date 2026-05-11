# CHAT: Medusa Stage 3a Implementation & Syntax Fix

**Date**: 2026-05-12  
**Session start**: ~07:12 UTC  
**Topic slug**: `medusa-stage3a-impl-fix`  
**Branch**: `mixed_minicpm_cudagraph`  
**Instance**: `ai-e7e98a7c52` (PAUSED)

---

## Session Summary

This session resumed from the prior session (which confirmed GLA state mismatch via NGRAM re-probe quick test and wrote the Stage 3a `medusa_worker.py`). Key work done:

1. **Identified and fixed a critical syntax error** in `medusa_worker.py` — the file had dangling Stage 2 docstring text (lines 318–500) causing `IndentationError` at import time.
2. **Deep analysis of `verify_tree_greedy` semantics** to confirm Stage 3a implementation correctness.

---

## Round 1: Resuming Session — State Assessment

### User Request
Start fcloud, continue testing Stage 3a Medusa. Previous session rewrote `medusa_worker.py`.

### Agent Actions
- Reviewed session summary: Stage 3a code written but NOT committed or tested.
- Read `medusa_worker.py` to understand current state.
- **Discovered syntax error**: file contained both Stage 3a code (lines 1–316) AND dangling Stage 2 docstring/code (lines 317–500). Python would fail to import it.
- Confirmed with `python3 -c "import py_compile; ..."` → `IndentationError` at line 318.

### Root Cause
The previous session's `replace_string_in_file` call was supposed to rewrite the entire file but only replaced part of it, leaving old Stage 2 content appended after the new Stage 3a class.

### Fix Applied
```bash
head -316 medusa_worker.py > /tmp/fixed.py && mv /tmp/fixed.py medusa_worker.py
```
Truncated at line 316 (after the closing `return GenerationBatchResult(...)` of `_forward_verify_k1`). Verified `SYNTAX_OK`.

---

## Round 2: Verify `_fill_requests` Semantics (K=1 Correctness Proof)

### Key Question
Does `_fill_requests` in `NgramVerifyInput.verify()` append the draft token (root = `req.output_ids[-1]`) causing a **duplicate** in `output_ids`? Or does it append the **next** token?

### Analysis of `verify_tree_greedy` + `_fill_requests`

From `ngram_info.py` (line 159, 203):
```python
predict_cpu = self.predict.tolist()
self.verified_id = self.predict[self.accepted_indices]
```

For K=6 test case (from unit tests):
- `candidates[0][0] = 0` (root value = draft token = x_t)
- `predicts[0] = 3` = **next token after root** = `candidates[0][3]` = x_{t+1}
- `accept_index[0] = [0, 3, 4, 5]` (accepted node global indices)
- `_fill_requests` appends: `predicts[0]=3, predicts[3]=4, predicts[4]=5, predicts[5]=18`
- = tokens `[x_{t+1}, x_{t+2}, x_{t+3}, corrected]`
- The **root x_t is NOT appended** (it was already in `output_ids` from previous step)

**For K=1 specifically:**
- `accept_index[0] = [0]` (only the root)
- `_fill_requests` loop: j=0, idx=0 → append `predicts[0]` = `x_{t+1}` (next after root)
- **No duplicate**: `x_t` (root) stays in `output_ids`, `x_{t+1}` (new token) is freshly appended
- `accept_length = 0`, `seq_lens += 0 + 1 = 1` ✓

### Step-by-Step Trace (K=1, Stage 3a)

**Initial state (after extend with N input tokens, output = x_N):**
- `output_ids = [x_N]`, `seq_lens = N` (KV has x_0..x_{N-1})

**Step 1 (first verify):**
- draft = `output_ids[-1]` = x_N (NOT yet in KV)
- `prepare_for_verify`: input_ids = [x_N], alloc KV slot at position N
- Forward: x_N attends to [x_0..x_{N-1}, x_N] → argmax = x_{N+1}
- `verify`: accept x_N (root), `predicts[0]` = x_{N+1}
- `_fill_requests`: append x_{N+1} → `output_ids = [x_N, x_{N+1}]` ✓
- `seq_lens += 1` → N+1 (KV has x_0..x_N) ✓

**Step 2 (next verify):**
- draft = `output_ids[-1]` = x_{N+1} (NOT yet in KV)
- Forward: x_{N+1} → x_{N+2}
- append x_{N+2} → `output_ids = [x_N, x_{N+1}, x_{N+2}]` ✓
- `seq_lens = N+2` ✓

**Conclusion**: Implementation is **correct**. No bootstrap step needed. No duplicate tokens.

---

## Round 3: `spec_algorithm=MEDUSA` in `process_batch_result_decode`

### Question
Does restoring `batch.spec_algorithm = MEDUSA` (vs NONE in Stage 2) cause issues in `process_batch_result_decode`?

### Analysis (`scheduler_output_processor_mixin.py` lines 371–453)

```python
if batch.spec_algorithm.is_none():
    req.output_ids.append(next_token_id)   # only for NONE
elif batch.is_spec_v2:
    req.output_ids.extend(next_token_id)    # only for spec_v2
# spec_v1 MEDUSA: no append (already done by _fill_requests) ✓

req.check_finished(new_accepted_len)        # called twice but idempotent ✓
```

For `_mamba_prefix_cache_update` (line 489): spec (not-none) path checks `accept_length_per_req_cpu` which is `accept_lens.tolist()` = [0, 0, ...]. Condition `actual_seq_len - 0 != actual_seq_len` = False → no spurious track update. ✓

**Conclusion**: `spec_algorithm=MEDUSA` for spec_v1 is safe in `process_batch_result_decode`.

---

## Files Changed This Session

| File | Change |
|------|--------|
| `python/sglang/srt/speculative/medusa_worker.py` | Fixed syntax error: truncated dangling Stage 2 content at line 316 |

---

## Status After This Session

- ✅ `medusa_worker.py` Stage 3a: syntax-valid, logic confirmed correct
- ❌ NOT committed yet
- ❌ NOT pushed to minicpm-src
- ❌ NOT tested on fcloud

## Immediate Next Steps

1. Commit Stage 3a + docs changes, push to `minicpm-src`
2. Verify console JWT not expired
3. Get user approval to start fcloud instance
4. Sync → restart Medusa server → run quick accuracy test
5. **Pass criterion**: accuracy ≥ 60% MCQ, avg_out ≤ 500 tokens/sample

---

## Cross-References

- `PROPOSAL_medusa_stage3_verify_rewind.en.md` (§14: GLA state mismatch confirmed)
- `TEST_RESULTS_TRACKING.md` (NGRAM-reprobe-quick row: 0% accuracy, 44916 avg_out)
- `CHANGE_0155_medusa_phase_r1b_stage2.en.md` (Stage 2 baseline: 80.11%, S1=118.28s)
