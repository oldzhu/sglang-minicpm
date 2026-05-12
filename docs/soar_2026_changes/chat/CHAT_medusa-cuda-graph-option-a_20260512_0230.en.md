# Chat Log: Medusa CUDA Graph Re-enable (Option A)

**Session**: 2026-05-13 (file timestamp: 20260512_0230)  
**Topic slug**: `medusa-cuda-graph-option-a`  
**Related change**: CHANGE_0157  
**Commit**: `dc7300710`

---

## 1. User Questions / Asks

**Context carried over from previous session**: Prior session ended with root cause
found for the MEDUSA + cuda-graph crash (`kv_indptr` AttributeError), but fix not yet
coded. User had already approved the Option A approach.

**This session opening**: (Implicit continuation — user returned to the session
expecting Option A implementation to proceed.)

**Binding directive (ongoing)**: "pls break the rule a little bit, I would like you
working on medusa feature until it can be tested in fcloud, just keep documents detail
and clear so we can review, reference and tracking the medusa implementation."

---

## 2. Agent Actions

### 2.1 Context verification

- Read `medusa_worker._forward_verify_k1()` (lines 80–307) to confirm the call flow:
  `target_worker.forward_batch_generation(model_worker_batch, is_verify=True)`
- Read `tp_worker.forward_batch_generation()` (lines 425–495) to understand what
  `is_verify=True` does: creates ForwardBatch, calls `model_runner.forward()`, skips
  sampling.
- Searched `_forward_raw()` dispatch to confirm that TARGET_VERIFY has no dispatch
  branch (would raise ValueError on fallthrough).
- Read `flashinfer_backend.init_forward_metadata()` (lines 420–543) to confirm
  TARGET_VERIFY uses `prefill_wrappers_verify` → `generate_attn_arg_prefill()`.
- Verified `piecewise_cuda_graph_runner.can_run()` — only checks `num_tokens`,
  but piecewise runner is `None` for our setup (`enable_piecewise_cuda_graph=False`
  because `enable_torch_compile=True` disables it).

### 2.2 Fix implementation (4 file edits)

**Fix 1: `cuda_graph_runner.can_run()`** (both main and submission copy)

Added `is_medusa_verify_ok` variable after `is_ngram_supported`:
```python
is_medusa_verify_ok = not (
    self.model_runner.spec_algorithm.is_medusa()
    and forward_batch.forward_mode.is_target_verify()
)
```
Added `and is_medusa_verify_ok` to return statement.

**Fix 2: `model_runner._forward_raw()` extend dispatch** (both main and submission copy)

Changed:
```python
elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
```
to:
```python
elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True) or (
    forward_batch.forward_mode.is_target_verify()
    and self.spec_algorithm.is_medusa()
):
```

### 2.3 Commit and push

```
[mixed_minicpm_cudagraph dc7300710] CHANGE_0157: re-enable cuda-graph for MEDUSA, run verify eagerly
4 files changed, 44 insertions(+), 2 deletions(-)
```
Pushed to `minicpm-src` (never `origin`).

### 2.4 Documentation

- Created `docs/soar_2026_changes/CHANGE_0157_medusa_cuda_graph_verify_fix.en.md`
- Created `docs/soar_2026_changes/CHANGE_0157_medusa_cuda_graph_verify_fix.zh.md`
- Updated `docs/soar_2026_changes/TEST_RESULTS_TRACKING.md` with
  `Stage3a-cgraph (PLANNED)` row (dc7300710, PENDING test results).

---

## 3. Key Decisions

| Decision | Rationale |
|----------|-----------|
| Option A (cuda-graph for DECODE, eager for verify) | Clean: DECODE path fast (cuda graph), verify correct (eager extend). No new graph captures needed. |
| Two-point fix (can_run + _forward_raw) | `can_run()` prevents incorrect DECODE graph replay; `_forward_raw()` prevents ValueError on fallthrough. Both needed. |
| Guard only `is_medusa()` in `_forward_raw` | Conservative: NGRAM/EAGLE unaffected (they never fall through their can_run). |
| `forward_extend()` for TARGET_VERIFY | `init_forward_metadata()` already has `is_target_verify()` branch → `prefill_wrappers_verify` → `generate_attn_arg_prefill()`. Correct path. |

---

## 4. Outcomes

- **CHANGE_0157 implemented and committed**: `dc7300710` on `mixed_minicpm_cudagraph`.
- **Pushed to minicpm-src** ✅
- **Docs created** ✅ (EN+ZH, TEST_RESULTS_TRACKING updated)
- **fcloud test PENDING** — needs user to start instance and approve test run.

---

## 5. Open Questions / Next Steps

1. **Fcloud validation needed**:
   - Start instance: `python3 scripts/fcloud/fcloud_workflow.py start-instance`
   - Sync + restart: `python3 scripts/fcloud/fcloud_workflow.py sync && restart-server`
   - Server uses default `SOAR_SPEC_MEDUSA_EAGER=0` (no override needed).
   - Quick MCQ test: expect ≥ 60% (≥ 12/20).
   - Full accuracy + speed: expect comparable to Stage 2 cgraph baseline.

2. **After validation**: Stage 3b (real Medusa heads, K=1 trained, W1≠0).

3. **Commit docs**: CHANGE_0157 docs and chat log should be committed to `minicpm-src`.

---

## 6. Cross-References

| Type | Reference |
|------|-----------|
| Change doc EN | [CHANGE_0157_medusa_cuda_graph_verify_fix.en.md](CHANGE_0157_medusa_cuda_graph_verify_fix.en.md) |
| Change doc ZH | [CHANGE_0157_medusa_cuda_graph_verify_fix.zh.md](CHANGE_0157_medusa_cuda_graph_verify_fix.zh.md) |
| Prior change | [CHANGE_0155_stage3a_gla_fix.en.md](CHANGE_0155_stage3a_gla_fix.en.md) |
| Test tracking | [TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md) row `Stage3a-cgraph (PLANNED)` |
| Commit | `dc7300710` on `mixed_minicpm_cudagraph` |
