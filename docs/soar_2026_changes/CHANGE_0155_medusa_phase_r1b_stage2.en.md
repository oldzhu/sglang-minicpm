# CHANGE_0155 — Medusa Phase R1b Stage 2 (Heads-Shadow Smoke Test)

- **Status**: Implemented. Awaiting fcloud validation.
- **Branch**: `mixed_minicpm_cudagraph`
- **Predecessor**: [CHANGE_0154_medusa_phase_r1b_design.en.md](CHANGE_0154_medusa_phase_r1b_design.en.md) (Stage 1 helpers).
- **Successor**: TBD (Stage 3 = full verify + rewind).

## 1. Purpose

Stage 2 of CHANGE_0154 R1b. The goal is **wiring validation**: prove that a
server launched with `SOAR_SPEC_MEDUSA=1` boots, instantiates the Medusa
heads on GPU, and produces output **byte-identical** to the no-spec baseline.

After Stage 2 passes fcloud, Stage 3 will add the real verify + rewind path
on top of the same wiring.

## 2. Scope (minimal by design)

Stage 2 does NOT touch the verify path. Specifically:

- `forward_mode` is never switched to `TARGET_VERIFY`.
- `capture_hidden_mode` is never set to `LAST`.
- The Stage 1 helpers (`snapshot_state_for_spec` / `restore_state_for_spec`)
  remain unused.
- No draft tokens are produced; `num_accepted_tokens` is always 0.

This makes Stage 2 a **pure delegation worker** plus a one-shot weight
allocation. If accuracy is not byte-identical, the bug is in
delegation/wiring (not in Medusa algorithm).

## 3. Why the design doc plan was simplified

CHANGE_0154 §4 originally proposed adding `is_medusa()` to six call sites
(scheduler, model_runner ×3, cuda_graph_runner ×3). Re-survey during this
session showed those sites only matter when MEDUSA actually triggers
TARGET_VERIFY capture. Stage 2 stays in normal DECODE mode → no TARGET_VERIFY
capture → no extra branches needed.

Net Stage 2 edit footprint: **2 files** (down from 6).

## 4. Implementation

### 4a. `python/sglang/srt/speculative/medusa_worker.py` (full rewrite)

Before: stub raising `NotImplementedError` from R1a (CHANGE_0153).

After: ~150 LOC implementation with these characteristics:

- Plain class (mirrors `NGRAMWorker`'s style; not an `ABC` subclass) so
  the scheduler's existing `draft_worker.forward_batch_generation(batch)`
  dispatch works without changes.
- `__init__` signature matches the scheduler's `draft_worker_kwargs`:
  `(server_args, gpu_id, tp_rank, dp_rank, moe_ep_rank, nccl_port, target_worker)`.
- Instantiates `MedusaHeads(hidden_size, num_heads, lm_head, dtype)` on GPU
  using `model_runner.dtype` (bfloat16 in the SOAR submission config).
- Validates `target_model.lm_head` exists (defensive guardrail; will only
  fail on unsupported model classes).
- `forward_batch_generation(batch)` calls
  `target_worker.forward_batch_generation(batch.get_model_worker_batch())`
  and wraps the result in `GenerationBatchResult` with
  `num_accepted_tokens=0, accept_lens=None`.
- Logs a one-line summary at init: `K`, `hidden`, `dtype`,
  `approx_weight_MiB`. K=1, hidden=4096, bf16 → ≈32 MiB.

### 4b. `python/sglang/srt/managers/scheduler.py` L866

Add `or self.spec_algorithm.is_medusa()` to the `init_disaggregation`
branch that skips `draft_token_to_kv_pool`. Medusa has no separate draft
model, so it must follow the same path as NGRAM.

This is defensive — `init_disaggregation` is only called in
PD-disaggregated setups (not SOAR), but adding the guard keeps the
contract correct.

### 4c. Files explicitly NOT modified in Stage 2

- `model_runner.py` L1702 (`_is_flashinfer_available` check) — MEDUSA can use
  the same target path as no-spec; no change needed.
- `model_runner.py` L1730 (`_dummy_run` warmup) — Stage 2 stays in DECODE,
  so the `TARGET_VERIFY` branch is not reached.
- `model_runner.py` L1860 (`get_spec_info` in dummy_run) — Stage 2 has no
  spec_info on the batch.
- `cuda_graph_runner.py` L281 (capture forward mode) — Stage 2 stays in
  DECODE; existing default `num_tokens_per_bs=1` is correct.
- `cuda_graph_runner.py` L431 (`is_ngram_supported`) — Medusa Stage 2 has
  `input_ids.numel() == batch_size * 1`; the existing default
  `is_ngram_supported = True` (when MEDUSA is the algorithm) is correct.
- `cuda_graph_runner.py` L909 (capture-side `get_spec_info`) — Stage 2 has
  no spec_info to capture.

Stage 3 will revisit each of these once verify-path forwarding lands.

## 5. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| `target_worker.forward_batch_generation` signature mismatch on `is_verify` kwarg | LOW | Stage 2 never passes `is_verify=True`; verified call site matches default usage in `NGRAMWorker` |
| `MedusaHeads(lm_head=ref)` double-counts parameters during weight load | LOW | `_lm_head` stored as a 1-element list (non-submodule); confirmed in `minicpm_medusa_heads.py` |
| `model.lm_head` missing on edge models | LOW | Explicit `hasattr` guard raises `RuntimeError` with a clear message |
| Server fails to find `--speculative-num-medusa-heads` flag | LOW | Already wired in `server_args.py` (R1a, CHANGE_0153) |
| GPU OOM from 32 MiB head allocation | NEGLIGIBLE | 0.04% of 84 GB GDDR7 |
| `init_disaggregation` regression | LOW | Single-line guard, single-instance SOAR setups never reach it |

## 6. Validation plan (fcloud)

After sync to fcloud and server restart:

```bash
# 1. Baseline check (must remain green from the prior test):
SOAR_SPEC_MEDUSA=0 SOAR_SPEC_NGRAM=0 \
  python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy   # expect ≈79.29% ori_acc

# 2. Stage 2 smoke test:
SOAR_SPEC_MEDUSA=1 SOAR_SPEC_MEDUSA_HEADS=1 \
  python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy   # MUST equal baseline byte-for-byte
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py server-logs --lines 200   # verify "MedusaWorker Stage 2 ready"
```

### Pass criteria

- ✅ Server boots without crashing.
- ✅ Server log contains a line matching `MedusaWorker Stage 2 ready`.
- ✅ Accuracy equal to baseline. **Byte-identical predictions** are
  acceptable; even sub-1% drift indicates a wiring bug.
- ✅ S1 latency regression ≤ 3% vs baseline. Heads are allocated but
  never invoked, so the only overhead is `GenerationBatchResult`
  construction + a Python function call per step.

### Fail-mode triage table

| Symptom | Likely cause | Stage-3 implication |
|---|---|---|
| Server hangs on init | `MedusaHeads` import or weight allocation | fix Stage 2 before Stage 3 |
| Accuracy = baseline but speed ≫ baseline | Some hidden code path treats MEDUSA as EAGLE/NGRAM | review `is_*` checks |
| Accuracy ≠ baseline | `forward_batch_generation` wrapper mutated batch state | fix Stage 2 delegation |
| Boot crash with `KeyError: 'MEDUSA'` | Server-args dispatch regression | re-verify CHANGE_0153 R1a |

## 7. Rollback

- `SOAR_SPEC_MEDUSA=0` in `prepare_env.sh` (the default) fully disables
  the path. The Stage 2 worker class is imported only when MEDUSA is
  selected (via `create_worker`).
- To revert source-level: `git revert <stage2-commit-sha>` reinstalls the
  R1a `NotImplementedError` stub. Reverts cleanly with no dependency
  fallout.

## 8. Hand-off to Stage 3

If fcloud validation passes, Stage 3 will:

1. Set `mwb.capture_hidden_mode = CaptureHiddenMode.LAST` and read
   `logits_output.hidden_states` after the baseline decode forward.
2. Run `self.medusa_heads(hidden)` to produce K draft tokens per request.
3. Wrap drafts into a `NgramVerifyInput` (decision per CHANGE_0154 §7 q2:
   **reuse** to save ~200 LOC; the K=1 chain is structurally compatible).
4. Snapshot SimpleGLA state via the Stage 1 helpers.
5. Call `target_worker.forward_batch_generation(..., is_verify=True)`.
6. Verify, restore state on partial accept, replay accepted prefix.
7. Wire `is_medusa()` into the 6 call sites listed in §4c.

Estimated Stage 3 footprint: ~400 LOC in `medusa_worker.py` plus the
six 1-line guards.

## 9. References

- [CHANGE_0153_medusa_phase_r1_design.en.md](CHANGE_0153_medusa_phase_r1_design.en.md)
- [CHANGE_0154_medusa_phase_r1b_design.en.md](CHANGE_0154_medusa_phase_r1b_design.en.md)
- Reference impl: [ngram_worker.py L213](../../python/sglang/srt/speculative/ngram_worker.py#L213) (forward_batch_generation pattern)
