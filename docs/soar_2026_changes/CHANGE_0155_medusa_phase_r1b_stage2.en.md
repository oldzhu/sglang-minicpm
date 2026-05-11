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

## 10. Stage 2 fcloud bring-up addendum (2026-05-11)

**First fcloud restart with `SOAR_SPEC_MEDUSA=1`** confirmed the worker
instantiates correctly:

```
[2026-05-11 00:41:32] MedusaWorker Stage 2 ready: K=1, hidden=4096,
  dtype=torch.bfloat16, device=cuda:0, approx_weight_MiB=32.0
```

but the **first prefill request crashed**:

```
File .../medusa_worker.py L150, in forward_batch_generation
    batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
File .../model_runner.py L2251, in _forward_raw
    ret = self.graph_runner.replay(...)
File .../input_buffers.py L156, in populate_from_forward_batch
    self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
RuntimeError: output with shape [1] doesn't match the broadcast shape [7]
```

**Root cause.** Two pieces of sglang internal logic interact badly when
MEDUSA is registered but no draft tokens are produced:

1. [cuda_graph_runner.py L281](../../python/sglang/srt/model_executor/cuda_graph_runner.py#L281)
   only sets `num_tokens_per_bs = speculative_num_draft_tokens` for
   EAGLE / STANDALONE / NGRAM. MEDUSA falls through to the default branch
   (`num_tokens_per_bs = 1`), so all 16 graph captures (936 s wall) used the
   normal-decode shape.
2. But `--speculative-algorithm MEDUSA` is set, so
   [scheduler.py L2225](../../python/sglang/srt/managers/scheduler.py#L2225)
   takes the spec-v1 branch and passes the raw `ScheduleBatch` (not a
   `ModelWorkerBatch`) into `model_worker.forward_batch_generation`. Combined
   with `--enable-torch-compile`, the prefill request flows into
   `graph_runner.replay`, where `raw_num_token = 1 * 1 = 1` cannot broadcast
   the 7-token prefill `input_ids`.

**Fix (chosen path).** Per CHANGE_0154 §2 we **already committed to eager-only
in R1b**; CUDA-graph capture for the verify path is deferred to R1c. So we
simply make `prepare_env.sh` strip `--enable-torch-compile` /
`--torch-compile-max-bs N` and append `--disable-cuda-graph` whenever
`SOAR_SPEC_MEDUSA=1`:

```bash
if [[ "$SOAR_SPEC_MEDUSA" == "1" ... ]]; then
    NUM_DRAFT_TOKENS=$(( SOAR_SPEC_MEDUSA_HEADS + 1 ))
    export SGLANG_SERVER_ARGS="... --speculative-algorithm MEDUSA \
        --speculative-num-medusa-heads ${SOAR_SPEC_MEDUSA_HEADS} \
        --speculative-num-draft-tokens ${NUM_DRAFT_TOKENS}"
    # Eager-only for Stage 2/3; CUDA-graph capture lands in R1c.
    export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS//--enable-torch-compile/}"
    export SGLANG_SERVER_ARGS="$(echo "$SGLANG_SERVER_ARGS" | sed -E 's/--torch-compile-max-bs [0-9]+//g')"
    export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS} --disable-cuda-graph"
fi
```

**Implications for benchmarks.** With CUDA graph + torch.compile disabled,
Stage 2 will be slower than the v22 baseline (rough estimate: 1.3–2× regression
on S1/S8/Smax). This is **expected and acceptable for Stage 2 validation**:
the goal is byte-identical accuracy and a working dispatch path. The speed
regression will be removed in **R1c** when we add a dedicated TARGET_VERIFY
graph capture branch for MEDUSA in `cuda_graph_runner.py`.

**Validation expectation (re-run).** Accuracy: byte-identical to v22 baseline.
Speed: significant regression (eager mode); we record but do not gate on it.

**Files touched (delta).**
- [benchmark/soar/demo_sala/prepare_env.sh](../../benchmark/soar/demo_sala/prepare_env.sh)
  — strip torch-compile + append `--disable-cuda-graph` in the MEDUSA branch.

## 11. Stage 2 fcloud bring-up addendum #2 — decode-prep bug (2026-05-11)

**Re-test with eager mode** progressed past the cuda-graph crash: the **first
prefill request succeeded** (Marlin GEMM ran at M=7). But the **second
forward call** (the first decode step) crashed:

```
RuntimeError: Number of tokens in position_ids must match QKV
```

at `fused_qk_norm_rope` with `M=7` (i.e. the decode batch still carried 7
input_ids from the prefill).

**Root cause.** [schedule_batch.py L1948](../../python/sglang/srt/managers/schedule_batch.py#L1948):

```python
def prepare_for_decode(self):
    self.forward_mode = ForwardMode.DECODE
    ...
    if not self.spec_algorithm.is_none():
        # if spec decoding is used, the decode batch is prepared inside
        # `forward_batch_speculative_generation` after running draft models.
        return   # ← early return
```

The scheduler calls `batch.prepare_for_decode()` between iterations. When
`spec_algorithm != NONE`, it flips `forward_mode` to `DECODE` but **skips
the field updates** (`input_ids = output_ids`, `seq_lens += 1`, `alloc_for_decode`,
etc.) because EAGLE/NGRAM workers do that themselves inside
`_prepare_for_speculative_decoding` → `prepare_for_verify`.

Stage 2 MedusaWorker is a **pure pass-through** and has no
`_prepare_for_speculative_decoding`. So the decode batch reaches our worker
with `forward_mode = DECODE` but `input_ids` still holding the stale prefill
[7 tokens]. Downstream `_forward_raw` then dispatches to `forward_decode`,
but `positions = clamp_position(seq_lens=[7])` has length 1, while
`input_ids` has length 7 → fused_qk_norm_rope assertion fires.

**Fix.** In `MedusaWorker.forward_batch_generation`, when the incoming
batch is `DECODE`, temporarily set `batch.spec_algorithm = NONE` and call
`prepare_for_decode()` again to finish the skipped prep, then restore the
original spec_algorithm:

```python
if batch.forward_mode.is_decode():
    saved_spec_algo = batch.spec_algorithm
    batch.spec_algorithm = SpeculativeAlgorithm.NONE
    try:
        batch.prepare_for_decode()
    finally:
        batch.spec_algorithm = saved_spec_algo
```

This is **diagnostic-grade evidence** captured via temporary debug logging
(committed `db839a4b1`, removed in the fix commit). The two log lines
showed:

```
[MedusaWorker.fbg] batch.forward_mode=EXTEND ... input_ids_len=7   ← prefill OK
[MedusaWorker.fbg] batch.forward_mode=DECODE ... input_ids_len=7   ← stale, crashes
```

**Why this is safe for Stage 3.** Stage 3 (full verify+rewind) will add
`_prepare_for_speculative_decoding` that runs BEFORE the decode dispatch
and flips `forward_mode` to `TARGET_VERIFY`. The `is_decode()` branch added
here will be skipped naturally because `forward_mode == TARGET_VERIFY` at
that point. So this Stage 2 patch does not block Stage 3.

**Files touched (delta #2).**
- [python/sglang/srt/speculative/medusa_worker.py](../../python/sglang/srt/speculative/medusa_worker.py)
  — finish the scheduler-skipped `prepare_for_decode` when the batch is in
  decode mode.

## 12. Stage 2 fcloud bring-up addendum #3 — runaway decode / output_ids never appended (2026-05-11)

**Re-test after addendum #2** removed the crash but exposed a worse bug:
**every decode runs forever**. The warmup request (configured for
`max_new_tokens=30`) generated **28,000+ tokens** at ~85 tok/s and only
stopped when the scheduler was killed manually.

**Root cause.** A *second* scheduler-side gate on `spec_algorithm`:
[scheduler_output_processor_mixin.py L398-L405](../../python/sglang/srt/managers/scheduler_output_processor_mixin.py#L398-L405)

```python
if batch.spec_algorithm.is_none():
    req.output_ids.append(next_token_id)
elif batch.is_spec_v2:
    req.output_ids.extend(next_token_id)
    new_accepted_len = len(next_token_id)
# else: NOTHING — req.output_ids is never appended
```

Medusa v1 satisfies **neither** `is_none()` **nor** `is_spec_v2`, so
`req.output_ids` is never grown. The very next line
`req.check_finished(new_accepted_len=1)` therefore sees `len(output_ids)` stuck
at zero and never fires `max_new_tokens` / EOS conditions → infinite decode.

**Fix.** Stage 2 is a *pure pass-through* with no real spec activity, so
we flip `batch.spec_algorithm = NONE` **permanently** on entry to
`MedusaWorker.forward_batch_generation`. This:

1. Makes `process_batch_result_decode` take the `is_none()` branch →
   `req.output_ids.append(next_token_id)` runs → `check_finished()` works.
2. Makes `scheduler.update_running_batch` call `prepare_for_decode` on the
   **next** iteration with `spec_algorithm = NONE` → full body runs natively
   (no manual re-run needed for iterations after the first).
3. Skips `update_spec_metrics` (no spec is actually happening — accurate).

The first-decode prep gap from addendum #2 still needs handling because
the scheduler already called the early-return form before our worker was
entered. So we keep the conditional `prepare_for_decode()` re-run for the
first decode call:

```python
needs_redo_prep = (
    batch.forward_mode.is_decode() and not batch.spec_algorithm.is_none()
)
if not batch.spec_algorithm.is_none():
    batch.spec_algorithm = SpeculativeAlgorithm.NONE  # permanent flip
if needs_redo_prep:
    batch.prepare_for_decode()  # finish what scheduler skipped
```

**Why Stage 3 is unaffected.** Stage 3 has a real draft + verify loop. It
will install its own `_prepare_for_speculative_decoding` that flips
`forward_mode` to `TARGET_VERIFY` *before* our worker is entered. The
`is_decode()` gate above is False in verify mode, so `needs_redo_prep` is
False and the spec_algorithm flip is bypassed by `is_none()` check too
(it'll never be `NONE` going in because Stage 3 sets up a Medusa-specific
processing path that respects `is_spec_v2` or equivalent).

**Files touched (delta #3).**
- [python/sglang/srt/speculative/medusa_worker.py](../../python/sglang/srt/speculative/medusa_worker.py)
  — flip `spec_algorithm` to NONE permanently on entry; keep first-decode
  `prepare_for_decode` re-run only when prep was skipped.
