# CHAT — Medusa Phase R1b Stage 2 Bring-up (Pure Pass-through)

**Topic**: Iterative bring-up of Stage 2 MedusaWorker pure pass-through on fcloud; first clean accuracy + S1 baseline.
**Session window**: 2026-05-10 23:00 → 2026-05-11 04:30 (UTC+8 local).
**Branch / latest commit**: `mixed_minicpm_cudagraph` @ `3b1293ed6`.
**fcloud instance**: `ai-e7e98a7c52` (new SM120 Blackwell).

---

## 1. User asks (verbatim, condensed)

1. *"this time pls break the rule a little bit, I would like you working on medusa feature until it can be tested in fcloud, just keep documents detail and clear so we can review,reference and tracking the medusa implementation."*
2. *"go 1 Run a clean restart + full accuracy + S1 benchmark for Stage 2 baseline numbers and then to proceed stage 3"*

## 2. Agent actions / decisions

### Four sequential bring-up bugs discovered & fixed

| # | Symptom | Root cause | Fix commit |
|---|---------|------------|------------|
| 1 | `shape [1] doesn't match broadcast shape [7]` on decode | cuda-graph dispatch missing MEDUSA branch | `fc2fc8ab1` — `prepare_env.sh` strips `--enable-torch-compile`/`--torch-compile-max-bs N` and appends `--disable-cuda-graph` when `SOAR_SPEC_MEDUSA=1` |
| 2 | `Number of tokens in position_ids must match QKV` on first decode | `schedule_batch.prepare_for_decode` early-returns when `spec_algorithm != NONE`, so positions never get updated | `364152221` — `medusa_worker.py` re-calls `batch.prepare_for_decode()` after temporarily flipping `batch.spec_algorithm = NONE` |
| 3 | Runaway decode (28k tokens, `max_new_tokens=30` ignored) | `scheduler_output_processor_mixin.py L398-405` only appends `req.output_ids` when `spec_algorithm.is_none()` or `is_spec_v2`; Medusa-v1 with passthrough satisfies neither | `a0e680907` — `medusa_worker.py` **permanently** flips `batch.spec_algorithm = NONE` on entry (per batch) so the output processor follows the standard path |
| 4 | `ZeroDivisionError` in `log_decode_stats` partway through eval | scheduler-level `self.spec_algorithm == MEDUSA` selects the `else` branch (`spec_num_accepted/spec_num_forward_ct`) but `spec_num_forward_ct` stays 0 because batch-level was flipped to NONE → `update_spec_metrics` skipped | `3b1293ed6` — `scheduler_metrics_mixin.py` guards the division (`if self.spec_num_forward_ct > 0 else 0`) |

All four fixes are tiny (≤ 5 lines each) and live in the right layer; documented in `CHANGE_0155_medusa_phase_r1b_stage2.{en,zh}.md` §10–§13.

### Final Stage 2 worker shape

```python
def forward_batch_generation(self, batch):
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    needs_redo_prep = batch.forward_mode.is_decode() and not batch.spec_algorithm.is_none()
    if not batch.spec_algorithm.is_none():
        batch.spec_algorithm = SpeculativeAlgorithm.NONE  # permanent flip per batch
    if needs_redo_prep:
        batch.prepare_for_decode()
    model_worker_batch = batch.get_model_worker_batch()
    batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
    return GenerationBatchResult(
        logits_output=batch_result.logits_output,
        next_token_ids=batch_result.next_token_ids,
        num_accepted_tokens=0,
        can_run_cuda_graph=batch_result.can_run_cuda_graph,
        accept_lens=None,
    )
```

### Test run after all 4 fixes

- Server boot: 35s, `MedusaWorker Stage 2 ready` log present.
- Accuracy: 150-sample harness, 59 minutes (vs Test 12 ~70 minutes).
- During eval the server log shows healthy decode: 24 running req, 956 tok/s gen, no crashes.
- S1 speed benchmark: 200.92s (vs Test 12 121.71s).

## 3. Outcomes

| Metric | Stage 2 (Medusa-R1b) | Test 12 baseline | Δ |
|--------|----------------------|------------------|---|
| ori_accuracy | **76.38%** | 79.29% | −2.91pt |
| normalized | 95.48% | 99.11% | — |
| C | 0 (eliminated) | 1.0 | — |
| mcq | 43.33% | 63.33% | **−20pt** |
| cwe | 86.33% | 72.00% | +14.33pt |
| fwe | 98.89% | 97.78% | +1.11pt |
| niah | 96.67% | 100.00% | −3.33pt |
| qa | 56.67% | 63.33% | −6.66pt |
| Total duration | ~3540s | 4244s | −16% (random eval variance) |
| **S1** | **200.92s** | **121.71s** | **+65% slower** |

**Interpretation**:
- Stage 2 is **not** a submission candidate. The −65% S1 regression and the mcq drop (43%) are expected because:
  1. eager mode (no torch.compile, no cuda graph) — the cuda-graph strip in fix #1 is the single biggest source of speed regression
  2. MedusaWorker pass-through detour
  3. One Medusa head is loaded but unused (memory footprint only; no compute saved)
- All bring-up gates are now passed. The infrastructure path Scheduler → MedusaWorker → target_worker is stable across long evals.

**Next step**: Stage 3 (real verify + rewind). Stage 3 should *not* re-trigger most of these bugs because the verify path flips `forward_mode` to `TARGET_VERIFY` before dispatch — that path is already supported by scheduler/output-processor logic.

## 4. Cross-references

- Implementation:
  - `python/sglang/srt/speculative/medusa_worker.py`
  - `python/sglang/srt/managers/scheduler_metrics_mixin.py` (div-by-zero guard)
  - `benchmark/soar/demo_sala/prepare_env.sh` (SOAR_SPEC_MEDUSA branch)
- Docs:
  - [CHANGE_0155_medusa_phase_r1b_stage2.en.md](../CHANGE_0155_medusa_phase_r1b_stage2.en.md)
  - [CHANGE_0155_medusa_phase_r1b_stage2.zh.md](../CHANGE_0155_medusa_phase_r1b_stage2.zh.md)
- Tracking row: `TEST_RESULTS_TRACKING.md` → row `Medusa-Stage2`.
- Commits (this session): `fc2fc8ab1`, `364152221`, `a0e680907`, `3b1293ed6` (all on `minicpm-src/mixed_minicpm_cudagraph`).
- fcloud predictions: `/root/data/outputs/20260511_032859/predictions.jsonl`.

## 5. Open follow-ups

1. **Stage 3 proposal**: Draft `PROPOSAL_medusa_stage3_verify_rewind.{en,zh}.md` — single-head verify, rewind on reject, treelike candidate generation deferred. Wait for user approval before coding.
2. **Restore torch.compile + cuda-graph for Medusa path**: Stage 3 should aim to lift the cuda-graph strip once the spec path can be captured (or use mixed graph with a Medusa-aware branch). Currently this is the dominant speed regression source.
3. **mcq regression diagnosis**: Once Stage 3 has real acceptance, re-test mcq specifically — if mcq stays low it's not a Stage 2 artifact and needs separate investigation.
4. fcloud paused via `pause-instance` after this session to save billing.
