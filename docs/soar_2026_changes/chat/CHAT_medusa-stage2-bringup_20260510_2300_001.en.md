# CHAT — Medusa Phase R1b Stage 2 — cuda-graph re-enable (continuation _001)

**Topic**: Diagnose why Stage 2 eager baseline was +65% slower on S1; re-enable cuda-graph + torch.compile by gating the strip behind an opt-in env var; validate.
**Session window**: 2026-05-11 07:30 → 2026-05-11 09:00 local.
**Branch / latest commit**: `mixed_minicpm_cudagraph` @ `46553947b` (this round) → `<pending>` (this commit).
**fcloud instance**: `ai-e7e98a7c52`.

## 1. User asks

1. *"before go stage3, how about the crash when cuda enabled at stage2? because the s1 speed +65% slower, I am just concern if don't fix the crash with cuda, medusa itself can't compensate the speed lost without cuda."*
2. *"go"* (approving the proposed plan: try fix A first — flip `spec_algorithm=NONE` earlier; but my actual analysis showed the cuda-graph crash was the same root cause as bug #2, so the fix is just to remove the eager-mode strip).

## 2. Agent analysis

Re-read addendum #1's comment in `prepare_env.sh`:

> the cuda_graph_runner does NOT recognize MEDUSA in its is_eagle/is_standalone/is_ngram branch → captures graphs at num_tokens_per_bs=1. But because spec_algorithm != NONE, scheduler.run_batch routes the ScheduleBatch directly into model_worker.forward_batch_generation, and the prefill batch ends up replayed against a [1]-shape input buffer → "shape [1] doesn't match broadcast shape [N]" crash.

That description is consistent with bug #2 (stale prefill metadata leaking into first decode). After addendum #2 (worker re-runs `prepare_for_decode`) and addendum #3 (permanent flip `batch.spec_algorithm = NONE` before `get_model_worker_batch()`):

- `ForwardBatch.spec_algorithm = NONE`, `num_tokens_per_seq = 1`
- `cuda_graph_runner.get_spec_info()` returns None for MEDUSA (no eagle/standalone/ngram branch) → captured graphs are byte-identical to a normal-decode graph with `num_tokens_per_bs=1`
- Replay: `forward_batch.batch_size=N` → `graph_key=N` → matches captured graph

**Conclusion**: the eager-mode strip in addendum #1 was overkill. Removing it should produce no crash and recover Test 12 speed.

## 3. Implementation

`prepare_env.sh`: introduced `SOAR_SPEC_MEDUSA_EAGER` (default `0`). When unset (the new default), the Medusa path keeps the v22 baseline args (`--enable-torch-compile --torch-compile-max-bs 24`, cuda-graph on). When set to `1`, the old eager-mode strip is reapplied as a rollback toggle.

Single commit: `46553947b` (`feat(medusa-r1b): re-enable cuda-graph + torch.compile for Medusa pass-through (Stage 2 §14)`).

## 4. Results

Server boot: ~257s (cuda-graph capture across 16 bs buckets + torch.compile autotune).
No crashes on first request; eval ran to completion.

| Metric | **Stage 2 (eager)** | **Stage 2 + cuda-graph + torch.compile** | Test 12 baseline | Δ vs Test 12 |
|--------|---------------------|--------------------------------------------|------------------|-------------|
| ori_accuracy | 76.38% | **80.11%** | 79.29% | **+0.82pt** |
| normalized | 95.48% | **100.14%** | 99.11% | — |
| C | 0 | **1.0** | 1.0 | — |
| mcq | 43.33% | 56.67% | 63.33% | −6.66pt (local noise band) |
| cwe | 86.33% | 85.00% | 72.00% | +13.00pt |
| fwe | 98.89% | 98.89% | 97.78% | +1.11pt |
| niah | 96.67% | 100.00% | 100.00% | = |
| qa | 56.67% | 60.00% | 63.33% | −3.33pt |
| Total acc duration | ~3540s | **2845.05s** | 4244s | −33% |
| S1 | 200.92s | **118.28s** | 121.71s | **−2.8%** |
| S8 | n/a | **43.87s** | 44.09s | **−0.5%** |
| Smax | n/a | **35.75s** | 35.86s | **−0.3%** |
| TPOT (S1) | 11.73ms | **6.74ms** | — | — |

**Hypothesis confirmed.** Bug #1 had the same root cause as bug #2; fixing #2/#3 made the eager-mode strip unnecessary. Stage 2 is now infrastructure-equivalent to Test 12 on all 3 speed tiers AND beats it slightly on accuracy.

This is the **clean Stage 2 baseline** — Medusa worker is wired in, cuda-graph + torch.compile work end-to-end, and there is zero speed penalty for going through the MedusaWorker pass-through. Stage 3 (real verify + rewind) can now be evaluated fairly against this baseline.

## 5. Cross-references

- Code: `benchmark/soar/demo_sala/prepare_env.sh` (this round). `python/sglang/srt/speculative/medusa_worker.py` (unchanged — bug #2/#3 fixes from prior round did the heavy lifting).
- Docs: [CHANGE_0155_medusa_phase_r1b_stage2.en.md §14](../CHANGE_0155_medusa_phase_r1b_stage2.en.md) / [.zh.md](../CHANGE_0155_medusa_phase_r1b_stage2.zh.md).
- Tracking row: `TEST_RESULTS_TRACKING.md` → row `Medusa-Stage2-cgraph` (the prior `Medusa-Stage2` eager row marked as superseded).
- Commit: `46553947b` (re-enable change) + this doc commit.
- fcloud predictions: `/root/data/outputs/20260511_075837/predictions.jsonl`.

## 6. Next steps

1. **Proceed to Stage 3** on this baseline. Stage 3 = real Medusa verify (1 head → 1 draft token per seq) + rewind on reject. With Test 12 ≈ Stage 2 here, any S1/S8/Smax improvement from Stage 3 will be cleanly attributable to Medusa acceptance.
2. Stage 3 design constraints (already documented in CHANGE_0154):
   - Draft = 1 (Stage 3); Stage 4 will widen to multi-head tree.
   - Verify forward_mode = `TARGET_VERIFY`; will reuse existing scheduler/output-processor handling, so most R1b bugs won't reappear.
   - Acceptance check: compare draft head's argmax against target's argmax. Accept → write both tokens, advance seq_lens by 2. Reject → write target token only, advance by 1.
3. **Open issue**: Medusa head is still untrained; Stage 3 will show ~50% acceptance ceiling at best (random head ≈ random token). Real gain comes after we either (a) fine-tune the head against MiniCPM-SALA outputs, or (b) initialize the head from the LM head + skip-connection (CHANGE_0154 §6).
