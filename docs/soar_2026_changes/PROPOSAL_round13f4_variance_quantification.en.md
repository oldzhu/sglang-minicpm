# PROPOSAL — Round 13f-4: Variance quantification before declaring Round 13f-1 dead

## Status: PROPOSAL (awaiting approval)

## Background

Round 13f-1 (`SOAR_BACKEND_VARIANT=flashinfer`, no force-dense, no
dense-as-sparse) measured ori_acc=76.91% with significant speed gain
(S₁=110.76 / S₈=40.50 / Smax=33.66; −9% / −8% / −6% vs Test 12).
Round 13f-2 (76.91 → 75.80) and 13f-3 (74.73) added flag combinations and
both regressed further; **but** all three were single runs.

The conclusion "13f-1 is C=0 dead" rests on comparing 13f-1's 76.91% to
**Test 12's old-fcloud 79.29%**. That reference predates the current fcloud
instance. On this new instance, the same Test 12 config has been re-tested
**four times** with the following results:

| Test | Date | Same config (Test 12 baseline) | acc |
|---|---|---|---|
| Test 12 (old fcloud) | 2026-04-12 | reference | 79.29% |
| Test 29 | 2026-04-22 | new fcloud baseline | 78.73% |
| Test 30 | 2026-04-22 | KV e4m3 (close) | 77.96% |
| Test 33 | 2026-04-22 | no torch.compile | 76.98% |
| Test 34a | 2026-04-23 | new-instance replay | 77.51% |

**Mean of 4 new-fcloud runs ≈ 77.8%, range 76.98–78.73%.**
13f-1's 76.91% sits at the bottom of that noise band — within ±0.9pt of
Test 33's 76.98% under nominally the same baseline config.

The TEST_RESULTS_TRACKING already records (Test 33 row):
*"no single-variable knob fixes the ~77–79% local floor — confirms high
intrinsic eval variance."*

So before pursuing expensive remedies (re-calibration, re-quantization,
attention-kernel patching), we should answer one question: **is the
13f-1↔Test 12 gap real, or noise?**

## Hypothesis

If 13f-1's 76.91% is within Test 12's true noise band on this fcloud,
13f-1 is a viable submission candidate — possibly C=0.92 (norm 92–97%)
or C=1.0 if the official private-set lottery breaks favorably — and
the speed gain (~9% S₁) is free.

If 13f-1's mean is reproducibly below Test 12's mean by ≥1.5pt, only
then is re-calibration / re-quantization justified.

## Plan

Single fcloud session, 4 alternating accuracy runs (no code changes,
only env-var flips between restarts). Total ≈ 3.5 hours.

| Run | Backend variant | Cmdline marker | Expected timing |
|---|---|---|---|
| **A1** | Test 12 baseline | `--attention-backend minicpm_flashinfer --force-dense-minicpm --dense-as-sparse` | ~50 min |
| **A2** | 13f-1 | `--attention-backend flashinfer` (no force-dense, no DAS) | ~48 min |
| **A3** | Test 12 baseline (re-run) | same as A1 | ~50 min |
| **A4** | 13f-1 (re-run) | same as A2 | ~48 min |

Restart server between each run. Record ori_acc and per-task acc for all 4.

### Statistical decision

Compute `gap = mean(A2,A4) − mean(A1,A3)`.

| Outcome | gap | Decision |
|---|---|---|
| **Variance-dominated** | \|gap\| ≤ 1.0pt | **13f-1 is statistically equivalent to Test 12 on this fcloud.** Greenlight 13f-1 as v20 submission candidate (provisional). Speed gain captured. Run S1/S8/Smax (already done in 13f-1: 110.76/40.50/33.66). Cut tarball. Submit. Let private-set decide C tier. |
| **Borderline** | 1.0–1.5pt | Run A5/A6 (one more cycle) to tighten. If still borderline, submit BOTH variants for direct A/B on official harness (one v20a = Test 12, v20b = 13f-1). |
| **Real regression** | > 1.5pt | 13f-1 has a true acc gap. Proceed to **Phase B (per-sample diff)** — `predictions.jsonl` from A1+A2 runs already on disk. Diff sample-by-sample to characterize divergence type (1-token vs whole-sentence). This costs zero fcloud time. Only after Phase B suggests "kernel numerical noise" do we propose re-calibration. |

### Rule compliance

- No code changes (only env-var flips).
- No eval-script changes.
- Accuracy gate: 13f-1 must reach ≥77% in at least one of A2/A4 to be
  considered a candidate (otherwise C=0 risk too high regardless of variance).
- Compatible with submission constraints (≤2GB, ≤5h, no prefix cache).

## Test commands (after user approval)

```bash
# (User starts fcloud)

# A1 — Test 12 baseline
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server   # no env -> Test 12 default
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_exec.py exec "pgrep -af sglang.launch_server | head -1"
# verify: --attention-backend minicpm_flashinfer --force-dense-minicpm --dense-as-sparse
python3 scripts/fcloud/fcloud_workflow.py accuracy
# record A1 ori_acc, save predictions.jsonl path

# A2 — 13f-1
python3 scripts/fcloud/fcloud_workflow.py restart-server --env SOAR_BACKEND_VARIANT=flashinfer
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_exec.py exec "pgrep -af sglang.launch_server | head -1"
# verify: --attention-backend flashinfer (NO --force-dense-minicpm, NO --dense-as-sparse)
python3 scripts/fcloud/fcloud_workflow.py accuracy

# A3 — Test 12 re-run
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy

# A4 — 13f-1 re-run
python3 scripts/fcloud/fcloud_workflow.py restart-server --env SOAR_BACKEND_VARIANT=flashinfer
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy

python3 scripts/fcloud/fcloud_workflow.py shutdown
```

## Deliverable

After all 4 runs, agent will:
1. Update `TEST_RESULTS_TRACKING.md` with rows R13f4-A1 / A2 / A3 / A4.
2. Compute mean ± half-range for each variant.
3. Apply decision matrix above and recommend next step (submit, repeat, or Phase B).

## Why this before re-quantization

- **Re-calibration cost**: 5+ hours of fcloud time (calibration + quantization
  + boot test + accuracy run). Today the official-eval calibration script
  uses sparse-mode forward; switching it to flashinfer would also require
  source edits to `gptqmodel_minicpm_sala.py`.
- **Variance run cost**: ~3.5 hours, no code changes, decisive answer either
  direction.
- **Information value**: re-calibration only helps if the bug is "calibration
  saw different activations than runtime sees." Phase A tells us if there
  IS a bug at all.
- **Failure mode if we skip A**: spend 5+ hours re-calibrating, find
  acc=77.5% (within noise of plain 13f-1), conclude nothing.

## Rollback

No code changes. Nothing to roll back.

## Cross-references

- 13f-1 result: `R13f1-flashinfer` row.
- 13f-2 result: `R13f2-flashinfer-keepforce` row.
- 13f-3 result: `R13f3-flashinfer-keepall` row.
- New-fcloud baseline noise: Tests 29/30/33/34a rows.
- Failed prior 13f line of reasoning: [RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.en.md](RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.en.md), [PROPOSAL_round13f2_flashinfer_keep_force_dense.en.md](PROPOSAL_round13f2_flashinfer_keep_force_dense.en.md).
