# CHANGE 0151 — Phase B FourOverSix, continuation 003

Companion to [CHANGE_0151_phase_b_four_over_six_002.en.md](CHANGE_0151_phase_b_four_over_six_002.en.md).

Iter-4 isolates **calibration sequence length** as a variable while holding
calibration *content* (stratified 90 qa,mcq,cwe with `FOS=1`, seed=20260320)
and Tier1 scheduling constant.

## Background

| Iter | Calib content | calib_seq_len | Scheduling | Mean ori | Δ vs iter-1 |
|------|---------------|---------------|------------|----------|-------------|
| 1    | sequential 32 (mixed) | **4096** | Tier1 | ~73.13% | — |
| 2    | stratified 90 qa,mcq,cwe FOS=1 | **16384** | Conservative | ~62.02% | −11pt |
| 3    | stratified 90 qa,mcq,cwe FOS=1 | **16384** | Tier1 | 68.20% | −5pt |
| 4    | stratified 90 qa,mcq,cwe FOS=1 | **4096** | Tier1 | **66.00%** | **−7.13pt** |

Iter 3 demonstrated that scheduling alone recovers ~6pt of the iter-1→iter-2
−11pt regression. Remaining −5pt was attributed to "calibration content
and/or seqlen". Iter 4 holds content fixed at iter-2's choice and reverts
seqlen to iter-1's choice (4096) to attribute that remaining −5pt.

## Implementation

No source patches in this iteration — only a per-invocation environment
override.

Re-quantize on fcloud:

```bash
python3 scripts/fcloud/fcloud_exec.py exec \
  'cd /root/submission_sim && \
   SOAR_QUANT_PROFILE=nvfp4_fos \
   SOAR_NVFP4_FOUR_OVER_SIX=1 \
   SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096 \
   SOAR_GPTQ_CALIBRATION_SAMPLES=90 \
   SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,mcq,cwe \
   SOAR_GPTQ_CALIBRATION_SAMPLING=stratified \
   SOAR_GPTQ_CALIBRATION_SEED=20260320 \
   python3 -u preprocess_model.py \
     --input /root/models/openbmb/MiniCPM-SALA \
     --output /root/models/MiniCPM-SALA-NVFP4-FOS \
     --mode nvfp4'
```

This run also exercised the new **`_init_rope` upstream patcher**
(CHANGE_0152) — verified by:

```
[preprocess][init-rope-patch] mode=nvfp4 dst:
patched /root/models/MiniCPM-SALA-NVFP4-FOS/modeling_minicpm_sala.py
(replaced 2 _init_rope headers, 2 else-branches)
```

`grep -c "transformers>=4.43 standardizes rope_scaling" .../modeling_minicpm_sala.py`
returns **2** as expected.

Iter-4 server launch (identical to iter-3):

```bash
python3 scripts/fcloud/fcloud_workflow.py restart-server \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --env SOAR_QUANT_PROFILE=nvfp4_fos \
  --env SOAR_NVFP4_FOUR_OVER_SIX=1 \
  --env SOAR_TIER1_LONG_CONTEXT=1 \
  --env SOAR_TORCH_COMPILE_MAX_BS=24
```

Live `/get_server_info` confirmed iter-3 family server config
(chunk=65536, prefill_max_req=4, sched_cons=0.8, max_run=24, modelopt_fp4,
KV fp8_e5m2).

## Pre-set abort gate

Per user instruction (carried forward from iter-3): if run-1
ori_accuracy < 70% → skip run-2 + speed bench, pause instance, document, decide
next direction.

## Result — iter-4 run-1

`outputs/20260508_043812/predictions.jsonl`

| Metric | Value |
|--------|-------|
| ori_accuracy (Average Score) | **66.00%** |
| cwe | 66.67% |
| fwe | 80.00% |
| mcq | 46.67% |
| niah | 93.33% |
| qa | 43.33% |
| Total Duration | 3084.38 s |
| Output TPS | 348.97 |
| FOS pct_m4 | 43.14% (unchanged — weight-only) |

**ABORT GATE TRIGGERED.** Run-2 + speed bench skipped. Instance paused.

## Comparison

| Metric | Iter-1 (mean) | Iter-2 (mean) | Iter-3 run-1 | **Iter-4 run-1** |
|--------|---------------|---------------|--------------|------------------|
| ori_accuracy | ~73.13% | ~62.02% | 68.20% | **66.00%** |
| Δ vs iter-1 | — | −11pt | −5pt | **−7.13pt** |
| cwe | (n/a) | ~67.3 | 77.67 | 66.67 |
| fwe | (n/a) | ~76.1 | 76.67 | 80.00 |
| mcq | (n/a) | ~51.7 | 56.67 | **46.67** |
| niah | (n/a) | ~71.7 | 80.00 | **93.33** |
| qa | (n/a) | ~43.3 | 50.00 | 43.33 |

Reducing calib_seq_len from 16384→4096 while keeping the iter-2 stratified
content **did not recover** iter-1 accuracy — in fact accuracy regressed
2.20pt vs iter-3.

## Verdict

The remaining ~5–7pt deficit relative to iter-1 is **NOT** attributable to
calibration sequence length. With Tier1 scheduling held constant and seqlen
matched to iter-1, the mean is still ~7pt below iter-1.

The dominant variable separating iter-1 (73%) from iter-{2,3,4} (62–68%) is
therefore the **calibration content**:
- iter-1: `SOAR_GPTQ_CALIBRATION_SAMPLES=32` sequential mix (default
  task distribution, 5 tasks present)
- iter-{2,3,4}: stratified 90 with `TASK_INCLUDE=qa,mcq,cwe` and `FOS=1`
  (only 3 of 5 tasks; samples picked by 4-over-6 score)

Hypotheses for why iter-2/3/4 calibration is worse:
1. **Task imbalance**: dropping `niah` and `fwe` from calibration starves the
   activation distribution for those two tasks at quantization time. Result
   is matched in iter-4: niah=93.33% (great when Tier1) but mcq/qa drop.
2. **FOS sample bias**: `FOUR_OVER_SIX=1` selects calibration samples whose
   token statistics favor M=4 mantissa scale. This may shrink coverage for
   long-tail activations in mcq/qa.
3. **Sequence-length distribution**: stratified 90 may concentrate length
   buckets differently than sequential 32, even at matched cap.

## Next-step decision

FOS is **parked**. Three forward options:

| Option | Description | Cost | Expected outcome |
|--------|-------------|------|------------------|
| A | Re-quantize: `SAMPLES=32` sequential (no `TASK_INCLUDE`, no `FOS_SCORE`) — exact iter-1 calibration | 1 fcloud iter (~30 min quant + 1× accuracy ~50 min) | Recover iter-1 accuracy → confirms hypothesis. If not → other variable hidden. |
| B | Re-quantize: `SAMPLES=90` stratified across **all 5 tasks** (drop `TASK_INCLUDE` filter, keep `FOS=0`) | 1 fcloud iter | Tests whether task imbalance alone is the regression source. |
| C | Abandon FOS entirely → switch to plain NVFP4 (no four-over-six patch) and apply Phase A-style W4A8 instead | Larger refactor | Different optimization track. |

Recommendation: try **Option A** first as the cheapest direct test of the
content hypothesis. If A recovers iter-1 accuracy (~73%), then Option B (with
all 5 tasks) likely brings further gains.

## Validation commands

```bash
# Verify _init_rope patch present in re-quantized model dir
grep -c "transformers>=4.43 standardizes rope_scaling" \
  /root/models/MiniCPM-SALA-NVFP4-FOS/modeling_minicpm_sala.py
# Expect: 2

# Re-run accuracy
python3 scripts/fcloud/fcloud_workflow.py accuracy \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS
```

## Rollback

No source change in this iteration. To revert the model artefact, simply
re-quantize without `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096` (default 16384
yields iter-3 behavior).

## Cross-references

- Continuation 002: [CHANGE_0151_phase_b_four_over_six_002.en.md](CHANGE_0151_phase_b_four_over_six_002.en.md)
- `_init_rope` patcher: [CHANGE_0152_init_rope_transformers5_compat.en.md](CHANGE_0152_init_rope_transformers5_compat.en.md)
- Test row: TEST_RESULTS_TRACKING.md → NVFP4-FOS-4
- Chat log: chat/CHAT_phase-b-fos-iter4_20260508_1230.en.md
