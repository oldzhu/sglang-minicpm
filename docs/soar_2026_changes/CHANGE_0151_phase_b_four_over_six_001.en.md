# CHANGE_0151 — Phase B FourOverSix NVFP4 (continuation 001: iter 2 + verdict)

Continuation of [CHANGE_0151_phase_b_four_over_six.en.md](CHANGE_0151_phase_b_four_over_six.en.md).
Branch HEAD at end of this iteration: `a6b34a41a` on `minicpm-src/mixed_minicpm_cudagraph`.

## Background

Iter 1 (CHANGE_0151) produced **ori 75.98%** (< 77% gate, C=0). Round 2 of this
session re-ran the SAME iter-1 ckpt and got **70.27%** with `mcq` runaway-think
(avg_out=10124), confirming the result was scheduling-induced run-to-run
variance — same pattern documented for GPTQ Tests 30/32/33 in
[TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md).

Iter 2 hypothesized two stabilizers:
1. **Conservative scheduling** (Test 12 family): `chunk=32K`, `prefill-max-req=1`,
   `schedule-conservativeness=1.0`, `torch-compile-max-bs=8`. Eliminates
   aggressive batching that interacts badly with long thinking blocks.
2. **Longer + task-balanced calibration**: `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=16384`
   (was the 4096 default in `preprocess_model.py`) and 90 stratified samples
   from `qa,mcq,cwe` tasks instead of the 32 sequential samples that the
   default `SOAR_GPTQ_CALIBRATION_*` env produces.

## Implementation

`benchmark/soar/demo_sala/prepare_env.sh` (commit `a6b34a41a`) — inside the
`SOAR_QUANT_PROFILE=nvfp4_fos` branch:

```bash
export SOAR_NVFP4_FOUR_OVER_SIX="${SOAR_NVFP4_FOUR_OVER_SIX:-1}"
export SOAR_NVFP4_MAX_CALIB_SEQ_LEN="${SOAR_NVFP4_MAX_CALIB_SEQ_LEN:-16384}"
export SOAR_TIER1_LONG_CONTEXT=0   # forces conservative scheduling branch
export SOAR_TORCH_COMPILE_MAX_BS="${SOAR_TORCH_COMPILE_MAX_BS:-8}"
```

`SOAR_GPTQ_CALIBRATION_*` defaults later in the same file (samples=90,
stratified, task_include=qa,mcq,cwe) are inherited because the
`load_calibration_texts` helper is shared between the GPTQ and NVFP4 paths.

## Validation commands

```bash
# Quantize (sync exec — no nohup, runs ~3 min)
python3 scripts/fcloud/fcloud_exec.py exec \
  'cd /root/submission_sim && export SOAR_QUANT_PROFILE=nvfp4_fos SOAR_NVFP4_FOUR_OVER_SIX=1 \
   SOAR_QUANT_FORCE=1 && source ./prepare_env.sh >/tmp/prep.log 2>&1; \
   python3 -u preprocess_model.py --input /root/models/openbmb/MiniCPM-SALA \
     --output /root/models/MiniCPM-SALA-NVFP4-FOS --mode nvfp4' --timeout 2400

# Copy tokenizer (modelopt manual export does not include it)
python3 scripts/fcloud/fcloud_exec.py exec \
  'cp /root/models/openbmb/MiniCPM-SALA/tokenizer.* \
      /root/models/openbmb/MiniCPM-SALA/special_tokens_map.json \
      /root/models/MiniCPM-SALA-NVFP4-FOS/'

# Server, accuracy x2, speed
python3 scripts/fcloud/fcloud_workflow.py restart-server \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --env SOAR_QUANT_PROFILE=nvfp4_fos --env SOAR_NVFP4_FOUR_OVER_SIX=1
python3 scripts/fcloud/fcloud_workflow.py accuracy --model-path /root/models/MiniCPM-SALA-NVFP4-FOS  # x2
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## Results — iter 2 is a regression on iter 1

### Accuracy (2 runs)

| Run | ori_acc | norm_acc | duration | mcq | qa | niah | cwe | fwe | TPS |
|----:|--------:|---------:|---------:|----:|---:|-----:|----:|----:|----:|
| Iter 1 run 1 | 75.98% | (gate) | — | 63.33 | 56.67 | 93.33 | 74.33 | 92.22 | — |
| Iter 1 run 2 | 70.27% | — | 2536.95s | 46.67 | 50.00 | 83.33 | 74.67 | 96.67 | 319.44 |
| **Iter 2 run 1** | **60.73%** | 75.92% | 3614.90s | 50.00 | 50.00 | **73.33** | 60.33 | **70.00** | 227.36 |
| **Iter 2 run 2** | **63.31%** | 79.14% | 3631.80s | 53.33 | **36.67** | 70.00 | 74.33 | 82.22 | 262.18 |

- **Iter 2 mean ori = 62.02%** (vs iter 1 mean ≈ 73.13%) — **−11pt regression**.
- Variance is still high (60.73 vs 63.31, ±1.3pt) but the FAILURE MODE
  has shifted: in iter 1 mcq runs away; in iter 2 fwe (run 2) or niah/cwe
  (run 1) collapse instead. The runaway-think generation is not eliminated,
  it just rotates between tasks.
- FOS pct_m4 = **43.14%** in all three quantize runs of this session
  (sequential-32 default, stratified-90+4096, stratified-90+16384) —
  confirming FOS scale selection is **deterministic from weights only**;
  calibration data does not affect M=4 vs M=6 picks.
- The accuracy regression therefore comes from **changed activation amax**
  during calibration (more `qa/mcq/cwe`-biased + longer context → different
  per-tensor scales → degraded `niah/fwe` performance).

### Speed

| | S1 | S8 | Smax |
|---|---:|---:|---:|
| GPTQ T12 baseline | 121.71s | 44.09s | 35.86s |
| Iter 1 (NVFP4-FOS, Tier1 long-ctx) | 175.08s | 47.37s | 31.01s |
| **Iter 2 (NVFP4-FOS, conservative)** | **173.69s** | **45.95s** | **34.39s** |

- S1/S8 essentially unchanged from iter 1; conservative scheduling did not
  recover S1 throughput on the local short-prompt set.
- Smax slowed −11% (31.01s → 34.39s) because `torch-compile-max-bs=8`
  drops compiled CUDA graphs for batches in [9, 24].

## Verdict — Phase B FOS does not work as a bolt-on

Two iterations failed the C ≠ 0 gate:
- **Iter 1**: 75.98% / 70.27% (variance straddling the 77% line, mean 73%).
- **Iter 2**: 60.73% / 63.31% (clean regression below 77%, mean 62%).

The hypothesis that iter 2 (conservative scheduling + longer/balanced calib)
would stabilize accuracy is **rejected**. Both changes either had no effect on
the variance source (scheduling) or made things worse (calibration).

Root cause analysis points to **activation outliers in MiniCPM-SALA's
`niah`/`fwe`/`qa` long-context paths that NVFP4's blockwise FP4 cannot
represent at 16-element granularity**. FOS only adjusts the scale per block
(weight-side); it cannot widen the dynamic range of the activations passing
through `input_quantizer`. The model needs either:

1. **Mixed precision**: keep `q_proj/k_proj/v_proj` (or specifically the
   layers feeding sparse/lightning attention) at int8 or bf16, NVFP4 only
   on MLP. Iter 1's "next options" listed this as option B (large effort).
2. **NVFP4 with INT8 `input_quantizer`** instead of FP8 input scales — but
   this requires modelopt config changes we have not yet validated.
3. **Skip Phase B entirely** and return to GPTQ + FP8 KV speed work, which
   is the current best-known config (T12: ori 79.29%, norm 99.11%, C=1.0).

## Rollback

```bash
git revert a6b34a41a   # iter-2 prepare_env settings
# OR explicitly switch profile back to gptq:
export SOAR_QUANT_PROFILE=gptq
```

The iter-2 quantized ckpt at `/root/models/MiniCPM-SALA-NVFP4-FOS` can be
deleted (`rm -rf`) — iter 1's CHANGE_0151 ckpt was overwritten in this
iteration, but the recipe to regenerate either is the prepare_env profile
switch.

## Next-step suggestions

Given two iterations of FOS have failed to clear the 77% gate, recommended
priority is:

1. **Park Phase B FOS** until we have a clear plan for fixing activation
   quantization on niah/fwe paths (mixed precision or different
   `input_quantizer` config).
2. **Return to GPTQ+FP8+dense** baseline (T12 family). Optimization vectors
   from `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` that have not been
   exhausted: scheduling sweep at long context, sparse attention re-enable
   probe, kernel fusions.
3. If we MUST pursue NVFP4: try plain NVFP4 (no FOS, `SOAR_QUANT_PROFILE=nvfp4`)
   with the iter-1 default calibration (32 sequential, 4096 seq) — tests the
   "iter 1 worked because of LUCKY calibration sample selection" hypothesis
   without the overhead of designing a new mixed-precision recipe.

Updated test-results rows are in [TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md)
under Phase B (NVFP4-FOS-2 and NVFP4-FOS-2b).
