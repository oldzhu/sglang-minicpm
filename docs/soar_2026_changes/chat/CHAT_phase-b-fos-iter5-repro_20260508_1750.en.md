# CHAT — Phase B FOS iter-5 reproducibility + speed bench

Started: 2026-05-08 17:50 local. Ended: 2026-05-08 19:05 local.

## Round 1 — User: "approve"

User approved iter-5 run-2 + S1/S8/Smax speed bench (re-quant iter-5
recipe, then accuracy + speed). See
[CHANGE_0151_phase_b_four_over_six_005.en.md](../CHANGE_0151_phase_b_four_over_six_005.en.md)
Next-steps #1 + #3.

## Agent actions

1. `start-instance` (HTTP 200), wait for JupyterLab (~80 s).
2. `sync` — pulled `cce6b5a7e` into `/root/sglang-minicpm`; no source changes.
3. Re-quantize iter-5 recipe on fcloud:
   `SAMPLES=32 sequential MAX_CALIB_SEQ_LEN=4096 SOAR_QUANT_PROFILE=nvfp4_fos
   SOAR_NVFP4_FOUR_OVER_SIX=1 SEED=20260320`. Quant log:
   `pct_m4=43.14%`, `tokenizer.save_pretrained complete`. Tokenizer-save
   fix from CHANGE_0151_005 worked first try (no manual `cp`).
4. `restart-server --quant-mode gptq --model-path .../MiniCPM-SALA-NVFP4-FOS
   --env SOAR_QUANT_PROFILE=nvfp4_fos --env SOAR_NVFP4_FOUR_OVER_SIX=1
   --env SOAR_TIER1_LONG_CONTEXT=1 --env SOAR_TORCH_COMPILE_MAX_BS=24`,
   `wait-server` ready after second pass (torch.compile warmup ~80 s).
5. `accuracy`: **70.98%** (vs iter-5 run-1 71.24%, Δ −0.26pt). Duration
   2513.84 s.
6. `speed --variant all`: **S1=173.83 s, S8=46.05 s, Smax=31.07 s**.
7. `pause-instance` (504 → retry 200 OK).

## Outcome

### Accuracy reproducibility

iter-5 run-1 / run-2 = 71.24% / 70.98%. **Reproducible at the overall
level (±0.26pt)** but per-task variance is ±10pt:

| task | r1 | r2 | Δ |
|------|---:|---:|---:|
| cwe  | 70.67 | 76.00 | +5.33 |
| fwe  | 92.22 | 85.56 | −6.66 |
| mcq  | 53.33 | 46.67 | −6.66 |
| niah | 90.00 | 100.00 | +10.00 |
| qa   | 50.00 | 46.67 | −3.33 |

### Speed comparison (NVFP4-FOS vs Test 12 GPTQ baseline)

| Tier | Weight | NVFP4-FOS | GPTQ (Test 12) | Δ |
|------|-------:|----------:|---------------:|---:|
| S1   | 40% | 173.83 s | 121.71 s | +52.12 s (NVFP4 slower) |
| S8   | 30% |  46.05 s |  44.09 s |  +1.96 s |
| Smax | 30% |  31.07 s |  35.86 s |  −4.79 s (NVFP4 faster) |

Score head-to-head: GPTQ 96.0 vs NVFP4 86.7 (S1's 40% weight dominates).
With NVFP4-FOS accuracy at ~71% (normalized 89% < 97%) → **C = 0 → Final
score = 0**. **NVFP4-FOS not submission-ready.**

## Decision

1. NVFP4-FOS stays on side branch; not in submission package.
2. GPTQ baseline (Test 12) remains submission baseline.
3. Park further FOS calibration tuning. Resume GPTQ optimization catalog.

## Cross-references

- [CHANGE_0151_phase_b_four_over_six_006.en.md](../CHANGE_0151_phase_b_four_over_six_006.en.md)
- [CHANGE_0151_phase_b_four_over_six_006.zh.md](../CHANGE_0151_phase_b_four_over_six_006.zh.md)
- [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md) row `NVFP4-FOS-7`
- Validates tokenizer-save fix from `CHANGE_0151_005` on a real FOS=1 quant

## Open questions / follow-ups

1. mcq runaway-think mitigation via server-side `generation_config` —
   could recover 5–10pt across tasks (because failure mass redistributes).
   Apply to GPTQ baseline too (no eval-script edits).
2. NVFP4 S1 latency profiling — 52 s overhead vs GPTQ on a single-request
   workload. Likely kernel launch / dequant / KV lookup overhead.
3. Resume GPTQ catalog priorities (OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md).
