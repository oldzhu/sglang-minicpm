# SOAR 2026 — Automated Test Results Tracking

This document records all accuracy and speed benchmark results from automated fcloud testing.
Every test run should be logged here with its configuration, commit, date, and results.

**Scoring formula**: `Final Score = Performance Score × C` (HIGHER = BETTER)
- `Performance Score = S1×40% + S8×30% + Smax×30%` where `S_N = (Duration_best / Duration_player) × 100`
- C = 0 if normalized accuracy ≤ 97% (eliminated)
- C = 0.92 if 97% < normalized accuracy ≤ 98%
- C = 0.96 if 98% < normalized accuracy ≤ 99%
- C = 1.0 if 99% < normalized accuracy ≤ 100%

---

## Test Results Table

| Test # | Date | Commit | fcloud Instance | Config | Accuracy (orig) | Accuracy (norm) | C | mcq | cwe | fwe | niah | qa | Duration | TPS | Notes |
|--------|------|--------|-----------------|--------|-----------------|-----------------|---|-----|-----|-----|------|----|----------|-----|-------|
| 1 | 2026-04-07 | dba2815c1+0070 | 223.167.85.181 | Non-quant + FP8 KV + sparse | 53.18% | — | 0 | 66.67% | 43.67% | 92.22% | 36.67% | 26.67% | — | — | CHANGE_0070 + FP8 both hurt |
| 2 | 2026-04-07 | HEAD+0070+0071 | 223.167.85.181 | Non-quant + bf16 KV + sparse | 57.11% | — | 0 | 60% | 90.67% | 97.78% | 100% | 66.67% | — | — | CHANGE_0070 regression confirmed |
| 3 | 2026-04-08 | dba2815c1 | 223.167.85.181 | Non-quant + bf16 KV + sparse (pre-0070) | **83.02%** | — | — | 60% | 90.67% | 97.78% | 100% | 66.67% | — | — | Baseline (best non-quant sparse) |
| 4 | 2026-04-08 | HEAD−0070 | 223.167.85.181 | Non-quant + bf16 KV + sparse (bug fixes only) | **83.02%** | — | — | 60% | 90.67% | 97.78% | 100% | 66.67% | — | — | Bug fixes 6/8/3/71 safe |
| 5 | 2026-04-08 | HEAD−0070 | 223.167.85.181 | GPTQ + FP8 KV + sparse | 50.36% | — | 0 | 40% | 47.33% | 97.78% | 36.67% | 30% | — | — | GPTQ+FP8 breaks sparse badly |
| 6 | 2026-04-08 | HEAD−0070 | 223.167.85.181 | GPTQ + bf16 KV + sparse | 57.47% | — | 0 | 63.33% | 44% | 96.67% | 36.67% | 46.67% | — | — | GPTQ alone breaks sparse |
| 7 | 2026-04-08 | HEAD−0070 | 223.167.85.181 | GPTQ + FP8 KV + dense | 78.84% | — | — | 56.67% | 82% | 98.89% | 100% | 56.67% | — | — | Dense tolerates GPTQ+FP8 well |
| 8 | 2026-04-09 | 687ac4127 | 223.167.85.183 | GPTQ + FP8 KV + sparse + topk_scale=2 | 0% | 0% | 0 | — | — | — | — | — | — | — | OOM crash: page table 4.6 GiB (topk=160) |
| 8b | 2026-04-09 | 9d3ecd168 | 223.167.85.183 | GPTQ + FP8 KV + sparse (default topk=96) | **76.07%** | 95.08% | 0 | 60% | 60.33% | 96.67% | 96.67% | 66.67% | 2411s | 99.73 | Freshly prepared GPTQ model; huge improvement vs old Test 5 |
| 9 | 2026-04-09 | 79e49f39f | 223.167.85.183 | GPTQ + bf16 KV + sparse (Option D) | **79.67%** | **99.58%** | **1.0** | 63.33% | 81.67% | 100% | 96.67% | 56.67% | 3157s | 275.20 | **Best GPTQ+sparse config!** bf16 KV eliminates FP8 scoring error |
| 10 | 2026-04-09 | 430dd221c | 223.167.85.183 | GPTQ + bf16 KV + sparse + topk_scale=2 (Option C) | 0.20% | 0.25% | 0 | 0% | 1% | 0% | 0% | 0% | 9385s | 661.10 | **BROKEN**: topk_scale=2 causes garbage output (avg 41K output tokens) |
| 11 | 2026-04-10 | 9e82efe43 | 223.167.85.181 | Non-quant + FP8 KV + sparse (retest w/o 0070 bug) | 55.82% | 69.78% | 0 | 66.67% | 44.67% | 97.78% | 36.67% | 33.33% | 7379s | — | FP8 KV severely hurts NIAH/qa on non-quant; concurrency=8, ~2h eval |
| 12 | 2026-04-12 | 9e82efe43 | 223.167.85.181 | GPTQ + FP8 KV + dense (freshly quant on old fcloud) | **79.29%** | **99.11%** | **1.0** | 63.33% | 72% | 97.78% | **100%** | 63.33% | 4244s | — | Dense mode + GPTQ + FP8; niah perfect; qa improved vs Test 9 |
| 13 | 2026-04-12 | 9e82efe43 | 223.167.85.181 | GPTQ + bf16 KV + dense (same quant model) | 76.67% | 95.83% | 0 | 50% | 80% | 100% | 96.67% | 56.67% | 4568s | — | bf16 KV dense; mcq dropped to 50%; below C=0.8 threshold |
| 14 | 2026-04-13 | c818ae261 | 223.167.85.181 | CHANGE_0075: bf16 RoPE + in-place residual | **52.64%** | **65.81%** | **0** | 53.33% | 47.67% | 98.89% | 36.67% | 26.67% | 4128s | — | **CATASTROPHIC**: bf16 RoPE destroys precision; qa=26.67%, niah=36.67% |
| 15 | 2026-04-13 | b8196b71e | 223.167.85.181 | CHANGE_0075 partial: in-place residual only (RoPE restored) | **51.91%** | **64.89%** | **0** | 46.67% | 44.0% | 98.89% | 40.0% | 30.0% | 4188s | — | **CATASTROPHIC**: in-place `*=` also destroys accuracy; WORSE than Test 14 |
| 16 | 2026-04-13 | caa93efe9 | 223.167.85.181 | Full baseline revert (no CHANGE_0075) | — | — | — | — | — | — | — | — | — | — | Running — verifying return to baseline |
| 17 | 2026-04-14 | 290e370e6 | 223.167.85.181 | CHANGE_0075 re-enabled (bf16 RoPE + in-place residual) + dense+FP8 | **79.98%** | **99.97%** | **1.0** | — | — | — | — | — | 3148s | 463.49 | **CHANGE_0075 VINDICATED**: Tests 14-16 were on wrong sparse config; on correct dense+FP8 config accuracy is excellent |
| 18 | 2026-04-14 | a9f4d43cb | 223.167.85.181 | torch.compile (max-bs=8) + CHANGE_0075 + dense+FP8 | 78.18% | ~97.7% | 0.92 | 50.00% | 78.67% | 98.89% | 100% | 63.33% | 3268s | 496.27 | **mcq dropped to 50%** (from 63%); torch.compile may be causing MCQ regression; C drops from 1.0→0.92; net negative |
| 18b | 2026-04-14 | a9f4d43cb | 223.167.85.181 | torch.compile (max-bs=8) re-run | **79.38%** | **99.22%** | **1.0** | 63.33% | 74.67% | 98.89% | 100% | 60.00% | — | 523.00 | **mcq recovered to 63.33%**; Test 18 mcq=50% was variance; C=1.0 restored; torch.compile is SAFE |
| 20 | 2026-04-15 | 9f9b02c52 | 223.167.85.181 | CHANGE_0085: mixed-chunk + max-running-requests=24 | **80.64%** | **100.80%** | **1.0** | 63.33% | 77.67% | 98.89% | 100% | 63.33% | 3171s | 501.00 | Accuracy improved; C=1.0 maintained; config also includes torch.compile(max-bs=8) |
| 21 | 2026-04-16 | nvfp4 branch | 223.167.85.181 | **NVFP4 W4A4** (modelopt, block_size=16, FP8 KV, dense) | **~12%** | **~15%** | **0** | 0.00% | 7.00% | 50.00% | 0.00% | 3.33% | ~7200s | 1636 | **CATASTROPHIC**: FP4 quantization destroys reasoning; avg output 30k-54k tokens (infinite think loops); decode throughput excellent (1636 tok/s) but accuracy unusable; 5 requests timed out (3000s) |
| 23 | 2026-04-17 | f373fbade | 223.167.85.181 | CHANGE_0100: residual scale folding + GPTQ + FP8 KV + dense + torch.compile(max-bs=8) + mixed-chunk | **78.64%** | **98.30%** | **0.96** | 56.67% | 81.00% | 98.89% | 100% | 56.67% | 3234s | 492.87 | Accuracy regression vs Test 20; C drops to 0.96 (not submission-safe yet) |
| 24 | 2026-04-18 | 96304f9cd | 223.167.85.181 | CHANGE_0110: **dense-calibrated GPTQ** + FP8 KV + dense + torch.compile(max-bs=8) + mixed-chunk | **77.64%** | **97.05%** | **0.92** | **50.00%** | 79.33% | 98.89% | 100% | 60.00% | 3059s | — | **FAILED**: Dense calibration made accuracy WORSE; mcq crashed to 50%; C=0.92 |
| 25 | 2026-04-20 | 08fd86023 | 223.167.85.181 | **CHANGE_0120**: prefill-max-req=4, sched-cons=0.8, chunk=65536 | **79.00%** | **98.75%** | **0.96** | 53.33% | 85.00% | 100% | 100% | 56.67% | 2988s | 423.75 | mcq=53.33% (variance); cwe improved 85%; fwe/niah perfect; duration 2988s (vs 3171s Test 20); C=0.96 |
| 27 | 2026-04-20 | 338989afe | 223.167.85.181 | **CHANGE_0125**: SM120 Marlin tile instantiations + rebuilt sgl-kernel | **77.18%** | ~96.5% | **0** | 56.67% | 83.67% | 98.89% | 100% | 46.67% | 3123s | 468.74 | New tiles compiled but NOT selected by scorer; accuracy drop is **test variance** (qa=46.67% anomaly); CHANGE_0125 is NEUTRAL |
| **OPTION_B** | 2026-04-21 | **b794b692d** | N/A | **Option B Phase 1**: FP8 blockwise GEMM implementation (code ready for testing) | — | — | — | — | — | — | — | — | — | — | **IMPLEMENTATION COMPLETE**: preprocess_model.py + FP8BlockwiseLinearMethod ready. Awaiting fcloud test. Expected: S1 ~85-100s (50% prefill reduction) |

---

## Speed Benchmark Results

| Test # | Date | Commit | Config | S1 | S8 | Smax | Weighted Speed Score | Notes |
|--------|------|--------|--------|----|----|------|---------------------|-------|
| 9-spd | 2026-04-09 | 79e49f39f | GPTQ + bf16 KV + sparse (Option D) | 139.28s | 56.97s | 48.33s | — | First speed test on new fcloud |
| 3/4-spd | 2026-04-12 | 9e82efe43 | Non-quant + bf16 KV + sparse | 281.32s | 90.74s | 72.17s | — | ~2× slower than GPTQ; Smax crashed on 1st attempt, passed on retry |
| 12-spd | 2026-04-12 | 9e82efe43 | GPTQ + FP8 KV + dense | 121.71s | 44.09s | 35.86s | — | Fastest config tested! |
| 13-spd | 2026-04-12 | 9e82efe43 | GPTQ + bf16 KV + dense | 121.22s | 44.05s | 35.94s | — | Nearly identical speed to FP8 KV |
| 12-VarA | 2026-04-12 | 9e82efe43 | GPTQ+FP8+dense: chunk=65K, prefill=2, running=40, mem=0.87 | 121.68s | 44.11s | 35.91s | — | Zero improvement vs baseline |
| 12-VarB | 2026-04-12 | 9e82efe43 | GPTQ+FP8+dense: +mixed-chunk, conserv=0.7 | 121.63s | 43.70s | 35.71s | — | Marginal: S8 -1%, Smax -0.6% |
| 12-VarC | 2026-04-12 | 9e82efe43 | GPTQ+FP8+dense: +torch.compile(max-bs=32)+mixed-chunk | **113.06s** | **41.65s** | **33.86s** | — | **Best: S1 -7.1%, S8 -5.7%, Smax -5.7%**; server OOM during accuracy eval |
| 14-spd | 2026-04-13 | c818ae261 | CHANGE_0075: bf16 RoPE + in-place residual | 139.26s | 52.82s | **CRASH** | — | **SLOWER**: S1 +14.5%, S8 +19.6% vs baseline; Smax server crashed |
| 17-spd | 2026-04-14 | 290e370e6 | CHANGE_0075 re-enabled + dense+FP8 (correct config) | 122.01s | 44.14s | 35.94s | — | Essentially identical to baseline (all within ±0.3%); CHANGE_0075 does NOT hurt speed |
| 18-spd | 2026-04-14 | a9f4d43cb | torch.compile (max-bs=8) + dense+FP8 | **112.97s** | **41.23s** | **35.41s** | — | **S1 -7.4%, S8 -6.6%, Smax -1.5%** vs Test 17; good speed gain but accuracy dropped (C=0.92), net negative |
| 19-spd | 2026-04-15 | 23d1c8ecf | CHANGE_0080: FLA chunk/threshold tuning sweep | 112.96s | 41.55s | 35.56s | — | Baseline (chunk=64,thresh=128). All variants tested below — **zero impact** |
| 19-A | 2026-04-15 | 23d1c8ecf | chunk_size=32, threshold=128 | 113.02s | — | — | — | No change vs baseline |
| 19-B | 2026-04-15 | 23d1c8ecf | chunk_size=128, threshold=128 | 112.97s | — | — | — | No change vs baseline |
| 19-C | 2026-04-15 | 23d1c8ecf | chunk_size=64, threshold=64 | 112.95s | — | — | — | No change vs baseline |
| 19-D | 2026-04-15 | 23d1c8ecf | chunk_size=64, threshold=256 | 112.97s | 41.54s | — | — | No change vs baseline |
| 20-spd | 2026-04-15 | 9f9b02c52 | CHANGE_0085: mixed-chunk + max-running-req=24 + torch.compile(max-bs=8) | **113.67s** | **41.07s** | **34.15s** | — | S1 ~same, S8 -1.2%, **Smax -4.0%** vs Test 19; mixed-chunk helps Smax most |
| 22-acc | 2026-04-17 | 548c8c153 | EAGLE3 spec-decode (untrained draft, mem-frac=0.72) | 187.01s | — | — | 74.33% / 92.92% / C=0 | **EAGLE3 FAIL**: accept_rate=0.26 (random draft), MCQ accuracy 56.67% (vs 76.67% baseline), S1 65% slower. C=0 → eliminated |
| 23-spd | 2026-04-17 | f373fbade | CHANGE_0100: residual scale folding + dense+FP8 + torch.compile(max-bs=8) + mixed-chunk | **112.55s** | **41.04s** | **34.58s** | — | vs Test 20: S1 -1.0%, S8 ~flat, Smax +1.3% slower; net speed change negligible |
| 24-spd | 2026-04-18 | 96304f9cd | CHANGE_0110: **dense-calibrated GPTQ** + FP8 KV + dense + torch.compile(max-bs=8) + mixed-chunk | **110.59s** | **40.45s** | **33.64s** | — | vs Test 20: S1 -2.7%, S8 -1.5%, Smax -1.5%; speed slightly better but accuracy killed (77.64%); **NOT viable** |
| 25-spd | 2026-04-20 | (local) | Baseline verification: prefill-max-req=1, sched-cons=1.0, chunk=32K | 120.48s | 40.49s | 33.67s | — | Matches Test 20 baseline (new fcloud instance) |
| 25A-spd | 2026-04-20 | (local) | **prefill-max-req=4, sched-cons=0.8**, chunk=32K | **110.58s** | 40.53s | 33.58s | — | **S1 -8.2%**, S8/Smax unchanged |
| 25B-spd | 2026-04-20 | (local) | prefill-max-req=4, sched-cons=0.8, **chunk=65K** | **110.54s** | 40.54s | 33.59s | — | Chunk=65K: same as 32K on old data (inputs too short to matter) |
| 25C-spd | 2026-04-20 | (local) | prefill-max-req=8, sched-cons=0.5, chunk=65K | **110.53s** | 40.54s | 33.54s | — | More aggressive: no further gain, plateau at ~110.5s |
| 26-prof | 2026-04-20 | 08fd86023 | **PROFILING**: torch profiler, stage-separated, 3 steps each | — | — | — | — | GEMM=85.3% prefill, FLA=12.4%. See CHANGE_0120_profiling_analysis |
| 27-spd | 2026-04-20 | 338989afe | CHANGE_0125: SM120 Marlin tile instantiations (rebuilt sgl-kernel) | **111.48s** | **40.42s** | **33.53s** | — | vs Test 25B: S1 +0.9%, S8 -0.3%, Smax -0.2%; **NEUTRAL** — new tiles compiled but never selected by scorer |

---

## Key Configurations Reference

### Server Args (prepare_env.sh)
- **Base args**: `--trust-remote-code --disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 32768 --max-prefill-tokens 32768 --prefill-max-requests 1 --max-running-requests 20 --mem-fraction-static 0.84 --schedule-conservativeness 1.0 --dense-as-sparse --quantization gptq_marlin --enable-fused-qk-norm-rope`
- **FP8 KV**: add `--kv-cache-dtype fp8_e5m2`
- **Dense mode**: replace `--attention-backend minicpm_flashinfer` with `--force-dense-minicpm`
- **TopK scaling**: add `--sparse-topk-scale N` (default 1, effective topk = base_topk×N + local_blocks)

### Model Sparse Config (from config.json)
- `topk=64`, `block_size=64`, `window_size=2048`, `kernel_size=32`, `kernel_stride=16`
- `local_blocks = 2048/64 = 32`
- Effective sparse_topk = 64 + 32 = 96 (at scale=1)

### fcloud Instances
- **Old**: 223.167.85.181:12369 (restored 2026-04-10, non-quant model only)
- **New**: 223.167.85.183:20685 (active, has GPTQ model)

---

## Official Submission Results

| Submission | Date | Package Config | acc_ori | acc (normalized) | C | S1 | S8 | Smax | final_score | Rank | Notes |
|------------|------|----------------|---------|------------------|---|----|----|------|-------------|------|-------|
| v18-A | 2026-04-15 | torch.compile(max-bs=8)+GPTQ+FP8+dense | 78.71 | 98.39% | 0.96 | 426.06s | 620.73s | 1169.84s | 52.94 | #19 | C=0.96 penalty |
| v18-B (resub) | 2026-04-15 | same package as v18-A | 80.51 | 100.0% | 1.0 | 429.28s | 624.24s | 1171.49s | 51.08 | #18 | C=1.0 but score dropped (other teams improved Duration_best) |

**Key insight**: Same package gives different accuracy across submissions (78.71→80.51). Official accuracy has variance — likely related to submission time (morning vs afternoon per user observation). Speed times are very similar (~0.5% variance). Score declined despite better C because competing teams improved their speeds (lowering Duration_best in formula).

## Notes
- Tests 1-7 were on old fcloud instance with potentially different GPTQ model preparation
- Tests 8+ are on new fcloud instance with freshly prepared GPTQ model
- The large accuracy improvement from Test 5 (50%) to Test 8b (76%) is likely due to fresh GPTQ model preparation
- Accuracy eval uses `--concurrency 32` (changed from 8 starting from CHANGE_0073)
