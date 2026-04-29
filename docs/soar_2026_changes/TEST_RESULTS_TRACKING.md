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
| **OPTION_B-crash** | 2026-04-21 | b794b692d | N/A (new fcloud SM120) | **Option B** FP8 blockwise (Bug 9 present - OOM) | — | — | — | — | — | — | — | — | — | — | **INVALID**: Server crashed mid-eval (CUDA OOM from float32 activation upcast). Partial 19/150: mcq=16.67%, niah=30%, cwe/fwe/qa=0%. Bug 9 fixed in next run. |
| **OPTION_B-final** | 2026-04-21 | **6b0492021+2d958dd95** | N/A (new fcloud SM120) | **Option B** FP8 blockwise (both bugs fixed) | — | — | **0** | — | ~0% | — | — | ~0% | **TIMEOUT** (3600s+) | — | **FAILED**: Eval reached only 139/150 in 46+ min (fcloud 3600s timeout); final 11 samples projected far longer. Avg time/sample ~26s vs baseline ~8s → model generates near-max-length (65536) degenerate output for CWE/FWE/QA. No valid predictions.jsonl produced. FP8 blockwise quantization destroys model instruction-following on long-context tasks. **DO NOT PURSUE.** |
| 28 | 2026-04-22 | 2d958dd95 | new fcloud SM120 | **CHANGE_0130**: SM120 Marlin prefill-aware dispatch + GPTQ + FP8 KV + dense + torch.compile(max-bs=8) + mixed-chunk + prefill-max-req=4 + sched-cons=0.8 + chunk=65536 | **76.40%** | **95.50%** | **0** | 50.00% | 82.00% | 100.00% | 100.00% | 50.00% | 2976.83s | 392.66 | **FAILED**: disqualifying accuracy regression. `mcq`/`qa` collapsed to 50%, `mcq` avg output length ballooned to 11143 tokens. No speed run performed. CHANGE_0130 reverted locally; do not pursue this heuristic in submission path. |
| 29 | 2026-04-22 | reverted (pre-CHANGE_0130) | new fcloud SM120 | **CHANGE_0130 reverted** + CHANGE_0125 retained + GPTQ + FP8 KV + dense | **78.73%** | **98.42%** | **0.96** | 56.67% | 83.67% | 96.67% | 100.00% | 56.67% | 3005.79s | 439.04 | Baseline revalidation after CHANGE_0130 revert. Wheel rebuilt from clean `gptq_marlin.cu` (CHANGE_0125 SM120 tile table retained). ori_accuracy=78.73% is within normal test variance vs Test 12 (79.29%). **C=0.96** (normalized=98.42% < 99%) — borderline, same as official v18-A. Speed run pending. |
| 30 | 2026-04-22 | 5ab5787e8 | new fcloud SM120 | **Test 30: KV fp8_e5m2 → fp8_e4m3** (GPTQ + dense + v19 scheduling) | **77.96%** | **98.32%** | **0.96** | 53.33% | 82.00% | 97.78% | 100.00% | 56.67% | 2878.90s | 385.72 | Step B bisect #1. mcq collapsed (96.67→53.33, avg_out=11368), cwe jumped (56.67→82), fwe jumped (83.67→97.78). e4m3 did NOT fix runaway mcq chains; in fact worse. Overall acc slightly down vs Test 29. C unchanged (0.96). **Hypothesis refuted**: KV dtype precision direction unclear; huge per-task variance suggests eval noise dominates. |
| 32 | 2026-04-22 | fc8029005 | new fcloud SM120 | **Test 32: v19 scheduling reverted to v18** (chunk 32768, prefill-max-req 1, sched-cons 1.0, running 20, KV fp8_e5m2) | **75.73%** | **94.66%** | **0** | 40.00% | 82.00% | **100.00%** | 100.00% | 56.67% | 2909.48s | 424.71 | Step B bisect #2. **FAILED**: acc dropped to 75.73%, C=0 (eliminated). Conservative scheduling did NOT fix accuracy — mcq collapsed further (40%, avg_out=11130), fwe hit 100% but with runaway 11536 avg_out tokens. cwe/niah/qa unchanged. Scheduling aggression is NOT the dominant source of variance. |
| 33 | 2026-04-22 | e625363a8 | new fcloud SM120 | **Test 33: disable torch.compile** (v19 scheduling + fp8_e5m2 KV, no `--enable-torch-compile --torch-compile-max-bs 8`) | **76.98%** | **96.22%** | **0** | 46.67% | 82.67% | 98.89% | 100.00% | 56.67% | 3016.59s | 423.29 | Step B bisect #3. Server boot 36s (vs 219s with compile). acc=76.98% (Test 29 with compile was 78.73%). mcq=46.67 (vs 56.67), runaway chain persists (avg_out=10390). Duration ~equal to Test 29 (3016 vs 3005). **Conclusion**: torch.compile is not the source of the local noise. Combined with Tests 30/32, no single-variable knob fixes the ~77-79% local floor — confirms high intrinsic eval variance. Suggests pivoting to multi-seed noise quantification or speed optimization. |
| 34a | 2026-04-23 | a639cb857 | **new fcloud 223.167.85.183** | **Test 34a: Test 20 best config replay** (chunk=32K, prefill-max-req=1, running=24, sched-cons=1.0, mem=0.84, mixed-chunk, torch.compile bs=8, GPTQ+FP8_e5m2+dense). Wheel = Mar-29 pre-CHANGE_0125 (kept for speed). | **77.51%** | **96.89%** | **0** | 56.67% | 85.33% | 98.89% | 100.00% | **46.67%** | 2817.47s | 389.34 | **New-instance replay of Test 20 best combo FAILED**: acc 77.51% (vs Test 20 80.64%, Test 29 78.73% same config). QA collapsed to 46.67% (Test 20 was 63.33%, Test 29 was 56.67%). mcq at 56.67% (runaway persists). Within the ~±2-3pt local noise floor but lands on wrong side of C=0 threshold (norm 96.89% < 97%). **Hypothesis**: new fcloud silicon lottery + Mar-29 wheel missing CHANGE_0125 Marlin tiles. Same config gave C=1.0 on official v18-A submission; local noise dominates on 150-sample concurrency=32 eval. |
| **v18-revert** | 2026-04-26 | **8d1e4d12b** | 223.167.85.181 | **v18 baseline surgical revert** (minicpm.py + preprocess_model.py reverted to a9f4d43cb; FLA chunk + Test 20 best server combo + eval flags KEPT). GPTQ+FP8_e5m2+dense, chunk=32K, prefill-max-req=1, running=24, sched-cons=1.0, mixed-chunk, torch.compile bs=8. | **77.44%** | **97.66%** | **0.92** | 56.67% | 81.67% | 98.89% | 100.00% | 50.00% | 2878.47s | 423.11 | Validates surgical revert is safe vs v19 hang. **Catastrophic 3000s hang ELIMINATED** (worst sample 137 was 197s). mcq runaway still present (avg_out=10479, acc=56.67%) — milder mcq slowdown at 136-138 (115s/197s/153s). qa=50% within local noise. norm=97.66% → C=0.92. Local floor remains ~77-79% (consistent with Tests 29/30/33/34a). **Speed**: S1=110.51s, S8=40.46s, Smax=33.61s. |
| **W4A8-#1** *(MISLABEL: actually W8A8 FP8)* | 2026-04-27 | **7ce21c3f5** | 223.167.85.181 | **CHANGE_W4A8_001**: load-time GPTQ INT4 → **FP8 (8-bit) storage** + cutlass `fp8_blockwise_scaled_mm` for std-attn QKV/O + MLP gate_up/down (env-gated `SOAR_W4A8_FP8_GEMM=1`). Lightning + sparse_qkv (INT8) untouched. GPTQ + FP8_e5m2 KV + dense + torch.compile bs=8 + Test 20 server args. **Note: this is W8A8 FP8 blockwise, NOT true W4A8 — see iteration_002 doc.** | **79.20%** | **99.00%** | **1.0** | 53.33% | 82.67% | 100.00% | 96.67% | 63.33% | 3158.44s | 437.47 | **Accuracy preserved** (Δ −0.09pt vs v18 Test 12 79.29%). **Speed: NET REGRESSION** — S1=265.32s (+118%), S8=68.88s (+56%), Smax=46.44s (+30%). Root cause: implementation upcast INT4 weights to FP8 storage at load time, doubling weight memory footprint vs Marlin INT4 baseline (W4A16). At decode bs=1 the kernel is memory-bandwidth-bound on weights → FP8 weight = 2× slower than INT4 weight. **The W4A8 hypothesis is NOT refuted** — true W4A8 (INT4 storage + FP8 MMA) requires a different kernel and is filed as `PROPOSAL_W4A8_REAL_001`. |
| **R13-FP4-KV-smoke** | 2026-04-28 | **8a0976593** (chain: d4608f170, fd7e797ea, 252cc4d64) | 223.167.85.181 | **CHANGE_0131**: NVFP4 (MXFP4) KV cache plumbing in `MiniCPMAttentionBackend` + `SOAR_FP4_KV_CACHE=1` toggle in `prepare_env.sh`. GPTQ + dense (`--force-dense-minicpm`) + `--kv-cache-dtype fp4_e2m1` + torch.compile bs=8 + Test 20 server args. | n/a | n/a | n/a | — | — | — | — | — | — | — | **RED — server fails to boot.** Bug chain: (1) `_handle_kv4_compatibility` rejects `flashinfer` backend → fixed (`fd7e797ea`+`252cc4d64`); (2) `torch.zeros(dtype=fp4_e2m1fn_x2)` not implemented → routed to `MHATokenToKVPoolFP4` (`8a0976593`). KV pool now allocates 27 GB packed K + 27 GB V correctly. (3) **Architectural blocker (NOT fixed)**: cudagraph capture → `KeyError: torch.float4_e2m1fn_x2` in `flashinfer.decode.get_batch_decode_uri`. Root cause: `--force-dense-minicpm` rewrites `attention_backend` `minicpm_flashinfer` → `flashinfer`; stock FlashInfer has no FP4 KV decode kernel; CHANGE_0131 plumbing only lives in `MiniCPMAttentionBackend`. **Next**: CHANGE_0132 — skip the rewrite when `kv_cache_dtype=="fp4_e2m1"`. |
| **R13e-prof-32k**  | 2026-04-29 | **f4097eef6** | 223.167.85.181 | **CHANGE_0135 Option-B profile**: BF16 (`SOAR_QUANT_MODE=noquant`) + native sparse + FP8_e5m2 KV; chunk=32K, prefill-max-req=1, running=8, mem=0.78, sched-cons=1.0, mixed-chunk; **no** `--quantization`, `--force-dense-minicpm`, `--dense-as-sparse`. torch.profiler `/start_profile`+`/stop_profile`, single `/generate`, max_new_tokens=64. Sample idx 129, prompt_tokens=31744. | n/a | n/a | n/a | — | — | — | — | — | wall=9.7s | — | **Profile-only run.** GPU kernel time 5658ms. Top kernel: BF16 `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128` = **66.7%** (3771.9ms, 160 launches). Sparse FA `BatchPrefillWithPagedKVCacheKernel` = 3.6% (201.6ms). compress_k1/k2 fill (post-CHANGE_0133) = 1.0% (57.1ms, 1720 launches). Cat: GEMM=83.6% / FLA+sparse=7.7% / Other=7.7%. GPU active 58%. Trace `round13e_32k.trace.json.gz` (56MB) on fcloud `/root/profile_round13e/`. See [CHANGE_0135_001](CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.en.md). |
| **R13e-prof-64k**  | 2026-04-29 | **f4097eef6** | 223.167.85.181 | Same config as R13e-prof-32k. Sample idx 49, prompt_tokens=63683. | n/a | n/a | n/a | — | — | — | — | — | wall=24.7s | — | **Profile-only.** GPU kernel time 10344ms. Top: BF16 cuTLASS GEMM **73.2%** (7571.4ms, 320 launches). Sparse FA = 4.0% (416.5ms). compress fill = 1.1%. Cat: GEMM=82.8% / FLA+sparse=8.3% / Other=7.9%. GPU active 42%. Trace `round13e_64k.trace.json.gz` (112MB). |
| **R13e-prof-128k** | 2026-04-29 | **f4097eef6** | 223.167.85.181 | Same config as R13e-prof-32k. Sample idx 149, prompt_tokens=127732. | n/a | n/a | n/a | — | — | — | — | — | wall=49.3s | — | **Profile-only.** GPU kernel time 19775ms. Top: BF16 cuTLASS GEMM **76.9%** (15216.6ms, 640 launches). Sparse FA = 4.3% (848.0ms, 544 launches). compress fill = 1.1% (225.7ms, 1888 launches). Cat: GEMM=82.4% / FLA+sparse=8.7% / Other=8.1%. GPU active 40%. Trace `round13e_128k.trace.json.gz` (211MB). **Decision**: BF16 GEMM is the bottleneck under sparse routing — not sparse attention itself. Resolves CHANGE_0135 decision tree to branch (d): keep dense+GPTQ+FP8 KV (Test 12) as submission baseline; sparse line cannot beat it without weight quantization. Raw breakdown: [profile_data/round13e_analyze.txt](profile_data/round13e_analyze.txt). |

## v18-revert Speed (2026-04-26, commit 8d1e4d12b, fcloud 223.167.85.181)

| Variant | Duration | vs Test 34a (108.91s S1) | Notes |
|---------|----------|--------------------------|-------|
| S1 | 110.51s | +1.5% slower | Concurrency=1, 48 req, 552 tok/s total throughput, TPOT=6.26ms |
| S8 | 40.46s | — | Concurrency=8 |
| Smax | 33.61s | — | No concurrency cap, 96 req, 3907 tok/s total throughput, peak concurrent 96 |

---

## Test 34a Speed (2026-04-23, same config, same fcloud)

| Variant | Duration | Notes |
|---------|----------|-------|
| S1 | 108.91s | vs Test 25A (110.58s), 1.5% faster |
| S8 | 39.99s | vs Test 20 (41.07s), 2.6% faster |
| Smax | 33.44s | vs Test 20 (34.15s), 2.1% faster |

Speed is slightly better than Test 20 reference, but accuracy disqualifies (C=0). Same config gave C=1.0 on official v18-A → local 150-sample eval has large variance, official 10× larger sample size should reproduce C=1.0.

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
| W4A8#1-spd *(MISLABEL: W8A8 FP8)* | 2026-04-27 | 7ce21c3f5 | **CHANGE_W4A8_001**: load-time GPTQ INT4 → **FP8 storage** + cutlass `fp8_blockwise_scaled_mm` (env-gated, std-attn + MLP only). Tested as if it were W4A8, but actually W8A8 FP8 blockwise. | **265.32s** | **68.88s** | **46.44s** | — | **REGRESSION**: S1 +118%, S8 +56%, Smax +30% vs Test 12 baseline (121.71/44.09/35.86). Doubled weight memory footprint (INT4 → FP8) on memory-bandwidth-bound decode workload. **Does NOT invalidate true W4A8** (INT4 storage + FP8 MMA) — see PROPOSAL_W4A8_REAL_001. Accuracy preserved (79.20% vs 79.29%). |

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
| pre-v18 (resub 2026-04-23) | 2026-04-23 | pre-v18 package (config TBD — earlier than torch.compile) | 79.24 | 99.06% | 1.0 | 596.25s | 1066.35s | 2746.62s | **39.92** | — | **Baseline drift reference**: same-era package scored healthy C=1.0; speed essentially identical to v18-C (S1 -1.7%, S8 -2.1%, Smax -4.1%). Confirms (a) our local speed optimizations Tests 18-28 barely moved official numbers, (b) accuracy drift is a property of ALL packages, not v18-specific |
| v18-A | 2026-04-15 | torch.compile(max-bs=8)+GPTQ+FP8+dense | 78.71 | 98.39% | 0.96 | 426.06s | 620.73s | 1169.84s | 52.94 | #19 | C=0.96 penalty |
| v18-B (resub) | 2026-04-15 | same package as v18-A | 80.51 | 100.0% | 1.0 | 429.28s | 624.24s | 1171.49s | 51.08 | #18 | C=1.0 but score dropped (other teams improved Duration_best) |
| **v18-C (resub)** | **2026-04-23** | **same package as v18-A/B** | **76.64** | **95.81%** | **0** | **586.56s** | **1089.21s** | **2864.47s** | **0.0** | — | **ELIMINATED**: acc below 97% threshold (C=0). Speed TIMES ARE 2-5× v18-A/B despite identical package — suggests either fcloud hardware contention at submission time OR eval harness difference OR runaway generation hitting KV cache pressure harder. mcq runaway (per Test 34a avg_out=10,946) is the likely root cause of both accuracy drop (chains get truncated) and Smax blowup (queue stall). **Triggers pivot to Iteration A-0: mcq runaway fix.** |

### Package diff v1_007 → v18 (attribution of regression)
Local file-level diff (2026-04-23) shows **ONLY** the following v18 deltas matter in dense+FP8-KV mode (we run `--force-dense-minicpm`, so sparse-kernel fixes are dead code):

1. **`--enable-torch-compile --torch-compile-max-bs 8`** added to `prepare_env.sh` (line 133); `--skip-server-warmup` removed
2. **FP32 cast around `fused_qk_norm_rope` removed** in `python/sglang/srt/models/minicpm.py` (2 call sites; v1_007 did `q,k = q.float(),k.float()` ... `to(orig_dtype)`, v18 runs QK-norm in BF16 directly)

All other diffs (`preprocess_model.py` preset refactor, `minicpm_sparse_kernels.py` int32→int64 + k_scale, `minicpm_sparse_utils.py` cu_seqlens GQA fix, `minicpm_backend.py`/`server_args.py` `--sparse-topk-scale` addition) only affect sparse-mode code paths, which v18's submission config does NOT use.

**Primary regression suspect**: removed FP32 cast around qk-norm-rope under FP8 KV cache → loses ~8 mantissa bits in Q@K score accumulation on long-context (QA/CWE/FWE/NIAH all use 70K+ tokens).
**Secondary suspect**: torch.compile's known ±1-2pt accuracy drift (v18-A/B/C: 78.71/80.51/76.64 pattern) and compile-graph mismatch under high concurrency contributing to Smax blowup.

**Key insight**: Same package gives different accuracy across submissions (78.71→80.51). Official accuracy has variance — likely related to submission time (morning vs afternoon per user observation). Speed times are very similar (~0.5% variance). Score declined despite better C because competing teams improved their speeds (lowering Duration_best in formula).

### Variant A+B direct test on fcloud (2026-04-23) — Option 3 confirmation

Extracted both tarballs side-by-side into `/root/submission_sim_A` (v1_007) and `/root/submission_sim_B` (v18); swapped `/root/submission_sim` symlink between runs; identical preprocessed GPTQ model at `/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8` was shared. Accuracy @ concurrency 32, speed via `bench_serving.sh`.

| Variant | Config | Accuracy (public set) | S1 | S8 | Smax | Server ready |
|---|---|---|---|---|---|---|
| **A** (v1_007) | torch.compile OFF, FP32 qk-norm cast ON, `--skip-server-warmup` | **79.51%** (norm 99.39%, C=1.0) | 119.64s | 43.49s | 35.52s | 46s |
| **B** (v18) | torch.compile ON `--torch-compile-max-bs 8`, FP32 qk-norm cast OFF, warmup ON | **79.51%** (norm 99.39%, C=1.0) | 111.19s | 40.79s | 35.06s | 225s |

**Per-task accuracy breakdown** (A vs B): cwe 85.33 vs 82.00, fwe 98.89 vs 98.89, mcq 53.33 vs 53.33, niah 96.67 vs 100.00, qa 63.33 vs 63.33 — net accuracy identical but different distribution (B does NIAH better, A does CWE better).

**Findings**:
1. **Local accuracy is a wash**: both 79.51%. Neither the FP32 cast nor torch.compile changes the local-public-set score.
2. **Speed favors B**: S1 −7.1%, S8 −6.2%, Smax −1.3% — v18 really is faster. The speed gain comes from torch.compile + FP32-cast removal combined (cannot separate yet without variants C/D).
3. **The v18 official regression is NOT from local-measurable accuracy loss**. Sources remaining:
   - private-set sensitivity (unknown questions may expose torch.compile or FP8-KV precision differently)
   - hardware/concurrency contention at submission time (v18-C saw Smax 2864s locally-equivalent-package was 35s)
   - pure per-submission variance in private-set sampling
4. **mcq output length dropped**: v18 `avg_out=12267` vs v1_007 `avg_out=8505` in mcq — v18 generates ~44% more tokens per mcq even though both get same 53.33% correct. This is symptomatic of Iteration A-0 mcq runaway (early-answer-then-continue).

**Recommendation**: Keep v18 code path (B is 7% faster locally for free) and layer Iteration A-0 (mcq runaway fix) on top for v19 submission. Do NOT revert to v1_007 — it's slower with no accuracy benefit on local eval. The "regression" observed officially is most likely private-set variance + harness contention, which v19 should absorb naturally once mcq runaway (root cause of Smax blowup) is fixed.

### v19 isolation tests on fcloud (2026-04-24)

v19 tarball (uploaded earlier, Apr-20) extracted to `/root/submission_sim_C`; same preprocessed GPTQ model shared with A/B. Package-level diff vs v18:

- **`prepare_env.sh`**: `chunked-prefill-size 32768→65536`, `max-prefill-tokens 32768→65536`, `prefill-max-requests 1→4`, `max-running-requests 20→24`, `schedule-conservativeness 1.0→0.8`, added `--enable-mixed-chunk`, new env `SGLANG_FLA_CHUNK_SIZE=64`, new env `SOAR_GPTQ_FORCE_DENSE=1` / `SOAR_GPTQ_DAMP_PERCENT=0.05` / `SOAR_GPTQ_MSE=0.0`.
- **`preprocess_model.py`**: under `SOAR_GPTQ_FORCE_DENSE=1`, sets `sparse_config=null` so GPTQ calibration matches dense inference. (No effect on this run — reused v18 preprocessed model.)
- **sglang source** (excl. `__pycache__`): modified `srt/layers/attention/fla/chunk.py`, `fla/chunk_delta_h.py`, `fla/fused_recurrent.py`, `srt/layers/attention/hybrid_linear_attn_backend.py`, `srt/models/minicpm.py`, `srt/models/minicpm3.py`, `srt/speculative/eagle_worker.py`; new file `srt/models/minicpm_eagle3.py`.

| Variant | Config | acc_ori | acc_norm | C | mcq | cwe | fwe | niah | qa | Duration | Timeouts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **v19** | v19 source + v19 aggressive server args (chunk=65K, prefill-max-req=4, mixed-chunk, running=24, sched-cons=0.8) | **73.76%** | 93.03% | **0** | 40.00% | 81.00% | 97.78% | 96.67% | 53.33% | 3226.52s | 1× 3000s |
| **v19-a** | v19 source + **v18 server args** (chunk=32K, prefill-max-req=1, running=20, sched-cons=1.0, no mixed-chunk); kept `SGLANG_FLA_CHUNK_SIZE=64` | **78.04%** | 98.42% | **0.96** | 53.33% | 81.33% | 98.89% | 100.00% | 56.67% | 3032.90s | 1× 3000s |
| **v19-c-FLA** | v19-a **with v18 FLA+hybrid_linear_attn backend restored** (kept v19 minicpm.py/minicpm3.py/minicpm_eagle3.py/eagle_worker.py) | **76.29%** | 96.22% | **0** | 46.67% | 77.00% | 97.78% | 100.00% | 60.00% | 2880.01s | 1× (sample 138) |

**Findings**:
1. **v19's server args alone caused ~4pt regression** on mcq (40→53.33 fully recovered in v19-a). Aggressive prefill/mixed-chunk + `SGLANG_FLA_CHUNK_SIZE=64` is unstable on short mcq.
2. **v19 source still costs ~1.5pt vs v18** (v19-a=78.04% vs v18=79.51%). Distribution shifted: qa improved (+3.34) but cwe/fwe worsened slightly.
3. **Hang bug (3000s timeout)** reproduces in ALL three v19 runs — not a server-arg artifact and not FLA. A real concurrency-load-triggered hang lives in v19's non-FLA source (most likely `srt/models/minicpm.py`, since minicpm3.py / eagle_worker / minicpm_eagle3 are unused in this dense-FP8 submission path).
4. **v19 FLA changes are beneficial, not harmful** — reverting FLA to v18 DROPPED accuracy by −1.75pt (v19-a=78.04 → v19-c-FLA=76.29). Previous hypothesis "FLA is the culprit" was **wrong**. v19's FLA kernels and `hybrid_linear_attn_backend.py` improvements help on this config.
5. **v19-a is submission-viable in C=0.96 tier**, but still dominated by v18 (which is C=1.0). No reason to ship v19 or v19-a.

**Conclusion**: Keep **v18 as baseline**. If we cherry-pick from v19: the FLA files are SAFE and actually beneficial, but `srt/models/minicpm.py` is the prime suspect for the concurrency hang and should NOT be cherry-picked without further bisect. Do not use v19's aggressive `SGLANG_SERVER_ARGS` either.

## Notes
- Tests 1-7 were on old fcloud instance with potentially different GPTQ model preparation
- Tests 8+ are on new fcloud instance with freshly prepared GPTQ model
- The large accuracy improvement from Test 5 (50%) to Test 8b (76%) is likely due to fresh GPTQ model preparation
- Accuracy eval uses `--concurrency 32` (changed from 8 starting from CHANGE_0073)
| **R13d-sparse-retest** | 2026-04-28 | **613ea54e4** (chain: 85b52f3d2) | 223.167.85.181 | **Round 13d: GPTQ + FP8 KV + sparse retest** (`SOAR_SPARSE_MODE=1`). Drop `--force-dense-minicpm`. Sparse path requires also dropping `--enable-torch-compile` (else CUDA-graph capture crashes on `CUDAGeneratorImpl::current_seed`). Required also patching the quantized model's `config.json` to restore `sparse_config` (preprocess_model.py had stripped it). Server boots in 35s. Accuracy concurrency=32. | **ABANDONED** | — | — | — | — | — | — | — | — | — | **RED — accuracy run timed out at 1h with ~76/150 samples done; from sample 73+ individual requests hit harness 3000s read-timeout.** Per-sample latency cliff: samples 0–10 take 5–13 s/it, 31–34 take 115–325 s/it, 73+ time out. Long-context samples (niah/cwe) take >50 minutes each. Conclusions: (1) Sparse path on current HEAD without torch.compile is unworkably slow on long-context samples. (2) The Test 8b "fast" 2411s eval duration was misleading — that build had a different (now-regressed) sparse kernel path. (3) Champion-recipe hypothesis (sparse + mixed KV) cannot be reproduced on current sglang tree without significant kernel work. **Decision**: stay on dense submission baseline; defer sparse until torch.compile + sparse-attn CUDA-graph incompatibility is fixed. See `docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.{en,zh}.md` Round 13d outcome. |
