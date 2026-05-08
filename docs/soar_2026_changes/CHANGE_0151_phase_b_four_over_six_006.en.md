# CHANGE 0151 — Phase B FourOverSix, continuation 006

Companion to [CHANGE_0151_phase_b_four_over_six_005.en.md](CHANGE_0151_phase_b_four_over_six_005.en.md).

Iter-7 = **iter-5 reproducibility probe + speed bench**: re-quantize the
exact iter-5 NVFP4-FOS recipe (FOS=1, SAMPLES=32 sequential, calib_seq_len
=4096, default qa,mcq,cwe filter) and run a second accuracy pass plus the
S1 / S8 / Smax speed benchmarks that were skipped in iter-5.

This iteration also exercises the tokenizer-save fix from
CHANGE_0151_005 (commits `39c0045c5` + `83921b207`) on a real successful
NVFP4 quantization (iter-6 used FOS=0; this run is the first FOS=1
quantization with the fix in place).

## Background / motivation

Iter-5 produced a single 71.24% accuracy data point (no run-2, no speed).
Iter-6 (FOS=0 ablation) showed FOS is protective on short-prompt tasks
and confirmed the iter-5 → iter-1 ~1.89pt gap is not caused by the FOS
flag itself. Open questions before deciding whether NVFP4-FOS belongs in
the submission package:

1. Is iter-5's 71.24% reproducible, or was it a lucky single-shot? FOS-1 →
   FOS-1b earlier in this iteration showed a −5.71pt swing on the same
   ckpt, so we cannot trust a single run.
2. What are the S1 / S8 / Smax numbers for NVFP4-FOS at the iter-5 config?
   Required to compare against current GPTQ baseline for the submission
   decision.

## Implementation

No source change. Re-quantize iter-5 recipe and run run-2 + speed.

```bash
# (1) re-quant (deterministic seed)
SOAR_QUANT_PROFILE=nvfp4_fos \
SOAR_NVFP4_FOUR_OVER_SIX=1 \
SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096 \
SOAR_GPTQ_CALIBRATION_SAMPLES=32 \
SOAR_GPTQ_CALIBRATION_SAMPLING=sequential \
SOAR_GPTQ_CALIBRATION_SEED=20260320 \
python3 -u preprocess_model.py \
  --input /root/models/openbmb/MiniCPM-SALA \
  --output /root/models/MiniCPM-SALA-NVFP4-FOS \
  --mode nvfp4

# (2) restart server (Tier1 long-ctx, modelopt_fp4)
SOAR_QUANT_PROFILE=nvfp4_fos SOAR_NVFP4_FOUR_OVER_SIX=1 \
SOAR_TIER1_LONG_CONTEXT=1 SOAR_TORCH_COMPILE_MAX_BS=24 \
python3 -m sglang.launch_server --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  ${SGLANG_SERVER_ARGS[@]}

# (3) accuracy + speed
fcloud_workflow.py accuracy --quant-mode gptq --model-path .../MiniCPM-SALA-NVFP4-FOS
fcloud_workflow.py speed --variant all
```

Re-quant log confirmed FOS active and pct_m4 unchanged:
```
[preprocess] NVFP4 FourOverSix activating for export
[preprocess] NVFP4 FourOverSix summary: layers=224 blocks=521142272
  blocks_picked_m4=224801387 pct_m4=43.14%
[preprocess] NVFP4 tokenizer.save_pretrained complete
```

Tokenizer-save fix worked on first try (no manual `cp` needed).

## Result

### Accuracy (run-2 reproducibility probe)

| Task | iter-5 (run-1) | iter-5 (run-2) | Δ |
|------|---------------:|---------------:|---:|
| cwe  | 70.67% | 76.00% | +5.33 |
| fwe  | 92.22% | 85.56% | −6.66 |
| mcq  | 53.33% | 46.67% | −6.66 |
| niah | 90.00% | 100.00% | +10.00 |
| qa   | 50.00% | 46.67% | −3.33 |
| **avg ori_accuracy** | **71.24%** | **70.98%** | **−0.26** |
| acc duration | 2485.20 s | 2513.84 s | +28.64 s |

Bucket breakdown (run-2): len_0_4k 46.67%, len_4k_32k 62.25%, len_32k_128k
84.46%.

**Verdict on reproducibility**: overall accuracy stable (±0.26pt) — iter-5
is reproducible. **Per-task variance is large** (single tasks swing ±10pt
between runs); this matches the well-known scheduling/runaway-think
variance pattern. The fact that overall accuracy lands within ±0.3pt
between two runs while individual tasks swing ±10pt indicates the failure
distribution is roughly conserved (probability mass for runaway-think
gets redistributed across tasks but total error count stays similar).

### Speed bench (S1 / S8 / Smax)

| Concurrency | Iter-5 NVFP4-FOS (run-2 ckpt) | NVFP4-FOS-1b (variance probe) | Test 12 GPTQ baseline |
|-------------|------------------------------:|------------------------------:|----------------------:|
| S1   | 173.83 s | 175.08 s | 121.71 s |
| S8   |  46.05 s |  47.37 s |  44.09 s |
| Smax |  31.07 s |  31.01 s |  35.86 s |

NVFP4-FOS speed signature (vs Test 12 GPTQ baseline):
- **S1 = +52.12 s slower** (+42.8% slower)
- **S8 = +1.96 s slower** (+4.4% slower)
- **Smax = −4.79 s faster** (−13.4% faster)

This is the expected NVFP4 profile on SM120: FP4 weight loading + dequant
overhead dominates per-request decode at S1 (where prefill is short and
tail latency dominates), but the FP4 GEMM bandwidth advantage shows up at
high concurrency (Smax) where the model is compute-bound on long-context
prefill.

## Comparison vs GPTQ submission baseline (Test 12)

Performance score (relative, higher = better) using
`Final = S1×40% + S8×30% + Smax×30%` with each tier scoring
`Duration_best / Duration_player × 100` against itself as best:

| Tier | Weight | Test 12 GPTQ | NVFP4-FOS (iter-5 r2) | NVFP4 ratio |
|------|-------:|-------------:|----------------------:|------------:|
| S1   | 40%    | 121.71 s | 173.83 s | 70.0% (−12.0pt of weighted score) |
| S8   | 30%    |  44.09 s |  46.05 s | 95.7% (−1.3pt) |
| Smax | 30%    |  35.86 s |  31.07 s | 115.4% (+4.6pt to GPTQ if NVFP4 best) |

If NVFP4-FOS were submitted: **GPTQ would score 100×0.4 + 100×0.3 +
35.86/35.86×100×0.3 = 100, but NVFP4 would score 70.0×0.4 + 95.7×0.3 +
100×0.3 = 86.7** vs GPTQ's score against NVFP4 best at Smax: 100×0.4 +
100×0.3 + (31.07/35.86)×100×0.3 = 96.0.

In the head-to-head:
- GPTQ wins overall (96.0 vs 86.7) because S1's 40% weight dominates.
- NVFP4 only beats GPTQ if **Smax** weight grows past ~60% — i.e. only on
  workloads dominated by high-concurrency long-context prefill.

Combined with NVFP4-FOS accuracy ceiling (~71%, C=0.92 at best —
normalized 89% < 97% **fails the C threshold and gives final
score 0**), NVFP4-FOS is **NOT submission-ready** as currently
configured. GPTQ + sparse_qkv_w8 + FP8 KV remains the submission baseline.

## Decision

1. **NVFP4-FOS is NOT submission-ready** at this accuracy ceiling. Keeping
   it on a side branch for further investigation (see Next steps).
2. **GPTQ baseline (Test 12) remains the submission package**. Continue
   iterating on it for speed/accuracy gains.
3. iter-5 result is now confirmed reproducible: 70.98% / 71.24% on two
   separate runs, ±0.26pt overall variance. Park further FOS calibration
   tuning until we have a path to break the ~71% accuracy ceiling.

## Rollback

No code changes in this iteration. Tokenizer-save fix from
CHANGE_0151_005 is unchanged and has been validated by this iter on a
fresh quantization.

## Next steps

| # | Idea | Expected | Effort | Risk |
|---|------|----------|--------|------|
| 1 | Investigate NVFP4 mcq runaway-think on small short prompts: server-side `generation_config` (lower temperature/top-p, stop tokens) — currently mcq runs at avg_out=7000+ tokens for 30 questions of short prompts. | recover 5–10pt on mcq if generation discipline can avoid runaway | medium (must not regress long-context tasks) | medium |
| 2 | Profile NVFP4 S1 latency to find where the 52s overhead vs GPTQ comes from (kernel launch, dequant, KV lookup). Possible fixes: pre-pin weights, shrink torch_compile graph set. | data for NVFP4 path viability | 1–2h | low |
| 3 | GPTQ baseline optimization (return to OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md). Higher leverage given current accuracy gap. | cumulative speed gains on the actual submission package | varies | low–med |
| 4 | Consider abandoning NVFP4 path entirely until Phase C (bf16 calibration with NVFP4 export) can break the 71% ceiling. | clarity for resource allocation | – | – |

Recommendation: **(3)** is highest-leverage given that NVFP4-FOS is
currently submission-blocked by accuracy. Park NVFP4-FOS work; resume
GPTQ catalog priorities.
