# Chat log — Round 8: clarify W8A8 FP8 vs INT8, Phase 0 vs Proposal B, baseline flow (2026-04-28)

Continuation of W4A8 deep-dive session. Prior rounds:
- Rounds 1–6: `CHAT_w4a8-deepdive_20260427_1500.{en,zh}.md`
- Round 7: `CHAT_w4a8-fp8-reuse-int8_20260427_1730.{en,zh}.md`

## Round 8 — Clarifications + baseline flow

### User asks
1. "Previously we tested w8a8 is for 4-bit quantization model which will convert int4 weight to int8 and then convert activation type from bf16 to int8, and then we got not-bad accuracy but worse benchmark, am I right? We didn't test w8a8 for fp8, right?"
2. "For proposal B, I remember we did test FP8, BF16 and INT8 in Phase 0, not the same as the test in proposal B?"
3. "Pls do both now [patch docs]. And now I understand the process flow of w8a8 fp8 as the below: (1) at load time the loader dequantized INT4 → BF16 → re-quantized to FP8 e4m3 storage, then ran a CUTLASS FP8×FP8 BF16-out GEMM. (2) Activation path: BF16 → FP8 e4m3 per-token quantize on every forward. How about the current baseline flow? It is w4a16 (bf16), right?"

### Agent answers / actions

**Correction on the W8A8 history**:
- The earlier "W4A8 #1" test (commit `7ce21c3f5`) was **W8A8 FP8**, not INT8. The user's recollection had the right intuition (regression on a W4 model after going to A8) but the activation/weight type was FP8 e4m3, not INT8.
- We never tested W8A8 INT8 end-to-end. INT8 only existed as Phase 0 microbench (136 TF, killed).

**Phase 0 vs Proposal B distinction**:
- Phase 0 = dense FP8×FP8→BF16 GEMM via `torch._scaled_mm`. Both inputs already FP8 in registers. Measures hardware ceiling (281 TF).
- Proposal B = W4-weight × FP8-activation GEMM where INT4 is dequantized in-kernel to FP8 before MMA. Measures the **dequant tax** Phase 0 deliberately did not include. Pass thresholds (≥250 / 180–249 / <180 TF) tell us how much of the 281 TF ceiling survives the W4→FP8 dequant.

**Baseline flow (v18 W4A16 BF16)**:
1. Load: INT4 packed in HBM, fp16/bf16 group scales, no weight inflation.
2. Forward: Activations BF16 (no quant). Marlin K-loop: INT4 → in-register BF16 dequant → BF16×BF16 MMA at 148 TF → BF16 epilogue.
3. KV: FP8 e5m2 in HBM, dequantized to BF16 inside FlashAttention.

Compared to the W4A8 #1 mislabel:

| | v18 baseline | W4A8 #1 (W8A8 FP8) | Real W4A8 FP8 (parked) |
|---|---|---|---|
| Weight HBM | INT4 packed | FP8 (2× inflated) | INT4 packed |
| Activation | BF16 | FP8 | FP8 |
| MMA | BF16 148 TF | FP8 281 TF | FP8 281 TF |
| Weight bytes/param | 0.5 | 1.0 | 0.5 |

Why W4A8 #1 regressed: doubled weight bytes hurt decode (weight-BW-bound) more than the FP8 MMA peak helped.

**Doc patches applied**:
- `ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.{en,zh}.md` — added § 7 "Clarifications added 2026-04-28" with the three-way table (baseline / W4A8 #1 / real W4A8) and Phase 0 vs Proposal B comparison.
- `PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.{en,zh}.md` — added § 1.1 "How this differs from Phase 0 (already done)" with side-by-side table.

### Outcomes
- All three open ambiguities resolved in writing.
- User now has consistent baseline-flow understanding to compare against future iterations.

### Open items
- Same as Round 7: awaiting user choice between (a) NVFP4 KV proposal P1 survey, (b) W4-FP8 spike, (c) both in sequence, before any coding starts.

## Cross-references
- [ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md § 7](../ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md)
- [PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md § 1.1](../PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md)
- [PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md](../PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md)
- [CHANGE_W4A8_001_iteration_002.en.md](../CHANGE_W4A8_001_iteration_002.en.md) — mislabel notice
