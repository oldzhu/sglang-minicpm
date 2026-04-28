# Chat log — Round 9: Code-level deep dive on baseline vs W8A8 FP8 flows (2026-04-28 11:00)

Continuation of W4A8 deep-dive session. Prior rounds:
- 1–6: `CHAT_w4a8-deepdive_20260427_1500.{en,zh}.md`
- 7: `CHAT_w4a8-fp8-reuse-int8_20260427_1730.{en,zh}.md`
- 8: `CHAT_w4a8-clarifications-baseline_20260428_0930.{en,zh}.md`

## Round 9 — top-to-bottom code/instruction/hardware comparison

### User asks (verbatim, condensed)
1. Why does W8A8 load do `INT4 → BF16 → FP8` instead of keeping INT4 like baseline?
2. What does "fp16/bf16 group scales kept" in baseline mean? Does it cost registers/memory?
3. What is the "BF16 epilogue" — converting FP32 → BF16?
4. Marlin INT4→BF16 inline dequant vs W8A8's BF16→FP8 per-token activation quant — which is more efficient and why? Compute, resources, all aspects.
5. "FP8×FP8 to BF16, no additional FP32→BF16, right?"
6. For KV, since we do FP8×FP8 GEMM, does FlashAttention skip FP8→BF16 conversion?
7. "Please explain combined with source code lines, functions, compute instructions, registers, memory usage."

### Agent answers / actions

Used `runSubagent` (Explore) to trace both paths thoroughly, then wrote a comprehensive bilingual document.

**Key answers**:
- **Q1**: Existing FP8 GEMM kernel requires both inputs in FP8 format. No SM120 dense kernel directly takes INT4 weight + FP8 activation. The load-time conversion is a workaround that destroys the W4 packing advantage.
- **Q2**: Group scales are persistent HBM tensors (~700 KB per layer, negligible), not registers. They're needed because per-128-elem-group bf16 scales are how INT4 maps back to the original real-number range. Register cost during GEMM: handful of values per thread per K-tile.
- **Q3**: Yes. Tensor Core MMA always accumulates in FP32 (PTX has no BF16-accum option for `mma.sync.aligned.*.bf16.bf16.f32`). Epilogue casts FP32→BF16 (`cvt.rn.bf16.f32`) before writing to HBM.
- **Q4**: Marlin's dequant is FAR more efficient — fused inside K-loop, runs in registers, overlaps with MMA, rides for free on weight HBM traffic. Path B's activation quant is a SEPARATE Triton kernel with its own kernel-launch overhead, HBM round-trip on activations, and per-group reduce-max. Plus Path B has the 2× weight HBM bandwidth penalty.
- **Q5**: WRONG — FP8×FP8 MMA also accumulates in FP32 (`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`). FP32→BF16 cast still happens in epilogue, identical to BF16 path.
- **Q6**: WRONG on two counts: (a) the FP8×FP8 GEMM is the LINEAR layer, not attention; attention is a separate kernel doing Q·K^T then softmax-times-V; (b) our attention backend runs BF16 regardless of KV storage dtype, so FP8 KV is dequanted to BF16 inside FlashAttention. To skip that step you'd need an FP8 FlashAttention kernel — we don't have one.

**Doc created**:
- [EXPLAIN_baseline_vs_w8a8fp8_flow_20260428_1100.{en,zh}.md](../EXPLAIN_baseline_vs_w8a8fp8_flow_20260428_1100.en.md): full bilingual deep-dive with file:line refs to gptq.py, utils_w4a8_fp8.py, fp8_kernel.py, marlin_template.h, model_runner.py, minicpm_backend.py; actual MMA opcodes; instruction-level timeline; HBM/SMEM/register footprint comparison.

### Outcomes
- All 6 user questions answered with code-level evidence.
- Two important misconceptions corrected (Q5 about epilogue, Q6 about KV path).
- User now has a comprehensive reference doc to consult before approving NVFP4 KV / W4-FP8 spike.

### Open items (unchanged from Round 7/8)
- Awaiting user choice: NVFP4 KV proposal P1 survey and/or W4-FP8 spike.

## Cross-references
- [EXPLAIN_baseline_vs_w8a8fp8_flow_20260428_1100.en.md](../EXPLAIN_baseline_vs_w8a8fp8_flow_20260428_1100.en.md)
- [ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md](../ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md) — high-level cost analysis
- [PROPOSAL_W4A8_REAL_001.en.md](../PROPOSAL_W4A8_REAL_001.en.md) — what real W4A8 kernel would look like
