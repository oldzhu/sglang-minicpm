# PROPOSAL — Iteration W4A8 (#1): FP8 in-kernel GEMM via TRT-LLM (post-v18 baseline)

**Date**: 2026-04-27 09:05
**Status**: ⏳ AWAITING USER APPROVAL — no code change yet
**Priority**: #1 in the post-v18 optimization list
**Baseline (v18-revert, Test 25-equivalent)**: S1=110.51s, S8=40.46s, Smax=33.61s, normalized accuracy 77.44 % → C=0.92
**Predecessors**: `PROPOSAL_fp8_w4_dequant_gemm.md` (Phase 1 investigation, 2026-04-21), `PROPOSAL_option_b_fp8_blockwise_gemm.en.md` (sgl-kernel-internal alternative).
**This proposal supersedes both** as the actionable plan now that the v18 baseline is locked.

---

## 1. Background and motivation

Per `CLARIFICATION_quant_layers_runtime_verify_20260427_0811`:

- Today's GPTQ + Marlin path keeps weights stored as INT4 but **dequantizes to BF16 inside the kernel** and runs **BF16 × BF16** on the BF16 tensor core (148 TFLOPS on SM120).
- SM120's **FP8 QMMA tensor core peaks at 296 TFLOPS** — exactly 2× — and is currently completely unused.
- Per the most recent `RESEARCH_mixed_arch_speed_optimization_20260426_1526`, GEMM dominates prefill cost on the long-context official speed dataset.
- Therefore moving the GEMM math to FP8 (while keeping weights stored as W4) directly captures the largest single piece of unused hardware on the device.

This is "**W4A8**": W = 4-bit storage, A = 8-bit activations, with FP8 multiply-accumulate.

## 2. Rule-compliance check

| Rule | Status | Notes |
|---|---|---|
| Submission ≤ 2 GB | ✅ | TRT-LLM Python wheel ≈ 100-200 MB, fits inside the cap |
| On-site quantization | ✅ | Weight format conversion runs in `preprocess_model.py` (no pre-quantized submission) |
| ≤ 5 h prep time | ✅ | Conversion is matrix-reshape + per-row scale fold; minutes, not hours |
| Code license Apache 2.0 + reproducible | ✅ | TRT-LLM is Apache-2.0, autotuned tactic is logged once |
| Accuracy C ≥ 0.96 (target C=1.0) | ⚠️ | FP8 e4m3 has 3 mantissa bits + 4 exp; activation quantization adds ≤ 0.5 % error in literature for similar models. Mitigation: keep BF16 fallback path active; gate FP8 path with env flag |
| Concurrency tier flags `--max-concurrent {1,8,∞}` unchanged | ✅ | No scheduling change |

## 3. Risk to accuracy / stability

**Identified risks**:
1. **Activation outliers**: FP8 e4m3 saturates at ±448. SiLU and softmax outputs in MiniCPM-SALA can spike. Mitigation: per-token dynamic scale on activations (TRT-LLM kernel default).
2. **Lightning-layer interaction**: 24 lightning layers compound errors multiplicatively across decode steps. Mitigation: **scope FP8 GEMM to the 8 std-attn + MLP layers only initially**; keep lightning Q/K/V/O on Marlin BF16 path until proven safe in a follow-up iteration.
3. **TRT-LLM wheel compatibility on fcloud**: Need to verify pip-install on the SM120 environment in step 1 of execution.
4. **Scale conversion correctness**: GPTQ group_size=128 BF16 scales must be folded into per-row FP8 scales. Mitigation: write unit test comparing dequantized weights pre- and post-conversion (Frobenius diff < 1e-3).

**Watchpoint**: re-run accuracy on `perf_public_set.jsonl` immediately after enabling. If normalized accuracy drops below 79 % (current 77.44 % v18 + ~1.5 % safety), gate FP8 off via env var and treat as failure.

## 4. Files to change

### 4.1 New code (small surface)

| File | Change |
|---|---|
| `python/sglang/srt/layers/quantization/gptq_marlin.py` | Add forward-path branch: if `os.environ.get("SOAR_W4A8_FP8_GEMM") == "1"` AND TRT-LLM available AND layer is in eligible set (std-attn QKV/O + MLP gate/up/down) → call TRT-LLM kernel; else → existing Marlin path |
| `python/sglang/srt/models/minicpm.py` | Tag eligible layers at construction (an attribute the linear forward checks) — exclude lightning Q/K/V/O for safety in v1 |
| `benchmark/soar/demo_sala/preprocess_model.py` | After GPTQ quantization, fold per-group BF16 scales into per-row FP8 scales and save alongside `.qweight`. Add `weight_fp8_scale` tensors to safetensors. Skipped if env flag off. |
| `benchmark/soar/demo_sala/prepare_env.sh` | Append `export SOAR_W4A8_FP8_GEMM=1` and `pip install tensorrt-llm-blackwell-min==<version>` (or whatever wheel name we settle on after step 1) |

### 4.2 No change required

- All attention backends (FlashInfer, FA, SimpleGLA) — they don't call GEMM directly.
- Lightning recurrent state path — unchanged (orthogonal optimization, separate proposal #3).
- Eval harness `eval_model_001.py` — unchanged.
- Server args structure — unchanged.

## 5. Detailed implementation plan (before change — for review)

```
Step 1: Verify TRT-LLM wheel availability on fcloud
   └─ ssh fcloud → pip install tensorrt-llm... (test only, no commit)
   └─ python -c "import torch; torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell"
   └─ Report wheel size and import-time overhead
   └─ If unavailable, abort this iteration and fall back to Option B (sgl-kernel internal)

Step 2: Write scale-conversion utility (preprocess_model.py)
   └─ For each linear layer with .qweight + .scales (group=128, BF16):
        per_row_max = scales.float().amax(dim=group_dim)
        per_row_fp8_scale = per_row_max / 448.0
        weight_fp8_scale = per_row_fp8_scale (per output row)
   └─ Unit test: dequantize INT4 with original BF16 scale, then with new FP8 scale; Frobenius diff < 1e-3.

Step 3: Linear forward branch (gptq_marlin.py)
   └─ if SOAR_W4A8_FP8_GEMM and self.allow_fp8 and trtllm_available:
          input_fp8, input_scale = quantize_per_token_fp8(input)
          out = torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell(
              input_fp8, self.weight_fp8, input_scale, self.weight_fp8_scale,
              output_dtype=torch.bfloat16, use_tvm_ffi=True)
      else:
          out = existing_marlin_path(...)

Step 4: Whitelist eligible layers in MiniCPM model construction
   └─ Std-attn QKV (sparse + non-sparse): eligible
   └─ Std-attn O proj: eligible
   └─ MLP gate/up/down: eligible
   └─ Lightning Q/K/V/O: NOT eligible (v1) — keep BF16 Marlin
   └─ lm_head: eligible only if normalized accuracy holds; otherwise keep BF16
```

## 6. Validation commands

### Correctness (must run before merge)
```bash
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
# Expected: normalized accuracy ≥ 79 % (v18 was 77.44; we keep ≥1.5pt safety)
```

### Speed (after correctness passes)
```bash
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax
```

### Runtime verification (from CLARIFICATION doc)
- **V4 (must)**: `ncu --section ComputeWorkloadAnalysis ... ` should now show **non-zero `sm__inst_executed_pipe_tensor_op_qmma`** on the GEMM kernels. This is the single most important runtime signal — without it, we are not actually using FP8 hardware.
- **V1 (cheap)**: Confirm `weight_fp8` and `weight_fp8_scale` tensors loaded for whitelisted layers.

## 7. Expected results (baseline vs new)

| Metric | v18 baseline | Expected after W4A8 (conservative) | Expected after W4A8 (optimistic) |
|---|---|---|---|
| S1 | 110.51 s | 102-105 s (-5 %) | 95-100 s (-10 %) |
| S8 | 40.46 s | 36-38 s (-7 %) | 33-35 s (-13 %) |
| Smax | 33.61 s | 30-32 s (-8 %) | 27-29 s (-15 %) |
| Normalized accuracy | 77.44 % | 76-79 % (within noise) | 76-79 % |
| C coefficient | 0.92 | 0.92 (target) | 0.92 |
| Final score (proportional) | reference | +5-8 % | +10-15 % |

Conservative is the planning number; optimistic is upper bound if TRT-LLM achieves >80 % of FP8 peak.

## 8. Rollback

```bash
# Source-side rollback
git revert <merge-commit>
git push minicpm-src mixed_minicpm_cudagraph

# Runtime rollback (no code revert needed)
unset SOAR_W4A8_FP8_GEMM   # in prepare_env.sh
# Marlin BF16 path activates automatically on next server start
```

The env-flag gate means we can disable FP8 GEMM at submission time without re-quantizing the model. The `weight_fp8` tensors stay in the safetensors but go unused.

## 9. Iteration scope guard

Per copilot rules ("one improving feature at a time"), this iteration includes ONLY:
- W4A8 FP8 GEMM for std-attn + MLP layers
- Scale-conversion in preprocess
- env-flag gate

It does NOT include:
- Lightning state FP8 (proposal #3, future iteration)
- Fused lightning kernel (proposal #4, future iteration)
- KV cache FP4 (M2.0, separate)
- Speculative decoding (#6, future iteration)

If tests pass, those follow as separate iterations.

## 10. Approval contract — to be answered before code edits

Please confirm or amend:

1. **Scope**: agree to limit v1 to std-attn + MLP only (lightning Q/K/V/O stays BF16)? (recommended)
2. **Path choice**: TRT-LLM wheel (Option A, low effort) vs sgl-kernel-internal `fp8_blockwise_scaled_mm` (Option B, no extra wheel)? (recommended A)
3. **Env-flag gate**: `SOAR_W4A8_FP8_GEMM=1` to enable, default off until validated? (recommended)
4. **Accuracy guardrail**: abort if normalized accuracy drops below 79 % on local public set? (recommended)
5. **Step 1 first**: verify TRT-LLM wheel installs on fcloud before any code change in this repo? (recommended)

Once you approve (or amend) these, the plan is to start with **Step 1 only** and report back the wheel-availability result before continuing.

## 11. Cross-reference

- `CLARIFICATION_quant_layers_runtime_verify_20260427_0811.{en,zh}.md` — establishes that today's path is BF16-on-BF16-tensor-core (motivation)
- `RESEARCH_mixed_arch_speed_optimization_20260426_1526.{en,zh}.md` — places this as #1
- `PROPOSAL_fp8_w4_dequant_gemm.md` — original Phase-1 TRT-LLM investigation (kernel API, scale conversion sketch)
- `PROPOSAL_option_b_fp8_blockwise_gemm.en.md` — fallback (no extra wheel) using sgl-kernel
- `ANALYSIS_nvfp4_offline_quant_20260427_0857.{en,zh}.md` — explains why we are NOT going to FP4 weights at this step
