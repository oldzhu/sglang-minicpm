# PROPOSAL — Iteration W4A8 (#1): FP8 in-kernel GEMM via TRT-LLM (post-v18 baseline)

**Date**: 2026-04-27 09:05  
**Status**: ✅ APPROVED 2026-04-27 with amendments (Option B chosen, no TRT-LLM wheel, soft accuracy guardrail) — implementation can start
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

**Watchpoint**: re-run accuracy on `perf_public_set.jsonl` immediately after enabling. If normalized accuracy drops below 79 % (current 77.44 % v18 + ~1.5 % safety), do NOT auto-abort: weigh the accuracy delta against the measured speed gains and decide jointly with the user whether to keep, tune, or revert. Hard floor remains the 97 % rule (C=0); we never knowingly cross that.

## 4. Files to change

### 4.1 New code (small surface)

| File | Change |
|---|---|
| `python/sglang/srt/layers/quantization/gptq_marlin.py` | Add forward-path branch: if `os.environ.get("SOAR_W4A8_FP8_GEMM") == "1"` AND `sgl_kernel.fp8_blockwise_scaled_mm` is importable AND layer is in eligible set (std-attn QKV/O + MLP gate/up/down) → call sgl-kernel FP8 blockwise GEMM; else → existing Marlin path |
| `python/sglang/srt/models/minicpm.py` | Tag eligible layers at construction (an attribute the linear forward checks) — exclude lightning Q/K/V/O for safety in v1 |
| `benchmark/soar/demo_sala/preprocess_model.py` | After GPTQ quantization, dequantize INT4 → BF16 once, then re-quantize to FP8 e4m3 with 128×128 block-scale and save as `weight_fp8` + `weight_fp8_scale`. Skipped if env flag off |
| `benchmark/soar/demo_sala/prepare_env.sh` | Append `export SOAR_W4A8_FP8_GEMM=1`. **No new pip install** — reusing the sgl-kernel wheel we already ship |

### 4.2 No change required

- All attention backends (FlashInfer, FA, SimpleGLA) — they don't call GEMM directly.
- Lightning recurrent state path — unchanged (orthogonal optimization, separate proposal #3).
- Eval harness `eval_model_001.py` — unchanged.
- Server args structure — unchanged.

## 5. Detailed implementation plan (before change — for review)

```
Step 1: Locate the sgl-kernel FP8 blockwise op + confirm SM120 dispatch
   └─ grep sgl-kernel for `fp8_blockwise_scaled_mm` and `sm120_fp8_blockwise_dispatch_shape`
   └─ Confirm op signature: weight (K,N) FP8 e4m3 col-major + per-128x128 block scales
   └─ Confirm activation contract: per-token or per-128 dynamic FP8 quantize
   └─ No fcloud step needed — sgl-kernel is already in our build pipeline

Step 2: Write FP8-blockwise weight conversion (preprocess_model.py)
   └─ For each whitelisted linear layer:
        w_bf16 = gptq_dequantize(qweight, scales, qzeros, group_size=128)
        per 128x128 block of w_bf16:
            block_amax = block.abs().amax()
            scale_b   = block_amax / 448.0
            w_fp8_blk = (block / scale_b).clamp(-448, 448).to(torch.float8_e4m3fn)
        save weight_fp8 (K,N col-major) + weight_fp8_scale ((K/128, N/128) fp32)
   └─ Unit test: w_bf16_recovered = w_fp8.float() * scale_b_broadcast; Frobenius diff < 1e-3 vs original w_bf16.

Step 3: Linear forward branch (gptq_marlin.py)
   └─ if SOAR_W4A8_FP8_GEMM and self.allow_fp8 and sgl_kernel_fp8_available:
          input_fp8, input_scale = per_token_fp8_quantize(input)   # also from sgl-kernel
          out = sgl_kernel.fp8_blockwise_scaled_mm(
              input_fp8, self.weight_fp8,
              input_scale, self.weight_fp8_scale,
              out_dtype=torch.bfloat16)
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
- **V4 (must)**: `ncu --section ComputeWorkloadAnalysis ... ` should now show **non-zero `sm__inst_executed_pipe_tensor_op_qmma`** on the new `fp8_blockwise_scaled_mm` kernel. This is the single most important runtime signal — without it, we are not actually using FP8 hardware.
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

## 10. Approval record (resolved 2026-04-27)

| # | Item | User decision |
|---|---|---|
| 1 | Scope: v1 limited to std-attn + MLP only | ✅ Approved |
| 2 | Path choice (Option A TRT-LLM wheel vs Option B sgl-kernel built-in) | ✅ **Option B chosen** — concern: extra wheel risks dependency conflicts and inflates 2 GB submission package |
| 3 | Env-flag gate `SOAR_W4A8_FP8_GEMM=1`, default off | ✅ Approved |
| 4 | Accuracy guardrail: hard-abort below 79 % normalized | ⚠️ Amended — soft guardrail. Decide based on observed speed/accuracy trade-off. Hard floor stays the 97 % rule (C=0) |
| 5 | Step 1 = verify TRT-LLM wheel | ❌ Skipped — Option B uses sgl-kernel built-in op, no wheel needed |

**Implementation order updated**: skip the wheel-verification step. Begin at Step 1 (locate the sgl-kernel op signature) and proceed sequentially through Step 4.

## 11. Cross-reference

- `CLARIFICATION_quant_layers_runtime_verify_20260427_0811.{en,zh}.md` — establishes that today's path is BF16-on-BF16-tensor-core (motivation)
- `RESEARCH_mixed_arch_speed_optimization_20260426_1526.{en,zh}.md` — places this as #1
- `PROPOSAL_fp8_w4_dequant_gemm.md` — original Phase-1 TRT-LLM investigation (kernel API, scale conversion sketch)
- `PROPOSAL_option_b_fp8_blockwise_gemm.en.md` — fallback (no extra wheel) using sgl-kernel
- `ANALYSIS_nvfp4_offline_quant_20260427_0857.{en,zh}.md` — explains why we are NOT going to FP4 weights at this step

---

## 12. Step 1 findings (2026-04-27)

Reconnaissance of the sgl-kernel + sglang FP8 stack revealed that **most of the integration is already wired in upstream sglang**:

| Discovery | Location | Implication |
|---|---|---|
| `fp8_blockwise_scaled_mm` op confirmed for SM120 | `sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu:368, 453` (SM120 dispatch via `sm120_fp8_blockwise_dispatch_shape`) | No kernel writing needed; we ship this in our build |
| Op signature | `(a: (M,K) e4m3 row-major, b: (N,K) e4m3 col-major after .t(), scales_a: (M, K/128) fp32, scales_b: (N/128, K/128) fp32, out_dtype) -> bf16/fp16` | Activation = per-token + per-128-K groups |
| Existing high-level wrapper | `python/sglang/srt/layers/quantization/fp8_utils.py:342` `cutlass_w8a8_block_fp8_linear_with_fallback` | Already does `per_token_group_quant_fp8(input, 128) → fp8_blockwise_scaled_mm(q_input, weight.T, x_scale, weight_scale.T)`. **We can call this directly.** |
| Activation quantize op | `sglang_per_token_group_quant_fp8` (Triton kernel from `sglang.srt.layers.quantization.fp8_kernel`) | Already used by FP8 paths. No new kernel needed. |
| Gate point for the branch | `python/sglang/srt/layers/quantization/gptq.py:787` `GPTQMarlinLinearMethod.apply()` | Single function, one if/else. |

**Implication**: the runtime side of this iteration is ~30 lines of dispatch in `gptq.py` + a one-shot `weight_fp8` attribute set during weight post-processing. The bulk of the work is in `preprocess_model.py` (offline conversion).

**Updated runtime branch sketch** (replaces what's in §5 Step 3):

```python
# Inside GPTQMarlinLinearMethod.apply()
if (
    os.environ.get("SOAR_W4A8_FP8_GEMM") == "1"
    and getattr(layer, "_soar_w4a8_eligible", False)
    and hasattr(layer, "weight_fp8")
):
    return cutlass_w8a8_block_fp8_linear_with_fallback(
        input=x,
        weight=layer.weight_fp8,            # (N, K) e4m3
        block_size=[128, 128],
        weight_scale=layer.weight_fp8_scale,  # (N/128, K/128) fp32
        bias=bias,
    )
# else: original Marlin path below (unchanged)
```

**`_soar_w4a8_eligible`** is the whitelist tag set in `minicpm.py` model construction — std-attn QKV/O + MLP gate/up/down only.

**Next step (pending user go-ahead)**: implement Step 2 — the preprocess_model.py conversion that produces `weight_fp8` + `weight_fp8_scale` from GPTQ INT4 weights. After that, Step 3 (the gptq.py dispatch above) and Step 4 (whitelist tag in minicpm.py) are straightforward.
