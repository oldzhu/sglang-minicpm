# CHAT — Perf deep‑dive: NVFP4‑FOS vs GPTQ Marlin (accuracy + S1/S8 gap)

Timestamp: 20260509_0703  •  Branch: `mixed_minicpm_cudagraph` @ `d48793563`

## 1. User ask (verbatim)

> before we go to gptq catalog, let us dig more into on the below things:
> 1. why NVFP4-FOS weigths accurray is lower than gptq INT4?
> 2. why NVFP4-FOS s1/s8 speed benchmark is slow than gptq INT4 (FP8 kvcache) marlin? which looks wired seem NVFP4 tensor should have doubled TFLOPS than FP8?
> Code review of the different execution path (NVFP4 and marlin kernel) may be helpful or any other profiliing or ways can help here?
> answers of the two questions not only can help to improving NVFP4+FOS but also can help on improving on GPTQ_FP8_DENSE path if we decide to switch to it.
> better to have anaylsis/report on prefill/decode, compute and memory bound, Fused ops.

## 2. Agent actions

- Pure read‑only research. No code change. No fcloud test (instance still paused after iter‑7).
- Dispatched 2 parallel `Explore` subagents (medium thoroughness):
  - **A** mapped the NVFP4 / `modelopt_fp4` path (loader, `apply()`, `fp4_gemm`, CUTLASS SM120 dispatch).
  - **B** mapped the GPTQ Marlin path (loader, `process_weights_after_loading`, `apply_gptq_marlin_linear`, `gptq_marlin_gemm`, MMA, build targets, mixed‑precision).
- Cross‑checked one inconsistency (subagent B claimed SM120 was not a build target; subagent A confirmed `compute_120a/sm_120a` is enabled when CUDA ≥ 12.8). Used the latter (verified in `sgl-kernel/CMakeLists.txt`).
- Wrote bilingual analysis report:
  - [docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_007.en.md](../CHANGE_0151_phase_b_four_over_six_007.en.md)
  - [docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_007.zh.md](../CHANGE_0151_phase_b_four_over_six_007.zh.md)

## 3. Key findings (one‑pager)

### Q1 accuracy gap (88.73 % vs 99.11 %)
1. **NVFP4 is W4A4, GPTQ is W4A16.** modelopt FP4 quantizes activations to FP4 every forward (`fp4_quantize` in `apply`). This is the dominant factor.
2. **Calibration**: GPTQ uses Hessian‑aware OBQ with inter‑column error compensation; modelopt FP4 + FOS uses static per‑block max‑abs + a 2‑way scale search (M=4 vs M=6). No Hessian. No activation awareness.
3. **Mixed precision**: GPTQ baseline runs `sparse_qkv_w8` (24 sparse‑attn QKV linears at W8 g=128); NVFP4 path uses uniform W4 with no sensitive‑layer fallback.
4. **Code book**: FP4 E2M1 levels {0, 0.5, 1, 1.5, 2, 3, 4, 6} are non‑uniform; uniform INT4 is closer to RMSNorm‑normalized weight distribution.

FOS lifts ~3.6 pp (iter‑6 ablation 67.33 % vs iter‑5 70.98 %), real but small relative to the ~10 pp gap.

### Q2 speed gap at S1/S8 (despite higher FP4 peak)
1. **S1/S8 are memory‑ and launch‑bound, not compute‑bound.** NVFP4 weights are ~9 % heavier than Marlin weights at 4096×4096: 9.00 MB vs 8.25 MB (FP8 e4m3 block scales at g=16 cost 1 MB vs FP16 scales at g=128 cost 0.25 MB).
2. **FP4 GEMM has no small‑M tile specialization.** SM120 dispatch in `cutlass_fp4_bf16_gemm_dispatch_sm120` uses 128×128×128 for M ≤ 256 — at S1 decode M=1 the M dimension is ~128× over‑provisioned. Marlin carries 5 small‑batch configs + a runtime scorer tuned for M=1–8.
3. **NVFP4 pays a per‑forward `fp4_quantize` kernel for every linear**; Marlin pays nothing (BF16 activations straight into the GEMM).
4. **Compute peaks**: Marlin's `mma.sync.aligned.m16n8k16` BF16 path is locked at 148 TFLOPS on SM120; NVFP4's `OpClassBlockScaledTensorOp` reaches 296 TFLOPS (FP8) / 593 TFLOPS (FP4) — but only at M large enough to fill the 128×M tile.
5. This is exactly consistent with the observed cross‑over: S1 +43 % slower, S8 +4.4 % slower, **Smax −13.4 % faster**.

### Roofline summary

| Stage | M | Bound | Winner | Why |
|---|---|---|---|---|
| Decode | 1 | mem + tile under‑use + extra act‑quant kernel | **Marlin** | smaller weights, small‑M tile picker |
| Prefill chunk (S1 long ctx) | up to 65 536/step | mixed | tie at small chunks, **NVFP4** at large | FP4 peak only at large M |
| Decode (S8) | 8 | mem + small compute | **Marlin** | gap shrinks |
| Prefill (Smax / S∞) | thousands | compute | **NVFP4** | FP4 peak materializes |

### Fused ops

- Marlin: dequant + bias + accumulate fused inside the kernel; nothing per‑forward outside.
- NVFP4: dequant + alpha fused inside; pays one extra `fp4_quantize` kernel per linear per step.
- Server‑side fusions (`--enable-fused-qk-norm-rope`, `--enable-mixed-chunk`) are equally available to both.

## 4. Outcomes

- Decision: **park NVFP4‑FOS**, return to GPTQ_FP8_DENSE catalog as planned.
- Catalog priority shift recommended in §6 of the EN/ZH report:
  - **down‑prioritize** "rewrite Marlin kernel" (its Ampere‑style MMA caps SM120 at 148 TFLOPS BF16, but rewriting it is a major engineering item; weight bytes are already near‑optimal at 4.13 bits/elem)
  - **up‑prioritize** scheduler / KV / fused‑pre‑attn / mcq runaway‑think mitigation
- Optional follow‑up if we ever revisit NVFP4: try W4A16 NVFP4 (no activation quant), add `sparse_qkv_w8` analog for FP4, add small‑M tile configs to `cutlass_fp4_bf16_gemm_dispatch_sm120`, fuse `fp4_quantize` into the prior RMSNorm kernel, layer OBQ‑style calibration on top of FOS.

## 5. Cross‑references

- Companion measurements: CHANGE_0151_phase_b_four_over_six_006 (iter‑7), TEST_RESULTS_TRACKING row NVFP4‑FOS‑7
- Hardware: SM120_RTX_PRO_HARDWARE.md
- Catalog to refine next: OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md

## 6. Files created/modified

- created: `docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_007.en.md`
- created: `docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_007.zh.md`
- created: this chat log + ZH twin

## 7. Open questions / next round

- Validate Q1 by trying **NVFP4 W4A16** (modelopt option to disable activation quant) — would isolate the W4A4 vs W4A16 contribution (~major share of the gap by hypothesis).
- Validate Q2 with `nsys` + `ncu` on S1 vs Smax (commands listed in §7 of the report). Not blocking for the GPTQ catalog pivot.
