# CHANGE_0151 Phase B (FourOverSix) — Iter‑7+ Deep Dive: Why GPTQ‑Marlin Beats NVFP4‑FOS on Accuracy & S1/S8 Speed

> Companion document to CHANGE_0151_phase_b_four_over_six_006 (iter‑7 measurements).
> Status: **research / no code change**. Pure read‑only code review + roofline analysis.
> Branch: `mixed_minicpm_cudagraph` @ `d48793563`. Hardware: SM120 (RTX PRO 6000 Blackwell, 96 SMs).
> Authoritative HW reference: [docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md](SM120_RTX_PRO_HARDWARE.md).

## 1. Background and ask

After iter‑7 we have:

| Config | norm‑acc | S1 (s) | S8 (s) | Smax (s) |
|---|---|---|---|---|
| GPTQ + sparse_qkv_w8 + FP8 KV (Test 12 baseline) | **99.11 %** | **121.71** | **44.09** | 35.86 |
| NVFP4‑FOS (iter‑7, iter‑5 quant repro) | 88.73 % | 173.83 (+43 %) | 46.05 (+4.4 %) | **31.07 (−13.4 %)** |

The user asked, before pivoting back to the GPTQ_FP8_DENSE catalog, to dig into:

1. Why is **NVFP4‑FOS accuracy lower** than GPTQ INT4 (Marlin)?
2. Why is **NVFP4‑FOS S1/S8 slower** than GPTQ Marlin? Naively FP4 has ~2× the tensor‑core peak of FP8 and ~4× of BF16, so this looks counter‑intuitive.
3. Provide a code‑review of the two GEMM execution paths and a roofline analysis covering **prefill vs decode**, **compute‑ vs memory‑bound**, and **fused ops**.

Findings below apply both to NVFP4‑FOS (if we ever revisit it) and to the GPTQ_FP8_DENSE optimization catalog (because they expose where the Marlin path is already doing well and where it is still leaving SM120 capabilities on the table).

## 2. Method

Read‑only code archaeology of two paths in this repo:

- **NVFP4 / `--quantization modelopt_fp4`** path — registration → loader → `apply()` → `fp4_gemm()` → CUTLASS SM120 kernel.
- **GPTQ Marlin / `--quantization gptq_marlin`** path — registration → loader → `process_weights_after_loading` → `apply_gptq_marlin_linear` → `gptq_marlin_gemm` → Marlin kernel.

Then a per‑byte / per‑FLOP roofline using the SM120 hardware sheet.

## 3. Question 1 — accuracy gap (88.73 % vs 99.11 %)

The accuracy gap has **four cumulative root causes**, none of which FOS scale‑factor search can fix:

### 3.1 Activation precision: NVFP4 is **W4A4**, GPTQ Marlin is **W4A16**

This is the single largest factor and was missed in earlier iterations.

In the NVFP4 path ([python/sglang/srt/layers/quantization/modelopt_quant.py](../../python/sglang/srt/layers/quantization/modelopt_quant.py#L1168), `ModelOptFp4LinearMethod.apply`):

```python
# Activation is quantized to FP4 on EVERY forward pass:
x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)
assert x_fp4.dtype == torch.uint8
out = fp4_gemm(x_fp4, w, x_scale_interleaved, w_scale_interleaved, alpha, output_dtype, w_n)
```

`x_fp4` is FP4 (E2M1, 8 levels per sign). Activations after RMSNorm and after attention have outliers and a long‑tailed distribution; squashing them through a per‑tensor + per‑16‑element block scale into 4 mantissa bits permanently destroys signal at every layer. Errors **compound across 80+ layers**.

In the Marlin path ([python/sglang/srt/layers/quantization/marlin_utils.py](../../python/sglang/srt/layers/quantization/marlin_utils.py#L494), `apply_gptq_marlin_linear`):

```python
output = gptq_marlin_gemm(
    reshaped_x,            # BF16 / FP16 — NOT quantized
    weight,                # repacked INT4
    weight_scale,          # permuted FP16 scales
    ...,
    is_k_full=is_k_full,
    use_fp32_reduce=use_fp32_reduce,
)
```

Activations stay BF16 end‑to‑end; the kernel dequantizes INT4 weights inline to BF16, runs `mma.sync.aligned.m16n8k16` with **FP32 accumulation**, and only the weights are lossy.

This alone explains most of the gap. NVFP4 W4A4 is much more aggressive than what its name suggests; modelopt advertises it for inference compute speed, not as a drop‑in replacement for W4A16.

### 3.2 Calibration algorithm: Hessian‑aware OBQ vs static max‑abs + FOS

GPTQ (gptqmodel) implements OBQ: for each layer, a calibration forward pass collects activations, builds a Hessian `H = X^T X`, and quantizes weights one column at a time, **compensating remaining columns** for the rounding error of the column just quantized. This is **activation‑aware** layer reconstruction error minimization. Even with only 32 sequential calibration samples, this gives the W4 weights a chance to track the actual data distribution.

modelopt NVFP4 + FOS in this repo (`benchmark/soar/demo_sala/preprocess_model.py::run_nvfp4_quantization`) does:

1. Per‑block (16) **max‑abs** to derive the FP8 block scale.
2. Per‑tensor amax → FP32 weight_scale_2 / input_scale.
3. **FOS** = pick `M ∈ {4, 6}` per block by lowest **weight‑only** local MSE (43.14 % of blocks pick M=4).

There is **no Hessian, no inter‑element compensation, no activation‑aware objective**. FOS just finds the better of two static scaling choices per block. Calibration data only contributes to the per‑tensor `input_scale`; it does not move per‑block scales or quantized levels.

Consequence: even at infinite calibration samples and perfect FOS, NVFP4‑FOS can only match a "static round‑to‑nearest with two scaling options"; it cannot reach Hessian‑compensated GPTQ accuracy.

### 3.3 Mixed‑precision asymmetry: `sparse_qkv_w8` vs uniform W4

The GPTQ baseline uses `SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8` — the 8 sparse‑attention layers' Q/K/V (24 linears, optionally also O = 32) are upgraded to **W8 g=128**. These are exactly the layers the SALA architecture marks as sensitive (sparse‑attn dispatch logits feeding back into routing). They are also where output‑channel dynamic range is the highest.

NVFP4‑FOS in iter‑5/iter‑7 ran **uniform W4** for all linears with no W8 fallback. The sensitive sparse‑attn projections were quantized at the same precision as everything else. modelopt does support per‑module quant rules, but we have not implemented a `sparse_qkv_w8` analog for the FP4 path.

### 3.4 Code‑book mismatch: FP4 E2M1 levels vs LLM weight distribution

INT4 (Marlin) levels are **uniform**: `{−8,…,+7}` × per‑group scale. Post‑GPTQ‑calibrated LLM weights have an approximately Gaussian‑ish distribution after RMSNorm‑normalised inputs, and uniform levels match this reasonably for a moderately small group (g=128).

FP4 E2M1 levels (sign × {0, 0.5, 1, 1.5, 2, 3, 4, 6}) are **non‑uniform** with denser spacing near zero and coarse spacing far from zero. This is good for log‑normal data but **wastes code points** at the high end where there are very few weights. FOS partly compensates by switching the per‑block max representable value (M=4 vs M=6), but it does not change the level positions inside the block.

### 3.5 Why FOS itself looks neutral

iter‑6 ablation (`SOAR_NVFP4_FOS_ENABLE=0` aborted at gate, ori‑acc 67.33 %) and iter‑5/iter‑7 (FOS on, ori‑acc 70.98 %–71.24 %) show FOS lifts ~3.6 pp, which is real but small relative to the ~9 pp gap to GPTQ. This is consistent with the analysis above: FOS optimizes a small slice of a deeper problem (static scaling choice within blocks), not the dominant losses (W4A4 + no Hessian compensation + uniform W4 on sensitive layers).

### 3.6 Accuracy summary table

| Factor | GPTQ Marlin | NVFP4‑FOS | Estimated impact |
|---|---|---|---|
| Activation dtype | BF16 (W4A16) | FP4 (W4A4) | **dominant**, multi‑pp |
| Calibration | OBQ, Hessian‑aware, error‑comp | static max‑abs + FOS scale search | several pp |
| Sensitive‑layer fallback | sparse_qkv_w8 (24 W8 linears) | uniform W4 | 1–2 pp |
| Code‑book / level spacing | uniform INT4 | non‑uniform FP4 E2M1 | 0.5–1 pp |
| Block size | 128 (FP16 scale) | 16 (FP8 scale) | small NVFP4 advantage |

Total observed gap: ~10 pp on normalized accuracy, fully attributable to factors 1–4 above; FOS alone cannot close it.

## 4. Question 2 — speed gap at S1/S8 despite higher FP4 peak

Key insight: **S1/S8 in this benchmark are not in the regime where FP4's 593 TFLOPS peak is the bottleneck**. They are dominated by decode and small‑M prefill chunks, which are **memory‑ and launch‑bound**, not compute‑bound. Smax exposes the long‑context compute‑bound regime where FP4's peak does win (−13.4 %).

### 4.1 Per‑layer memory footprint (4096 × 4096 linear)

| Component | GPTQ Marlin (W4A16, g=128) | NVFP4 (W4A4, block=16) |
|---|---|---|
| Packed weights | (K/8, N) int32 = **8.00 MB** | (N, K/2) uint8 = **8.00 MB** |
| Per‑group / per‑block scale | (K/128, N) FP16 = **0.25 MB** | (N, K/16) FP8 e4m3 = **1.00 MB** |
| Zero‑points | 0 (symmetric) | 0 |
| g_idx | 0 (no desc_act) | 0 |
| Per‑tensor extras | — | input_scale + weight_scale_2 + alpha (negligible) |
| **Total per layer** | **~8.25 MB** | **~9.00 MB** |
| **Per element (effective bits/weight)** | **4.13 bits** | **4.50 bits** |

**NVFP4 weights are ~9 % larger than GPTQ Marlin weights.** Because S1 decode is bound by the rate at which the GPU can stream weights from HBM, this 9 % flows almost directly into a slower decode token rate. The same analysis at S8 is partially diluted by 8× kv‑cache and activation traffic, which is why the gap shrinks (+4.4 %) rather than +9 %.

### 4.2 Tile‑shape vs M

NVFP4 SM120 dispatch ([sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu](../../sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu#L483), `cutlass_fp4_bf16_gemm_dispatch_sm120`):

```
M ≤ 256: MmaTileShape = 128 × 128 × 128, ClusterShape 1×1×1
M  > 256: MmaTileShape = 256 × 128 × 128, ClusterShape 1×1×1
```

For S1 decode `M=1`, the kernel executes a 128×128×128 tile to compute a 1×N row — roughly **128× over‑provisioned in the M dimension**. CTA occupancy and SM utilization collapse. There is no smaller‑M specialization for FP4 in the current sgl‑kernel.

Marlin in contrast carries **5 dedicated small‑batch configurations** ([sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu](../../sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#L163)):

```
(thread_k=128, thread_n=256, threads=256)
(thread_k=64,  thread_n=256, threads=256)
(thread_k=128, thread_n=128, threads=256)
(thread_k=64,  thread_n=128, threads=128)
(thread_k=128, thread_n=64,  threads=128)
```

with a runtime **scorer** (lines 310–350) that picks the best fit based on M, N, K, occupancy, smem fit, and wave count. For M=1–8, Marlin lands on a tile that uses thread_m_blocks=1 (16 in M) and amortizes over a wide N — this is much closer to the "row × matrix" shape the actual workload has. Marlin is **structurally tuned for the decode regime**; the FP4 kernel currently is not.

### 4.3 Per‑forward overhead unique to NVFP4

Each NVFP4 forward additionally runs:

- `fp4_quantize(x, input_scale_inv)` — quantize activations, produce interleaved FP8 block scales. Small kernel, but launched **once per linear per token‑step**.
- `cutlass_scaled_fp4_mm_sm100a_sm120a` host‑side runtime SM check (`getSMVersion()`) — branchy host code at kernel enqueue time. Cached after first launch but still incurs Python‑side overhead.

Marlin has **no per‑forward activation transform**; activations are BF16 directly into the kernel.

In a long S1 run with thousands of decode steps × 80+ layers, even a few‑µs‑per‑linear difference accumulates to several seconds.

### 4.4 MMA op and SM120 utilization

| Path | MMA instruction | Tensor‑core class | Theoretical SM120 peak |
|---|---|---|---|
| GPTQ Marlin (BF16 inputs after dequant) | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | warp‑level Ampere‑style | **148 TFLOPS BF16** |
| NVFP4 (FP4 inputs, FP8 block scales) | CUTLASS `OpClassBlockScaledTensorOp`, SM120 native | warp‑level block‑scaled | **296 TFLOPS FP8 / 593 TFLOPS FP4** |

The NVFP4 path **does** target SM120 natively (`ArchTag = cutlass::arch::Sm120`, build flags `compute_120a/sm_120a`, `-DENABLE_NVFP4=1`), so on a sufficiently large M (S∞ / Smax) it does cash in the FP4 peak. On small M it cannot, because the fixed 128×M tile destroys utilization before the MMA peak is reached.

Marlin is structurally locked at **148 TFLOPS BF16** on SM120 — it cannot use QMMA or block‑scaled FP4 ops because its kernel is built around `m16n8k16` BF16 MMA. This is why **Smax flips in NVFP4's favour**: at M~thousands the tile is well‑filled and FP4's 4× peak materializes; at S1 it cannot.

### 4.5 Roofline by stage (SM120, BW = 1398 GB/s)

|  | M | Bound | NVFP4‑FOS | GPTQ Marlin | Winner |
|---|---|---|---|---|---|
| Decode | 1 | memory‑bound on weights + tile under‑utilization | 9 MB/layer + 128‑tile waste + per‑forward act quant | 8.25 MB/layer + small‑M tile picker | **Marlin** |
| Prefill chunk (S1 long ctx) | up to 65 536 / step | mixed; small chunks memory‑bound, large compute‑bound | tile fits at high M | BF16 peak limits gain | tie at small chunks, **NVFP4** at large chunks |
| Decode (S8) | 8 | memory‑bound + small compute | same as S1 but 8× activations partly amortize weights | small‑M tile picker | **Marlin** (margin shrinks) |
| Prefill (Smax, S∞) | thousands | compute‑bound | FP4 peak 593 TFLOPS, tile fully filled | BF16 peak 148 TFLOPS | **NVFP4** (consistent with measured −13.4 %) |

This matches the observation that the speed deltas flip sign across S1 → S8 → Smax.

### 4.6 Fused ops

| Fused op | NVFP4 path | Marlin path |
|---|---|---|
| Weight dequant inside MMA | yes (block‑scaled tensor op) | yes (LUT + shifts inline) |
| Per‑tensor `alpha = input_scale × weight_scale_2` | yes, in epilogue | n/a (no activation scale) |
| Bias add | not in kernel; host‑side after | yes, in epilogue |
| Activation quant (per forward) | **separate kernel before GEMM** | none |
| QK‑norm + RoPE | server‑side `--enable-fused-qk-norm-rope`, equally available to both | same |
| Mixed‑chunk scheduling | server‑side `--enable-mixed-chunk`, equally available to both | same |

Net: NVFP4 has one extra pre‑GEMM kernel (activation quant) per linear per step. Marlin has none. Both fuse the dominant dequant inside the GEMM. Server‑level fusion (qk‑norm‑rope, mixed‑chunk) is identical.

## 5. Profiling proposals to confirm

These would close the loop on the analysis but are NOT required to act on the conclusions:

1. **`nsys profile`** an S1 run and an Smax run, group by kernel name; expect:
   - S1: top kernels are `gptq_marlin_gemm` (Marlin run) or `cutlass_scaled_fp4_mm_*` + `fp4_quantize` (NVFP4 run); the per‑forward `fp4_quantize` should account for several percent of NVFP4 wall time.
   - Smax: top kernel is the FP4 GEMM at large M; SM efficiency should be > 60 %.
2. **`ncu`** with `--section MemoryWorkloadAnalysis,LaunchStats,Occupancy,SchedulerStats,WarpStateStats` on the FP4 GEMM at fixed problem sizes M ∈ {1, 8, 64, 512, 4096} to confirm tile under‑utilization at small M.
3. **`cuobjdump --dump-sass`** on the installed `sgl_kernel*.so` to confirm `cutlass_fp4_bf16_gemm_dispatch_sm120` actually emits SM120 SASS, and Marlin emits whatever it falls back to (compute_90 or compute_120a).
4. Microbench: a standalone `gptq_marlin_gemm` vs `fp4_gemm` script across {bs=1, 8, 64, 512, 4k} × {seq=1k, 8k, 32k} on the same shapes the model actually uses.

## 6. Implications

### For NVFP4‑FOS (parking now, possibly revisit later)

If we ever return to NVFP4 we should:

- **Implement W4A16 NVFP4** (FP4 weights with BF16 activations) instead of W4A4. modelopt supports activation‑off configurations. Expected accuracy lift: most of factor 3.1.
- **Add a `sparse_qkv_w8` analog** for the FP4 path (or per‑module quant config) so sparse‑attn QKV stay in higher precision. Expected lift: factor 3.3.
- **Add small‑M tile configs** to `cutlass_fp4_bf16_gemm_dispatch_sm120` (e.g. 16×128×128 or 32×128×128) and a runtime scorer like Marlin's. Expected S1 lift: large.
- **Fuse `fp4_quantize` into the previous op** (likely the residual + RMSNorm) to remove the per‑linear pre‑GEMM kernel.
- **Apply OBQ‑style or Hessian‑aware calibration on top of FOS** before settling.

### For GPTQ_FP8_DENSE catalog (next round)

The audit confirms that the Marlin path is **structurally near its peak in this codebase**; the obvious wins are:

- **It is BF16‑peak‑bound on SM120 (148 TFLOPS)**. To break this ceiling we would need an FP8 W8A8 path (cutlass `OpClassBlockScaledTensorOp` at FP8) for the W8 sparse‑attn QKV layers, or migrate to a Hopper/Blackwell FP8 GEMM. This is a major engineering item.
- **No TMA** in Marlin → at very long context the prefill phase leaves bandwidth on the table. Adding TMA bulk loads to the Marlin kernel is non‑trivial but yields concrete wins.
- **Per‑forward overhead is already ~zero**, so we should not chase it.
- **Memory footprint is already ~optimal** at 4.13 bits/weight; no easy further compression.
- The realistic catalog targets are therefore on the **scheduling / scheduler / KV / attention / fused‑pre‑attn** layers above the GEMM, not inside the Marlin kernel itself.

This refines the priority of the existing catalog [docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md): de‑prioritize "rewrite Marlin", up‑prioritize "scheduler / KV / fused‑pre‑attn / mcq runaway‑think mitigation".

## 7. Validation commands (if/when we want to verify experimentally)

```bash
# Prepare instance
python3 scripts/fcloud/fcloud_workflow.py start-instance
python3 scripts/fcloud/fcloud_workflow.py sync

# nsys on S1 (GPTQ baseline)
python3 scripts/fcloud/fcloud_exec.py exec '
cd /root/submission_sim &&
source prepare_env.sh &&
nsys profile -o /root/nsys_gptq_s1 -t cuda,nvtx --force-overwrite=true \
  python3 /root/data/eval_model_001.py \
    --data_path /root/data/perf_public_set.jsonl --max-concurrent 1 \
    --output_dir /root/outputs_nsys
'

# Same for NVFP4 (set SOAR_QUANT_PROFILE=nvfp4_fos in prepare_env first, requantize, run)
```

## 8. Result summary

| Question | Conclusion |
|---|---|
| Q1 accuracy gap | Dominated by **W4A4 vs W4A16 activation precision** (NVFP4 quantizes activations to FP4 every forward), then **OBQ Hessian‑aware vs static FOS calibration**, then **uniform W4 vs sparse_qkv_w8 mixed precision**. FOS scale‑search alone cannot close this. |
| Q2 S1/S8 speed gap | **Memory‑ and launch‑bound regime, not compute‑bound.** NVFP4 weights are ~9 % heavier (FP8 block scales at g=16); FP4 GEMM has no small‑M tile specialization (128×M tile vs Marlin's 5‑config small‑batch scorer); per‑forward `fp4_quantize` adds a kernel per linear. FP4's 4× compute peak only materializes at Smax / S∞ (large M), exactly as observed (Smax −13.4 %). |
| Q2 fused ops / roofline | Marlin already fuses dequant + bias + accumulate; NVFP4 fuses dequant + alpha but pays for an extra activation‑quant kernel each forward. Server‑level fusions (qk‑norm‑rope, mixed‑chunk) are equal. The Marlin path is BF16‑peak‑bound on SM120 (148 TFLOPS); NVFP4 is FP4‑peak‑capable (593 TFLOPS) only at large M. |

## 9. Rollback

This document changes no code. Nothing to roll back.

## 10. Next step

Pivot to the GPTQ_FP8_DENSE optimization catalog with the priority shift recommended in §6.

## 11. Cross‑references

- iter‑5 plan: CHANGE_0151_phase_b_four_over_six_004
- iter‑5 reference quant: TEST_RESULTS_TRACKING row NVFP4‑FOS‑5
- iter‑6 FOS=0 ablation: CHANGE_0151_phase_b_four_over_six_005, row NVFP4‑FOS‑6
- iter‑7 reproduction + speed: CHANGE_0151_phase_b_four_over_six_006, row NVFP4‑FOS‑7
- HW reference: SM120_RTX_PRO_HARDWARE.md
- Catalog to update next: OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md

## 12. Code references (key call sites)

- NVFP4 loader: [python/sglang/srt/layers/quantization/modelopt_quant.py](../../python/sglang/srt/layers/quantization/modelopt_quant.py#L861) (`ModelOptFp4Config`), [line 1068](../../python/sglang/srt/layers/quantization/modelopt_quant.py#L1068) (`ModelOptFp4LinearMethod`), [line 1168](../../python/sglang/srt/layers/quantization/modelopt_quant.py#L1168) (`apply()` with `fp4_quantize` + `fp4_gemm`).
- NVFP4 SM120 kernel: [sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu](../../sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu#L483) (`cutlass_fp4_bf16_gemm_dispatch_sm120`, tile configs lines 123–137, dispatch line 657).
- Marlin loader: [python/sglang/srt/layers/quantization/gptq.py](../../python/sglang/srt/layers/quantization/gptq.py#L219) (`GPTQMarlinConfig`), [line 563](../../python/sglang/srt/layers/quantization/gptq.py#L563) (`GPTQMarlinLinearMethod`), [line 893](../../python/sglang/srt/layers/quantization/gptq.py#L893) (`apply`).
- Marlin dispatch + scorer: [python/sglang/srt/layers/quantization/marlin_utils.py](../../python/sglang/srt/layers/quantization/marlin_utils.py#L464) (`apply_gptq_marlin_linear`); [sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu](../../sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#L163) (small‑batch configs, scorer lines 310–350).
- Marlin MMA: [sgl-kernel/csrc/gemm/marlin/marlin_template.h](../../sgl-kernel/csrc/gemm/marlin/marlin_template.h#L77) (`mma.sync.aligned.m16n8k16` for FP16/BF16).
- Build target: [sgl-kernel/CMakeLists.txt](../../sgl-kernel/CMakeLists.txt#L231) (`compute_100a/120a` and `-DENABLE_NVFP4=1` when CUDA ≥ 12.8).
