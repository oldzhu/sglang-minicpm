# Deep dive — Baseline (W4A16 BF16) vs W8A8-on-W4 (W8A8 FP8) execution flow

**Date**: 2026-04-28
**Status**: Explanation document. No code change. Written to answer user questions about code-level/instruction-level/hardware-level differences between the two paths.

This document complements `ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.{en,zh}.md`. Read this **after** that document for the cost analysis; this one is the line-by-line execution trace.

---

## 0. Roadmap

We compare two paths end-to-end:
- **Path A (v18 baseline, in production)**: GPTQ W4 + Marlin GEMM + BF16 activations + FP8 e5m2 KV
- **Path B (W4A8 #1 mislabel test, env-gated off)**: same GPTQ W4 checkpoint, but loader re-quantizes weights to FP8 e4m3 → CUTLASS FP8×FP8 blockwise GEMM + FP8 e4m3 per-token activations + (still) FP8 e5m2 KV

For each path we trace: **load → forward GEMM → KV path**, with file:line references, the actual MMA opcode, and register/smem footprint.

---

## 1. Path A — Baseline v18: GPTQ W4A16 + Marlin

### 1.1 Load time (no weight inflation)

[gptq.py](python/sglang/srt/layers/quantization/gptq.py) `GPTQMarlinLinearMethod.create_weights` — line ~545:

```python
qweight = PackedvLLMParameter(
    data=torch.empty(
        input_size_per_partition // self.quant_config.pack_factor,  # K // 8
        output_size_per_partition,                                    # N
        dtype=torch.int32,            # 8 INT4 values packed per int32
    ),
    ...
)
scales = ChannelQuantScaleParameter(
    data=torch.empty(
        scales_and_zp_size,           # K // group_size = K // 128
        output_size_per_partition,    # N
        dtype=params_dtype,           # torch.bfloat16
    ),
    ...
)
qzeros = PackedColumnParameter(
    data=torch.empty(
        scales_and_zp_size,            # K // 128
        output_size_per_partition // 8,
        dtype=torch.int32,             # 8 INT4 zero-points packed per int32
    ),
)
```

After load, `process_weights_after_loading` calls `gptq_marlin_repack` (line ~720) to permute the int32 packed layout into a Marlin tile-friendly layout — **the storage stays packed INT4**, only the byte layout changes.

**HBM footprint per layer** (example: K=11008, N=4096, group_size=128):

| Tensor | Shape | Dtype | Bytes |
|---|---|---|---|
| qweight | (K/8, N) | int32 | (1376 × 4096) × 4 = **~22.5 MB** |
| scales | (K/128, N) | bf16 | (86 × 4096) × 2 = **704 KB** |
| qzeros | (K/128, N/8) | int32 | (86 × 512) × 4 = **176 KB** |
| **Total HBM** | | | **~23.4 MB** |

### 1.2 What "fp16/bf16 group scales kept" means — and resource cost

"Group scales" = **per-128-element-group fp16/bf16 multipliers** stored alongside the INT4 weights. They are **persistent HBM tensors** (not registers), allocated once at load.

**Storage cost (per layer, example numbers above):** 704 KB. For the whole model (~24 std-attn linears + 32 MLP linears with various K/N): on the order of tens of MB total — negligible vs ~6 GB total weights.

**Register cost (during GEMM, transient):** During each Marlin K-loop iteration, every warp loads a **subset** of scales for the current group via `ldmatrix` or `cp.async` into shared memory, then broadcasts to thread registers (typically ~4–8 fp16/bf16 values per thread per K-tile). This is in the noise; Marlin's hot register usage is dominated by accumulators and weight fragments, not scales.

**Why scales are needed at all**: GPTQ INT4 only encodes 16 levels per group. The fp16/bf16 scale tells you what `[−scale × 8, scale × 7]` range those 16 levels span. Without the scale, the int4 values are meaningless. The scale is per-group rather than per-channel because per-group gives much better accuracy at the same compression ratio (this is GPTQ's contribution).

### 1.3 Forward — call chain

```
Linear.forward(x)               # x: (M, K) bf16, M = batch*seq
  → GPTQMarlinLinearMethod.apply(layer, x, bias)         # gptq.py L871
    → apply_gptq_marlin_linear(...)                      # marlin_utils.py L464
      → gptq_marlin_gemm(...)                            # sgl-kernel binding
        → CUDA kernel in sgl-kernel/csrc/gemm/marlin/    # actual GEMM
```

The kernel does **fused INT4→BF16 dequant inside the K-loop**:

```cuda
// sgl-kernel/csrc/gemm/marlin/* (simplified)
for (int ki = 0; ki < K; ki += 16) {
    // 1. Load packed INT4 from shared memory (came from HBM via cp.async)
    uint32_t w_packed = *(weight_smem + ki/8);

    // 2. Unpack 8 int4 values; subtract zero; multiply by bf16 scale
    //    Result lives in registers as bf16 fragments (8 per int32 packed word)
    bf16_w_frag = (int4_unpack(w_packed) - zero) * bf16_scale;

    // 3. Feed bf16 weight fragment + bf16 activation fragment to Tensor Cores
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ...");
    //                                ^^^^   ^^^^^^^^^   ^^^
    //                                accum   inputs A,B  initial
}
```

The **MMA opcode** is in [marlin_template.h](sgl-kernel/csrc/gemm/marlin/marlin_template.h) lines 57–68:
```cuda
asm volatile(
  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
  : "=f"(c[0..3])    // 4× FP32 accumulators per warp slot
  : "r"(a[0..3]),    // 4× uint32 (each holds 2× bf16) — A frag
    "r"(b[0..1]),    // 2× uint32 (each holds 2× bf16) — B frag
    "f"(c[0..3]));   // initial accumulator (FP32)
```

**Critical**: the FP32 accumulator is hardware-fixed for BF16 MMA — there is no BF16 accumulator option in PTX. The MMA always accumulates in FP32. This is why "epilogue" exists.

### 1.4 What "BF16 epilogue" means

After the K-loop ends, each warp holds its tile's accumulators as **FP32** (4 fp32 values per thread per output fragment). The epilogue is a small post-processing step that:

1. Reads FP32 accumulators from registers.
2. Optionally adds bias (broadcast bf16 → fp32 → fmadd).
3. Casts FP32 → BF16 via `__float2bfloat16` intrinsic (one PTX `cvt.rn.bf16.f32` per element).
4. Writes BF16 result to global memory.

So yes, "BF16 epilogue" = **the FP32 accumulator gets cast to BF16 before write-back**. There's no avoiding the FP32→BF16 step on Tensor Cores; the question is only whether it happens fused in the kernel (it does in Marlin) or as a separate kernel (it doesn't).

### 1.5 KV cache (FP8 e5m2)

KV is allocated as `torch.float8_e5m2` (1 byte/value) — see [model_runner.py](python/sglang/srt/model_executor/model_runner.py) line ~1541. Inside the FlashAttention backend ([minicpm_backend.py](python/sglang/srt/layers/attention/minicpm_backend.py) line ~834):

```python
if self.kv_cache_dtype_str.startswith("fp8"):
    if query_layer.dtype != torch.bfloat16:
        query_layer = query_layer.to(torch.bfloat16)  # implicit dequant FP8 → BF16
    # the actual attention matmul runs on BF16
```

So **the attention matmul (Q·K^T then softmax-times-V) uses BF16 tensor cores**, even though K/V are stored as FP8 in HBM. The FP8 storage is purely a memory/bandwidth optimization; the math runs in BF16.

---

## 2. Path B — W4A8 #1 mislabel (W8A8 FP8 blockwise)

### 2.1 Load time — why it does INT4 → BF16 → FP8 instead of keeping INT4

[utils_w4a8_fp8.py](python/sglang/srt/layers/quantization/utils_w4a8_fp8.py) `gptq_int4_dequantize` and `fp8_blockwise_quantize`, called from `_soar_maybe_setup_w4a8_fp8` in [gptq.py](python/sglang/srt/layers/quantization/gptq.py) line ~796:

```python
# 1. Unpack INT4 → INT32
q_unpacked = (qweight.unsqueeze(1) >> shifts.view(1, -1, 1)) & 0xF  # (K, N) int32
# 2. Dequant: (q - zero) * scale → BF16 dense
w = (q_unpacked - zeros_full).to(scales.dtype) * scales_full  # (K, N) bf16
# 3. Per-block FP8 quant (128×128 blocks)
block_amax = w_blocked.abs().amax(dim=(1,3))                  # (N/128, K/128) fp32
weight_fp8_scale = (block_amax / 448.0).to(torch.float32)
w_scaled = (w_f32 / scale_full).clamp(-448, 448)
weight_fp8 = w_scaled.to(torch.float8_e4m3fn)                 # (N, K) FP8 e4m3
```

**Why it does this** (and not keep INT4): the runtime GEMM target is `cutlass_w8a8_block_fp8_linear` from sgl-kernel, which **expects both inputs to already be FP8**. There is no SM120 dense kernel that takes INT4 weight + FP8 activation as inputs (that's exactly the kernel that would need to be built — see PROPOSAL_W4A8_REAL_001 / Option A). So the loader has to **pre-convert** the INT4 weights to a format the existing FP8 kernel can consume.

This is the **mislabel root cause**: we used a W4 source checkpoint, dequantized it, and re-quantized to FP8 storage just to feed an FP8 GEMM. The kernel never sees the original INT4 representation, so the W4 packing advantage is destroyed at load time.

**HBM footprint after conversion** (same K, N as Path A):

| Tensor | Shape | Dtype | Bytes |
|---|---|---|---|
| weight_fp8 | (N, K) | float8_e4m3fn | 4096 × 11008 = **~44 MB** |
| weight_fp8_scale | (N/128, K/128) | float32 | 32 × 86 × 4 = **~11 KB** |
| **Total HBM** | | | **~44 MB** |

Compare to Path A: **44 MB vs 23.4 MB. Weight HBM nearly doubled.** This is the bandwidth penalty that caused the regression.

### 2.2 Forward — call chain

```
Linear.forward(x)
  → GPTQMarlinLinearMethod.apply(layer, x, bias)             # gptq.py L871
    if layer._soar_w4a8_active:                              # gated branch
      → cutlass_w8a8_block_fp8_linear_with_fallback(...)     # fp8_utils.py L343
        → per_token_group_quant_fp8(input_2d, 128, ...)      # quantize activations
        → fp8_blockwise_scaled_mm(q_input, weight.T, x_scale, weight_scale.T, ...)
          → CUDA kernel (CUTLASS FP8 GEMM)
```

The two distinct steps in forward:

**A. Activation quantization** — Triton kernel, runs **every forward**, [fp8_kernel.py](python/sglang/srt/layers/quantization/fp8_kernel.py) line ~464:
```python
sgl_per_token_group_quant_fp8(
    x,                # (M, K) bf16 input
    x_q,              # (M, K) FP8 e4m3 output (allocated)
    x_s,              # (M, K/128) fp32 scales (allocated)
    group_size=128, eps=1e-10, fp8_min=-224, fp8_max=224,
)
```
For each row, for each 128-element group: compute amax, derive fp32 scale, divide, clamp, cast to FP8. This is a **separate kernel launch**, with its own load/store of the activation tensor.

**B. FP8×FP8 GEMM** — CUTLASS, with both A (activations) and B (weights) already in FP8:
```cuda
asm volatile(
  "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 ..."
  //                                ^^^^^^^^^^^^^^^
  //                                FP8 inputs, FP32 accumulator
);
```
FP32 accumulator → BF16 cast in epilogue (same FP32→BF16 step as Path A; the FP8 cores still accumulate in FP32).

### 2.3 The user's exact question — Marlin INT4→BF16 dequant vs Path B's BF16→FP8 quant: which is more efficient?

These are **not the same kind of operation**, even though both involve type conversion:

| | Marlin INT4 → BF16 (Path A) | Path B activation BF16 → FP8 |
|---|---|---|
| When | **Inside** the GEMM K-loop | **Before** the GEMM, separate kernel launch |
| Operand | Weights (constant per inference) | Activations (different every token) |
| Operation | unpack int4 + sub zero + multiply bf16 scale | reduce-max over 128 elems + divide + clamp + cast |
| Done in | Registers, same SMs as MMA, fully fused | Separate Triton kernel = its own kernel-launch + HBM I/O |
| Per-element cost | ~3 simple int/fp ops, no global reduce | reduce + divide + clamp = ~5 ops + a reduction across 128 elements |
| HBM traffic | Reads INT4 (already needed for GEMM) | Reads BF16 input (M·K·2 bytes), writes FP8 (M·K·1 byte) + scales (M·K/128·4 bytes) |
| Latency hidden by | Overlapped with MMA in same warp | NOT overlapped — runs as separate kernel before GEMM |

**Verdict: Path A's dequant is much cheaper.** It rides for free on the K-loop's existing memory traffic and is overlapped with MMA. Path B's activation quantization is a **separate kernel** that pays its own HBM round-trip (reads BF16 input, writes FP8 + scales) and adds its own kernel-launch latency. At decode bs=1 with M=1, the M·K activation quant is small but still non-zero overhead; at prefill it's a substantial extra kernel pass over the activation tensor.

This is on top of the **2× weight HBM bandwidth** issue (Path B's FP8 weight is 44 MB vs Marlin's 23 MB). At decode (bandwidth-bound), the 2× weight bandwidth is the dominant penalty.

### 2.4 Your question — "FP8×FP8 to BF16, no additional FP32→BF16, right?"

**No, the FP32→BF16 cast still happens** in Path B's epilogue. The hardware FP8 MMA always produces an FP32 accumulator; you cannot get BF16 directly from `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`. The `f32` after `row.col` is the **accumulator dtype**, which is non-negotiable. So the epilogue still does FP32 → BF16 (`cvt.rn.bf16.f32`) before writing to global memory, exactly as in Path A.

The output of the GEMM itself is FP32 in registers; only the **stored** result is BF16. No path on either Tensor Core (BF16 MMA or FP8 MMA) skips the FP32 accumulator.

### 2.5 Your question — "for KV, no need to convert FP8 to BF16 inside FlashAttention since we do FP8×FP8 GEMM, right?"

**No.** The KV cache and the linear-layer GEMM are **separate operations on separate tensors**:

- The **W8A8 FP8 GEMM** is the linear projection (e.g. q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj). Its inputs are `linear_input × linear_weight`. FP8×FP8 here.
- The **attention** is `softmax(Q · K^T) · V`. Q, K, V are activations *output by* the q/k/v projections. The K and V tensors get **stored** in the KV cache. The next-token Q gets matmul'd against the stored K's, then softmax, then matmul'd against V's.
- The **attention kernel** (FlashAttention backend) is a different kernel from the linear-layer GEMM. It does its own MMA. In our build, the FlashAttention path runs **BF16** Q/K/V matmul.
- So when KV is stored as FP8 e5m2 in HBM (memory savings), the attention kernel still has to **dequant FP8 → BF16** before its own BF16 matmul. There is no end-to-end FP8 path through attention here.

Path B kept `--kv-cache-dtype fp8_e5m2` unchanged. The W4A8 #1 implementation only touched linear layers, not the attention kernel. So FP8→BF16 dequant inside FlashAttention happens in **both** Path A and Path B exactly the same way.

If you wanted attention to actually run in FP8 cores, you'd need an **FP8 FlashAttention** kernel (e.g. cuDNN's flash-attention-3 FP8 path on Hopper, or a Blackwell variant). We don't currently use one. NVFP4 KV cache doesn't fix this either — it's still storage-only, dequanted to BF16 for attention compute.

---

## 3. Side-by-side timeline at instruction granularity

For one linear-layer call with shape M × K → M × N (e.g. M=1 decode token, K=14336, N=4096):

| Step | Path A (Marlin) | Path B (W8A8 FP8) |
|---|---|---|
| 1. Read input | (M, K) bf16 from HBM (28 KB) | (M, K) bf16 from HBM (28 KB) |
| 2. Activation prep | none — pass-through | **separate kernel**: read bf16 (28 KB), reduce-max per 128-elem group, compute fp32 scale, divide+clamp, cast to FP8, write FP8 (14 KB) + scales (448 B) → **+~85 KB HBM** |
| 3. Read weights | Marlin packed INT4 (~22.5 MB) + bf16 scales (~700 KB) | FP8 (~44 MB) + fp32 scales (~11 KB) |
| 4. K-loop dequant | inline INT4→bf16 in regs (free, fused with cp.async) | none (weights already FP8) |
| 5. MMA | `m16n8k16.f32.bf16.bf16.f32` × ⌈K/16⌉ iters × tiles, peak 148 TF | `m16n8k32.f32.e4m3.e4m3.f32` × ⌈K/32⌉ iters × tiles, peak 281 TF |
| 6. Epilogue | FP32 accum → bf16 cast → write (M, N) bf16 (8 KB) | FP32 accum → bf16 cast → write (M, N) bf16 (8 KB) |
| Dominant cost @ M=1 | Reading 22.5 MB of weights from HBM | Reading **44 MB** of weights from HBM (~2× the bandwidth) |
| Dominant cost @ M=large | Compute (MMA throughput) | Compute (MMA throughput, 2× peak in theory) |

**Why Path B regresses at decode (M=1)**: bandwidth-bound regime, weights are 2× larger in HBM → ~2× longer wall-clock for the GEMM. The 2× MMA peak is irrelevant because the MMA isn't the bottleneck.

**Why Path B *might* win at large M**: compute-bound regime, FP8 MMA is 2× faster. But for SOAR, even Smax doesn't get into the regime where FP8's MMA win can overcome the 2× weight-bandwidth handicap, because GPTQ-Marlin's W4 weights are already very efficient and we're rarely fully compute-bound.

**Why real W4A8 FP8 (Option A in PROPOSAL_W4A8_REAL_001) would win**: it would keep weights as INT4 in HBM (no inflation) AND use FP8 MMA. That requires writing the dequant (INT4 → FP8) inside the K-loop, exactly like Marlin does for INT4 → BF16, but with FP8 encode at the end instead of just leaving the bf16 result. That's the kernel project — 3–4 weeks (see ANALYSIS_w4a8_fp8_kernel_feasibility).

---

## 4. Direct answers to your questions (recap)

**Q1: Why does W8A8 load do INT4→BF16→FP8 instead of keeping INT4 like baseline?**
Because the existing FP8 GEMM kernel (`cutlass_w8a8_block_fp8_linear`) requires both inputs to already be in FP8 format. There is no SM120 dense kernel that takes INT4 weights + FP8 activations directly. The loader does the conversion as a workaround, which destroys the W4 packing advantage. A real W4A8 kernel would do the int4→FP8 conversion inside the K-loop (preserving INT4 HBM storage).

**Q2: What does "fp16/bf16 group scales kept" mean? Does it use registers/memory?**
It means the GPTQ checkpoint's per-128-element-group bf16 scale tensors stay in HBM alongside the int4 weights. HBM cost: ~700 KB per layer (negligible). Register cost during GEMM: a few values per thread per K-tile (negligible). They're kept because INT4 quantization is meaningless without the per-group scale to map back to the original real-number range.

**Q3: What does "BF16 epilogue" mean — convert FP32 to BF16?**
Yes exactly. Tensor Core MMA always accumulates in FP32 (hardware constraint). The epilogue is the post-MMA step that casts FP32 → BF16 (and adds bias if any) before writing to HBM.

**Q4: Marlin INT4→BF16 dequant vs Path B's BF16→FP8 activation quant — which is more efficient?**
Marlin's, by a large margin. Marlin's dequant is **fused inside the GEMM K-loop**, runs in registers, overlaps with MMA, and rides for free on weight memory traffic that's already happening. Path B's activation quant is a **separate Triton kernel** with its own kernel-launch overhead, its own HBM round-trip on the activation tensor, and a per-group reduce-max + scale-compute step. Path B also has the additional 2× weight-bandwidth penalty at HBM level.

**Q5: FP8×FP8 → BF16, no additional FP32→BF16, right?**
Wrong — the FP32→BF16 cast still happens. The PTX MMA opcode is `mma.sync...f32.e4m3.e4m3.f32`, which means **FP32 accumulator with FP8 inputs**. Hardware does not produce BF16 directly. The "BF16 output" comes from the epilogue's FP32→BF16 cast, identical to the BF16 path.

**Q6: For KV, no need to convert FP8 to BF16 in attention since we do FP8×FP8, right?**
Wrong on two counts:
1. The "FP8×FP8" was the **linear layer** GEMM, not attention. The attention kernel is a separate kernel that does Q·K^T and softmax-times-V.
2. Our attention backend runs in **BF16**, not FP8, regardless of KV storage dtype. So FP8 KV is dequanted to BF16 inside FlashAttention. To skip that step you'd need an FP8 FlashAttention kernel — we don't have one.

---

## 5. Cross-references

- [ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md](ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md) — high-level cost analysis
- [PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md](PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md) — hardware ceiling measurement
- [PROPOSAL_W4A8_REAL_001.en.md](PROPOSAL_W4A8_REAL_001.en.md) — what real W4A8 kernel would look like
- [gptq.py](../../python/sglang/srt/layers/quantization/gptq.py) lines 520–860 — both load paths
- [utils_w4a8_fp8.py](../../python/sglang/srt/layers/quantization/utils_w4a8_fp8.py) — the dequant→requant code that caused the mislabel
- [marlin_template.h](../../sgl-kernel/csrc/gemm/marlin/marlin_template.h) lines 50–120 — Marlin BF16 MMA opcode
- [model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) line ~1541 — KV dtype config
- [minicpm_backend.py](../../python/sglang/srt/layers/attention/minicpm_backend.py) lines 830–900 — KV dequant before attention
