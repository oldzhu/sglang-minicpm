# CHANGE_0120: Profiling Results & Kernel Breakdown Analysis

## 概述 / Overview

**Date**: 2026-04-20  
**Commit**: 08fd86023 (CHANGE_0120 config)  
**Config**: GPTQ + FP8 KV + dense + torch.compile(max-bs=8) + mixed-chunk + prefill-max-req=4 + sched-cons=0.8 + chunk=65536  
**fcloud Instance**: 223.167.85.181  
**Purpose**: Profile forward pass kernel breakdown to identify optimization targets

---

## Profiling Method

### Commands Used

**Step 1: Start profiling (stage-separated, 3 steps per stage)**
```python
# Via HTTP API on fcloud
requests.post("http://localhost:30000/start_profile", json={
    "output_dir": "/tmp/minicpm_profile",
    "num_steps": 3,
    "profile_by_stage": True,
    "activities": ["CPU", "GPU"],
    "with_stack": True,
    "record_shapes": True
})
```

**Step 2: Send inference requests**
- Short-context run: 5 MCQ samples (avg 127 tokens input, 8441 tokens output)
- Long-context run: 1 NIAH (30K tokens) + 1 QA (25K tokens), concurrency=1

**Step 3: Analyze traces**
```bash
python3 /tmp/analyze_profile.py /tmp/minicpm_profile
```

### Output Files
| File | Size | Context | Description |
|------|------|---------|-------------|
| `1776651754-TP-0-EXTEND.trace.json.gz` | 598K | Short (127 tok) | Short-context prefill |
| `1776651754-TP-0-DECODE.trace.json.gz` | 155K | Short (127 tok) | Short-context decode |
| `1776652067-TP-0-EXTEND.trace.json.gz` | 838K | Long (25K-30K tok) | **Long-context prefill** |
| `1776652067-TP-0-DECODE.trace.json.gz` | 153K | Long (25K-30K tok) | Long-context decode |

---

## Results: Long-Context Prefill (25K-30K tokens) — THE KEY TRACE

**Total GPU kernel time: 4989.6 ms** (3 prefill forward passes)

### Category Breakdown

| Category | Time (ms) | % of Total | Description |
|----------|----------|------------|-------------|
| **GEMM (Marlin/GPTQ)** | **4257.8** | **85.3%** | GPTQ W4A16 dequant+GEMM |
| **FLA/SimpleGLA** | **620.1** | **12.4%** | FLA chunk kernels + FlashInfer attention |
| Other | 87.6 | 1.8% | Type casts, sigmoid, indexing |
| RMSNorm | 23.4 | 0.5% | Fused QK-norm-RoPE |
| GEMM (other) | 0.5 | 0.0% | lm_head GEMV |
| Memory ops | 0.1 | 0.0% | DtoH memcpy |

### Top Kernels (Long-Context Prefill)

| Rank | % | Time (ms) | Count | Kernel | Category |
|------|---|----------|-------|--------|----------|
| 1 | 80.8% | 4029.7 | 2056x | `Marlin<bf16, ..., 128, 4, 4, 8, false, ...>` | GPTQ GEMM (main) |
| 2 | 8.8% | 439.8 | 8x | `BatchPrefillWithRaggedKVCacheKernel<MaskMode=1, 128, 2, 2, 8>` | FlashInfer attention (8 std layers) |
| 3 | 3.2% | 161.9 | 32x | `cutlass_80_tensorop_bf16_s16816gemm_relu_256x128_32x3` | Gate GEMM (MLP gate_proj×up_proj) |
| 4 | 1.5% | 72.5 | 32x | `act_and_mul_kernel<bf16, silu>` | SiLU activation |
| 5 | 1.3% | 66.2 | 248x | `Marlin<bf16, ..., 128, 4, 4, 8, false, ...>` (variant) | GPTQ GEMM (8-bit layers) |
| 6 | 1.0% | 49.3 | 64x | `FusedAddRMSNormKernel<8, bf16>` | Fused residual+norm |
| 7 | 0.7% | 33.9 | 80x | `direct_copy_kernel_cuda` | Type cast (bf16→fp8, etc.) |
| 8 | 0.6% | 28.0 | 24x | `chunk_fwd_kernel_o` | FLA chunk output kernel |
| 9 | 0.5% | 23.4 | 24x | `fusedQKNormRopeKernel<128>` | Fused QK-norm + RoPE |
| 10 | 0.5% | 22.5 | 65x | `AUnaryFunctor<bf16>` | Elementwise (residual_scale?) |
| 11 | 0.4% | 22.0 | 24x | `chunk_fwd_kernel_h` | FLA chunk hidden state kernel |

### Key Observations — Prefill

1. **GEMM dominates at 85.3%** — The Marlin GPTQ kernel is called 2056 times across 3 prefill steps (~685 GEMM calls per forward = 32 layers × ~7 GEMM per layer × ~3). Each call processes the entire sequence length.

2. **FLA/SimpleGLA is only 12.4%** — This is surprisingly low. The FLA chunk kernels (`chunk_fwd_kernel_o` at 28ms + `chunk_fwd_kernel_h` at 22ms = 50ms) are small relative to the total. FlashInfer attention (8 standard layers) at 440ms is the larger FLA component.

3. **FlashInfer attention (8 std layers) = 8.8%** — `BatchPrefillWithRaggedKVCacheKernel` called only 8 times (once per standard attention layer) but each takes ~55ms for 25-30K tokens.

4. **Gate GEMM (cutlass) = 3.2%** — This is the gate_proj×up_proj fused GEMM for the MLP, using cutlass rather than Marlin (likely because these are dense FP16 weights?).

5. **RMSNorm is negligible at 0.5%** — Already well-fused. K4 (fuse residual_scale into RMSNorm) would save <0.5%.

6. **Type casts at 0.7%** — 80 copy operations, likely bf16→fp8 for KV cache storage.

---

## Results: Decode (single token per step)

**Total GPU kernel time: 18.5 ms** (3 decode steps)

### Category Breakdown

| Category | Time (ms) | % of Total | Description |
|----------|----------|------------|-------------|
| **GEMM (Marlin/GPTQ)** | **11.7** | **63.5%** | GPTQ W4A16 dequant+GEMM |
| **Other (torch.compile fused)** | **5.4** | **29.4%** | Fused GEMM+activation, RMSNorm, state I/O |
| **FLA/SimpleGLA** | **0.9** | **4.9%** | fused_recurrent_fwd + FlashInfer paged KV |
| RMSNorm | 0.3 | 1.5% | fusedQKNormRopeKernel |
| Activation (SiLU) | 0.1 | 0.5% | triton fused SiLU |
| Memory ops | 0.0 | 0.1% | DtoD memcpy |

### Key Observations — Decode

1. **torch.compile fused kernels = 29.4%** — The triton kernels from `torch.compile` fuse GEMM+activation+RMSNorm together:
   - `triton_red_fused__to_copy_add_mean_mm_mul_pow_rsqrt_sigmoid_t_1`: 10.5% (72x, 1.9ms) — GEMM + RMSNorm + sigmoid (output gate)
   - `triton_red_fused_div_mm_permute_0`: 7.3% (3x, 1.3ms) — GEMM + permute (lm_head?)
   - `triton_red_fused_mm_mul_sigmoid_t_0`: 3.3% (23x, 0.6ms) — GEMM + mul + sigmoid

2. **GEMM still dominates decode at 63.5%** — Even for single-token decode, GEMM is the bottleneck.

3. **FLA/SimpleGLA is tiny at 4.9%** — `fused_recurrent_fwd_kernel` (72x, 0.2ms) is extremely fast for single-token. The state I/O (index_kernel + index_put_kernel) adds 3.0% via indexing operations.

4. **State scatter/gather = ~3%** — `index_elementwise_kernel` for index_kernel (75x, 0.3ms) + index_put_kernel (75x, 0.2ms). This is the state load/store that A1 (state contiguity) aims to optimize.

---

## Decision Analysis

### What the data tells us

**The overwhelming bottleneck is GEMM (Marlin/GPTQ) at 85% of prefill time.**

For the new competition dataset (68% inputs 32K-512K), prefill time will be even more dominant (proportionally more GEMM). This means:

1. **FLA kernel optimization (Path B) has LIMITED impact for prefill** — only 12.4% of prefill. Even a 50% improvement in FLA kernels would only give ~6% total prefill speedup.

2. **Marlin GEMM optimization (Path C) is THE highest-impact target** — 85.3% of prefill. Even a 10% improvement in Marlin GEMM would give 8.5% total prefill speedup.

3. **FP8 weight quantization (Path D) could be transformative** — Moving from GPTQ W4A16 Marlin to FP8 W8A8 would:
   - Use native FP8 tensor cores instead of dequant+FP16 GEMM
   - Potentially 2× GEMM throughput (FP8 tensor cores = 2× FP16)
   - But 2× weight size (8-bit vs 4-bit) = more memory bandwidth needed
   - Net effect depends on whether GEMM is compute-bound or memory-bound at these sequence lengths

4. **For decode, torch.compile is already doing great** — The fused triton kernels (GEMM+activation+RMSNorm) show torch.compile is effectively fusing operations during decode.

### Revised Priority Order

| Priority | Path | Expected Impact | Justification |
|----------|------|----------------|---------------|
| **1** | **Marlin GEMM profiling & tuning** | **5-15%** | 85.3% of prefill; check if SM120 auto-config is optimal |
| **2** | **FP8 weight quantization (W8A16)** | **10-30%** | Replace Marlin W4+dequant with native FP8 GEMM |
| **3** | **FlashInfer attention optimization** | **5-8%** | 8.8% of prefill (8 std layers); sequence-dependent |
| **4** | **FLA chunk kernel optimization** | **1-3%** | Only 1% of prefill (chunk_fwd_o + chunk_fwd_h) |
| **5** | **State contiguity (A1)** | **<1% prefill, ~3% decode** | State I/O is 3% of decode but negligible in prefill |
| **6** | **K4 fused RMSNorm+residual_scale** | **<0.5%** | RMSNorm already negligible at 0.5% of prefill |

### Why FLA optimization is deprioritized

The earlier optimization catalog assumed "24 SimpleGLA layers = 75% of forward pass time." **The profiling data disproves this.** During prefill:
- SimpleGLA layers contribute ~12.4% total (FLA chunk + fused_recurrent + state I/O)
- The remaining ~87.6% is GEMM + attention + other
- Each SimpleGLA layer's FLA kernel takes ~2ms per prefill step vs ~50ms per GEMM

The "75% of forward pass" estimate was for decode-dominated workloads. With the new long-context dataset being prefill-dominant, **GEMM is the true bottleneck.**

---

## Recommended Next Steps

1. **Profile Marlin GEMM occupancy** on Blackwell SM120:
   - Is the auto-config selecting optimal tile sizes?
   - Are we using the right Marlin kernel variant (4-thread, 8-thread groups)?
   - What's the SM utilization and memory bandwidth utilization?

2. **Investigate FP8 weight quantization (W8A16)**:
   - sglang supports `--quantization fp8` natively
   - Would use cutlass FP8 GEMM kernels → native Blackwell tensor cores
   - Need accuracy test before committing (but this is W8A16, much safer than W4A4 NVFP4)

3. **Submit v19 with current config** — even small config improvements matter

---

## 概述（中文）

### 分析结论

**前缀填充（Prefill）的瓶颈是GEMM（85.3%）**，而不是FLA/SimpleGLA内核（12.4%）。

- Marlin GPTQ GEMM在25K-30K token输入时占据85.3%的GPU时间
- FLA块内核（chunk_fwd_kernel_o + chunk_fwd_kernel_h）仅占1%
- FlashInfer标准注意力（8层）占8.8%

### 优先级调整

1. **Marlin GEMM调优**（影响最大，85.3%的时间）
2. **FP8权重量化**（用原生FP8张量核替代W4+反量化+FP16 GEMM）
3. **FlashInfer注意力优化**（8层，8.8%的时间）
4. **FLA块内核优化**（降级，仅1%）
5. **状态连续性A1**（降级，prefill中可忽略不计）
