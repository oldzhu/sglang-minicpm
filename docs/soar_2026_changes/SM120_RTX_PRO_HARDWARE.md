# SM120 RTX PRO 6000 Hardware Reference

> Competition GPU for SOAR 2026. All optimization and kernel decisions should be made with reference to these specs.

## Hardware Specifications

| Feature | Value |
|---------|-------|
| Architecture | SM120 (Blackwell) |
| BF16/FP16 Tensor Core | **148 TFLOPS** |
| FP8 Tensor Core | **296 TFLOPS** |
| FP4 Tensor Core | **593 TFLOPS** |
| GPU Memory | **84 GB GDDR7** |
| Memory Bandwidth | **1398 GB/s** |
| L2 Cache | **112 MB** |
| SM Count | 96 SMs |

Note: Verified mapping from the official event hardware table is FP8=296 TFLOPS and FP4=593 TFLOPS.

## Key CUDA Programming Differences vs Prior Generations

### vs Hopper (SM90)
- **Optimization strategy is basically similar to Hopper**
- **MMA level**: Warp-level (NOT warpgroup-level like Hopper)
  - Hopper: `wgmma` (warpgroup MMA, 128-thread tiles)
  - SM120: `mma.sync` at warp level (32 threads per warp)
- **TMA**: Tensor Memory Accelerator supported (async bulk load/store)
- **QMMA**: Supported — `mxfp8` block-scaled quantized MMA

### vs Ampere (SM80)
- Marlin GPTQ currently uses SM80's `mma.sync.aligned.m16n8k16` — does NOT leverage SM120 advantages
- SM120 has ~2× FP8 vs BF16 throughput, but only accessible through SM120-native kernels

## CUDA Compilation

- Compile with `-arch=sm_120` or check capability at runtime
- PyTorch/CUDA must support sm120 (CUDA 12.8+ recommended)
- sgl-kernel: build with `CMAKE_CUDA_ARCHITECTURES=120` (default in our CMakeLists)

## Optimization Opportunities (SM120-Specific)

| Technique | Throughput Gain | Notes |
|-----------|----------------|-------|
| FP8 GEMM (QMMA/mxfp8) | Up to 2× vs BF16 | 296 vs 148 TFLOPS |
| FP4 GEMM | Up to 4× vs BF16 | 593 TFLOPS; accuracy risk |
| TMA (async loads) | Latency hiding | Better pipelining vs shared-mem staging |
| SM120 warp-level MMA | Replace SM80 path | Rewrite Marlin kernel to use SM120 MMA |
| CUTLASS 3.x SM120 kernels | High | NVIDIA's optimized path for Blackwell |
| TRT-LLM SM120 reference | High | Official reference for SM120 GEMM |

## TRT-LLM SM120 Reference

Competition organizers have confirmed TRT-LLM provides reference usage and integration guides for SM120. Key reference file:

```
https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/unittest/_torch/thop/parallel/test_fp8_block_scale_gemm.py
```
- Lines 133–171 show FP8 block-scale GEMM usage on SM120
- Search `sm120` in TRT-LLM repo for more examples

## Bandwidth Analysis (Memory-Bound vs Compute-Bound)

For decode (small M, M=1–8), GEMM is **memory-bandwidth limited**:
- Roofline crossover for W4 GPTQ: arithmetic intensity ≈ K/4 FLOPs/byte
- For K=4096: intensity ≈ 1024 FLOP/byte → need 1024 × 1398 GB/s = 1.43 PFLOPS to be compute-bound
- **148 TFLOPS << 1.43 PFLOPS** → decode is always memory-bound
- Implication: more SM utilization (more tiles) is better than fewer large tiles

For prefill (large M, M≥64), GEMM transitions toward compute-bound:
- SM120's FP8 or FP4 path can provide real speedup in this regime

## Relevant Files

- `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu` — current GPTQ Marlin GEMM (SM80 MMA)
- `sgl-kernel/CMakeLists.txt` — SM120 compile target configuration
- `docs/soar_2026_changes/CHANGE_0125_sm120_marlin_tiles_001.en.md` — investigation: why SM80 Marlin tile changes don't help on SM120
- `docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` — full optimization roadmap

## Investigation Findings (CHANGE_0125)

Adding SM120 tile instantiations to Marlin GPTQ produced **no speed improvement** because:
1. Marlin uses SM80 `mma.sync.aligned.m16n8k16` — not SM120's warp-level MMA
2. MiniCPM-SALA's N dimensions (1024–28672) favor narrow tiles (thread_n=64) over wide (thread_n=256)
3. The scorer correctly identifies narrow tiles maximize 96-SM utilization
4. FP8/FP4 throughput (296/593 TFLOPS) is inaccessible through SM80 instruction paths

**Conclusion**: To truly benefit from SM120, kernel-level rewrite using SM120-native MMA, TMA, or switch to CUTLASS/TRT-LLM SM120 kernels is required.
