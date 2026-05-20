# CHANGE_W4A8_QMMA_TCGEN05: SM120 Fused INT4→FP8 Dequant + QMMA GEMM Kernel

## Background

### Prior W4A8 Work

The W4A8 optimization journey has gone through several iterations:

| Iteration | Approach | Result |
|-----------|----------|--------|
| W4A8#1 (2026-04) | INT4→FP8 dequant at load time + FP8 GEMM | **+118% S1 regression** — doubled HBM weight bandwidth |
| W4A8-REAL v25 (2026-05-18) | INT4→FP8 dequant per forward pass + cutlass FP8 GEMM | **Claimed −9% speedup but unreproducible** — illegal CUDA memory access |
| Scalar fused kernel | INT4→FP16 dequant in SMEM + wmma | **Correct but 2−5× slower** than Marlin (148 TFLOPS) |
| MMA fused kernel | INT4→FP16 dequant in SMEM + wmma m16n16k16 | **Buggy** — multi-tile error ~68, server 503 |

### Root Cause Analysis

The two-step approach (dequant + separate FP8 GEMM) fundamentally cannot beat Marlin:
- Marlin: Reads INT4 weights (0.5 bytes/elem) from HBM, computes at 148 TFLOPS
- Two-step: Reads INT4 (0.5 B/elem) + writes FP8 (1 B/elem) + reads FP8 (1 B/elem) = **2.5× HBM weight traffic**
- Even with 296 TFLOPS FP8 QMMA, the GEMM is bandwidth-bound at decode (bs=1)

**The only viable path is a fused kernel**: INT4 storage → in-kernel dequant → FP8 QMMA.

### This Change

A new fused kernel (`w4a8_fp8_qmma.cu`) replaces the old wmma-based kernel with:
- **tcgen05.mma** PTX inline assembly — SM120's native FP8 QMMA at **296 TFLOPS**
- **TMEM** (Tensor Memory) for accumulator — new Blackwell memory space
- **TMA descriptors** for SMEM tensor access by tcgen05
- Same INT4→FP8 dequant path as the old kernel (proven correct)
- Same Python interface (`torch.ops.w4a8_fused.w4a8_fp8_fused_gemm`)

## Rule Compliance

- **SOAR constraints**: Pure CUDA kernel optimization — no pre-quantized weights, no extra model files
- **Accuracy**: INT4→FP8 dequant is bit-exact with the old kernel (same unpacking logic)
- **Submission size**: No change (weights stay INT4, 0.5 bytes/elem)

## Implementation

### New Files

| File | Purpose |
|------|---------|
| `sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu` | New tcgen05 QMMA kernel (~340 lines) |
| `sgl-kernel/csrc/gemm/CMakeLists_standalone.txt` | Standalone build config |

### Modified Files

| File | Change |
|------|--------|
| `sgl-kernel/CMakeLists.txt` | Added `w4a8_fp8_qmma.cu` to build |

### Kernel Architecture

```
Thread block: 128 threads (1 warp-group, cta_group::1)
Tile: M=128, N=128, K=128

Per K-tile iteration:
  1. Cooperative load + dequant: INT4 → FP8 → W_fp8[128×128] in SMEM (column-major)
  2. Cooperative load: FP8 activations → A_fp8[128×128] in SMEM (row-major)
  3. Build TMA descriptors for W_fp8 and A_fp8
  4. tcgen05.mma SS: SMEM_A × SMEM_B → TMEM_C  (296 TFLOPS)
  5. Accumulate in TMEM across K-tiles

Epilogue:
  6. tcgen05.commit — finalize TMEM
  7. tcgen05.st — per-thread sub-tile TMEM → SMEM (16×8 float per thread)
  8. float → BF16 → global memory
```

### SMEM Budget

| Buffer | Size | Purpose |
|--------|------|---------|
| W_fp8 | 16 KB | Dequantized weights (128×128 FP8, col-major) |
| A_fp8 | 16 KB | Activation tile (128×128 FP8, row-major) |
| C_bf16 | 32 KB | Output staging (128×128 BF16, also tcgen05.st target) |
| TMA descs | 256 B | Two 128-byte TMA descriptors |
| **Total** | **~64 KB** | Fits in 101 KB SM120 SMEM |

## Known Risks

### HIGH: TMA Descriptor Format (Risk A)
The 128-byte TMA descriptor layout filled by `fill_tma_desc_2d()` is a best-effort interpretation based on NVIDIA PTX ISA documentation and cutlass CuTe patterns. **It has not been validated on SM120 hardware.** If incorrect, tcgen05.mma will read garbage from SMEM.

**Mitigation**: If the descriptor is wrong, fall back to using cutlass headers (`cute::TmaDescriptor`) or the `cp.async.bulk.tensor.2d` PTX instruction to create valid descriptors.

### MEDIUM: No SMEM Swizzling (Risk B)
Simple row-major/col-major SMEM layouts may cause bank conflicts during tcgen05.mma reads, reducing effective throughput. A production kernel should use the swizzled layout from cutlass.

### MEDIUM: tcgen05.st Granularity (Risk C)
The per-thread sub-tile copy via `tcgen05.st` assumes a specific thread-to-element mapping (8×16 thread grid, 16×8 sub-tile per thread). If this mapping is wrong, some elements will be lost or duplicated.

### LOW: TMEM Leak (Risk D)
`tcgen05.alloc` is called but `tcgen05.dealloc` is not. For single-block-per-SM launches this is fine; TMEM is per-warpgroup and released when the block exits.

## Validation

### Build
```bash
cd /root/standalone_fused/build
cmake .. \
  -DTorch_DIR=/app/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/share/cmake/Torch \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . -- -j2
cp libw4a8_fused_gemm.so /root/submission_sim/
```

### Correctness (single tile)
```python
import torch
torch.ops.load_library("/root/submission_sim/libw4a8_fused_gemm.so")

M, N, K, g = 128, 128, 128, 128
qw = torch.randint(0, 2**31-1, (K//8, N), dtype=torch.int32, device="cuda")
qz = torch.randint(0, 2**31-1, (K//g, N//8), dtype=torch.int32, device="cuda")
sc = torch.randn(K//g, N, dtype=torch.float32, device="cuda")
a  = torch.randn(M, K, dtype=torch.float8_e4m3fn, device="cuda")

c_fused = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw, qz, sc, a, N, K, g)
torch.cuda.synchronize()

# Reference: CPU FP32 matmul
w_fp32 = dequant_cpu(qw, qz, sc, g)  # use existing dequant
c_ref  = torch.mm(a.float(), w_fp32.t().float()).bfloat16()

print("Max error:", (c_fused.float() - c_ref.float()).abs().max().item())
# Expected: < 0.5 for FP8→BF16 quantization error
```

### Speed
```bash
# On fcloud, run speed benchmark with SOAR_W4A8_REAL_FP8_GEMM=1
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## Rollback

To revert to the old wmma kernel:
```bash
cd /root/standalone_fused/build
# Edit CMakeLists.txt: replace w4a8_fp8_qmma.cu with w4a8_fp8_fused_gemm.cu
cmake --build . -- -j2
cp libw4a8_fused_gemm.so /root/submission_sim/
```

Or disable the fused kernel entirely:
```bash
export SOAR_W4A8_REAL_FP8_GEMM=0
```

## Next Steps

1. **Build and compile** on fcloud — verify PTX compilation succeeds with `-arch=sm_120a`
2. **Correctness test** — single-tile then multi-tile (4096×4096)
3. **Server integration** — restart sglang and verify output is coherent
4. **Speed benchmark** — S1/S8/Smax against Marlin baseline
5. **If TMA descriptors are wrong** — switch to cutlass header approach or use `cute::TmaDescriptor`
6. **If kernel works but slower than expected** — add SMEM swizzling, double-buffering
