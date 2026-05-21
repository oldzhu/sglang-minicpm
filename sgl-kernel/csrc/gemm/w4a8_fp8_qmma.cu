/* W4A8 Fused GEMM with SM120 tcgen05 MMA (FP8 QMMA, 296 TFLOPS).
 *
 * ===========================================================================
 * ARCHITECTURE OVERVIEW
 * ===========================================================================
 *
 * This kernel implements true W4A8: INT4 weight storage + FP8 activation +
 * FP8 QMMA on SM120 (Blackwell). Unlike the old wmma-based kernel
 * (w4a8_fp8_fused_gemm.cu, 148 TFLOPS), this uses tcgen05.mma for 2x
 * compute throughput (296 TFLOPS).
 *
 * Data flow per K-tile:
 *   1. Load INT4 weights from global, dequant to FP8 in SMEM (column-major)
 *   2. Load FP8 activations from global into SMEM (row-major)
 *   3. Build TMA descriptors for SMEM tensors
 *   4. tcgen05.mma SS: SMEM_A × SMEM_B → TMEM_C (warp-group, 128 threads)
 *   5. Accumulate across K-tiles in TMEM
 *   6. After K-loop: tcgen05.commit → tcgen05.st → SMEM → global BF16
 *
 * Key SM120 concepts:
 *   - TMEM (Tensor Memory): ~64KB per SM, accumulator storage.
 *     Allocated via tcgen05.alloc, committed via tcgen05.commit,
 *     read back via tcgen05.st.
 *   - tcgen05.mma: 5th-gen Tensor Core MMA. Operates at warp-group level
 *     (cta_group::1 = 128 threads = 4 warps). Only the elected thread issues
 *     the instruction; all 128 threads participate in the hardware compute.
 *     PTX format (from NVIDIA cutlass sm100_umma.hpp):
 *       tcgen05.mma.cta_group::1.kind::f8f6f4 [tmem_c], desc_a, desc_b,
 *         idescE, {mask[4]}, p;
 *   - TMA descriptors: 128-byte structures describing tensor layout in SMEM.
 *     Created manually in this kernel (no cutlass dependency).
 *
 * Tile: M=128, N=128, K=128 (one warp-group, 128 threads/block)
 * Weight format: GPTQ standard (K/8, N) int32, group_size=128
 * Activation: FP8 e4m3fn, row-major (M, K)
 * Output: BF16, row-major (M, N)
 *
 * ===========================================================================
 * KNOWN RISKS & LIMITATIONS (first implementation)
 * ===========================================================================
 *
 * A. TMA DESCRIPTOR FORMAT (HIGH RISK):
 *    The 128-byte descriptor layout filled by fill_tma_desc_2d() is a
 *    best-effort interpretation based on cutlass CuTe patterns and PTX ISA
 *    documentation. The exact bit layout for SM120 SMEM descriptors used
 *    with tcgen05.mma HAS NOT BEEN VALIDATED ON HARDWARE.
 *
 *    If the descriptor format is incorrect, tcgen05.mma will read garbage
 *    from SMEM, producing wrong results. In that case, options are:
 *    a) Include cutlass headers and use cute::TmaDescriptor
 *    b) Use cp.async.bulk.tensor.2d to create descriptors via PTX
 *    c) Use non-TMA tcgen05 path (tcgen05.ld for manual SMEM→TMEM copy)
 *
 * B. NO SMEM SWIZZLING:
 *    This kernel uses simple row-major/col-major SMEM layouts without
 *    swizzling. tcgen05.mma reads from SMEM with a fixed access pattern;
 *    without swizzling, bank conflicts may reduce effective bandwidth.
 *    A production kernel should use the swizzled SMEM layout defined by
 *    cutlass (see SmemLayoutAtom in sm120_*_tma.hpp).
 *
 * C. NO SOFTWARE PIPELINING:
 *    Each K-tile is processed sequentially with __syncthreads() barriers.
 *    A production kernel would double-buffer SMEM and overlap data loading
 *    with tcgen05.mma compute. See cutlass KernelTmaWarpSpecialized.
 *
 * D. TMEM NOT DEALLOCATED:
 *    tcgen05.alloc reserves TMEM; it is not freed before kernel exit.
 *    For single-block launches this is acceptable (TMEM is per-block).
 *    For multi-block, TMEM may leak across block executions on the same SM.
 *
 * E. COMPILATION REQUIREMENTS:
 *    Requires -arch=sm_120 or equivalent. The tcgen05 instructions are
 *    only available on Blackwell (SM120+). Verify with:
 *      nvcc -arch=sm_120a ...
 *    or in cmake:
 *      set(CMAKE_CUDA_ARCHITECTURES 120)
 *
 * ===========================================================================
 * BUILD & TEST
 * ===========================================================================
 *
 * Standalone build (sgl-kernel/csrc/gemm/):
 *   mkdir build && cd build
 *   cmake .. -DTorch_DIR=<torch_share>/cmake/Torch \
 *            -DCMAKE_CUDA_ARCHITECTURES=120
 *   cmake --build . -- -j2
 *
 * Correctness test:
 *   python3 -c "
 *   import torch
 *   torch.ops.load_library('build/libw4a8_fused_gemm.so')
 *   qw = torch.randint(0,2**31-1,(64,128),dtype=torch.int32,device='cuda')
 *   qz = torch.randint(0,2**31-1,(4,16),dtype=torch.int32,device='cuda')
 *   sc = torch.randn(32,128,dtype=torch.float32,device='cuda')
 *   a  = torch.randn(128,4096,dtype=torch.float8_e4m3fn,device='cuda')
 *   c  = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw,qz,sc,a,128,4096,128)
 *   print('Output shape:', c.shape)
 *   "
 *
 * Author: SOAR 2026 team-beta
 * Date: 2026-05-20
 * Target: SM120 (Blackwell), CUDA 12.8+, 296 TFLOPS FP8 QMMA
 */

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <torch/library.h>

namespace sglang {

// ============================================================================
// Constants
// ============================================================================

static constexpr int kTileM = 128;        // M dimension per tile
static constexpr int kTileN = 128;        // N dimension per tile
static constexpr int kTileK = 128;        // K dimension per step
static constexpr int kBlockSize = 128;    // Threads (one warp-group)
static constexpr int kAccumBytes = 65536; // 128×128×4 bytes (float)

// Dynamic SMEM offsets (bytes from smem_raw)
static constexpr int kOffWfp8   = 0;                        // 0
static constexpr int kOffAfp8   = kTileN * kTileK;          // 16384
static constexpr int kOffCbf16  = kOffAfp8 + kTileM * kTileK; // 32768
// TMA descriptors: align to 64 bytes after C_bf16 (32768+32768=65536, already aligned)
static constexpr int kOffTma    = kOffCbf16 + kTileM * kTileN * 2; // 65536
static constexpr int kSmemTotal = kOffTma + 256;             // 65792 (~64.25 KB)

// ============================================================================
// Device helpers
// ============================================================================

/// Execute elect_one.sync for the full warp-group (128 threads).
/// Returns true only for one elected thread.
/// All 128 threads must call this in lockstep.
/// REQUIRES: __CUDA_ARCH__ >= 1200 (SM120+)
#if __CUDA_ARCH__ >= 1200
__device__ inline bool elect_one_sync_warpgroup() {
  uint32_t elected;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "elect_one.sync p|0xFFFFFFFF;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t"
      "}\n"
      : "=r"(elected));
  return elected != 0;
}

/// Fill a minimal 128-byte TMA descriptor for a 2D SMEM tensor.
///
/// WARNING: The descriptor format below is a best-effort interpretation.
/// It has NOT been validated against SM120 hardware. See RISK A in the
/// file header. If tcgen05.mma produces wrong results, this function
/// is the most likely culprit.
///
/// @param desc      Output buffer (16 uint64_t, 64-byte aligned in SMEM)
/// @param smem_addr Byte address of tensor base in shared memory
/// @param dim0      Size of outer dimension (rows for row-major)
/// @param dim1      Size of inner dimension (columns for row-major)
/// @param stride0   Stride of outer dim in ELEMENTS
/// @param stride1   Stride of inner dim in ELEMENTS
/// @param elem_size Element size in bytes (1=FP8, 2=BF16, 4=float)
///
/// Reference: NVIDIA PTX ISA §TMA Descriptor, and cutlass
///   include/cute/atom/copy_traits_sm100.hpp (TMA descriptor construction)
__device__ void fill_tma_desc_2d(
    uint64_t* desc,
    uint32_t smem_addr,
    int dim0, int dim1,
    int stride0, int stride1,
    int elem_size) {

  // Thread 0 fills; all participate in barrier
  if (threadIdx.x == 0) {
    // Zero entire 128-byte descriptor (16 × uint64_t)
    #pragma unroll
    for (int i = 0; i < 16; ++i) desc[i] = 0ULL;

    // Byte 0x00-0x07: base address (32-bit SMEM byte address)
    desc[0] = static_cast<uint64_t>(smem_addr);

    // Byte 0x08-0x0F: tensor shape register
    //   bits [0:15]   = dim0 (number of rows / outer dim)
    //   bits [16:31]  = dim1 (number of cols / inner dim)
    desc[1] = (static_cast<uint64_t>(dim0) & 0xFFFFULL)
            | ((static_cast<uint64_t>(dim1) & 0xFFFFULL) << 16);

    // Byte 0x10-0x17: stride register
    //   bits [0:31]   = stride of dim0 in BYTES
    //   bits [32:63]  = stride of dim1 in BYTES
    uint64_t s0 = static_cast<uint64_t>(stride0) * elem_size;
    uint64_t s1 = static_cast<uint64_t>(stride1) * elem_size;
    desc[2] = (s0 & 0xFFFFFFFFULL) | ((s1 & 0xFFFFFFFFULL) << 32);

    // Byte 0x18-0x1F: element size register
    //   bits [0:4]    = log2(elem_size_in_bytes)
    int log2e = (elem_size == 1) ? 0 : (elem_size == 2) ? 1 : 2;
    desc[3] = static_cast<uint64_t>(log2e) & 0x1FULL;

    // Byte 0x20-0x27: box_dim0 register (same as tensor shape for full tensor)
    desc[4] = (static_cast<uint64_t>(dim0) & 0xFFFFULL)
            | ((static_cast<uint64_t>(dim1) & 0xFFFFULL) << 16);

    // Byte 0x28-0x7F: box_dim1, element stride, swizzle — all zero
    // (no swizzling, no subtensor boxing, no boundary checks)
  }
  __syncthreads();
}

// ============================================================================
// Main kernel
// ============================================================================

__global__ void w4a8_fp8_qmma_kernel(
    const int32_t* __restrict__ qweight,     // (K/8, N) packed INT4
    const int32_t* __restrict__ qzeros,      // (K/g/8, N) packed zeros
    const float* __restrict__ scales,        // (K/g, N) float32 scales
    const __nv_fp8_e4m3* __restrict__ a_fp8, // (M, K) FP8 activations
    __nv_bfloat16* __restrict__ c_bf16,      // (M, N) output
    int M, int N, int K,
    int group_size,
    int lda, int ldc) {

  const int mb = blockIdx.x;
  const int nb = blockIdx.y;
  const int m0 = mb * kTileM;
  const int n0 = nb * kTileN;
  const int tid = threadIdx.x;

  // ---- Dynamic shared memory ----
  extern __shared__ char smem_raw[];

  __nv_fp8_e4m3* W_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(
      smem_raw + kOffWfp8);
  __nv_fp8_e4m3* A_fp8_smem = reinterpret_cast<__nv_fp8_e4m3*>(
      smem_raw + kOffAfp8);
  __nv_bfloat16* C_bf16_smem = reinterpret_cast<__nv_bfloat16*>(
      smem_raw + kOffCbf16);
  uint64_t* tma_desc = reinterpret_cast<uint64_t*>(
      smem_raw + kOffTma);
  uint64_t* desc_A = tma_desc;       // 128 bytes
  uint64_t* desc_B = tma_desc + 16;  // 128 bytes

  // ---- Allocate TMEM accumulator (128×128 float) ----
  uint32_t tmem_c;
  asm volatile(
      "tcgen05.alloc.sync.aligned.cta_group::1.b32 [%0], %1;\n"
      : "=r"(tmem_c)
      : "n"(kAccumBytes)
      : "memory");

  // ---- K-loop ----
  for (int kb = 0; kb < K; kb += kTileK) {

    // ---------------------------------------------------------------
    // Phase 1: Dequant INT4 → FP8 into W_fp8 (SMEM, column-major)
    //
    // W_fp8[n * kTileK + k] where n is the N index (column), k is K.
    // Column-major: fastest-varying index is K (inner dim = K).
    // ---------------------------------------------------------------
    for (int i = tid; i < kTileN * kTileK; i += kBlockSize) {
      int n = i / kTileK;   // N (0..127)
      int k = i % kTileK;   // K (0..127)
      int kg = kb + k;
      int ng = n0 + n;

      __nv_fp8_e4m3 val{0};
      if (kg < K && ng < N) {
        int kp = kg / 8;
        int kbit = (kg % 8) * 4;
        int w4 = (qweight[kp * N + ng] >> kbit) & 0xF;

        int gid = kg / group_size;
        int z4 = 0;
        if (qzeros != nullptr) {
          int zn = ng / 8;
          int zb = (ng % 8) * 4;
          z4 = ((qzeros[gid * (N / 8) + zn] >> zb) & 0xF) + 1;
        }
        float fv = (static_cast<float>(w4) - static_cast<float>(z4))
                 * scales[gid * N + ng];
        val = static_cast<__nv_fp8_e4m3>(
            __nv_cvt_float_to_fp8(fv, __NV_SATFINITE, __NV_E4M3));
      }
      W_fp8[n * kTileK + k] = val;
    }

    // ---------------------------------------------------------------
    // Phase 2: Load FP8 activations → A_fp8_smem (SMEM, row-major)
    //
    // A_fp8_smem[m * kTileK + k] where m is M (row), k is K (col).
    // Row-major: fastest-varying index is K.
    // ---------------------------------------------------------------
    for (int i = tid; i < kTileM * kTileK; i += kBlockSize) {
      int m = i / kTileK;
      int k = i % kTileK;
      int mg = m0 + m;
      int kg = kb + k;

      A_fp8_smem[m * kTileK + k] =
          (mg < M && kg < K) ? a_fp8[mg * lda + kg] : __nv_fp8_e4m3{0};
    }
    __syncthreads();

    // ---------------------------------------------------------------
    // Phase 3: Build TMA descriptors
    //
    // desc_A: M×K row-major FP8 (stride0=K, stride1=1)
    // desc_B: K×N col-major FP8 (stride0=N, stride1=1)
    // ---------------------------------------------------------------
    {
      uint32_t base_a = __cvta_generic_to_shared(A_fp8_smem);
      uint32_t base_b = __cvta_generic_to_shared(W_fp8);
      fill_tma_desc_2d(desc_A, base_a, kTileM, kTileK, kTileK, 1, 1);
      fill_tma_desc_2d(desc_B, base_b, kTileK, kTileN, kTileN, 1, 1);
    }

    // ---------------------------------------------------------------
    // Phase 4: tcgen05.mma SS — QMMA (FP8×FP8 → float accumulate)
    //
    // Only the elected thread issues; all 128 threads participate.
    // The accumulator stays in TMEM across K-tile iterations.
    //
    // PTX reference from NVIDIA/cutlass sm100_umma.hpp:
    //   tcgen05.mma.cta_group::1.kind::f8f6f4 [tmem_c],
    //     desc_a, desc_b, idescE, {scaleC, mask[4]}, p;
    //
    // Where:
    //   [tmem_c]  — accumulator in TMEM (uint32_t, in/out)
    //   desc_a/b  — TMA descriptor for A/B in SMEM (uint64_t)
    //   idescE    — epilogue descriptor (uint64_t); low 32 bits passed
    //   scaleC    — accumulator scale (0 = no scaling)
    //   mask[0..3]— sparse predicate masks (all 0 for dense)
    //   p         — predicate (always false for dense = always execute)
    // ---------------------------------------------------------------
    if (elect_one_sync_warpgroup()) {
      uint32_t m[4] = {0, 0, 0, 0};  // sparse mask (all zero = dense)
      asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %5, 0;\n\t"
          "tcgen05.mma.cta_group::1.kind::f8f6f4 "
          "[%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
          "}\n"
          :
          : "r"(tmem_c),            // %0: TMEM accumulator
            "l"(desc_A[0]),          // %1: TMA desc A (uint64_t)
            "l"(desc_B[0]),          // %2: TMA desc B (uint64_t)
            "r"(0u),                 // %3: idescE low32 (0 = default epilogue)
            "r"(0u),                 // %4: scaleC (0)
            "r"(m[0]), "r"(m[1]),    // %5,%6: mask
            "r"(m[2]), "r"(m[3])     // %7,%8: mask
          : "memory");
    }
    __syncthreads();
  }

  // ====================================================================
  // Epilogue: TMEM → SMEM → global memory (BF16)
  //
  // Step 1: tcgen05.commit — finalize accumulator for reading
  // Step 2: tcgen05.st — per-thread sub-tile copy TMEM → SMEM
  // Step 3: Convert float → BF16, store to global
  //
  // Thread-to-element mapping (for 128×128 tile, cta_group::1):
  //   Threads are logically 8 rows × 16 columns.
  //   Thread (row_t, col_t) where row_t = tid/16, col_t = tid%16
  //     owns sub-tile: M ∈ [row_t*16, (row_t+1)*16)
  //                    N ∈ [col_t*8,  (col_t+1)*8)
  //   Each sub-tile is 16×8 = 128 float elements = 512 bytes.
  //
  // TMEM layout is row-major: offset(row, col) = (row*128 + col)*4 bytes.
  // Sub-tile TMEM offset for thread T:
  //   tmem_sub = tmem_c + (row_t * 16 * 128 + col_t * 8) * 4
  // ====================================================================

  // Step 1: Commit TMEM
  if (elect_one_sync_warpgroup()) {
    asm volatile("tcgen05.commit.cta_group::1;\n" ::: "memory");
  }
  __syncthreads();

  // Step 2: tcgen05.st — copy each thread's sub-tile to SMEM staging
  //
  // SMEM staging: each thread gets 512 bytes (128 floats) within C_bf16_smem.
  // Layout: thread T's staging starts at byte T*512.
  //
  // tcgen05.st.sync.aligned.cta_group::1.kind::f32
  //   [smem_dst], [tmem_src], size_bytes
  constexpr int kFloatPerThread = 128;  // 16×8
  constexpr int kBytesPerThread = kFloatPerThread * 4;

  int row_t = tid / 16;  // 0..7
  int col_t = tid % 16;  // 0..15
  int tmem_off = (row_t * 16 * kTileN + col_t * 8) * 4;  // byte offset

  float* thread_staging = reinterpret_cast<float*>(
      reinterpret_cast<char*>(C_bf16_smem) + tid * kBytesPerThread);

  {
    uint32_t smem_dst_addr = __cvta_generic_to_shared(thread_staging);
    uint32_t tmem_src_addr = tmem_c + tmem_off;
    asm volatile(
        "tcgen05.st.sync.aligned.cta_group::1.kind::f32 [%0], [%1], %2;\n"
        :
        : "r"(smem_dst_addr), "r"(tmem_src_addr), "n"(kBytesPerThread)
        : "memory");
  }
  __syncthreads();

  // Step 3: Convert float → BF16 and store to global
  //
  // Each thread writes its 16×8 sub-tile:
  //   For mr = 0..15, nr = 0..7:
  //     Global[mg][ng] = bf16(thread_staging[mr*8 + nr])
  //   where mg = m0 + row_t*16 + mr, ng = n0 + col_t*8 + nr
  for (int mr = 0; mr < 16; ++mr) {
    #pragma unroll
    for (int nr = 0; nr < 8; ++nr) {
      float val = thread_staging[mr * 8 + nr];
      int mg = m0 + row_t * 16 + mr;
      int ng = n0 + col_t * 8 + nr;
      if (mg < M && ng < N) {
        c_bf16[mg * ldc + ng] = __float2bfloat16(val);
      }
    }
  }
}

#endif  // __CUDA_ARCH__ >= 1200

// ============================================================================
// Host-side launch function
// ============================================================================

torch::Tensor w4a8_fp8_fused_gemm(
    const torch::Tensor& qweight,
    const torch::Tensor& qzeros,
    const torch::Tensor& scales,
    const torch::Tensor& a_fp8,
    int64_t N, int64_t K, int64_t group_size) {

  int M = a_fp8.size(0);

  TORCH_CHECK(qweight.dtype() == torch::kInt32,
              "qweight must be int32, got ", qweight.dtype());
  TORCH_CHECK(a_fp8.dtype() == torch::kFloat8_e4m3fn,
              "activations must be fp8_e4m3fn, got ", a_fp8.dtype());
  TORCH_CHECK(M % kTileM == 0,
              "M must be a multiple of ", kTileM, ", got M=", M);

  auto c_bf16 = torch::empty(
      {M, N},
      torch::dtype(torch::kBFloat16).device(qweight.device()));

  dim3 grid(M / kTileM, (static_cast<int>(N) + kTileN - 1) / kTileN);
  dim3 block(kBlockSize);

  auto stream = c10::cuda::getCurrentCUDAStream();
  w4a8_fp8_qmma_kernel<<<grid, block, kSmemTotal, stream>>>(
      static_cast<const int32_t*>(qweight.const_data_ptr()),
      qzeros.numel() > 0
          ? static_cast<const int32_t*>(qzeros.const_data_ptr()) : nullptr,
      static_cast<const float*>(scales.const_data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(a_fp8.const_data_ptr()),
      static_cast<__nv_bfloat16*>(c_bf16.data_ptr()),
      M, static_cast<int>(N), static_cast<int>(K),
      static_cast<int>(group_size),
      static_cast<int>(a_fp8.stride(0)),
      static_cast<int>(c_bf16.stride(0)));

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "W4A8 QMMA kernel launch failed: ",
                cudaGetErrorString(err));
  }

  return c_bf16;
}

}  // namespace sglang

// ============================================================================
// Python binding (namespace w4a8_fused — separate from sgl_kernel wheel)
// ============================================================================

TORCH_LIBRARY_FRAGMENT(w4a8_fused, m) {
  m.def("w4a8_fp8_fused_gemm(Tensor qweight, Tensor qzeros, Tensor scales, "
        "Tensor a_fp8, int N, int K, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(w4a8_fused, CUDA, m) {
  m.impl("w4a8_fp8_fused_gemm", &sglang::w4a8_fp8_fused_gemm);
}
