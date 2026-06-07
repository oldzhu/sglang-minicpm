/* W4A8 Fused GEMM v3 — cp.async multi-stage pipeline + register-only dequant.
 *
 * Phase 2 from PROPOSAL_optimized_fused_w4a8_v2: the v1/v2 kernels stalled
 * because each K-tile serialized {global load -> __syncthreads -> MMA}, leaving
 * the FP8 tensor cores idle during HBM loads. v3 adds a 3-stage cp.async ring
 * buffer (Marlin's technique): it prefetches tile k+2's raw INT4 weights + FP8
 * activations + scales/zeros while the MMA computes on tile k, overlapping
 * memory latency with tensor-core work.
 *
 * Differences vs v2:
 *   - SMEM weight layout transposed to [K-group][N] so the kTileN columns are
 *     contiguous in BOTH global and shared memory -> enables 16-byte vectorized
 *     cp.async (v2's [N][K-group] layout has N-strided columns, not vectorizable).
 *   - Triple-buffered SMEM ring; cp.async + __pipeline_commit/__pipeline_wait_prior.
 *   - Requires full tiles (N % kTileN == 0, K % kTileK == 0) so cp.async can run
 *     without per-element predication. SALA GEMM shapes satisfy this.
 *
 * The MMA inner loop, register dequant, and epilogue are IDENTICAL to v2 (the
 * validated-correct math), only the weight-SMEM indexing changes ([g][n]).
 *
 * Tile: M=128, N=128, K=64. MMA: m16n8k32 FP8. Threads: 128 (4 warps).
 * SMEM: 3 stages x 12.3 KB = ~37 KB.
 */

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstring>

namespace sglang {

static constexpr int kTileM = 128;
static constexpr int kTileN = 128;
static constexpr int kTileK = 64;
static constexpr int kMmaM = 16;
static constexpr int kMmaN = 8;
static constexpr int kMmaK = 32;
static constexpr int kWarpSize = 32;
static constexpr int kWarps = 4;
static constexpr int kStages = 3;
static constexpr int kKGroupsPerTile = kTileK / 8;  // 8 int32 K-groups per tile column

// Per-stage SMEM byte layout: [A | Wq | Sc | Zq], each sub-array 16-byte aligned.
static constexpr int kA_bytes = kTileM * kTileK;                 // 8192 (FP8 = 1 byte)
static constexpr int kW_bytes = kTileN * kKGroupsPerTile * 4;    // 4096 (int32)
static constexpr int kS_bytes = kTileN * 2;                      // 256  (bf16)
static constexpr int kZ_bytes = (kTileN / 8) * 4;               // 64   (int32)
static constexpr int kStageBytes = kA_bytes + kW_bytes + kS_bytes + kZ_bytes;  // 12608
static constexpr int kSmemBytes = kStageBytes * kStages;          // ~37 KB

// Dequant 4 consecutive-K INT4 weights (one N column) into 4 packed FP8 bytes.
// Identical to v2.
__device__ __forceinline__ uint32_t dequant4_to_fp8(int32_t wq, int shift, int z4, float sc) {
  uint32_t packed = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int w4 = (wq >> (shift + j * 4)) & 0xF;
    float fv = ((float)w4 - (float)z4) * sc;
    __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(fv);
    unsigned char b;
    memcpy(&b, &f8, 1);
    packed |= (uint32_t)b << (j * 8);
  }
  return packed;
}

// Issue cp.async copies for one K-tile into the given stage buffer.
// All copies are 16 bytes; sources are 16-byte aligned for SALA shapes.
__device__ __forceinline__ void load_tile_async(
    char* stage_base, int kb, int gid, int kg0, int m0, int n0,
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_fp8_e4m3* __restrict__ a_fp8,
    int N, int lda, int tid) {

  __nv_fp8_e4m3* A_smem = reinterpret_cast<__nv_fp8_e4m3*>(stage_base);
  int32_t* Wq_smem = reinterpret_cast<int32_t*>(stage_base + kA_bytes);          // [g][n]
  __nv_bfloat16* Sc_smem = reinterpret_cast<__nv_bfloat16*>(stage_base + kA_bytes + kW_bytes);
  int32_t* Zq_smem = reinterpret_cast<int32_t*>(stage_base + kA_bytes + kW_bytes + kS_bytes);

  // Activations [kTileM][kTileK] FP8: 512 chunks of 16 bytes, 4 per thread.
  // chunk c -> row m = c/4, kk = (c%4)*16.
#pragma unroll
  for (int c = tid; c < kTileM * kTileK / 16; c += kWarps * kWarpSize) {
    int m = c >> 2;
    int kk = (c & 3) * 16;
    const void* src = a_fp8 + (size_t)(m0 + m) * lda + kb + kk;
    void* dst = A_smem + m * kTileK + kk;
    __pipeline_memcpy_async(dst, src, 16);
  }

  // Weights [kTileN][kKGroupsPerTile] stored transposed as [g][n]: 256 chunks
  // (4 int32 = 4 columns each). chunk c -> g = c/32, col4 = (c%32)*4.
#pragma unroll
  for (int c = tid; c < kTileN * kKGroupsPerTile / 4; c += kWarps * kWarpSize) {
    int g = c >> 5;            // c / 32
    int col4 = (c & 31) * 4;   // (c % 32) * 4
    const void* src = qweight + (size_t)(kg0 + g) * N + n0 + col4;
    void* dst = Wq_smem + g * kTileN + col4;
    __pipeline_memcpy_async(dst, src, 16);
  }

  // Scales [kTileN] bf16: 16 chunks of 16 bytes (8 bf16 each).
  for (int c = tid; c < kTileN / 8; c += kWarps * kWarpSize) {
    int n8 = c * 8;
    const void* src = scales + (size_t)gid * N + n0 + n8;
    void* dst = Sc_smem + n8;
    __pipeline_memcpy_async(dst, src, 16);
  }

  // Zeros [kTileN/8] int32: 4 chunks of 16 bytes (4 int32 each). Optional.
  if (qzeros != nullptr) {
    int n_zwords = N / 8;
    for (int c = tid; c < (kTileN / 8) / 4; c += kWarps * kWarpSize) {
      int z4 = c * 4;
      const void* src = qzeros + (size_t)gid * n_zwords + n0 / 8 + z4;
      void* dst = Zq_smem + z4;
      __pipeline_memcpy_async(dst, src, 16);
    }
  }
}

__global__ void w4a8_fp8_qmma_v3_kernel(
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_fp8_e4m3* __restrict__ a_fp8,
    __nv_bfloat16* __restrict__ c_bf16,
    int M, int N, int K, int group_size, int lda, int ldc) {

  const int mb = blockIdx.x, nb = blockIdx.y;
  const int m0 = mb * kTileM, n0 = nb * kTileN;
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize, lane_id = tid % kWarpSize;
  const int warp_m0 = warp_id * (kTileM / kWarps);
  const bool has_zeros = (qzeros != nullptr);

  extern __shared__ char smem_raw[];

  float c_regs[2][16][4];
  for (int ms = 0; ms < 2; ++ms)
    for (int ns = 0; ns < 16; ++ns)
#pragma unroll
      for (int r = 0; r < 4; ++r) c_regs[ms][ns][r] = 0.0f;

  const int num_ktiles = K / kTileK;

  auto stage_ptr = [&](int s) -> char* { return smem_raw + s * kStageBytes; };
  auto tile_gid = [&](int kt) { return (kt * kTileK) / group_size; };
  auto tile_kg0 = [&](int kt) { return (kt * kTileK) / 8; };

  // --- Prologue: issue loads for the first kStages-1 tiles ---
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < num_ktiles) {
      load_tile_async(stage_ptr(s), s * kTileK, tile_gid(s), tile_kg0(s),
                      m0, n0, qweight, qzeros, scales, a_fp8, N, lda, tid);
    }
    __pipeline_commit();
  }

  // --- Main loop ---
  for (int kt = 0; kt < num_ktiles; ++kt) {
    __pipeline_wait_prior(kStages - 2);  // wait until tile kt is resident
    __syncthreads();

    const int cur = kt % kStages;
    char* base = stage_ptr(cur);
    __nv_fp8_e4m3* A_fp8_smem = reinterpret_cast<__nv_fp8_e4m3*>(base);
    int32_t* Wq_smem = reinterpret_cast<int32_t*>(base + kA_bytes);                 // [g][n]
    __nv_bfloat16* Sc_smem = reinterpret_cast<__nv_bfloat16*>(base + kA_bytes + kW_bytes);
    int32_t* Zq_smem = reinterpret_cast<int32_t*>(base + kA_bytes + kW_bytes + kS_bytes);

    // FP8 warp-level mma.sync m16n8k32 with register-only weight dequant.
    for (int sk = 0; sk < kTileK; sk += kMmaK) {
      for (int ns = 0; ns < 16; ++ns) {
        int wn = ns * kMmaN;

        // B fragment: dequant raw INT4 -> FP8 in registers (depends only on
        // sk, ns — hoisted out of the ms loop, same as v2).
        uint32_t b_regs[2];
        {
          int n_idx = wn + lane_id / 4;
          int k0 = sk + (lane_id % 4) * 4;
          int k1 = k0 + 16;

          float sc = __bfloat162float(Sc_smem[n_idx]);
          int z4 = 0;
          if (has_zeros) {
            int zb = (n_idx % 8) * 4;
            z4 = ((Zq_smem[n_idx / 8] >> zb) & 0xF) + 1;
          }

          int32_t wq0 = Wq_smem[(k0 >> 3) * kTileN + n_idx];
          int32_t wq1 = Wq_smem[(k1 >> 3) * kTileN + n_idx];
          b_regs[0] = dequant4_to_fp8(wq0, (k0 & 7) * 4, z4, sc);
          b_regs[1] = dequant4_to_fp8(wq1, (k1 & 7) * 4, z4, sc);
        }

        for (int ms = 0; ms < 2; ++ms) {
          int wm = warp_m0 + ms * kMmaM;

          uint32_t a_regs[4];
          {
            int row0 = wm + lane_id / 4;
            int row1 = row0 + 8;
            int col0 = sk + (lane_id % 4) * 4;
            int col1 = col0 + 16;
            memcpy(&a_regs[0], &A_fp8_smem[row0 * kTileK + col0], sizeof(uint32_t));
            memcpy(&a_regs[1], &A_fp8_smem[row1 * kTileK + col0], sizeof(uint32_t));
            memcpy(&a_regs[2], &A_fp8_smem[row0 * kTileK + col1], sizeof(uint32_t));
            memcpy(&a_regs[3], &A_fp8_smem[row1 * kTileK + col1], sizeof(uint32_t));
          }

          float* cp = c_regs[ms][ns];
          asm volatile(
              "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
              : "+f"(cp[0]), "+f"(cp[1]), "+f"(cp[2]), "+f"(cp[3])
              : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                "r"(b_regs[0]), "r"(b_regs[1]),
                "f"(cp[0]), "f"(cp[1]), "f"(cp[2]), "f"(cp[3]));
        }
      }
    }

    // Prefetch tile kt + kStages-1 into its ring buffer slot.
    const int fetch = kt + kStages - 1;
    if (fetch < num_ktiles) {
      load_tile_async(stage_ptr(fetch % kStages), fetch * kTileK,
                      tile_gid(fetch), tile_kg0(fetch),
                      m0, n0, qweight, qzeros, scales, a_fp8, N, lda, tid);
    }
    __pipeline_commit();
    __syncthreads();  // ensure compute(cur) done before its slot is reused
  }

  // Epilogue: accumulators -> BF16 (identical to v2).
  for (int ms = 0; ms < 2; ++ms) {
    int wm = warp_m0 + ms * kMmaM;
    for (int ns = 0; ns < 16; ++ns) {
      int wn = ns * kMmaN;
      float* cp = c_regs[ms][ns];
      int row0 = wm + lane_id / 4;
      int row1 = row0 + 8;
      int col0 = wn + 2 * (lane_id % 4);
      int col1 = col0 + 1;
      if (m0 + row0 < M && n0 + col0 < N)
        c_bf16[(m0 + row0) * ldc + n0 + col0] = __float2bfloat16(cp[0]);
      if (m0 + row0 < M && n0 + col1 < N)
        c_bf16[(m0 + row0) * ldc + n0 + col1] = __float2bfloat16(cp[1]);
      if (m0 + row1 < M && n0 + col0 < N)
        c_bf16[(m0 + row1) * ldc + n0 + col0] = __float2bfloat16(cp[2]);
      if (m0 + row1 < M && n0 + col1 < N)
        c_bf16[(m0 + row1) * ldc + n0 + col1] = __float2bfloat16(cp[3]);
    }
  }
}

torch::Tensor w4a8_fp8_fused_gemm(
    const torch::Tensor& qweight, const torch::Tensor& qzeros,
    const torch::Tensor& scales, const torch::Tensor& a_fp8,
    int64_t N, int64_t K, int64_t group_size) {

  int M = a_fp8.size(0);
  TORCH_CHECK(qweight.dtype() == torch::kInt32, "qweight must be int32");
  TORCH_CHECK(a_fp8.dtype() == torch::kFloat8_e4m3fn, "act must be fp8_e4m3");
  TORCH_CHECK(M % kTileM == 0, "M must be multiple of ", kTileM);
  TORCH_CHECK(N % kTileN == 0, "v3 requires N multiple of ", kTileN, " (got N=", N, ")");
  TORCH_CHECK(K % kTileK == 0, "v3 requires K multiple of ", kTileK, " (got K=", K, ")");
  TORCH_CHECK(group_size >= kTileK, "group_size must be >= ", kTileK,
              " (tile must not cross a quant group boundary)");

  auto c_bf16 = torch::empty({M, N},
      torch::dtype(torch::kBFloat16).device(qweight.device()));

  dim3 grid(M / kTileM, (int)N / kTileN);
  dim3 block(kWarps * kWarpSize);

  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(w4a8_fp8_qmma_v3_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    attr_set = true;
  }

  auto stream = c10::cuda::getCurrentCUDAStream();
  w4a8_fp8_qmma_v3_kernel<<<grid, block, kSmemBytes, stream>>>(
      static_cast<const int32_t*>(qweight.const_data_ptr()),
      qzeros.numel() > 0 ? static_cast<const int32_t*>(qzeros.const_data_ptr()) : nullptr,
      static_cast<const __nv_bfloat16*>(scales.const_data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(a_fp8.const_data_ptr()),
      static_cast<__nv_bfloat16*>(c_bf16.data_ptr()),
      M, (int)N, (int)K, (int)group_size,
      (int)a_fp8.stride(0), (int)c_bf16.stride(0));

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    TORCH_CHECK(false, "Kernel launch failed: ", cudaGetErrorString(err));
  return c_bf16;
}

}  // namespace sglang

TORCH_LIBRARY_FRAGMENT(w4a8_fused, m) {
  m.def("w4a8_fp8_fused_gemm(Tensor qweight, Tensor qzeros, Tensor scales, "
        "Tensor a_fp8, int N, int K, int group_size) -> Tensor");
}
TORCH_LIBRARY_IMPL(w4a8_fused, CUDA, m) {
  m.impl("w4a8_fp8_fused_gemm", &sglang::w4a8_fp8_fused_gemm);
}
