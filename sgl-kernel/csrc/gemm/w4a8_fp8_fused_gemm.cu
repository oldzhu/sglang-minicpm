/* W4A8 Fused GEMM with warp-level MMA (FP16, m16n16k16).
 *
 * Dequants INT4->FP16 in shared memory, then uses warp MMA (nvcuda::wmma)
 * for the matrix multiply. Eliminates the FP8/FP16 weight HBM round-trip.
 *
 * Tile: 128x128x128, 128 threads/block (4 warps). SMEM: ~40KB.
 * MMA: m16n16k16 FP16 (148 TFLOPS on SM120).
 */

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <torch/library.h>

#include <mma.h>

namespace sglang {

static constexpr int kTileM = 128, kTileN = 128, kTileK = 128;
static constexpr int kWarpSize = 32, kWarps = 4;
static constexpr int kMmaM = 16, kMmaN = 16, kMmaK = 16;

__global__ void w4a8_fp8_fused_gemm_kernel(
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const float* __restrict__ scales,
    const __nv_fp8_e4m3* __restrict__ a_fp8,
    __nv_bfloat16* __restrict__ c_bf16,
    int M, int N, int K, int group, int lda, int ldc) {

  const int mb = blockIdx.x, nb = blockIdx.y;
  const int m0 = mb * kTileM, n0 = nb * kTileN;
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;

  // Shared memory: weight dequant buffer (FP16) + activation tile (FP16)
  __shared__ __half W[kTileN][kTileK];        // 128x128 half = 32KB
  __shared__ __half A[kTileM][kMmaK];          // 128x16 half = 4KB
  // Output staging: each warp writes its 32x128 sub-result here
  __shared__ float C_smem[kTileM][kTileN];     // 128x128 float = 64KB

  // Each warp handles 32 rows of M
  const int warp_m0 = warp_id * (kTileM / kWarps);  // 0, 32, 64, or 96

  // Initialize output SMEM (all warps participate)
  for (int i = tid; i < kTileM * kTileN; i += blockDim.x)
    C_smem[i / kTileN][i % kTileN] = 0.0f;
  __syncthreads();

  for (int kb = 0; kb < K; kb += kTileK) {
    // Phase 1: Dequant INT4 -> FP16 into W[][]
    for (int i = tid; i < kTileN * kTileK; i += blockDim.x) {
      int n = i / kTileK, k = i % kTileK;
      int kg = kb + k, ng = n0 + n;
      if (kg < K && ng < N) {
        int kp = kg / 8, kbit = (kg % 8) * 4;
        int w4 = (qweight[kp * N + ng] >> kbit) & 0xF;
        int gid = kg / group;
        int z4 = 0;
        if (qzeros != nullptr) {
          int zn = ng / 8, zb = (ng % 8) * 4;
          z4 = ((qzeros[gid * (N / 8) + zn] >> zb) & 0xF) + 1;
        }
        float v = ((float)w4 - (float)z4) * scales[gid * N + ng];
        W[n][k] = __float2half(v);
      } else {
        W[n][k] = __float2half(0.0f);
      }
    }
    __syncthreads();

    // Phase 2: Loop over K sub-tiles, do MMA
    for (int sk = 0; sk < kTileK; sk += kMmaK) {
      // Load activation sub-tile A[128][16] from FP8 global
      __syncthreads();
      for (int i = tid; i < kTileM * kMmaK; i += blockDim.x) {
        int m = i / kMmaK, k = i % kMmaK;
        int kg = kb + sk + k, mg = m0 + m;
        A[m][k] = (kg < K && mg < M)
            ? __float2half(static_cast<float>(a_fp8[mg * lda + kg]))
            : __float2half(0.0f);
      }
      __syncthreads();

      // Each warp computes its 32x128 output using m16n16k16 MMA
      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, kMmaM, kMmaN, kMmaK, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, kMmaM, kMmaN, kMmaK, half, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, kMmaM, kMmaN, kMmaK, float> c_frag;

      // 2 M-steps (32/16), 8 N-steps (128/16)
      for (int ms = 0; ms < 2; ++ms) {
        int wm = warp_m0 + ms * kMmaM;
        for (int ns = 0; ns < 8; ++ns) {
          int wn = n0 + ns * kMmaN;

          // Load A from SMEM
          wmma::load_matrix_sync(a_frag, &A[wm][0], kMmaK);
          // Load B from SMEM (W is [N][K] row-major; MMA uses col-major B)
          wmma::load_matrix_sync(b_frag, &W[wn][sk], kTileK);
          // Load accumulator from output SMEM
          wmma::fill_fragment(c_frag, 0.0f);

          // MMA: C += A * B
          wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

          // Accumulate back to SMEM
          wmma::store_matrix_sync(&C_smem[wm][wn], c_frag, kTileN, wmma::mem_row_major);
        }
      }
    }
  }

  // Phase 4: Write output from SMEM to global (BF16)
  __syncthreads();
  for (int i = tid; i < kTileM * kTileN; i += blockDim.x) {
    int m = i / kTileN, n = i % kTileN;
    int mg = m0 + m, ng = n0 + n;
    if (mg < M && ng < N)
      c_bf16[mg * ldc + ng] = __float2bfloat16(C_smem[m][n]);
  }
}

torch::Tensor w4a8_fp8_fused_gemm(
    const torch::Tensor& qweight,
    const torch::Tensor& qzeros,
    const torch::Tensor& scales,
    const torch::Tensor& a_fp8,
    int64_t N, int64_t K, int64_t group_size) {

  int M = a_fp8.size(0);
  TORCH_CHECK(qweight.dtype() == torch::kInt32, "qweight must be int32");
  TORCH_CHECK(a_fp8.dtype() == torch::kFloat8_e4m3fn, "act must be fp8_e4m3");
  TORCH_CHECK(M % kTileM == 0, "M must be multiple of 128");

  auto c_bf16 = torch::empty({M, N},
      torch::dtype(torch::kBFloat16).device(qweight.device()));

  dim3 grid(M / kTileM, (N + kTileN - 1) / kTileN);
  dim3 block(128);

  auto stream = c10::cuda::getCurrentCUDAStream();
  w4a8_fp8_fused_gemm_kernel<<<grid, block, 0, stream>>>(
      static_cast<const int32_t*>(qweight.const_data_ptr()),
      qzeros.numel() > 0 ? static_cast<const int32_t*>(qzeros.const_data_ptr()) : nullptr,
      static_cast<const float*>(scales.const_data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(a_fp8.const_data_ptr()),
      static_cast<__nv_bfloat16*>(c_bf16.data_ptr()),
      M, (int)N, (int)K, (int)group_size,
      (int)a_fp8.stride(0), (int)c_bf16.stride(0));

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "Kernel launch failed: ", cudaGetErrorString(err));
  }

  return c_bf16;
}

}  // namespace sglang

// Python binding — use unique namespace to avoid conflict with sgl_kernel wheel
TORCH_LIBRARY_FRAGMENT(w4a8_fused, m) {
  m.def("w4a8_fp8_fused_gemm(Tensor qweight, Tensor qzeros, Tensor scales, "
        "Tensor a_fp8, int N, int K, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(w4a8_fused, CUDA, m) {
  m.impl("w4a8_fp8_fused_gemm", &sglang::w4a8_fp8_fused_gemm);
}
