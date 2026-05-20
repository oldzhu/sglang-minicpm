/* W4A8 Fused GEMM with warp-level MMA (FP16 accumulator).
 *
 * Eliminates the FP8 weight HBM round-trip by dequantizing INT4→FP16
 * in shared memory and computing the GEMM with warp MMA instructions.
 *
 * Tile: 128×128×128, 128 threads per block (4 warps).
 * SMEM: 32KB weight tile (FP16) + 4KB activation sub-tile (FP16) = 36KB.
 *
 * MMA: m16n8k16 FP16 (148 TFLOPS on SM120). Future: QMMA tcgen05 FP8 (296 TF).
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
static constexpr int kWarpSize = 32;
static constexpr int kWarps = 4;
static constexpr int kMmaM = 16, kMmaN = 8, kMmaK = 16;

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

  // Weight SMEM: FP16 [kTileN][kTileK] (128×128 half = 32KB)
  __shared__ __half W[kTileN][kTileK];
  // Activation SMEM: FP16 [kTileM][kMmaK] (128×16 half = 4KB)
  __shared__ __half A[kTileM][kMmaK];

  // Each warp handles a 32-row slice of the output
  const int warp_m_start = warp_id * 32;  // 0, 32, 64, or 96

  // Accumulator for this warp's 32×128 output slice
  // Organized as [2 M-steps][16 N-steps][8 elts per 16x8 block]
  float acc[2][16][8] = {0.0f};

  for (int kb = 0; kb < K; kb += kTileK) {
    // Phase 1: Dequant INT4 -> FP16 into SMEM
    for (int i = tid; i < kTileN * kTileK; i += 128) {
      int n = i / kTileK;
      int k = i % kTileK;
      int kg = kb + k;
      int ng = n0 + n;
      if (kg < K && ng < N) {
        int kp = kg / 8, kbit = (kg % 8) * 4;
        int32_t pw = qweight[kp * N + ng];
        int w4 = (pw >> kbit) & 0xF;
        int gid = kg / group;
        int z4 = 0;
        if (qzeros != nullptr) {
          int zn = ng / 8, zb = (ng % 8) * 4;
          z4 = ((qzeros[gid * (N / 8) + zn] >> zb) & 0xF) + 1;
        }
        float s = scales[gid * N + ng];
        float v = ((float)w4 - (float)z4) * s;
        W[n][k] = __float2half(v);
      } else {
        W[n][k] = __float2half(0.0f);
      }
    }
    __syncthreads();

    // Phase 2+3: MMA loop over K sub-tiles
    for (int sk = 0; sk < kTileK; sk += kMmaK) {
      // Load activation sub-tile A[128][16] from FP8 global memory
      __syncthreads();
      for (int i = tid; i < kTileM * kMmaK; i += 128) {
        int m = i / kMmaK;
        int k = i % kMmaK;
        int kg = kb + sk + k;
        int mg = m0 + m;
        if (kg < K && mg < M) {
          A[m][k] = __float2half(static_cast<float>(a_fp8[mg * lda + kg]));
        } else {
          A[m][k] = __float2half(0.0f);
        }
      }
      __syncthreads();

      // Each warp does 2 M-steps (32 rows / 16) × 16 N-steps (128 cols / 8)
      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, kMmaM, kMmaN, kMmaK, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, kMmaM, kMmaN, kMmaK, half, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, kMmaM, kMmaN, kMmaK, float> c_frag;

      for (int m_step = 0; m_step < 2; ++m_step) {
        int local_m = warp_m_start + m_step * kMmaM;
        for (int n_step = 0; n_step < 16; ++n_step) {
          // Load A: 16×16 from A[local_m..local_m+15][0..15]
          wmma::load_matrix_sync(a_frag, &A[local_m][0], kMmaK);
          // Load B: 16×8 from W[n0 + n_step*8..][sk..sk+15]
          // W is [N][K] row-major; MMA needs col-major for B
          // Manual load into fragment
          for (int i = 0; i < a_frag.num_elements; ++i)
            a_frag.x[i] = A[local_m + (i % 4) * 4 + (i / 4) % 4][(i / 16) % 16];
          // Actually use simpler pattern: lane_id maps to element
          // lane 0..3: row, lane 4..31: distributed
          // Use wmma's built-in load which handles the mapping
          wmma::load_matrix_sync(b_frag, &W[n0 + n_step * kMmaN][sk], kTileK);
          // Load accumulator
          wmma::load_matrix_sync(c_frag, &acc[m_step][n_step][0], kMmaN);
          // MMA
          wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
          // Store accumulator
          wmma::store_matrix_sync(&acc[m_step][n_step][0], c_frag, kMmaN, wmma::mem_row_major);
        }
      }
    }
  }

  // Phase 4: Write output (BF16)
  for (int m_step = 0; m_step < 2; ++m_step) {
    int mg_base = m0 + warp_m_start + m_step * kMmaM;
    for (int mi = 0; mi < kMmaM; ++mi) {
      int mg = mg_base + mi;
      if (mg >= M) continue;
      for (int n_step = 0; n_step < 16; ++n_step) {
        int ng_base = n0 + n_step * kMmaN;
        for (int ni = 0; ni < kMmaN; ++ni) {
          int ng = ng_base + ni;
          if (ng < N) {
            c_bf16[mg * ldc + ng] = __float2bfloat16(acc[m_step][n_step][mi * kMmaN + ni]);
          }
        }
      }
    }
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

TORCH_LIBRARY_FRAGMENT(w4a8_fused, m) {
  m.def("w4a8_fp8_fused_gemm(Tensor qweight, Tensor qzeros, Tensor scales, "
        "Tensor a_fp8, int N, int K, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(w4a8_fused, CUDA, m) {
  m.impl("w4a8_fp8_fused_gemm", &sglang::w4a8_fp8_fused_gemm);
}
