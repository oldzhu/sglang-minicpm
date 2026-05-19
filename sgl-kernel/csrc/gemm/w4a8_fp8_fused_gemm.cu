/* W4A8 Fused GEMM: INT4 dequant + FP8 matmul in a single kernel.
 *
 * Eliminates the FP8 weight HBM round-trip by dequantizing INT4->FP8
 * in shared memory and computing the GEMM in the same kernel.
 *
 * Tile: 128x128x128, 128 threads per block.
 * SMEM: 16KB weight tile + 4KB activation sub-tile = 20KB.
 *
 * For max throughput, a warp-level QMMA (tcgen05) version is planned.
 * This kernel prioritizes correctness and bandwidth elimination.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <torch/all.h>

#include "utils.h"

namespace sglang {

static constexpr int kTileM = 128, kTileN = 128, kTileK = 128, kSubK = 32;
static constexpr int kMaxFp8 = 448;

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

  __shared__ __nv_fp8_e4m3 W[kTileN][kTileK];
  __shared__ __nv_fp8_e4m3 A[kSubK][kTileM];
  float acc[kTileN] = {0.0f};

  for (int kb = 0; kb < K; kb += kTileK) {
    // Phase 1: Dequant INT4 -> FP8 into SMEM
    for (int sk = 0; sk < kTileK; sk += kSubK) {
      for (int i = tid; i < kTileN * kSubK; i += 128) {
        int kl = i % kSubK, nl = i / kSubK;
        int kg = kb + sk + kl, ng = n0 + nl;
        if (kg < K && ng < N) {
          int kp = kg / 8, kbit = (kg % 8) * 4;
          int32_t pw = qweight[kp * N + ng];
          int w4 = (pw >> kbit) & 0xF;
          int gid = kg / group;
          int z4 = 0;
          if (qzeros != nullptr) {
            int zp = gid / 8, zb = (gid % 8) * 4;
            z4 = (qzeros[zp * N + ng] >> zb) & 0xF;
          }
          float s = scales[gid * N + ng];
          float v = ((float)w4 - (float)z4) * s;
          if (v > (float)kMaxFp8) v = (float)kMaxFp8;
          if (v < -(float)kMaxFp8) v = -(float)kMaxFp8;
          W[nl][sk + kl] = __nv_cvt_float_to_fp8(v);
        }
      }
    }
    __syncthreads();

    // Phase 2+3: Load activation sub-tile and compute partial dot products
    for (int sk = 0; sk < kTileK; sk += kSubK) {
      __syncthreads();
      for (int i = tid; i < kTileM * kSubK; i += 128) {
        int kl = i % kSubK, ml = i / kSubK;
        int kg = kb + sk + kl, mg = m0 + ml;
        A[kl][ml] = (kg < K && mg < M) ? a_fp8[mg * lda + kg]
                     : __nv_cvt_float_to_fp8(0.0f);
      }
      __syncthreads();

      for (int n = 0; n < kTileN; ++n) {
        float dot = 0.0f;
        for (int k = 0; k < kSubK; ++k)
          dot += __nv_cvt_fp8_to_float(W[n][sk + k]) *
                 __nv_cvt_fp8_to_float(A[k][tid]);
        acc[n] += dot;
      }
    }
  }

  // Phase 4: Write output
  for (int n = 0; n < kTileN; ++n) {
    int mg = m0 + tid, ng = n0 + n;
    if (mg < M && ng < N)
      c_bf16[mg * ldc + ng] = __float2bfloat16(acc[n]);
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

  auto stream = at::cuda::getCurrentCUDAStream();
  w4a8_fp8_fused_gemm_kernel<<<grid, block, 0, stream>>>(
      static_cast<const int32_t*>(qweight.const_data_ptr()),
      qzeros.numel() > 0 ? static_cast<const int32_t*>(qzeros.const_data_ptr()) : nullptr,
      static_cast<const float*>(scales.const_data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(a_fp8.const_data_ptr()),
      static_cast<__nv_bfloat16*>(c_bf16.data_ptr()),
      M, (int)N, (int)K, (int)group_size,
      (int)a_fp8.stride(0), (int)c_bf16.stride(0));

  return c_bf16;
}

}  // namespace sglang

// Python binding
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("w4a8_fp8_fused_gemm(Tensor qweight, Tensor qzeros, Tensor scales, "
        "Tensor a_fp8, int N, int K, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("w4a8_fp8_fused_gemm", &sglang::w4a8_fp8_fused_gemm);
}
