/* GPTQ INT4 -> FP8 e4m3 blockwise dequantization kernel for SM120.
 *
 * Converts GPTQ packed INT4 weights to FP8 e4m3 with 128x128 blockwise
 * scales, in the format expected by the SM120 FP8 blockwise GEMM.
 *
 * Grid: (N_tiles, K_tiles) where each tile is 128x128 elements.
 * Each block handles one tile.
 *
 * Output layout:
 *   weight_fp8:  (N, K) float8_e4m3fn, column-major contiguous
 *   weight_scale: (N/128, K/128) float32
 *
 * GPTQ input layout:
 *   qweight: (K/8, N) int32  — 8 INT4 values packed per int32 along K
 *   qzeros:  (K/g/8, N) int32 — packed zero points
 *   scales:  (K/g, N) float32 — per-group scales (converted by caller)
 *   g = group_size (typically 128)
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>
#include <torch/all.h>

#include "utils.h"

namespace sglang {

static constexpr float kFp8E4m3Max = 448.0f;

__global__ void gptq_int4_to_fp8_blockwise_kernel(
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const float* __restrict__ scales,
    __nv_fp8_e4m3* __restrict__ weight_fp8,
    float* __restrict__ weight_scales,
    int K,
    int N,
    int group_size) {

  const int n_tile = blockIdx.x;
  const int k_tile = blockIdx.y;
  const int n_start = n_tile * 128;
  const int k_start = k_tile * 128;

  constexpr int kBlockSize = 256;
  constexpr int kElemsPerTile = 128 * 128;
  constexpr int kItersPerThread = kElemsPerTile / kBlockSize;

  __shared__ float smem_vals[kElemsPerTile];
  __shared__ float smem_reduce[kBlockSize];

  const int tid = threadIdx.x;
  float tile_amax = 0.0f;

  // Step 1: Unpack INT4 → float.
  for (int iter = 0; iter < kItersPerThread; ++iter) {
    int elem_idx = tid * kItersPerThread + iter;
    int k_local = elem_idx % 128;
    int n_local = elem_idx / 128;
    int k_global = k_start + k_local;
    int n_global = n_start + n_local;

    int k_packed_idx = k_global / 8;
    int k_shift = (k_global % 8) * 4;

    int32_t packed_w = qweight[k_packed_idx * N + n_global];
    int32_t w4 = (packed_w >> k_shift) & 0xF;

    int group = k_global / group_size;
    int z_packed_idx = group / 8;
    int z_shift = (group % 8) * 4;

    int32_t packed_z = qzeros[z_packed_idx * N + n_global];
    int32_t z4 = (packed_z >> z_shift) & 0xF;

    float scale_val = scales[group * N + n_global];
    float val = (static_cast<float>(w4) - static_cast<float>(z4)) * scale_val;

    smem_vals[elem_idx] = val;
    float a = fabsf(val);
    if (a > tile_amax) tile_amax = a;
  }

  __syncthreads();

  // Step 2: Block-reduce tile_amax.
  smem_reduce[tid] = tile_amax;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float other = smem_reduce[tid + stride];
      if (other > smem_reduce[tid]) smem_reduce[tid] = other;
    }
    __syncthreads();
  }
  float block_amax = smem_reduce[0];
  if (block_amax < 1e-12f) block_amax = 1e-12f;
  float scale_val = block_amax / kFp8E4m3Max;  // multiply: val/scale_val to quantize
  float scale_inv = 1.0f / scale_val;

  // Step 3: Quantize + write.
  int n_stride = K;

  for (int iter = 0; iter < kItersPerThread; ++iter) {
    int elem_idx = tid * kItersPerThread + iter;
    int k_local = elem_idx % 128;
    int n_local = elem_idx / 128;
    int k_global = k_start + k_local;
    int n_global = n_start + n_local;

    float val = smem_vals[elem_idx];
    float scaled = val * scale_inv;
    scaled = fmaxf(-kFp8E4m3Max, fminf(kFp8E4m3Max, scaled));

    // Convert float → fp8 e4m3 via CUDA intrinsic.
    __nv_fp8_e4m3 fp8_val = static_cast<__nv_fp8_e4m3>(
        __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));

    weight_fp8[n_global * n_stride + k_global] = fp8_val;
  }

  if (tid == 0) {
    int scale_stride = K / 128;
    weight_scales[n_tile * scale_stride + k_tile] = scale_val;
  }
}

std::tuple<torch::Tensor, torch::Tensor> gptq_int4_to_fp8_blockwise(
    const torch::Tensor& qweight,
    const torch::Tensor& qzeros,
    const torch::Tensor& scales,
    int64_t K,
    int64_t N,
    int64_t group_size) {

  TORCH_CHECK(qweight.is_cuda(), "qweight must be CUDA");
  TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "K,N must be multiples of 128");

  // Accept scales in float16 or bfloat16; convert to float32 for the kernel.
  auto scales_f32 = scales.to(torch::kFloat32).contiguous();

  auto weight_fp8 = torch::empty(
      {N, K},
      torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(qweight.device()));
  auto weight_scales = torch::empty(
      {N / 128, K / 128},
      torch::TensorOptions().dtype(torch::kFloat32).device(qweight.device()));

  dim3 grid(N / 128, K / 128);
  dim3 block(256);
  auto stream = at::cuda::getCurrentCUDAStream(qweight.get_device());

  gptq_int4_to_fp8_blockwise_kernel<<<grid, block, 0, stream>>>(
      static_cast<const int32_t*>(qweight.data_ptr()),
      static_cast<const int32_t*>(qzeros.data_ptr()),
      static_cast<const float*>(scales_f32.data_ptr()),
      static_cast<__nv_fp8_e4m3*>(weight_fp8.data_ptr()),
      static_cast<float*>(weight_scales.data_ptr()),
      static_cast<int>(K),
      static_cast<int>(N),
      static_cast<int>(group_size));

  return std::make_tuple(weight_fp8, weight_scales);
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "gptq_int4_to_fp8_blockwise(Tensor "
      "qweight, Tensor qzeros, Tensor "
      "scales, int K, int N, int "
      "group_size) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("gptq_int4_to_fp8_blockwise", &sglang::gptq_int4_to_fp8_blockwise);
}

}  // namespace sglang
