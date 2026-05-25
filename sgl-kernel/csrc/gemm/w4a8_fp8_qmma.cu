/* W4A8 Fused GEMM with SM120 warp-level FP8 MMA (296 TFLOPS).
 *
 * Uses warp-level mma.sync with FP8 e4m3 types. tcgen05 is SM100-only.
 * SM120 MMA is warp-level (32 threads), not warpgroup.
 *
 * Tile: M=128, N=128, K=64. MMA: m16n8k32 FP8. Threads: 128 (4 warps).
 * SMEM: ~16KB (128x64x2 FP8 buffers).
 */

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <torch/library.h>

namespace sglang {

static constexpr int kTileM = 128;
static constexpr int kTileN = 128;
static constexpr int kTileK = 64;
static constexpr int kMmaM = 16;
static constexpr int kMmaN = 8;
static constexpr int kMmaK = 32;
static constexpr int kWarpSize = 32;
static constexpr int kWarps = 4;

__global__ void w4a8_fp8_qmma_kernel(
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

  extern __shared__ char smem_raw[];
  __nv_fp8_e4m3* W_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(smem_raw);
  __nv_fp8_e4m3* A_fp8_smem = W_fp8 + kTileN * kTileK;

  float c_regs[2][16][4];
  for (int ms = 0; ms < 2; ++ms)
    for (int ns = 0; ns < 16; ++ns)
#pragma unroll
      for (int r = 0; r < 4; ++r) c_regs[ms][ns][r] = 0.0f;

  for (int kb = 0; kb < K; kb += kTileK) {

    // Phase 1: INT4 -> FP8 dequant into W_fp8 [N][K]
    for (int i = tid; i < kTileN * kTileK; i += blockDim.x) {
      int n = i / kTileK, k = i % kTileK;
      int kg = kb + k, ng = n0 + n;
      __nv_fp8_e4m3 val;
      if (kg < K && ng < N) {
        int kp = kg / 8, kbit = (kg % 8) * 4;
        int w4 = (qweight[kp * N + ng] >> kbit) & 0xF;
        int gid = kg / group_size, z4 = 0;
        if (qzeros != nullptr) {
          int zn = ng / 8, zb = (ng % 8) * 4;
          z4 = ((qzeros[gid * (N / 8) + zn] >> zb) & 0xF) + 1;
        }
        float fv = ((float)w4 - (float)z4) * __bfloat162float(scales[gid * N + ng]);
        // Directly cast float->__nv_fp8_e4m3 to avoid the double-conversion bug:
        // __nv_cvt_float_to_fp8() returns uint8_t; static_cast<fp8>(uint8_t) calls
        // the float constructor (uint8_t→float→fp8), reinterpreting the raw byte as
        // a float value and reconverting, producing the wrong FP8 byte.
        val = __nv_fp8_e4m3(fv);
      }
      W_fp8[n * kTileK + k] = val;
    }

    // Phase 2: Load FP8 activations -> A_fp8_smem [M][K]
    for (int i = tid; i < kTileM * kTileK; i += blockDim.x) {
      int m = i / kTileK, k = i % kTileK;
      int mg = m0 + m, kg = kb + k;
      A_fp8_smem[m * kTileK + k] =
          (mg < M && kg < K) ? a_fp8[mg * lda + kg] : __nv_fp8_e4m3{0};
    }
    __syncthreads();

    // Phase 3: FP8 warp-level mma.sync m16n8k32
    for (int sk = 0; sk < kTileK; sk += kMmaK) {
      for (int ms = 0; ms < 2; ++ms) {
        int wm = warp_m0 + ms * kMmaM;
        for (int ns = 0; ns < 16; ++ns) {
          int wn = ns * kMmaN;

          // A fragment: m16n8k32 FP8 row-major, PTX ISA layout.
          // Thread t: a[0]=A[t/4][(t%4)*4..+4], a[1]=A[t/4][(t%4)*4+16..+4],
          //           a[2]=A[t/4+8][(t%4)*4..+4], a[3]=A[t/4+8][(t%4)*4+16..+4]
          uint32_t a_regs[4];
          {
            int row0 = wm + lane_id / 4;
            int row1 = row0 + 8;
            int col0 = sk + (lane_id % 4) * 4;
            int col1 = col0 + 16;
            memcpy(&a_regs[0], &A_fp8_smem[row0 * kTileK + col0], sizeof(uint32_t));
            memcpy(&a_regs[1], &A_fp8_smem[row0 * kTileK + col1], sizeof(uint32_t));
            memcpy(&a_regs[2], &A_fp8_smem[row1 * kTileK + col0], sizeof(uint32_t));
            memcpy(&a_regs[3], &A_fp8_smem[row1 * kTileK + col1], sizeof(uint32_t));
          }

          // B fragment: m16n8k32 FP8 col-major, shape [32, 8] (K=32, N=8).
          // Col-major: B[k][n] stored at k*8+n.
          // Thread t: b[0]=B[t/4][(t%4)*4..+4], b[1]=B[t/4][(t%4)*4+16..+4]
          //   = 4 consecutive bytes at K=t/4, N=(t%4)*4..+4
          // Our SMEM: W_fp8[n][k] stored as row-major with stride kTileK.
          // For col-major B[k][n], we need to read from W_fp8[n][k] with
          // k = K_idx (inner), n = N_idx (outer).
          // b_regs[0] = B[K_idx][N_start..N_start+3] = W_fp8[N_start..N_start+3][K_idx]
          // But W_fp8 is row-major [N][K], so W_fp8[n][k] = W_fp8[n * kTileK + k]
          // B[K_idx][N_start..N_start+3] = W_fp8[N_start..N_start+3][K_idx]
          // These are NOT consecutive in SMEM (stride = kTileK)!
          // We need to gather 4 individual bytes.
          uint32_t b_regs[2];
          {
            int k_idx = sk + lane_id / 4;  // K index within tile (0..31)
            int n_start = wn + (lane_id % 4) * 4;  // N start (0,4,8,12 within 8-wide tile)
            uint8_t b0_bytes[4], b1_bytes[4];
            for (int i = 0; i < 4; ++i) {
              b0_bytes[i] = (uint8_t)W_fp8[(n_start + i) * kTileK + k_idx];
              b1_bytes[i] = (uint8_t)W_fp8[(n_start + 16 + i) * kTileK + k_idx];
            }
            memcpy(&b_regs[0], b0_bytes, sizeof(uint32_t));
            memcpy(&b_regs[1], b1_bytes, sizeof(uint32_t));
          }

          float* cp = c_regs[ms][ns];
          float c0 = cp[0], c1 = cp[1], c2 = cp[2], c3 = cp[3];
          asm volatile(
              "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
              : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
              : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                "r"(b_regs[0]), "r"(b_regs[1]),
                "f"(c0), "f"(c1), "f"(c2), "f"(c3));
          cp[0] = c0; cp[1] = c1; cp[2] = c2; cp[3] = c3;
        }
      }
    }
    __syncthreads();
  }

  // Epilogue: accumulators -> BF16
  // m16n8 D layout per thread t: rows {t/4, t/4+8}, cols {2*(t%4), 2*(t%4)+1}
  //   cp[0]->(t/4,   2*(t%4))  cp[1]->(t/4,   2*(t%4)+1)
  //   cp[2]->(t/4+8, 2*(t%4))  cp[3]->(t/4+8, 2*(t%4)+1)
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

  auto c_bf16 = torch::empty({M, N},
      torch::dtype(torch::kBFloat16).device(qweight.device()));

  dim3 grid(M / kTileM, ((int)N + kTileN - 1) / kTileN);
  dim3 block(kWarps * kWarpSize);
  constexpr int kSmemBytes = kTileN * kTileK * 2;  // 16KB

  auto stream = c10::cuda::getCurrentCUDAStream();
  w4a8_fp8_qmma_kernel<<<grid, block, kSmemBytes, stream>>>(
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
