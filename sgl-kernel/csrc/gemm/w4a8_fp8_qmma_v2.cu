/* W4A8 Fused GEMM v2 — register-only INT4 dequant + SM120 warp-level FP8 MMA.
 *
 * Path B from PROPOSAL_optimized_fused_w4a8_v2: eliminate the shared-memory
 * FP8 round-trip used by w4a8_fp8_qmma.cu (v1). Instead of dequanting the whole
 * INT4 weight tile into an 8 KB FP8 SMEM buffer (write) and reading it back
 * (read) during the MMA, v2 stages only the COMPACT raw INT4 weights (4 KB) in
 * SMEM and dequants each MMA B-fragment directly in registers immediately
 * before the mma.sync instruction.
 *
 * Identical to v1 in tile sizes, warp/MMA partition, A-fragment loading, and
 * epilogue layout (validated correct). The ONLY change is how the B (weight)
 * fragment is produced: v1 reads pre-dequanted FP8 from SMEM; v2 dequants raw
 * INT4 from SMEM in registers.
 *
 * Tile: M=128, N=128, K=64. MMA: m16n8k32 FP8. Threads: 128 (4 warps).
 * SMEM: ~12.3 KB (4 KB raw INT4 weights + 8 KB FP8 acts + scales/zeros).
 */

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
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
static constexpr int kKGroupsPerTile = kTileK / 8;  // int32 words per column (8 INT4 per int32)

// Dequant 4 consecutive-K INT4 weights (for one N column) into 4 packed FP8 bytes.
//   wq    : int32 holding 8 INT4 values for the column's K-group
//   shift : bit offset of the first of the 4 nibbles (0 or 16)
//   z4    : zero point for this column/group (already +1 adjusted)
//   sc    : dequant scale for this column/group (float)
// Returns uint32_t = {fp8(k0), fp8(k0+1), fp8(k0+2), fp8(k0+3)} little-endian,
// matching the byte order v1 read from the FP8 SMEM buffer.
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

__global__ void w4a8_fp8_qmma_v2_kernel(
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
  // Compact raw INT4 weights: [kTileN][kKGroupsPerTile] int32 = 128*8 = 4 KB
  int32_t* Wq_smem = reinterpret_cast<int32_t*>(smem_raw);
  // Per-tile scales: [kTileN] bf16 (gid constant within a tile)
  __nv_bfloat16* Sc_smem = reinterpret_cast<__nv_bfloat16*>(Wq_smem + kTileN * kKGroupsPerTile);
  // Per-tile packed zeros: [kTileN/8] int32
  int32_t* Zq_smem = reinterpret_cast<int32_t*>(Sc_smem + kTileN);
  // FP8 activations: [kTileM][kTileK] = 128*64 = 8 KB
  __nv_fp8_e4m3* A_fp8_smem = reinterpret_cast<__nv_fp8_e4m3*>(Zq_smem + kTileN / 8);

  float c_regs[2][16][4];
  for (int ms = 0; ms < 2; ++ms)
    for (int ns = 0; ns < 16; ++ns)
#pragma unroll
      for (int r = 0; r < 4; ++r) c_regs[ms][ns][r] = 0.0f;

  const int kb_groups = K / 8;            // total int32 K-groups in qweight
  const int n_zwords = N / 8;             // int32 words per zeros row

  for (int kb = 0; kb < K; kb += kTileK) {
    const int gid = kb / group_size;      // group id (constant within this tile)
    const int kg0 = kb / 8;               // first int32 K-group for this tile

    // --- Load raw INT4 weights into SMEM (compact, no dequant) ---
    // Wq_smem[n * kKGroupsPerTile + g] = qweight[(kg0+g)*N + (n0+n)]
    for (int i = tid; i < kTileN * kKGroupsPerTile; i += blockDim.x) {
      int n = i / kKGroupsPerTile, g = i % kKGroupsPerTile;
      int ng = n0 + n, kg = kg0 + g;
      Wq_smem[i] = (ng < N && kg < kb_groups) ? qweight[kg * N + ng] : 0;
    }

    // --- Load per-tile scales into SMEM ---
    for (int n = tid; n < kTileN; n += blockDim.x) {
      int ng = n0 + n;
      Sc_smem[n] = (ng < N) ? scales[gid * N + ng] : __nv_bfloat16(0);
    }

    // --- Load per-tile packed zeros into SMEM ---
    for (int zn = tid; zn < kTileN / 8; zn += blockDim.x) {
      int zng = n0 / 8 + zn;
      Zq_smem[zn] = (qzeros != nullptr && zng < n_zwords) ? qzeros[gid * n_zwords + zng] : 0;
    }

    // --- Load FP8 activations into SMEM [M][K] (same as v1) ---
    for (int i = tid; i < kTileM * kTileK; i += blockDim.x) {
      int m = i / kTileK, k = i % kTileK;
      int mg = m0 + m, kg = kb + k;
      A_fp8_smem[i] = (mg < M && kg < K) ? a_fp8[mg * lda + kg] : __nv_fp8_e4m3{0};
    }
    __syncthreads();

    // --- FP8 warp-level mma.sync m16n8k32 with register-only weight dequant ---
    for (int sk = 0; sk < kTileK; sk += kMmaK) {
      for (int ns = 0; ns < 16; ++ns) {
        int wn = ns * kMmaN;

        // B fragment: dequant raw INT4 -> FP8 in registers (hoisted out of ms
        // loop — depends only on (sk, ns), identical for both M sub-tiles).
        // Thread t: column n_idx = wn + t/4; K0 = sk + (t%4)*4, K1 = K0 + 16.
        uint32_t b_regs[2];
        {
          int n_idx = wn + lane_id / 4;
          int k0 = sk + (lane_id % 4) * 4;
          int k1 = k0 + 16;

          float sc = __bfloat162float(Sc_smem[n_idx]);
          int z4 = 0;
          if (qzeros != nullptr) {
            int zb = (n_idx % 8) * 4;
            z4 = ((Zq_smem[n_idx / 8] >> zb) & 0xF) + 1;
          }

          int32_t wq0 = Wq_smem[n_idx * kKGroupsPerTile + (k0 / 8)];
          int32_t wq1 = Wq_smem[n_idx * kKGroupsPerTile + (k1 / 8)];
          b_regs[0] = dequant4_to_fp8(wq0, (k0 % 8) * 4, z4, sc);
          b_regs[1] = dequant4_to_fp8(wq1, (k1 % 8) * 4, z4, sc);
        }

        for (int ms = 0; ms < 2; ++ms) {
          int wm = warp_m0 + ms * kMmaM;

          // A fragment: m16n8k32 FP8 row-major (identical to v1).
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
    __syncthreads();
  }

  // Epilogue: accumulators -> BF16 (identical to v1)
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
  TORCH_CHECK(group_size >= kTileK, "group_size must be >= ", kTileK,
              " (tile must not cross a quant group boundary)");

  auto c_bf16 = torch::empty({M, N},
      torch::dtype(torch::kBFloat16).device(qweight.device()));

  dim3 grid(M / kTileM, ((int)N + kTileN - 1) / kTileN);
  dim3 block(kWarps * kWarpSize);
  // SMEM: raw INT4 weights + scales + zeros + FP8 activations
  constexpr int kSmemBytes =
      kTileN * kKGroupsPerTile * (int)sizeof(int32_t)   // 4096
      + kTileN * (int)sizeof(__nv_bfloat16)             // 256
      + (kTileN / 8) * (int)sizeof(int32_t)             // 64
      + kTileM * kTileK * (int)sizeof(__nv_fp8_e4m3);   // 8192

  auto stream = c10::cuda::getCurrentCUDAStream();
  w4a8_fp8_qmma_v2_kernel<<<grid, block, kSmemBytes, stream>>>(
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
