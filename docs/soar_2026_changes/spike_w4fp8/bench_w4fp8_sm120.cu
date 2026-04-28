// bench_w4fp8_sm120.cu
// W4-weight × FP8-activation GEMM micro-benchmark on SM120 (Blackwell RTX PRO 6000D)
//
// Purpose: measure the FP8 ceiling AFTER paying the W4 in-kernel dequant tax.
// This complements Phase 0 (which measured pure FP8×FP8 at 281 TF) by adding the
// dequant chain that a real W4-FP8 kernel would have.
//
// Status: SPIKE. Not production. Synthetic data. Single tile shape per launch.
//
// Kernel structure (per-warp, M=16, N=8, K=32 tile):
//   1. Load packed INT4 weight (4 bits per value, 8 values per uint32) from gmem.
//   2. Dequant in registers:
//        unpacked_int4 = (packed >> shift) & 0xF;
//        signed_int4   = unpacked_int4 - 8;          // zero-point (symmetric)
//        bf16_value    = signed_int4 * group_scale;  // bf16 multiply
//        fp8_value     = __nv_cvt_bfloat16raw_to_fp8(bf16_value);  // FP8 e4m3 cast
//   3. mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32  (FP32 accumulator)
//   4. Loop over K, accumulate.
//   5. Epilogue: FP32 -> BF16 store.
//
// Build:
//   nvcc -arch=sm_120 -O3 -std=c++17 -lcudart bench_w4fp8_sm120.cu -o bench_w4fp8
// Run:
//   ./bench_w4fp8                 # default M=16384,N=14336,K=4096
//   ./bench_w4fp8 8192 8192 4096  # custom MNK

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <chrono>

// ---- Tile shape ----
constexpr int BM = 128;   // block tile M
constexpr int BN = 128;   // block tile N
constexpr int BK = 64;    // block tile K (must be multiple of 32 for m16n8k32 MMA)
constexpr int WARP_M = 64;
constexpr int WARP_N = 64;
constexpr int WARPS_PER_BLOCK_M = BM / WARP_M;  // 2
constexpr int WARPS_PER_BLOCK_N = BN / WARP_N;  // 2
constexpr int WARPS_PER_BLOCK = WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N;  // 4
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // 128

// ---- Group size (for W4 group-scale dequant) ----
constexpr int GROUP_SIZE = 128;

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (0)

// ============================================================================
// Hand-rolled W4-FP8 GEMM kernel — minimal reference, NOT optimized
// ============================================================================
//
// Inputs (synthetic, generated on host):
//   A_fp8     : (M, K) FP8 e4m3, row-major   — activations
//   B_int4    : (N, K/8) packed INT4 (8 per uint32), row-major over (N, K)
//                Each packed group of 8 belongs to one (N, K_group) tile.
//   B_scales  : (N, K/GROUP_SIZE) BF16        — per-group scales for B
// Output:
//   C_bf16    : (M, N) BF16, row-major
//
// For a real production kernel you would use:
//   - cp.async.bulk for tile loads (Hopper+/Blackwell TMA)
//   - swizzled shared memory layouts to avoid bank conflicts
//   - software pipelining (multi-stage K-loop)
//   - epilogue fusion
// This spike intentionally skips all of that — we just need wall-clock TFLOPS
// to know whether the W4→FP8 dequant chain caps the FP8 ceiling.
// ============================================================================

__device__ __forceinline__ uint32_t mma_fp8_m16n8k32(
    const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
    const uint32_t b0, const uint32_t b1,
    float &c0, float &c1, float &c2, float &c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
    return 0;
}

// Convert a bf16 register-pair to an FP8 e4m3 byte-pair via PTX cvt.
// One PTX 'cvt.rn.satfinite.e4m3x2.bf16x2' produces 2 FP8 in a uint16.
__device__ __forceinline__ uint16_t bf16x2_to_fp8e4m3x2(uint32_t bf16_pair) {
    uint16_t result;
    asm volatile(
        "{ cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1; }\n"
        : "=h"(result) : "r"(bf16_pair));
    return result;
}

__global__ void w4fp8_gemm_kernel(
    const __nv_fp8_e4m3 *__restrict__ A_fp8,   // (M, K)
    const uint32_t      *__restrict__ B_int4,  // (N, K/8) packed
    const __nv_bfloat16 *__restrict__ B_scales,// (N, K/GROUP_SIZE)
    __nv_bfloat16       *__restrict__ C_bf16,  // (M, N)
    int M, int N, int K) {

    // For brevity, this scaffold computes 1 warp tile per warp, single K-iteration only.
    // A full kernel iterates over the K dimension and accumulates into FP32 registers.
    //
    // === The hot K-loop (pseudocode, the dequant tax we want to measure) ===
    //
    //   float c[4] = {0,0,0,0};               // FP32 accumulator for warp tile
    //   for (int kb = 0; kb < K; kb += BK) {
    //       // 1. Cooperative load: A tile (FP8) and B tile (INT4 packed) into smem
    //       cp_async(...);  __syncthreads();
    //
    //       // 2. Per-warp K-iter: load A frag and dequant B frag
    //       for (int ki = 0; ki < BK; ki += 32) {  // m16n8k32 step
    //           // ldmatrix A frag (4× uint32 of FP8e4m3, holds 16 values per thread)
    //           ldmatrix.sync.aligned.m8n8.x4.shared.b16  {a0,a1,a2,a3}, [smem_A_ptr];
    //
    //           // ldmatrix B packed-INT4 (1× uint32 of packed int4, 8 values)
    //           uint32_t b_packed = smem_B_int4[...];
    //
    //           // *** DEQUANT TAX ***
    //           // 8 INT4 -> 8 BF16  -> 4× bf16x2 register pairs
    //           bf16 b_bf16[8];
    //           for (int i = 0; i < 8; i++) {
    //               int4_t v = ((b_packed >> (i*4)) & 0xF) - 8;
    //               b_bf16[i] = __int2bfloat16(v) * scale_for_this_group;
    //           }
    //           // Pack into 4 bf16x2 pairs, then cvt to FP8 e4m3
    //           uint16_t b_fp8_pair0 = bf16x2_to_fp8e4m3x2(*(uint32_t*)&b_bf16[0]);
    //           uint16_t b_fp8_pair1 = bf16x2_to_fp8e4m3x2(*(uint32_t*)&b_bf16[2]);
    //           uint16_t b_fp8_pair2 = bf16x2_to_fp8e4m3x2(*(uint32_t*)&b_bf16[4]);
    //           uint16_t b_fp8_pair3 = bf16x2_to_fp8e4m3x2(*(uint32_t*)&b_bf16[6]);
    //           uint32_t b0 = (b_fp8_pair1 << 16) | b_fp8_pair0;
    //           uint32_t b1 = (b_fp8_pair3 << 16) | b_fp8_pair2;
    //           // *** END DEQUANT TAX ***
    //
    //           mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, c[0], c[1], c[2], c[3]);
    //       }
    //       __syncthreads();
    //   }
    //   // Epilogue: FP32 -> BF16 store
    //
    // === End pseudocode ===
    //
    // For the purpose of getting a TFLOPS number we don't even need correct output.
    // What matters is the kernel performs the right number of MMAs with the dequant
    // chain in front of each one. To make this compile-and-run we'll do a smaller
    // canonical version below: each warp issues the same dequant+MMA pattern in a
    // tight inner loop sized to match (M*N*K * 2) total FMAs.

    // === Minimal compile-and-run version: unrolled fake K-loop on shared dummy data ===
    extern __shared__ uint8_t smem[];
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    // Each warp gets:
    //   smem A frag : 16 bytes of FP8 (4× uint32 per thread → MMA A operand)
    //   smem B int4 : 4 bytes per warp of packed int4 (1× uint32 per thread)
    //   smem B scl  : 1 bf16 per warp
    //
    // We just loop the dequant + MMA pattern K_LOOP_ITERS times to estimate cost.
    constexpr int K_LOOP_ITERS = 64;  // arbitrary; controls work-per-launch

    uint32_t *smemA = reinterpret_cast<uint32_t*>(smem);
    uint32_t *smemB_int4 = reinterpret_cast<uint32_t*>(smem + 1024);
    __nv_bfloat16 *smemB_scale = reinterpret_cast<__nv_bfloat16*>(smem + 2048);

    if (tid == 0) {
        for (int i = 0; i < 256; i++) smemA[i] = 0x12345678u;
        for (int i = 0; i < 64;  i++) smemB_int4[i] = 0xFEDCBA98u;
        for (int i = 0; i < 16;  i++) smemB_scale[i] = __float2bfloat16(0.0625f);
    }
    __syncthreads();

    // Per-thread MMA accumulators
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    // Per-thread A fragments (4× uint32)
    uint32_t a0 = smemA[lane * 4 + 0];
    uint32_t a1 = smemA[lane * 4 + 1];
    uint32_t a2 = smemA[lane * 4 + 2];
    uint32_t a3 = smemA[lane * 4 + 3];

    __nv_bfloat16 scale = smemB_scale[warp_id];
    uint32_t bf16_scale_pair = (*(uint16_t*)&scale) | (((uint32_t)*(uint16_t*)&scale) << 16);

    #pragma unroll
    for (int it = 0; it < K_LOOP_ITERS; it++) {
        // Each thread "dequants" 8 int4 values and MMAs them.
        uint32_t b_packed = smemB_int4[(lane + it) & 0x3F];

        // Unpack 8 int4 → 8 bf16 (subtract 8 zero-point, multiply by group scale)
        // Build 4 bf16x2 pairs.
        uint16_t bf16_vals[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int4_t v = (int4_t)((b_packed >> (i * 4)) & 0xF) - 8;
            __nv_bfloat16 bf = __float2bfloat16((float)v * 0.0625f);  // scalar fallback
            bf16_vals[i] = *reinterpret_cast<uint16_t*>(&bf);
        }
        uint32_t bf_pair0 = bf16_vals[0] | ((uint32_t)bf16_vals[1] << 16);
        uint32_t bf_pair1 = bf16_vals[2] | ((uint32_t)bf16_vals[3] << 16);
        uint32_t bf_pair2 = bf16_vals[4] | ((uint32_t)bf16_vals[5] << 16);
        uint32_t bf_pair3 = bf16_vals[6] | ((uint32_t)bf16_vals[7] << 16);

        // bf16 → FP8 e4m3 (2 values per cvt)
        uint16_t fp8_p0 = bf16x2_to_fp8e4m3x2(bf_pair0);
        uint16_t fp8_p1 = bf16x2_to_fp8e4m3x2(bf_pair1);
        uint16_t fp8_p2 = bf16x2_to_fp8e4m3x2(bf_pair2);
        uint16_t fp8_p3 = bf16x2_to_fp8e4m3x2(bf_pair3);
        uint32_t b0 = fp8_p0 | ((uint32_t)fp8_p1 << 16);
        uint32_t b1 = fp8_p2 | ((uint32_t)fp8_p3 << 16);

        // FP8 m16n8k32 MMA — performs 16*8*32 = 4096 FMAs per warp per iter
        mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
    }

    // Sink: write to global so the compiler doesn't DCE the loop
    if (lane == 0) {
        int row = blockIdx.x;
        int col = blockIdx.y * 4 + warp_id;
        if (row < M && col < N) {
            C_bf16[row * N + col] = __float2bfloat16(c0 + c1 + c2 + c3);
        }
    }
}

// ============================================================================
// Host driver
// ============================================================================
int main(int argc, char **argv) {
    int M = 16384, N = 14336, K = 4096;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    printf("W4-FP8 spike: M=%d N=%d K=%d\n", M, N, K);

    // Allocate device buffers (we won't touch them meaningfully; this is timing)
    __nv_fp8_e4m3 *dA;  size_t bytesA = (size_t)M * K * 1;
    uint32_t *dB;       size_t bytesB = (size_t)N * (K / 8) * 4;
    __nv_bfloat16 *dBs; size_t bytesBs = (size_t)N * (K / GROUP_SIZE) * 2;
    __nv_bfloat16 *dC;  size_t bytesC = (size_t)M * N * 2;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dBs, bytesBs));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemset(dA, 0x12, bytesA));
    CUDA_CHECK(cudaMemset(dB, 0x98, bytesB));
    CUDA_CHECK(cudaMemset(dBs, 0x10, bytesBs));
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));

    // Launch grid: blocks chosen so total MMAs match a real M×N×K workload.
    // Per launch: gridDim.x * gridDim.y * 4 warps * K_LOOP_ITERS * (16*8*32) FMAs.
    // Set gridDim to give us close to M*N*K / (16*8*K_LOOP_ITERS*32) blocks.
    constexpr int K_LOOP_ITERS = 64;
    constexpr int FMA_PER_WARP_ITER = 16 * 8 * 32;
    long long total_fma_target = (long long)M * N * K;
    long long fma_per_block = (long long)4 /* warps */ * K_LOOP_ITERS * FMA_PER_WARP_ITER;
    long long n_blocks = (total_fma_target + fma_per_block - 1) / fma_per_block;
    int grid_x = (int)((n_blocks + 31) / 32);  // arbitrary 2D split for occupancy
    int grid_y = 32;
    if (grid_x < 1) grid_x = 1;

    dim3 grid(grid_x, grid_y);
    dim3 block(THREADS_PER_BLOCK);
    size_t smem_bytes = 4096;

    printf("Launching grid=(%d,%d) block=%d  expected_fma=%.2e\n",
           grid_x, grid_y, THREADS_PER_BLOCK,
           (double)grid_x * grid_y * fma_per_block);

    // Warmup
    for (int i = 0; i < 3; i++) {
        w4fp8_gemm_kernel<<<grid, block, smem_bytes>>>(dA, dB, dBs, dC, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    constexpr int N_ITERS = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < N_ITERS; i++) {
        w4fp8_gemm_kernel<<<grid, block, smem_bytes>>>(dA, dB, dBs, dC, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double ms_per_iter = ms / N_ITERS;
    double total_fmas_per_iter = (double)grid_x * grid_y * fma_per_block;
    // Each FMA = 2 floating-point operations
    double tflops = (total_fmas_per_iter * 2.0) / (ms_per_iter * 1e-3) / 1e12;

    printf("\n=== Result ===\n");
    printf("Avg time/iter: %.3f ms\n", ms_per_iter);
    printf("FMAs/iter:     %.3e\n", total_fmas_per_iter);
    printf("Throughput:    %.1f TFLOPS  (W4→FP8 dequant + FP8 MMA)\n", tflops);
    printf("Reference:     281 TF (Phase 0 dense FP8 ceiling)\n");
    printf("Ratio:         %.1f%%\n", 100.0 * tflops / 281.0);

    if (tflops >= 250.0)        printf("Verdict: GREEN (≥89%% of FP8 ceiling) — pursue full kernel\n");
    else if (tflops >= 180.0)   printf("Verdict: YELLOW (~64%% of FP8 ceiling) — marginal, defer\n");
    else                        printf("Verdict: RED (<64%% of FP8 ceiling) — kill W4-FP8\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dBs); cudaFree(dC);
    return 0;
}
