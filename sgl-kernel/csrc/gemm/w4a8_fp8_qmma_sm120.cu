/*
 * SM120 QMMA Fused W4A8 GEMM Kernel (v3 — rewritten with correct SM100 API)
 *
 * Uses SM100 CollectiveBuilder specialization (sm100_mixed_input_umma_builder.inl)
 * which is backward-compatible with SM120 hardware. Compile with -arch=sm_120a
 * for 296 TFLOPS via tcgen05 QMMA on Blackwell.
 *
 * Key API difference from SM90:
 *   - Layout tuples use plain types, NOT pointers: tuple<LayoutTag, LayoutTag>
 *   - ElementA = narrow/tuple type (our INT4 weights)
 *   - ElementB = wide type (our FP8 activations)
 *   - Arguments: ptr_A=weights, ptr_B=activations (A/B swapped vs SM90 convention)
 *
 * Input:  qweight [K, N/8] int32, qzeros [K/g, N/8] int32, scales [K/g, N] bf16,
 *         a_fp8 [M, K] fp8_e4m3
 * Output: [M, N] bf16
 */

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/library.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass_extensions/gemm/collective/collective_builder_mixed_input.hpp"

using namespace cute;

namespace {

// --- Type definitions ---
using MmaType            = cutlass::float_e4m3_t;    // FP8 activation (8-bit)
using QuantType          = cutlass::int4b_t;          // INT4 weight (4-bit packed)
using ElementAccumulator = float;                    // FP32 accumulate
using ElementScale       = cutlass::bfloat16_t;       // Scale storage type
using ElementC           = cutlass::bfloat16_t;       // Output type
using ElementD           = ElementC;

using ArchTag       = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

// Layout tags (plain types, no pointers)
using LayoutA = cutlass::layout::RowMajor;     // Activation: [M, K] row-major
using LayoutB = cutlass::layout::ColumnMajor;  // Weight: [K, N] column-major
using LayoutC = cutlass::layout::RowMajor;     // Output: [M, N] row-major

using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using LayoutC_Transpose = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
using LayoutScale = cutlass::layout::RowMajor;  // Scales: [K/g, N] row-major

// Alignment: TMA requires 128-byte boundaries. sizeof(float)*Alignment % 128 == 0
// sizeof(float)=4, so Alignment must be multiple of 32.
static constexpr int AlignmentA = 128;  // FP8: 128 elems = 128 bytes
static constexpr int AlignmentB = 128;  // int4: 128 storage units (packed)
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // bf16: 128/16=8

// Problem shape: {M, N, K, batch=1} for non-grouped GEMM
using ProblemShape = cute::Shape<int, int, int, int>;

// SM100 mixed-input schedule (must derive from KernelScheduleSm100MixedInputGemm)
using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecialized1SmMixedInputSm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;

// --- Kernel template ---
template <typename TileShape, typename ClusterShape>
struct sm120_qmma_w4a8_gemm {
  static constexpr int GroupSize = 128;
  static constexpr int PackedScalesNum = get<2>(TileShape{}) / GroupSize;
  using ElementScalePacked = cutlass::Array<ElementScale, PackedScalesNum>;

  // Epilogue: accumulator → bf16 output (SM100 TMA warp specialized)
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC_Transpose*, AlignmentC,
      ElementD, LayoutC_Transpose*, AlignmentC,
      EpilogueSchedule>::CollectiveOp;

  // Mainloop: INT4 weights (narrow/tuple side) × FP8 activations (wide side)
  // SM100 CollectiveBuilder maps:
  //   ElementA = narrow/tuple type  → our weights (QuantType + scale tuple)
  //   ElementB = wide type          → our activations (MmaType)
  //   GmemLayoutA tuple = (weight_layout, scale_layout) as plain types
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      cute::tuple<QuantType, ElementScalePacked>,          // ElementA: (INT4, packed_scale)
      cute::tuple<LayoutB_Transpose, LayoutScale>, AlignmentB, // LayoutA: (weight_layout, scale_layout)
      MmaType,                                               // ElementB: FP8 activation
      LayoutA_Transpose*, AlignmentA,                        // LayoutB: activation layout pointer
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Stride types (deduced automatically by builder from LayoutTags)
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using StrideS = typename CollectiveMainloop::StrideScale;
};

}  // anonymous namespace

// --- Torch entry point ---
namespace sglang {

torch::Tensor w4a8_fp8_fused_gemm(
    const torch::Tensor& qweight,   // [K, N/8] int32 — GPTQ packed INT4
    const torch::Tensor& qzeros,    // [K/g, N/8] int32 — packed zeros
    const torch::Tensor& scales,    // [K/g, N] bf16
    const torch::Tensor& a_fp8,     // [M, K] fp8_e4m3
    int64_t N, int64_t K, int64_t group_size) {

  int M = a_fp8.size(0);
  TORCH_CHECK(a_fp8.dtype() == torch::kFloat8_e4m3fn, "act must be fp8_e4m3");
  TORCH_CHECK(M >= 64, "M must be >= 64 for QMMA tile (min 64 for 128×128×128 tile)");

  using TileShape    = Shape<Int<128>, Int<128>, Int<128>>;
  using ClusterShape = Shape<Int<1>, Int<1>, Int<1>>;

  using Config     = sm120_qmma_w4a8_gemm<TileShape, ClusterShape>;
  using GemmKernel = typename Config::GemmKernel;
  using Gemm       = typename Config::Gemm;

  // Output tensor
  auto c_bf16 = torch::empty({M, N},
      torch::dtype(torch::kBFloat16).device(qweight.device()));

  // --- Pointers ---
  // SM100 mapping: ptr_A = weight data (narrow side), ptr_B = activation data (wide side)
  auto ptr_weight = static_cast<const QuantType*>(qweight.const_data_ptr());
  auto ptr_act    = static_cast<const MmaType*>(a_fp8.const_data_ptr());
  auto ptr_scale  = static_cast<const typename Config::ElementScalePacked*>(
      scales.const_data_ptr());
  auto ptr_zero   = (qzeros.numel() > 0)
      ? static_cast<const QuantType*>(qzeros.const_data_ptr()) : nullptr;
  auto ptr_out    = static_cast<ElementD*>(c_bf16.data_ptr());

  // --- Strides ---
  using StrideA = typename Config::StrideA;  // weight stride
  using StrideB = typename Config::StrideB;  // activation stride
  using StrideD = typename Config::StrideD;  // output stride
  using StrideS = typename Config::StrideS;  // scale stride

  auto stride_weight = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(N, K, 1));
  auto stride_act    = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(M, K, 1));
  auto stride_out    = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
  auto stride_scale  = cutlass::make_cute_packed_stride(StrideS{},
      cute::make_shape(static_cast<int>(K / group_size), static_cast<int>(N), 1));

  // --- Mainloop arguments (SM100 CollectiveMma::Arguments) ---
  typename GemmKernel::MainloopArguments mainloop_args{
      ptr_weight, stride_weight,   // A: INT4 weights (narrow/tuple side)
      ptr_act,    stride_act,      // B: FP8 activations (wide side)
      ptr_scale,  stride_scale,    // Scale factor (for dequant of A)
      ptr_zero};                   // Zero point (optional, nullptr if unused)

  // --- Epilogue arguments ---
  typename GemmKernel::EpilogueArguments epilogue_args{
      {},                             // fusion args (default)
      nullptr, stride_out,            // C source (beta * C, nullptr = no source)
      ptr_out, stride_out};           // D output (alpha * A*B + beta * C)
  epilogue_args.thread.alpha = 1.0f;
  epilogue_args.thread.beta  = 0.0f;

  // --- Assemble full arguments ---
  typename Gemm::Arguments args = {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},                    // problem_shape
      mainloop_args,
      epilogue_args,
  };

  // --- Launch ---
  Gemm gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace = torch::empty(workspace_size,
      torch::dtype(torch::kUInt8).device(a_fp8.device()));

  auto stream = at::cuda::getCurrentCUDAStream(a_fp8.device().index());

  cutlass::Status status = gemm_op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
      "SM120 QMMA GEMM: not supported for these dims (M=", M, " N=", N, " K=", K, ")");

  status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
      "SM120 QMMA GEMM: init failed");

  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
      "SM120 QMMA GEMM: run failed");

  return c_bf16;
}

}  // namespace sglang

// --- Torch op registration (drop-in replacement for old WMMA kernel) ---
TORCH_LIBRARY_FRAGMENT(w4a8_fused, m) {
  m.def("w4a8_fp8_fused_gemm(Tensor qweight, Tensor qzeros, Tensor scales, "
        "Tensor a_fp8, int N, int K, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(w4a8_fused, CUDA, m) {
  m.impl("w4a8_fp8_fused_gemm", &sglang::w4a8_fp8_fused_gemm);
}
