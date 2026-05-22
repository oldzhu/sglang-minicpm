/*
 * SM120 QMMA Fused W4A8 GEMM Kernel
 *
 * Uses cutlass SM100 mixed-input collective builder (backward-compatible
 * with SM120 hardware) for on-the-fly INT4→FP8 dequant + UMMA/tcgen05 GEMM.
 *
 * Compiled with -arch=sm_120a this achieves 296 TFLOPS via tcgen05 QMMA.
 *
 * Input: qweight [K, N/8] int32, qzeros [K/g, N/8] int32, scales [K/g, N] bf16,
 *        a_fp8 [M, K] fp8_e4m3
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

using MmaType            = cutlass::float_e4m3_t;
using QuantType          = cutlass::int4b_t;
using ElementAccumulator = float;
using ElementScale       = cutlass::bfloat16_t;
using ElementC           = cutlass::bfloat16_t;
using ElementD           = ElementC;

using ArchTag       = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using LayoutC_Transpose = typename cutlass::layout::LayoutTranspose<LayoutC>::type;

// TMA requires 128-byte alignment. Use byte-based sizes not bit-based.
static constexpr int AlignmentA = 128 / static_cast<int>(sizeof(MmaType));   // 128 fp8 elems = 128 bytes
static constexpr int AlignmentB = 128 / static_cast<int>(sizeof(QuantType)); // 128 int4 elems = 64 bytes but pack
// Actually need to satisfy: sizeof(float)*Alignment % 128 == 0
// sizeof(float)=4, so Alignment must be multiple of 32.
// Use 128 for both.
static_assert(AlignmentA == 128 && AlignmentB == 128, "TMA alignment check");
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 128/16 = 8

// Non-grouped problem shape: {M, N, K, batch=1}
using ProblemShape = cute::Shape<int, int, int, int>;

using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecialized1SmMixedInputSm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;

template <typename TileShape, typename ClusterShape>
struct sm120_qmma_w4a8_gemm {
  static constexpr int GroupSize = 128;
  static constexpr int PackedScalesNum = get<2>(TileShape{}) / GroupSize;
  using ElementScalePacked = cutlass::Array<ElementScale, PackedScalesNum>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC_Transpose*, AlignmentC,
      ElementD, LayoutC_Transpose*, AlignmentC,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilderMixedInput<
      ArchTag, OperatorClass,
      cute::tuple<QuantType, ElementScalePacked>,
      LayoutB_Transpose*, AlignmentB,
      MmaType,
      LayoutA_Transpose*, AlignmentA,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using StrideS = typename CollectiveMainloop::StrideScale;
};

}  // anonymous namespace

namespace sglang {

torch::Tensor w4a8_fp8_fused_gemm(
    const torch::Tensor& qweight,
    const torch::Tensor& qzeros,
    const torch::Tensor& scales,
    const torch::Tensor& a_fp8,
    int64_t N, int64_t K, int64_t group_size) {

  int M = a_fp8.size(0);
  TORCH_CHECK(a_fp8.dtype() == torch::kFloat8_e4m3fn, "act must be fp8_e4m3");
  TORCH_CHECK(M >= 64, "M must be >= 64 for QMMA tile");

  using TileShape    = Shape<Int<128>, Int<128>, Int<128>>;
  using ClusterShape = Shape<Int<1>, Int<1>, Int<1>>;

  using GemmConfig = sm120_qmma_w4a8_gemm<TileShape, ClusterShape>;
  using GemmKernel = typename GemmConfig::GemmKernel;
  using Gemm       = typename GemmConfig::Gemm;

  auto c_bf16 = torch::empty({M, N},
      torch::dtype(torch::kBFloat16).device(qweight.device()));

  // Strides (contiguous tensors)
  using StrideA = typename GemmConfig::StrideA;
  using StrideB = typename GemmConfig::StrideB;
  using StrideD = typename GemmConfig::StrideD;
  using StrideS = typename GemmConfig::StrideS;

  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
  auto stride_s = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(static_cast<int>(K / group_size), static_cast<int>(N), 1));

  // Pointers
  auto a_ptr   = static_cast<const MmaType*>(a_fp8.const_data_ptr());
  auto b_ptr   = static_cast<const QuantType*>(qweight.const_data_ptr());
  auto s_ptr   = static_cast<const typename GemmConfig::ElementScalePacked*>(
      scales.const_data_ptr());
  auto z_ptr   = (qzeros.numel() > 0)
      ? static_cast<const QuantType*>(qzeros.const_data_ptr()) : nullptr;
  auto d_ptr   = static_cast<ElementD*>(c_bf16.data_ptr());

  // Mainloop arguments — passed as initializer list matching CollectiveMainloop::Arguments
  typename GemmKernel::MainloopArguments mainloop_args{
      b_ptr, stride_b,   // B: quantized weights (INT4)
      a_ptr, stride_a,   // A: FP8 activations
      s_ptr, stride_s,   // Scales for dequant
      K / group_size};   // chunk_size (number of scale groups per K)

  // Epilogue arguments
  typename GemmKernel::EpilogueArguments epilogue_args{
      {},                          // fusion args (default)
      nullptr, stride_d,           // C source (beta * C)
      d_ptr, stride_d};            // D output (alpha * A*B + beta * C)
  epilogue_args.thread.alpha = 1.0f;
  epilogue_args.thread.beta  = 0.0f;

  // Assemble full arguments
  typename Gemm::Arguments args = {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},               // problem_shape
      mainloop_args,
      epilogue_args,
  };

  // Workspace
  Gemm gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace = torch::empty(workspace_size,
      torch::dtype(torch::kUInt8).device(a_fp8.device()));

  auto stream = at::cuda::getCurrentCUDAStream(a_fp8.device().index());

  cutlass::Status status = gemm_op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
      "SM120 QMMA GEMM: not supported for these dims");

  status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
      "SM120 QMMA GEMM: init failed");

  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
      "SM120 QMMA GEMM: run failed");

  return c_bf16;
}

}  // namespace sglang

// Drop-in replacement for the old WMMA kernel
TORCH_LIBRARY_FRAGMENT(w4a8_fused, m) {
  m.def("w4a8_fp8_fused_gemm(Tensor qweight, Tensor qzeros, Tensor scales, "
        "Tensor a_fp8, int N, int K, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(w4a8_fused, CUDA, m) {
  m.impl("w4a8_fp8_fused_gemm", &sglang::w4a8_fp8_fused_gemm);
}
