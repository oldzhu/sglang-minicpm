/*
 * SM120 Mixed-Input Collective Builder for W4A8 GEMM
 *
 * Extends the cutlass SM120 collective builder to support mixed-width
 * operands (INT4 weights + FP8 activations) with on-the-fly dequant.
 *
 * Based on:
 *   - sm90_gmma_builder_mixed_input.inl (SM90 pattern, from sgl-kernel)
 *   - sm120_mma_builder.inl (SM120 non-block-scaled, from NVIDIA cutlass)
 *
 * Key differences from SM90:
 *   - Uses rr_op_selector_sm120 instead of GMMA::rs_op_selector
 *   - Uses SM120 SMEM capacity (101KB) vs SM90 (232KB)
 *   - Uses sm120_rr_smem_selector / sm120_rr_smem_copy_selector
 *   - tcgen05.mma (warp-group FP8 QMMA at 296 TFLOPS)
 *
 * This builder handles the case where B operand is INT4 with per-group
 * scales and zero points. The INT4 data is loaded as uint8_t via TMA,
 * dequantized to FP8 in the mainloop transform phase, then fed into
 * tcgen05.mma.
 */

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/builders/sm120_common.inl"
#include "cutlass/gemm/collective/builders/sm120_mma_builder.inl"
#include "cutlass/gemm/collective/collective_builder_decl.hpp"
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

// SM120 Mixed-Input: handles INT4 weights (ElementB = tuple<int4b_t, float, int4b_t>)
// and FP8 activations (ElementA = float_e4m3_t).
// The narrow INT4 weights are TMA-loaded as uint8_t, then dequantized to FP8 in the
// register-to-register transform phase before MMA.
//
// This is a specialization for W4A8 where A is the wider type (FP8, 8 bits) and
// B is the narrow type (INT4, 4 bits).
template <
    class ElementA_,
    class GmemLayoutATag_,
    int AlignmentA,
    class ElementB_,
    class GmemLayoutBTag_,
    int AlignmentB,
    class ElementAccumulator,
    class TileShape_MNK,
    class ClusterShape_MNK,
    class StageCountType,
    class KernelScheduleType>
struct CollectiveBuilderMixedInput<
    arch::Sm120,
    arch::OpClassTensorOp,
    ElementA_,
    GmemLayoutATag_,
    AlignmentA,
    ElementB_,
    GmemLayoutBTag_,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
        (cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecialized> ||
         cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpong> ||
         cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative>) &&
        (sizeof_bits<detail::deduce_mixed_width_dtype_t<0, ElementA_>>::value !=
         sizeof_bits<detail::deduce_mixed_width_dtype_t<0, ElementB_>>::value
         || cute::is_tuple<ElementB_>::value)>> {

 private:
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementB_>;
  using ZeroB  = detail::deduce_mixed_width_dtype_t<2, ElementB_>;

 public:
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementA_>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementB_>;

  // For W4A8: B (INT4) is narrower than A (FP8)
  static constexpr bool IsBNarrow = sizeof_bits<ElementB>::value < sizeof_bits<ElementA>::value;
  static_assert(IsBNarrow || cute::is_tuple<ElementB_>::value,
      "SM120 mixed-input: B must be narrower than A (W4A8)");

  using GmemLayoutATag = GmemLayoutATag_;
  using GmemLayoutBTag = GmemLayoutBTag_;

  // B is the narrow operand — wrap with scale/zero for dequant
  using ElementPairA = ElementA_;
  using ElementPairB = cute::conditional_t<
      IsBNarrow && !cute::is_tuple<ElementB_>::value,
      cute::tuple<ElementB, ScaleB, ZeroB>,
      ElementB_>;

  using ElementScale = ScaleB;
  using ElementZero  = ZeroB;

  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);

  static constexpr cute::UMMA::Major UmmaMajorA = detail::sm1xx_tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute::UMMA::Major UmmaMajorB = detail::sm1xx_tag_to_umma_major_B<GmemLayoutBTag>();

  // No operand swap needed — B is always the narrow one
  static constexpr bool SwapAB = false;

  // MMA input types: both become FP8 e4m3 after dequant
  using ElementAMma = float_e4m3_t;
  using ElementBMma = float_e4m3_t;

  // For tcgen05.mma f8f6f4, use uint8_t for SMEM allocation (sub-byte)
  using SmemAllocTypeA = uint8_t;
  using SmemAllocTypeB = uint8_t;

  // TMA copy atoms
  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  // SMEM layout selection (SM120-specific)
  using SmemLayoutAtomA = decltype(detail::sm120_rr_smem_selector<
      SmemAllocTypeA, decltype(size<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::sm120_rr_smem_selector<
      SmemAllocTypeB, decltype(size<2>(TileShape_MNK{}))>());

  // SMEM copy atoms for SM120
  static constexpr bool UseF8f6f4 = true;  // FP8 QMMA
  using SmemCopyAtomA = Copy_Atom<
      decltype(detail::sm120_rr_smem_copy_selector_A<ElementA, ElementB, UseF8f6f4>()),
      SmemAllocTypeA>;
  using SmemCopyAtomB = Copy_Atom<
      decltype(detail::sm120_rr_smem_copy_selector_B<ElementA, ElementB, UseF8f6f4>()),
      SmemAllocTypeB>;

  // Tiled MMA: tcgen05 QMMA for FP8
  using AtomLayoutMNK = cute::conditional_t<
      cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative>,
      Layout<Shape<_2, _2, _1>>,
      Layout<Shape<_1, _1, _1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(
      cute::rr_op_selector_sm120<ElementAMma, ElementBMma, ElementAccumulator>(),
      AtomLayoutMNK{}));

  // SMEM capacity for SM120
  static constexpr int KernelSmemCarveout = 0;  // No extra carveout needed for mixed-input
  static constexpr int Sm120ReducedSmemCapacityBytes =
      detail::sm120_smem_capacity_bytes - KernelSmemCarveout;

  // Pipeline stages
  static constexpr int PipelineStages =
      detail::sm120_compute_stage_count_or_override_mixed_input<
          Sm120ReducedSmemCapacityBytes,
          SmemAllocTypeA, SmemAllocTypeB,
          TileShape_MNK,
          ElementScale, ElementZero>(StageCountType{});

  // Dispatch policy
  using DispatchPolicy = MainloopSm120TmaRrWarpSpecializedMixedInput<
      PipelineStages, ClusterShape_MNK, KernelScheduleType>;

  // Strides
  using StrideA = cute::conditional_t<
      cute::is_layout<cute::remove_pointer_t<GmemLayoutATag_>>::value,
      GmemLayoutATag_,
      TagToStrideA_t<GmemLayoutATag>>;
  using StrideB = cute::conditional_t<
      cute::is_layout<cute::remove_pointer_t<GmemLayoutBTag_>>::value,
      GmemLayoutBTag_,
      TagToStrideB_t<GmemLayoutBTag>>;

  // CollectiveOp — use the standard CollectiveMma, not the array variant
  // (mixed-input transform happens via ElementPairB tuple)
  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementPairA,
      StrideA,
      ElementPairB,
      StrideB,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,       // TransformA = identity (FP8→FP8)
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity>;      // TransformB = identity (dequant handled by ElementPairB)
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
