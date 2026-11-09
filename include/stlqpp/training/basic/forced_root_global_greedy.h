#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/training/basic/large_root_algebra.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

[[nodiscard]] std::vector<float> ComputeVirtualRootSampleNorms2(
    ColumnMajorMatrixF32View samples);

// DEV's exact initializer for UMAP virtual centers.  Layer zero is fixed to
// the owning CID; each remaining layer is selected globally by normalized
// absolute residual correlation.
template <int M>
struct ForcedRootGlobalGreedyResult final {
  std::vector<std::uint8_t> small_codes;
  std::vector<float> root_dots;
  // Immutable small-book projection, column-major [small_count, samples].
  // Refinement consumes the same projection used to initialize the support.
  std::vector<float> small_dots;
};

template <int M>
[[nodiscard]] StatusOr<ForcedRootGlobalGreedyResult<M>>
InitializeForcedRootGlobalGreedy(
    ColumnMajorMatrixF32View samples,
    std::uint32_t fixed_root,
    const BasicCodebookAlgebra<M>& algebra,
    const F32GemmBackend& gemm);

extern template StatusOr<ForcedRootGlobalGreedyResult<5>>
InitializeForcedRootGlobalGreedy<5>(
    ColumnMajorMatrixF32View,
    std::uint32_t,
    const BasicCodebookAlgebra<5>&,
    const F32GemmBackend&);
extern template StatusOr<ForcedRootGlobalGreedyResult<10>>
InitializeForcedRootGlobalGreedy<10>(
    ColumnMajorMatrixF32View,
    std::uint32_t,
    const BasicCodebookAlgebra<10>&,
    const F32GemmBackend&);

}  // namespace stlqpp
