#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/training/io/basic_block_source.h"
#include "stlqpp/training/initialization/cuda_spherical_assignment.h"
#include "stlqpp/training/initialization/cuda_normalized_source_cache.h"
#include "stlqpp/training/initialization/root_model_initializer.h"
#include "stlqpp/training/initialization/rvq_assignment_store.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

struct CudaStreamingSphericalLloydOptions final {
  std::size_t maximum_score_bytes = 256U * 1024U * 1024U;
  std::size_t maximum_normalized_cache_bytes =
      static_cast<std::size_t>(20U) * 1024U * 1024U * 1024U;
  bool collect_metrics = true;
};

struct CudaStreamingSphericalLloydResult final {
  std::vector<float> centers;
  SphericalRvqLayerStatistics statistics;
};

// Exact flat Lloyd/annealing driver for one already-seeded root layer. The
// large-layer seeder/accelerator is a separate owner; this class fixes the
// repeated full-source update semantics and final retained-code barrier.
template <typename Scalar = std::uint8_t>
class CudaStreamingSphericalLloyd final {
 public:
  [[nodiscard]] static StatusOr<CudaStreamingSphericalLloydResult> Train(
      CudaContext& context,
      const ParallelForExecutor& parallel,
      InitializationBasicBlockSourceFactory<Scalar>& source,
      const SphericalRvqInitializationRequest& request,
      ArrayView<const ColumnMajorMatrixF32View> prior_codebooks,
      std::uint32_t layer,
      RvqAssignmentStore& assignments,
      ColumnMajorMatrixF32View initial_centers,
      ColumnMajorMatrixF32View empty_cluster_fallback,
      CudaStreamingSphericalLloydOptions options);

  // Backend-owned form used across a complete M-layer RVQ initialization.
  // The normalized raw source is invariant across layers; residual projection
  // remains layer-local and is applied to mutable cache slots.
  [[nodiscard]] static StatusOr<CudaStreamingSphericalLloydResult>
  TrainWithNormalizedSource(
      CudaContext& context,
      const ParallelForExecutor& parallel,
      CudaNormalizedSourceCache<Scalar>& normalized_source,
      const SphericalRvqInitializationRequest& request,
      ArrayView<const ColumnMajorMatrixF32View> prior_codebooks,
      std::uint32_t layer,
      RvqAssignmentStore& assignments,
      ColumnMajorMatrixF32View initial_centers,
      ColumnMajorMatrixF32View empty_cluster_fallback,
      CudaStreamingSphericalLloydOptions options,
      double normalized_cache_wall_seconds = 0.0);
};

extern template class CudaStreamingSphericalLloyd<std::uint8_t>;
extern template class CudaStreamingSphericalLloyd<float>;

}  // namespace stlqpp
