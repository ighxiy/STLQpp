#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/training/io/basic_block_source.h"
#include "stlqpp/training/initialization/root_model_initializer.h"
#include "stlqpp/training/initialization/rvq_assignment_store.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace stlqpp {

struct CudaKmeansParallelSeedOptions final {
  std::size_t maximum_score_bytes = 256U * 1024U * 1024U;
  std::uint32_t candidate_batch = 256U;
  std::uint32_t score_tile_columns = 8'192U;
};

struct CudaKmeansParallelSeedStatistics final {
  std::uint64_t sample_start = 0U;
  std::uint32_t sample_count = 0U;
  std::uint32_t rounds = 0U;
  std::uint32_t picked_raw = 0U;
  std::uint32_t picked_new = 0U;
  std::uint32_t unique_candidates = 0U;
  std::uint32_t padded_candidates = 0U;
  std::uint32_t capped_candidates = 0U;
  std::uint32_t reduced_candidates = 0U;
  double sample_wall_seconds = 0.0;
  double candidate_wall_seconds = 0.0;
  double reduction_wall_seconds = 0.0;
  double fallback_wall_seconds = 0.0;
};

struct CudaKmeansParallelSeedResult final {
  std::vector<float> centers;
  std::vector<float> empty_cluster_fallback;
  CudaKmeansParallelSeedStatistics statistics;
};

// Production GPU-first kmeans|| seeding for one spherical RVQ layer. The
// bounded raw sample and candidate generation stay device resident; only the
// reduced candidate pool crosses to the deterministic CPU k-means++ reducer.
template <typename Scalar = std::uint8_t>
class CudaKmeansParallelSeeder final {
 public:
  [[nodiscard]] static StatusOr<CudaKmeansParallelSeedResult> Seed(
      CudaContext& context,
      InitializationBasicBlockSourceFactory<Scalar>& source,
      const SphericalRvqInitializationRequest& request,
      ArrayView<const ColumnMajorMatrixF32View> prior_codebooks,
      std::uint32_t layer,
      const RvqAssignmentStore& assignments,
      std::mt19937& random,
      const ParallelForExecutor& parallel,
      CudaKmeansParallelSeedOptions options);
};

extern template class CudaKmeansParallelSeeder<std::uint8_t>;
extern template class CudaKmeansParallelSeeder<float>;

}  // namespace stlqpp
