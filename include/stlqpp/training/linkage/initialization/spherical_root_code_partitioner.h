#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/linkage/initialization/root_to_edge_code_map.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

// Fixed legacy mapping used exactly once between root-only topology and the
// first C_one solve. It normalizes C_root[0], runs deterministic 256-way
// spherical k-means++ for at most 30 iterations, and returns the final root
// assignments. The caller supplies only the semantic training seed.
class SphericalRootCodePartitioner final {
 public:
  explicit SphericalRootCodePartitioner(
      const ParallelForExecutor& parallel) noexcept
      : parallel_(&parallel) {}

  [[nodiscard]] StatusOr<RootToEdgeCodeMap> Build(
      ColumnMajorMatrixF32View root_codebook,
      std::uint32_t training_seed) const;

  // The regular two-codebook initializer needs the same deterministic
  // centroid fit as an actual C_one[0] seed, not only its root assignments.
  [[nodiscard]] StatusOr<std::vector<float>> BuildEdgeSeed(
      ColumnMajorMatrixF32View root_codebook,
      std::uint32_t training_seed) const;

 private:
  [[nodiscard]] StatusOr<RootToEdgeCodeMap> BuildImpl(
      ColumnMajorMatrixF32View root_codebook,
      std::uint32_t training_seed,
      std::vector<float>* centers_output) const;

  const ParallelForExecutor* parallel_ = nullptr;
};

}  // namespace stlqpp
