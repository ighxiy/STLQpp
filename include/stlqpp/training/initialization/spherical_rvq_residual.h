#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/basic/basic_pipeline.h"
#include "stlqpp/training/initialization/rvq_assignment_store.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

struct SphericalRvqResidualBlock final {
  std::uint64_t first_id = 0U;
  std::uint32_t dimension = 0U;
  std::uint32_t count = 0U;
  std::vector<float> values;

  [[nodiscard]] ColumnMajorMatrixF32View view() const noexcept {
    return {values.data(), dimension, count};
  }
};

// CPU semantic oracle for the exact residual consumed by one spherical RVQ
// layer. Codebooks and assignments outlive this non-owning projector.
template <typename Scalar = std::uint8_t>
class SphericalRvqResidualProjector final {
 public:
  [[nodiscard]] static StatusOr<SphericalRvqResidualProjector<Scalar>> Create(
      const ModelShape& shape,
      ArrayView<const ColumnMajorMatrixF32View> codebooks,
      std::uint32_t prior_layer_count,
      const RvqAssignmentStore* assignments,
      const ParallelForExecutor& parallel);

  [[nodiscard]] StatusOr<SphericalRvqResidualBlock> Project(
      const BasicInputBlock<Scalar>& input) const;

 private:
  SphericalRvqResidualProjector() = default;

  ModelShape shape_;
  std::vector<ColumnMajorMatrixF32View> codebooks_{};
  std::uint32_t prior_layer_count_ = 0U;
  const RvqAssignmentStore* assignments_ = nullptr;
  const ParallelForExecutor* parallel_ = nullptr;
};

extern template class SphericalRvqResidualProjector<std::uint8_t>;
extern template class SphericalRvqResidualProjector<float>;

}  // namespace stlqpp
