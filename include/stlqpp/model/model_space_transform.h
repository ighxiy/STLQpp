#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

// Immutable canonical column-major R. DEV applies the model transform as the
// non-transposed product R*X; retaining that BLAS operation is part of the
// numerical contract, not merely an algebraic equivalence.
class ModelSpaceTransform final {
 public:
  [[nodiscard]] static Status ValidateView(
      ColumnMajorMatrixF32View rotation);
  [[nodiscard]] static StatusOr<ModelSpaceTransform> Create(
      ColumnMajorMatrixF32View rotation);

  [[nodiscard]] static ModelSpaceTransform SourceSpace(std::uint32_t dimension) {
    ModelSpaceTransform transform;
    transform.dimension_ = dimension;
    return transform;
  }
  [[nodiscard]] bool source_space() const noexcept { return values_.empty(); }

  ModelSpaceTransform(ModelSpaceTransform&&) noexcept = default;
  ModelSpaceTransform& operator=(ModelSpaceTransform&&) noexcept = default;
  ModelSpaceTransform(const ModelSpaceTransform&) = delete;
  ModelSpaceTransform& operator=(const ModelSpaceTransform&) = delete;

  [[nodiscard]] std::uint32_t dimension() const noexcept {
    return dimension_;
  }
  [[nodiscard]] ColumnMajorMatrixF32View matrix() const noexcept {
    if (source_space()) return {};
    return {values_.data(), dimension_, dimension_};
  }

 private:
  ModelSpaceTransform() = default;

  std::uint32_t dimension_ = 0U;
  std::vector<float> values_;
};

}  // namespace stlqpp
