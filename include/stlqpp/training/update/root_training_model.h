#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/model/root_codebooks.h"

#include <utility>
#include <vector>

namespace stlqpp {

// Immutable pre-C_one model. This is the only valid model state between the
// RVQ-like initialization and the first root-only topology solve.
template <int M>
class RootTrainingModel final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<RootTrainingModel<M>> Create(
      ColumnMajorMatrixF32View rotation,
      const RootCodebookView<M>& codebooks);

  RootTrainingModel(RootTrainingModel&&) noexcept = default;
  RootTrainingModel& operator=(RootTrainingModel&&) noexcept = default;
  RootTrainingModel(const RootTrainingModel&) = delete;
  RootTrainingModel& operator=(const RootTrainingModel&) = delete;

  [[nodiscard]] ColumnMajorMatrixF32View rotation() const noexcept {
    return {rotation_.data(), codebooks_.shape().dimension,
            codebooks_.shape().dimension};
  }
  [[nodiscard]] const RootCodebooks<M>& codebooks() const noexcept {
    return codebooks_;
  }
  [[nodiscard]] RootCodebookView<M> codebook_view() const noexcept(M != 0) {
    return codebooks_.view();
  }

 private:
  RootTrainingModel(std::vector<float> rotation,
                    RootCodebooks<M> codebooks) noexcept
      : rotation_(std::move(rotation)), codebooks_(std::move(codebooks)) {}

  std::vector<float> rotation_;
  RootCodebooks<M> codebooks_;
};

using RuntimeRootTrainingModel = RootTrainingModel<0>;
extern template class RootTrainingModel<0>;
extern template class RootTrainingModel<5>;
extern template class RootTrainingModel<10>;

}  // namespace stlqpp
