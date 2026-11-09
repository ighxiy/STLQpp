#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/model/product_codebooks.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {

// Immutable float model observed by one complete training round. The rotation
// is retained in its canonical R form; query-side R^T is derived only when an
// index is opened. Returning a new instance is the publication boundary
// between rounds, so an encoder can never observe a partially updated model.
template <int M>
class TrainingModel final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<TrainingModel<M>> Create(
      ColumnMajorMatrixF32View rotation,
      const ProductCodebookView<M>& codebooks);

  TrainingModel(TrainingModel&&) noexcept = default;
  TrainingModel& operator=(TrainingModel&&) noexcept = default;
  TrainingModel(const TrainingModel&) = delete;
  TrainingModel& operator=(const TrainingModel&) = delete;

  [[nodiscard]] ColumnMajorMatrixF32View rotation() const noexcept {
    return {rotation_.data(), codebooks_.shape().dimension,
            codebooks_.shape().dimension};
  }
  [[nodiscard]] const ProductCodebooks<M>& codebooks() const noexcept {
    return codebooks_;
  }
  [[nodiscard]] ProductCodebookView<M> codebook_view() const noexcept(M != 0) {
    return codebooks_.view();
  }

 private:
  TrainingModel(std::vector<float> rotation,
                ProductCodebooks<M> codebooks) noexcept
      : rotation_(std::move(rotation)), codebooks_(std::move(codebooks)) {}

  std::vector<float> rotation_;
  ProductCodebooks<M> codebooks_;
};

using RuntimeTrainingModel = TrainingModel<0>;
extern template class TrainingModel<0>;
extern template class TrainingModel<5>;
extern template class TrainingModel<10>;

}  // namespace stlqpp
