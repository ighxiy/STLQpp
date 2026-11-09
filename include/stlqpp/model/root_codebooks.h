#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/model/model_layers.h"
#include "stlqpp/model/runtime_codebooks.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {

template <int M>
struct RootCodebookView final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  ModelShape shape;
  ColumnMajorMatrixF32View root0;
  ModelLayers<ColumnMajorMatrixF32View, M, 1> root_small{};
};

// Immutable C_root ownership boundary. Initialization owns exactly this state
// before the first C_one exists; complete product codebooks compose it rather
// than representing the missing edge books with dummy storage.
template <int M>
class RootCodebooks final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static Status ValidateView(
      const RootCodebookView<M>& source);
  [[nodiscard]] static StatusOr<RootCodebooks<M>> Create(
      const RootCodebookView<M>& source);

  RootCodebooks(RootCodebooks&&) noexcept = default;
  RootCodebooks& operator=(RootCodebooks&&) noexcept = default;
  RootCodebooks(const RootCodebooks&) = delete;
  RootCodebooks& operator=(const RootCodebooks&) = delete;

  [[nodiscard]] const ModelShape& shape() const noexcept { return shape_; }
  [[nodiscard]] std::uint32_t m() const noexcept {
    if constexpr (M != 0) return M;
    return shape_.m;
  }
  [[nodiscard]] RuntimeRootCodebookView runtime_view() const noexcept {
    return {shape_, root0(), root_small_flat()};
  }
  [[nodiscard]] ColumnMajorMatrixF32View root0() const noexcept {
    return {root0_.data(), shape_.dimension, shape_.nlist};
  }
  [[nodiscard]] ColumnMajorMatrixF32View root_small(
      std::uint32_t layer) const noexcept {
    const std::size_t offset = static_cast<std::size_t>(layer - 1U) *
                               256U * shape_.dimension;
    return {root_small_.data() + offset, shape_.dimension, 256U};
  }
  [[nodiscard]] ColumnMajorMatrixF32View root_small_flat() const noexcept {
    return {root_small_.data(), shape_.dimension,
            (m() - 1U) * 256U};
  }
  [[nodiscard]] ArrayView<const float> root_inverse_norms() const noexcept {
    return root_inverse_norms_;
  }
  [[nodiscard]] RootCodebookView<M> view() const noexcept(M != 0) {
    RootCodebookView<M> output;
    output.shape = shape_;
    output.root0 = root0();
    if constexpr (M == 0) output.root_small.resize(shape_.m - 1U);
    for (std::uint32_t layer = 1U;
         layer < m(); ++layer) {
      output.root_small[layer - 1U] = root_small(layer);
    }
    return output;
  }

 private:
  RootCodebooks() = default;

  ModelShape shape_;
  std::vector<float> root0_;
  std::vector<float> root_small_;
  std::vector<float> root_inverse_norms_;
};

using RuntimeRootCodebooks = RootCodebooks<0>;
extern template class RootCodebooks<0>;
extern template class RootCodebooks<5>;
extern template class RootCodebooks<10>;

}  // namespace stlqpp
