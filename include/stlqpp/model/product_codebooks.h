#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/model/root_codebooks.h"
#include "stlqpp/model/runtime_codebooks.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {

template <int M>
struct ProductCodebookView final {
  ModelShape shape;
  ColumnMajorMatrixF32View root0;
  ModelLayers<ColumnMajorMatrixF32View, M, 1> root_small{};
  ModelLayers<ColumnMajorMatrixF32View, M> edge{};
};

// Immutable model-owned codebooks in the exact physical order used by query
// GEMMs. Root-small and edge books are flattened layer-major without padding.
template <int M>
class ProductCodebooks final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static Status ValidateView(
      const ProductCodebookView<M>& source);
  [[nodiscard]] static StatusOr<ProductCodebooks<M>> Create(
      const ProductCodebookView<M>& source);

  ProductCodebooks(ProductCodebooks&&) noexcept = default;
  ProductCodebooks& operator=(ProductCodebooks&&) noexcept = default;
  ProductCodebooks(const ProductCodebooks&) = delete;
  ProductCodebooks& operator=(const ProductCodebooks&) = delete;

  [[nodiscard]] const ModelShape& shape() const noexcept { return shape_; }
  [[nodiscard]] RuntimeProductCodebookView runtime_view() const noexcept {
    return {shape_, root0(), root_small_flat(), edge_flat()};
  }
  [[nodiscard]] ColumnMajorMatrixF32View root0() const noexcept {
    return root_.root0();
  }
  [[nodiscard]] ColumnMajorMatrixF32View root_small(
      std::uint32_t layer) const noexcept {
    return root_.root_small(layer);
  }
  [[nodiscard]] ColumnMajorMatrixF32View edge(
      std::uint32_t layer) const noexcept {
    const std::size_t offset = static_cast<std::size_t>(layer) *
                               256U * shape_.dimension;
    return {edge_.data() + offset, shape_.dimension, 256U};
  }
  [[nodiscard]] ColumnMajorMatrixF32View root_small_flat() const noexcept {
    return root_.root_small_flat();
  }
  [[nodiscard]] ColumnMajorMatrixF32View edge_flat() const noexcept {
    return {edge_.data(), shape_.dimension,
            root_.m() * 256U};
  }
  [[nodiscard]] ArrayView<const float> root_inverse_norms() const noexcept {
    return root_.root_inverse_norms();
  }
  [[nodiscard]] const RootCodebooks<M>& root_codebooks() const noexcept {
    return root_;
  }
  [[nodiscard]] RootCodebookView<M> root_view() const noexcept(M != 0) {
    return root_.view();
  }
  [[nodiscard]] ProductCodebookView<M> view() const noexcept(M != 0) {
    ProductCodebookView<M> output;
    output.shape = shape_;
    output.root0 = root0();
    if constexpr (M == 0) {
      output.root_small.resize(shape_.m - 1U);
      output.edge.resize(shape_.m);
    }
    for (std::uint32_t layer = 1U;
         layer < root_.m(); ++layer) {
      output.root_small[layer - 1U] = root_small(layer);
    }
    for (std::uint32_t layer = 0U;
         layer < root_.m(); ++layer) {
      output.edge[layer] = edge(layer);
    }
    return output;
  }

 private:
  ProductCodebooks(RootCodebooks<M> root,
                   std::vector<float> edge) noexcept
      : shape_(root.shape()), root_(std::move(root)), edge_(std::move(edge)) {}

  ModelShape shape_;
  RootCodebooks<M> root_;
  std::vector<float> edge_;
};

using RuntimeProductCodebooks = ProductCodebooks<0>;
extern template class ProductCodebooks<0>;
extern template class ProductCodebooks<5>;
extern template class ProductCodebooks<10>;

}  // namespace stlqpp
