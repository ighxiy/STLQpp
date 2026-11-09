#pragma once

#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/core/status.h"
#include "stlqpp/model/model_shape.h"

namespace stlqpp {

struct RuntimeRootCodebookView final {
  ModelShape shape;
  ColumnMajorMatrixF32View root0;
  ColumnMajorMatrixF32View root_small_flat;
  [[nodiscard]] Status Validate() const;
};

// Borrowed, contiguous model view. The stride is always the active runtime M;
// no padding to a compiled code length is permitted.
struct RuntimeProductCodebookView final {
  ModelShape shape;
  ColumnMajorMatrixF32View root0;
  ColumnMajorMatrixF32View root_small_flat;
  ColumnMajorMatrixF32View edge_flat;

  [[nodiscard]] ColumnMajorMatrixF32View root_small(
      std::uint32_t layer) const noexcept {
    return {root_small_flat.column((layer - 1U) * 256U), shape.dimension, 256U};
  }
  [[nodiscard]] ColumnMajorMatrixF32View edge(
      std::uint32_t layer) const noexcept {
    return {edge_flat.column(layer * 256U), shape.dimension, 256U};
  }
  [[nodiscard]] Status Validate() const;
  [[nodiscard]] RuntimeRootCodebookView roots() const noexcept {
    return {shape, root0, root_small_flat};
  }
};

}  // namespace stlqpp
