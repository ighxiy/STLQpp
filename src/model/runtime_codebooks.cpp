#include "stlqpp/model/runtime_codebooks.h"

#include <cmath>

namespace stlqpp {

namespace {
Status ValidateMatrix(ColumnMajorMatrixF32View matrix, std::uint32_t rows,
                      std::uint32_t columns) {
  if (matrix.values == nullptr || matrix.rows != rows ||
      matrix.columns != columns)
    return Status::InvalidArgument("runtime codebook matrix shape mismatch");
  for (std::size_t i = 0; i < static_cast<std::size_t>(rows) * columns; ++i) {
    if (!std::isfinite(matrix.values[i]))
      return Status::InvalidArgument(
          "runtime codebook contains non-finite values");
  }
  return Status::Ok();
}
}  // namespace

Status RuntimeRootCodebookView::Validate() const {
  if (const auto status = shape.Validate(); !status.ok()) return status;
  if (const auto s = ValidateMatrix(root0, shape.dimension, shape.nlist);
      !s.ok())
    return s;
  return ValidateMatrix(root_small_flat, shape.dimension,
                        (shape.m - 1U) * 256U);
}

Status RuntimeProductCodebookView::Validate() const {
  if (const auto s = roots().Validate(); !s.ok()) return s;
  return ValidateMatrix(edge_flat, shape.dimension, shape.m * 256U);
}

}  // namespace stlqpp
