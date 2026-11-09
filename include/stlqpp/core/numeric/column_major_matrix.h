#pragma once

#include <cstddef>
#include <cstdint>

namespace stlqpp {

// Immutable column-major matrix view shared by model, runtime and training.
// Vectors/codewords are columns, so one column is contiguous.
template <typename Scalar>
struct ColumnMajorMatrixView final {
  const Scalar* values = nullptr;
  std::uint32_t rows = 0U;
  std::uint32_t columns = 0U;

  [[nodiscard]] const Scalar* column(std::uint32_t index) const noexcept {
    return values + static_cast<std::size_t>(index) * rows;
  }
};

using ColumnMajorMatrixF32View = ColumnMajorMatrixView<float>;
using ColumnMajorMatrixU8View = ColumnMajorMatrixView<std::uint8_t>;

}  // namespace stlqpp
