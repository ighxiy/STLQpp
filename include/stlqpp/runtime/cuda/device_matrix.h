#pragma once

#include <cstddef>
#include <cstdint>

namespace stlqpp {

// Borrowed column-major device matrix. The owner controls stream ordering and
// must keep the allocation alive until the consuming operation completes.
struct CudaMatrixF32View final {
  const float* values = nullptr;
  std::uint32_t rows = 0U;
  std::uint32_t columns = 0U;
  std::uint32_t leading_dimension = 0U;

  [[nodiscard]] const float* column(std::uint32_t index) const noexcept {
    return values + static_cast<std::size_t>(index) * leading_dimension;
  }
};

}  // namespace stlqpp
