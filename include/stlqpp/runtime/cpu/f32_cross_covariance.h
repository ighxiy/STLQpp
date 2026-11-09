#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/core/numeric/column_major_matrix.h"

#include <cstdint>

namespace stlqpp {

// output += left * right^T for two (dimension x sample_count)
// column-major blocks.
struct CrossCovarianceF32 final {
  ColumnMajorMatrixF32View left;
  ColumnMajorMatrixF32View right;
  float* output = nullptr;
  std::uint32_t output_leading_dimension = 0U;
};

class F32CrossCovarianceBackend {
 public:
  virtual ~F32CrossCovarianceBackend() = default;
  F32CrossCovarianceBackend(const F32CrossCovarianceBackend&) = delete;
  F32CrossCovarianceBackend& operator=(const F32CrossCovarianceBackend&) =
      delete;

  [[nodiscard]] virtual Status Accumulate(
      const CrossCovarianceF32& operation) const = 0;

 protected:
  F32CrossCovarianceBackend() = default;
};

}  // namespace stlqpp
