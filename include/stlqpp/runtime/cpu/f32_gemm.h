#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/core/numeric/column_major_matrix.h"

#include <cstddef>
#include <cstdint>

namespace stlqpp {

// C = A B. A is a column-major (output_rows x shared_dimension)
// immutable model matrix. B and C are strided column-major blocks.
struct LeftGemmF32 final {
  ColumnMajorMatrixF32View left;
  const float* right = nullptr;
  std::uint32_t query_count = 0U;
  std::size_t right_leading_dimension = 0U;
  float* output = nullptr;
  std::uint32_t output_leading_dimension = 0U;
};

// C = A^T B. A is a column-major (shared_dimension x output_rows)
// immutable model matrix. B is a strided column-major query block. C is a
// column-major (output_rows x query_count) table block.
struct TransposedLeftGemmF32 final {
  ColumnMajorMatrixF32View left;
  const float* right = nullptr;
  std::uint32_t query_count = 0U;
  std::size_t right_leading_dimension = 0U;
  float* output = nullptr;
  std::uint32_t output_leading_dimension = 0U;
};

// Upper-triangle C = A^T A.  The untouched lower triangle is deliberately
// unspecified.  This is a distinct BLAS operation because the registered
// exact-parent candidate path uses SSYRK for a same-layer diagonal tile.
struct SelfTransposedUpperF32 final {
  ColumnMajorMatrixF32View input;
  float* output = nullptr;
  std::uint32_t output_leading_dimension = 0U;
};

class F32GemmBackend {
 public:
  virtual ~F32GemmBackend() = default;

  F32GemmBackend(const F32GemmBackend&) = delete;
  F32GemmBackend& operator=(const F32GemmBackend&) = delete;

  [[nodiscard]] virtual Status MultiplyLeft(
      const LeftGemmF32&) const {
    return Status::Unimplemented(
        "this F32 GEMM backend does not implement a non-transposed left operand");
  }

  [[nodiscard]] virtual Status MultiplyTransposedLeft(
      const TransposedLeftGemmF32& operation) const = 0;

  [[nodiscard]] virtual Status MultiplySelfTransposedUpper(
      const SelfTransposedUpperF32& operation) const {
    return MultiplyTransposedLeft({
        .left = operation.input,
        .right = operation.input.values,
        .query_count = operation.input.columns,
        .right_leading_dimension = operation.input.rows,
        .output = operation.output,
        .output_leading_dimension = operation.output_leading_dimension,
    });
  }

 protected:
  F32GemmBackend() = default;
};

}  // namespace stlqpp
