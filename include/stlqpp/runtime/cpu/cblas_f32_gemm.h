#pragma once

#include "stlqpp/runtime/cpu/f32_gemm.h"

#include <cstdint>

namespace stlqpp {

// Thin production adapter over the configured CBLAS implementation. Vendor
// selection and thread policy remain runtime/build concerns; query-table
// mathematics does not depend on MKL or OpenBLAS headers.
class CblasF32Gemm final : public F32GemmBackend {
 public:
  // Zero preserves the vendor runtime's current policy. A positive value is
  // the calling thread's MKL budget or a call-scoped OpenBLAS budget. Query
  // execution uses its worker count; training callers that manage BLAS
  // independently retain the zero default.
  explicit CblasF32Gemm(std::uint32_t thread_count = 0U) noexcept
      : thread_count_(thread_count) {}

  [[nodiscard]] Status MultiplyLeft(
      const LeftGemmF32& operation) const override;

  [[nodiscard]] Status MultiplyTransposedLeft(
      const TransposedLeftGemmF32& operation) const override;

  [[nodiscard]] Status MultiplySelfTransposedUpper(
      const SelfTransposedUpperF32& operation) const override;

 private:
  std::uint32_t thread_count_ = 0U;
};

}  // namespace stlqpp
