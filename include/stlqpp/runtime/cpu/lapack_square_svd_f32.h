#pragma once

#include "stlqpp/runtime/cpu/square_svd_f32.h"

namespace stlqpp {

// FP32 input/output boundary with a double-precision LAPACK decomposition.
// Scratch belongs to each call so concurrent callers never share workspace.
class LapackSquareSvdF32 final : public SquareSvdF32 {
 public:
  explicit LapackSquareSvdF32(std::uint32_t available_workers = 1U) noexcept
      : available_workers_(available_workers) {}

  [[nodiscard]] Status Solve(
      const SquareSvdSystemF32& system) const override;

 private:
  std::uint32_t available_workers_ = 1U;
};

}  // namespace stlqpp
