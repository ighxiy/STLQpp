#pragma once

#include "stlqpp/runtime/cpu/spd_solver_f32.h"

namespace stlqpp {

// Production adapter over the configured LAPACKE implementation. Vendor
// headers and integer ABI remain private to the runtime target.
class LapackSpdSolverF32 final : public SpdSolverF32 {
 public:
  explicit LapackSpdSolverF32(
      std::uint32_t available_workers = 1U) noexcept
      : available_workers_(available_workers) {}

  [[nodiscard]] StatusOr<SpdSolveOutcome> FactorAndSolve(
      const SpdSystemF32& system) const override;

 private:
  std::uint32_t available_workers_ = 1U;
};

}  // namespace stlqpp
