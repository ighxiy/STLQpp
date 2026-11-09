#pragma once

#include "stlqpp/runtime/cpu/spd_solver_f64.h"

namespace stlqpp {

// Production FP64 adapter used by the registered scalar coefficient codec.
class LapackSpdSolverF64 final : public SpdSolverF64 {
 public:
  LapackSpdSolverF64() = default;

  [[nodiscard]] StatusOr<SpdSolveOutcome> FactorAndSolve(
      const SpdSystemF64& system) const override;
};

}  // namespace stlqpp
