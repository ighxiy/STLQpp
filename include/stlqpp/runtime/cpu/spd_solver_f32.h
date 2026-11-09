#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cpu/spd_solve_outcome.h"

#include <cstdint>

namespace stlqpp {

struct SpdSystemF32 final {
  // Upper-triangular column-major matrix, overwritten by its Cholesky factor.
  float* matrix = nullptr;
  std::uint32_t order = 0U;
  std::uint32_t matrix_leading_dimension = 0U;
  // Column-major right-hand sides, overwritten by the solution.
  float* right_hand_sides = nullptr;
  std::uint32_t right_hand_side_count = 0U;
  std::uint32_t rhs_leading_dimension = 0U;
  // Only phase-isolated model updates may borrow the CPU worker budget.
  bool allow_phase_parallelism = false;
};

class SpdSolverF32 {
 public:
  virtual ~SpdSolverF32() = default;
  SpdSolverF32(const SpdSolverF32&) = delete;
  SpdSolverF32& operator=(const SpdSolverF32&) = delete;

  [[nodiscard]] virtual StatusOr<SpdSolveOutcome> FactorAndSolve(
      const SpdSystemF32& system) const = 0;

 protected:
  SpdSolverF32() = default;
};

}  // namespace stlqpp
