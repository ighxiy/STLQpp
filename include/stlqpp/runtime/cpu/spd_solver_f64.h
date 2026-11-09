#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cpu/spd_solve_outcome.h"

#include <cstdint>

namespace stlqpp {

struct SpdSystemF64 final {
  // Upper-triangular column-major matrix, overwritten by its Cholesky factor.
  double* matrix = nullptr;
  std::uint32_t order = 0U;
  std::uint32_t matrix_leading_dimension = 0U;
  // Column-major right-hand sides, overwritten by the solution.
  double* right_hand_sides = nullptr;
  std::uint32_t right_hand_side_count = 0U;
  std::uint32_t rhs_leading_dimension = 0U;
};

class SpdSolverF64 {
 public:
  virtual ~SpdSolverF64() = default;
  SpdSolverF64(const SpdSolverF64&) = delete;
  SpdSolverF64& operator=(const SpdSolverF64&) = delete;

  [[nodiscard]] virtual StatusOr<SpdSolveOutcome> FactorAndSolve(
      const SpdSystemF64& system) const = 0;

 protected:
  SpdSolverF64() = default;
};

}  // namespace stlqpp
