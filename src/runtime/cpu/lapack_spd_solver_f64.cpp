#include "stlqpp/runtime/cpu/lapack_spd_solver_f64.h"

#if defined(STLQPP_LAPACKE_MKL)
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include <cstddef>
#include <cstdint>
#include <limits>

namespace stlqpp {

StatusOr<SpdSolveOutcome> LapackSpdSolverF64::FactorAndSolve(
    const SpdSystemF64& system) const {
  constexpr std::size_t kLapackIntMax =
      static_cast<std::size_t>(std::numeric_limits<lapack_int>::max());
  if (system.matrix == nullptr || system.right_hand_sides == nullptr ||
      system.order == 0U || system.right_hand_side_count == 0U ||
      system.matrix_leading_dimension < system.order ||
      system.rhs_leading_dimension < system.order ||
      system.order > kLapackIntMax ||
      system.right_hand_side_count > kLapackIntMax ||
      system.matrix_leading_dimension > kLapackIntMax ||
      system.rhs_leading_dimension > kLapackIntMax) {
    return Status::InvalidArgument("LAPACK FP64 SPD system dimensions are invalid");
  }

  const lapack_int order = static_cast<lapack_int>(system.order);
  const lapack_int matrix_ld =
      static_cast<lapack_int>(system.matrix_leading_dimension);
  lapack_int info = LAPACKE_dpotrf(
      LAPACK_COL_MAJOR, 'U', order, system.matrix, matrix_ld);
  if (info > 0) {
    return SpdSolveOutcome::kNotPositiveDefinite;
  }
  if (info < 0) {
    return Status::Internal("LAPACKE_dpotrf rejected a validated argument");
  }
  info = LAPACKE_dpotrs(
      LAPACK_COL_MAJOR, 'U', order,
      static_cast<lapack_int>(system.right_hand_side_count),
      system.matrix, matrix_ld, system.right_hand_sides,
      static_cast<lapack_int>(system.rhs_leading_dimension));
  if (info != 0) {
    return Status::Internal("LAPACKE_dpotrs failed after factorization");
  }
  return SpdSolveOutcome::kSolved;
}

}  // namespace stlqpp
