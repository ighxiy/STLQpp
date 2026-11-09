#include "stlqpp/runtime/cpu/lapack_spd_solver_f32.h"
#include "stlqpp/core/execution/workload_thread_budget.h"

#if defined(STLQPP_LAPACKE_MKL)
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include <cstddef>
#include <cstdint>
#include <limits>

namespace stlqpp {
namespace {

#if defined(STLQPP_LAPACKE_MKL)
// MKL's local setting affects only calls issued by this thread. Restore its
// previous value even when factorization rejects a non-SPD system.
class ScopedLocalMklThreads final {
 public:
  explicit ScopedLocalMklThreads(std::uint32_t count) noexcept {
    if (count > 1U) {
      previous_ = mkl_set_num_threads_local(static_cast<int>(count));
      active_ = true;
    }
  }
  ~ScopedLocalMklThreads() {
    if (active_) mkl_set_num_threads_local(previous_);
  }
  ScopedLocalMklThreads(const ScopedLocalMklThreads&) = delete;
  ScopedLocalMklThreads& operator=(const ScopedLocalMklThreads&) = delete;

 private:
  int previous_ = 0;
  bool active_ = false;
};
#endif

}  // namespace

StatusOr<SpdSolveOutcome> LapackSpdSolverF32::FactorAndSolve(
    const SpdSystemF32& system) const {
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
    return Status::InvalidArgument("LAPACK SPD system dimensions are invalid");
  }

  const lapack_int order = static_cast<lapack_int>(system.order);
  const lapack_int matrix_ld =
      static_cast<lapack_int>(system.matrix_leading_dimension);
#if defined(STLQPP_LAPACKE_MKL)
  const std::uint32_t thread_count = system.allow_phase_parallelism
      ? SelectLargeSpdThreadCount(
            system.order, system.right_hand_side_count, available_workers_)
      : 1U;
  const ScopedLocalMklThreads local_threads(thread_count);
#endif
  lapack_int info = LAPACKE_spotrf(
      LAPACK_COL_MAJOR, 'U', order, system.matrix, matrix_ld);
  if (info > 0) {
    return SpdSolveOutcome::kNotPositiveDefinite;
  }
  if (info < 0) {
    return Status::Internal("LAPACKE_spotrf rejected a validated argument");
  }
  info = LAPACKE_spotrs(
      LAPACK_COL_MAJOR, 'U', order,
      static_cast<lapack_int>(system.right_hand_side_count),
      system.matrix, matrix_ld, system.right_hand_sides,
      static_cast<lapack_int>(system.rhs_leading_dimension));
  if (info != 0) {
    return Status::Internal("LAPACKE_spotrs failed after factorization");
  }
  return SpdSolveOutcome::kSolved;
}

}  // namespace stlqpp
