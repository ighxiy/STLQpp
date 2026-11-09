#pragma once

#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/least_squares_cost_mode.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/rsd/linkage_support.h"

namespace stlqpp {

// Shared fixed/runtime-M refit and objective used uniformly for every RSD
// candidate and every ICM trial.
template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
class LinkageSupportLeastSquares final {
 public:
  static_assert(M == 0 ||
                (M >= kMinimumModelLayers && M <= kMaximumModelLayers));

  [[nodiscard]] static bool Solve(
      const LinkageCodebookAlgebra<M>& algebra,
      const LinkageSupport<M, Capacity>& support, const float* atom_dots,
      float* coefficients, float* applied_diagonal_bump = nullptr) noexcept;

  [[nodiscard]] static float SquaredCost(
      const LinkageCodebookAlgebra<M>& algebra,
      const LinkageSupport<M, Capacity>& support, const float* atom_dots,
      const float* coefficients, float residual_squared_norm) noexcept;

  // Uses the normal equation satisfied by coefficients returned from Solve.
  // The bump must be the exact diagonal regularization used by that solve.
  [[nodiscard]] static float NormalEquationSquaredCost(
      const LinkageCodebookAlgebra<M>& algebra,
      const LinkageSupport<M, Capacity>& support, const float* atom_dots,
      const float* coefficients, float residual_squared_norm,
      float applied_diagonal_bump) noexcept;

  template <LeastSquaresCostMode Mode>
  [[nodiscard]] static float Cost(const LinkageCodebookAlgebra<M>& algebra,
                                  const LinkageSupport<M, Capacity>& support,
                                  const float* atom_dots,
                                  const float* coefficients,
                                  float residual_squared_norm,
                                  float applied_diagonal_bump) noexcept {
    if constexpr (Mode == LeastSquaresCostMode::kNormalEquation) {
      return NormalEquationSquaredCost(algebra, support, atom_dots,
                                       coefficients, residual_squared_norm,
                                       applied_diagonal_bump);
    }
    (void)applied_diagonal_bump;
    return SquaredCost(algebra, support, atom_dots, coefficients,
                       residual_squared_norm);
  }
};

extern template class LinkageSupportLeastSquares<5>;
extern template class LinkageSupportLeastSquares<10>;

}  // namespace stlqpp
