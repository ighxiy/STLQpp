#pragma once

#include "stlqpp/training/basic/large_root_algebra.h"
#include "stlqpp/training/least_squares_cost_mode.h"

#include <cstdint>

namespace stlqpp {

// Fixed-M trusted numerical primitive used by RSD post-processing, ICM and
// ILS. Inputs have already crossed the basic-encoding validation boundary.
template <int M>
class BasicSupportLeastSquares final {
 public:
  static_assert(M >= 2 && M <= 20);

  [[nodiscard]] static bool Solve(
      const BasicCodebookAlgebra<M>& algebra,
      std::uint32_t root,
      const std::uint8_t* small_codes,
      float root_dot,
      const float* small_dots,
      float* coefficients,
      float* applied_diagonal_bump = nullptr) noexcept;

  [[nodiscard]] static float SquaredCost(
      const BasicCodebookAlgebra<M>& algebra,
      std::uint32_t root,
      const std::uint8_t* small_codes,
      float root_dot,
      const float* small_dots,
      const float* coefficients,
      float sample_squared_norm) noexcept;

  // Uses the normal equation satisfied by coefficients returned from Solve.
  // The bump must be the exact diagonal regularization used by that solve.
  [[nodiscard]] static float NormalEquationSquaredCost(
      const BasicCodebookAlgebra<M>& algebra,
      const std::uint8_t* small_codes,
      float root_dot,
      const float* small_dots,
      const float* coefficients,
      float sample_squared_norm,
      float applied_diagonal_bump) noexcept;

  template <LeastSquaresCostMode Mode>
  [[nodiscard]] static float Cost(
      const BasicCodebookAlgebra<M>& algebra,
      std::uint32_t root,
      const std::uint8_t* small_codes,
      float root_dot,
      const float* small_dots,
      const float* coefficients,
      float sample_squared_norm,
      float applied_diagonal_bump) noexcept {
    if constexpr (Mode == LeastSquaresCostMode::kNormalEquation) {
      return NormalEquationSquaredCost(
          algebra, small_codes, root_dot, small_dots, coefficients,
          sample_squared_norm, applied_diagonal_bump);
    } else {
      (void)applied_diagonal_bump;
      return SquaredCost(
          algebra, root, small_codes, root_dot, small_dots, coefficients,
          sample_squared_norm);
    }
  }
};

class RuntimeBasicSupportLeastSquares final {
 public:
  [[nodiscard]] static bool Solve(
      const RuntimeBasicCodebookAlgebra& algebra,
      std::uint32_t root,
      const std::uint8_t* small_codes,
      float root_dot,
      const float* small_dots,
      float* coefficients,
      float* applied_diagonal_bump = nullptr) noexcept;

  template <std::uint32_t Capacity>
  [[nodiscard]] static bool SolveWithCapacity(
      const RuntimeBasicCodebookAlgebra&, std::uint32_t root,
      const std::uint8_t* codes, float root_dot, const float* small_dots,
      float* coefficients, float* applied_diagonal_bump = nullptr) noexcept;

  [[nodiscard]] static float SquaredCost(
      const RuntimeBasicCodebookAlgebra& algebra,
      std::uint32_t root,
      const std::uint8_t* small_codes,
      float root_dot,
      const float* small_dots,
      const float* coefficients,
      float sample_squared_norm) noexcept;

  // Uses the normal equation satisfied by coefficients returned from Solve.
  // The bump must be the exact diagonal regularization used by that solve.
  [[nodiscard]] static float NormalEquationSquaredCost(
      const RuntimeBasicCodebookAlgebra& algebra,
      const std::uint8_t* small_codes,
      float root_dot,
      const float* small_dots,
      const float* coefficients,
      float sample_squared_norm,
      float applied_diagonal_bump) noexcept;

  template <LeastSquaresCostMode Mode>
  [[nodiscard]] static float Cost(
      const RuntimeBasicCodebookAlgebra& algebra,
      std::uint32_t root,
      const std::uint8_t* small_codes,
      float root_dot,
      const float* small_dots,
      const float* coefficients,
      float sample_squared_norm,
      float applied_diagonal_bump) noexcept {
    if constexpr (Mode == LeastSquaresCostMode::kNormalEquation) {
      return NormalEquationSquaredCost(
          algebra, small_codes, root_dot, small_dots, coefficients,
          sample_squared_norm, applied_diagonal_bump);
    } else {
      (void)applied_diagonal_bump;
      return SquaredCost(
          algebra, root, small_codes, root_dot, small_dots, coefficients,
          sample_squared_norm);
    }
  }
};

extern template class BasicSupportLeastSquares<5>;
extern template class BasicSupportLeastSquares<10>;

namespace detail {
template <int M, std::uint32_t Capacity, typename Algebra>
bool SolveBasicSupport(const Algebra& algebra, std::uint32_t root,
    const std::uint8_t* codes, float root_dot, const float* small_dots,
    float* coefficients, float* bump) noexcept {
  if constexpr (M == 0)
    return RuntimeBasicSupportLeastSquares::SolveWithCapacity<Capacity>(
        algebra, root, codes, root_dot, small_dots, coefficients, bump);
  else
    return BasicSupportLeastSquares<M>::Solve(
        algebra, root, codes, root_dot, small_dots, coefficients, bump);
}
}  // namespace detail

}  // namespace stlqpp
