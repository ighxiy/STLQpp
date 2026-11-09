#pragma once

#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/rsd/linkage_support.h"
#include "stlqpp/training/linkage/rsd/linkage_support_least_squares.h"
#include "stlqpp/training/least_squares_cost_mode.h"

#include <cmath>
#include <cstdint>

namespace stlqpp::detail {

template <int M, LeastSquaresCostMode CostMode, std::uint32_t Capacity = kModelScratchLayers<M>>
[[nodiscard]] bool RefitLinkageSolutionWithCost(
    const LinkageCodebookAlgebra<M>& algebra,
    const LinkageSupport<M, Capacity>& support,
    const float* atom_dots,
    float residual_squared_norm,
    LinkageSolution<M, Capacity>* solution) noexcept {
  solution->support = support;
  float applied_diagonal_bump = 0.0F;
  if (!LinkageSupportLeastSquares<M, Capacity>::Solve(
          algebra, support, atom_dots, solution->coefficients.data(),
          &applied_diagonal_bump)) {
    return false;
  }
  solution->squared_error =
      LinkageSupportLeastSquares<M, Capacity>::template Cost<CostMode>(
          algebra, support, atom_dots, solution->coefficients.data(),
          residual_squared_norm, applied_diagonal_bump);
  return std::isfinite(solution->squared_error);
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
[[nodiscard]] bool RefitLinkageSolution(
    const LinkageCodebookAlgebra<M>& algebra,
    const LinkageSupport<M, Capacity>& support,
    const float* atom_dots,
    float residual_squared_norm,
    LinkageSolution<M, Capacity>* solution,
    LeastSquaresCostMode cost_mode =
        LeastSquaresCostMode::kNormalEquation) noexcept;

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
void RunLinkageIcmSweep(
    const LinkageCodebookAlgebra<M>& algebra,
    const float* atom_dots,
    float residual_squared_norm,
    const std::uint8_t* layer_order,
    LinkageSolution<M, Capacity>* solution,
    LeastSquaresCostMode cost_mode =
        LeastSquaresCostMode::kNormalEquation) noexcept;

extern template bool RefitLinkageSolution<5>(
    const LinkageCodebookAlgebra<5>&,
    const LinkageSupport<5>&,
    const float*, float, LinkageSolution<5>*, LeastSquaresCostMode) noexcept;
extern template bool RefitLinkageSolution<10>(
    const LinkageCodebookAlgebra<10>&,
    const LinkageSupport<10>&,
    const float*, float, LinkageSolution<10>*, LeastSquaresCostMode) noexcept;
extern template void RunLinkageIcmSweep<5>(
    const LinkageCodebookAlgebra<5>&,
    const float*, float, const std::uint8_t*,
    LinkageSolution<5>*, LeastSquaresCostMode) noexcept;
extern template void RunLinkageIcmSweep<10>(
    const LinkageCodebookAlgebra<10>&,
    const float*, float, const std::uint8_t*,
    LinkageSolution<10>*, LeastSquaresCostMode) noexcept;

}  // namespace stlqpp::detail
