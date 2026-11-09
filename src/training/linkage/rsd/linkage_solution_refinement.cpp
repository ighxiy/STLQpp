#include "stlqpp/training/linkage/rsd/detail/linkage_solution_refinement.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include "stlqpp/training/basic/detail/icm_candidate_selector.h"
#include <limits>

namespace stlqpp::detail {
namespace {

constexpr float kAcceptanceEpsilon = 1.0e-6F;

template <int M, LeastSquaresCostMode CostMode, std::uint32_t Capacity>
void RunLinkageIcmSweepWithCost(const LinkageCodebookAlgebra<M>& algebra,
                                const float* atom_dots,
                                float residual_squared_norm,
                                const std::uint8_t* layer_order,
                                LinkageSolution<M, Capacity>* solution) noexcept {
  const std::uint32_t m = M == 0 ? algebra.shape().m : M;
  for (std::uint32_t position = 0U; position < m; ++position) {
    const std::uint32_t layer = layer_order[position];
    const std::uint32_t old_code = solution->support.codes[layer];
    std::uint32_t best_code = old_code;
    const std::uint32_t begin = algebra.layer_offset(layer);
    alignas(32) std::array<float, 256> remaining;
    std::copy_n(atom_dots + begin, 256U, remaining.data());
    for (std::uint32_t other_layer = 0U; other_layer < m; ++other_layer) {
      if (other_layer == layer) continue;
      const std::uint32_t selected_atom = algebra.layer_offset(other_layer) +
          solution->support.codes[other_layer];
      const float* gram = algebra.gram().data() +
          static_cast<std::size_t>(selected_atom) * algebra.atom_count() + begin;
      const float coefficient = solution->coefficients[other_layer];
#pragma omp simd
      for (std::uint32_t code = 0U; code < 256U; ++code)
        remaining[code] -= coefficient * gram[code];
    }
    best_code = SelectIcmCandidate(remaining.data(),
        algebra.inverse_norms().data() + begin, old_code, true).code;
    if (best_code == old_code) {
      continue;
    }

    LinkageSupport<M, Capacity> trial_support = solution->support;
    trial_support.codes[layer] = static_cast<std::uint8_t>(best_code);
    LinkageSolution<M, Capacity> trial;
    if (RefitLinkageSolutionWithCost<M, CostMode, Capacity>(
            algebra, trial_support, atom_dots, residual_squared_norm, &trial) &&
        trial.squared_error + kAcceptanceEpsilon < solution->squared_error) {
      *solution = trial;
    }
  }
}

}  // namespace

template <int M, std::uint32_t Capacity>
bool RefitLinkageSolution(const LinkageCodebookAlgebra<M>& algebra,
                          const LinkageSupport<M, Capacity>& support,
                          const float* atom_dots, float residual_squared_norm,
                          LinkageSolution<M, Capacity>* solution,
                          LeastSquaresCostMode cost_mode) noexcept {
  return cost_mode == LeastSquaresCostMode::kNormalEquation
             ? RefitLinkageSolutionWithCost<M, LeastSquaresCostMode::kNormalEquation, Capacity>(
                   algebra, support, atom_dots, residual_squared_norm, solution)
             : RefitLinkageSolutionWithCost<M, LeastSquaresCostMode::kExplicitQuadratic, Capacity>(
                   algebra, support, atom_dots, residual_squared_norm,
                   solution);
}

template <int M, std::uint32_t Capacity>
void RunLinkageIcmSweep(const LinkageCodebookAlgebra<M>& algebra,
                        const float* atom_dots, float residual_squared_norm,
                        const std::uint8_t* layer_order,
                        LinkageSolution<M, Capacity>* solution,
                        LeastSquaresCostMode cost_mode) noexcept {
  if (cost_mode == LeastSquaresCostMode::kNormalEquation) {
    RunLinkageIcmSweepWithCost<M, LeastSquaresCostMode::kNormalEquation, Capacity>(
        algebra, atom_dots, residual_squared_norm, layer_order, solution);
    return;
  }
  RunLinkageIcmSweepWithCost<M, LeastSquaresCostMode::kExplicitQuadratic, Capacity>(
      algebra, atom_dots, residual_squared_norm, layer_order, solution);
}

template bool RefitLinkageSolution<0>(const LinkageCodebookAlgebra<0>&,
                                      const LinkageSupport<0>&, const float*,
                                      float, LinkageSolution<0>*,
                                      LeastSquaresCostMode) noexcept;
template void RunLinkageIcmSweep<0>(const LinkageCodebookAlgebra<0>&,
                                    const float*, float, const std::uint8_t*,
                                    LinkageSolution<0>*,
                                    LeastSquaresCostMode) noexcept;
template bool RefitLinkageSolution<5>(const LinkageCodebookAlgebra<5>&,
                                      const LinkageSupport<5>&, const float*,
                                      float, LinkageSolution<5>*,
                                      LeastSquaresCostMode) noexcept;
template bool RefitLinkageSolution<12>(const LinkageCodebookAlgebra<12>&,
                                       const LinkageSupport<12>&, const float*,
                                       float, LinkageSolution<12>*,
                                       LeastSquaresCostMode) noexcept;
template bool RefitLinkageSolution<6>(const LinkageCodebookAlgebra<6>&,
                                      const LinkageSupport<6>&, const float*,
                                      float, LinkageSolution<6>*,
                                      LeastSquaresCostMode) noexcept;
template bool RefitLinkageSolution<10>(const LinkageCodebookAlgebra<10>&,
                                       const LinkageSupport<10>&, const float*,
                                       float, LinkageSolution<10>*,
                                       LeastSquaresCostMode) noexcept;
template void RunLinkageIcmSweep<5>(const LinkageCodebookAlgebra<5>&,
                                    const float*, float, const std::uint8_t*,
                                    LinkageSolution<5>*,
                                    LeastSquaresCostMode) noexcept;
template void RunLinkageIcmSweep<12>(const LinkageCodebookAlgebra<12>&,
                                     const float*, float, const std::uint8_t*,
                                     LinkageSolution<12>*,
                                     LeastSquaresCostMode) noexcept;
template void RunLinkageIcmSweep<6>(const LinkageCodebookAlgebra<6>&,
                                    const float*, float, const std::uint8_t*,
                                    LinkageSolution<6>*,
                                    LeastSquaresCostMode) noexcept;
template void RunLinkageIcmSweep<10>(const LinkageCodebookAlgebra<10>&,
                                     const float*, float, const std::uint8_t*,
                                     LinkageSolution<10>*,
                                     LeastSquaresCostMode) noexcept;

template bool RefitLinkageSolution<0, kWideRuntimeScratchLayers>(const LinkageCodebookAlgebra<0>&,
                                      const LinkageSupport<0, kWideRuntimeScratchLayers>&, const float*,
                                      float, LinkageSolution<0, kWideRuntimeScratchLayers>*,
                                      LeastSquaresCostMode) noexcept;
template void RunLinkageIcmSweep<0, kWideRuntimeScratchLayers>(const LinkageCodebookAlgebra<0>&,
                                    const float*, float, const std::uint8_t*,
                                    LinkageSolution<0, kWideRuntimeScratchLayers>*,
                                    LeastSquaresCostMode) noexcept;

}  // namespace stlqpp::detail
