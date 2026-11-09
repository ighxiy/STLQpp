#include "stlqpp/training/linkage/rsd/detail/linkage_rsd_kernel.h"

#include <limits>

#include "stlqpp/training/linkage/rsd/detail/linkage_solution_refinement.h"
#include "stlqpp/training/linkage/rsd/ranked_single_deviation.h"

namespace stlqpp::detail {

namespace {

template <int M, LeastSquaresCostMode CostMode, std::uint32_t Capacity>
bool FitRsdOneWithCost(const LinkageCodebookAlgebra<M>& algebra,
                       const float* atom_dots, float residual_squared_norm,
                       LinkageSolution<M, Capacity>* solution,
                       std::uint32_t maximum_rank) noexcept {
  const RankedSingleDeviationSet<M, Capacity> rsd =
      RankedSingleDeviation<M, Capacity>::Build(algebra, atom_dots, maximum_rank);
  LinkageSolution<M, Capacity> best;
  best.squared_error = std::numeric_limits<float>::infinity();
  bool solved = false;
  for (std::uint32_t proposal = 0U; proposal < rsd.candidate_count;
       ++proposal) {
    LinkageSolution<M, Capacity> candidate;
    if (!RefitLinkageSolutionWithCost<M, CostMode, Capacity>(
            algebra, rsd.candidates[proposal], atom_dots, residual_squared_norm,
            &candidate)) {
      continue;
    }
    if (!solved || candidate.squared_error < best.squared_error) {
      best = candidate;
      solved = true;
    }
  }
  if (!solved) return false;
  *solution = best;
  return true;
}

template <int M, LeastSquaresCostMode CostMode, std::uint32_t Capacity>
bool FitRsdOneFixedLayerZeroWithCost(const LinkageCodebookAlgebra<M>& algebra,
                                     const float* atom_dots,
                                     float residual_squared_norm,
                                     std::uint8_t fixed_layer_zero_code,
                                     LinkageSolution<M, Capacity>* solution,
                                     std::uint32_t maximum_rank) noexcept {
  const RankedSingleDeviationSet<M, Capacity> rsd =
      RankedSingleDeviation<M, Capacity>::BuildFixedLayerZero(
          algebra, atom_dots, fixed_layer_zero_code, maximum_rank);
  LinkageSolution<M, Capacity> best;
  best.squared_error = std::numeric_limits<float>::infinity();
  bool solved = false;
  for (std::uint32_t proposal = 0U; proposal < rsd.candidate_count;
       ++proposal) {
    LinkageSolution<M, Capacity> candidate;
    if (!RefitLinkageSolutionWithCost<M, CostMode, Capacity>(
            algebra, rsd.candidates[proposal], atom_dots, residual_squared_norm,
            &candidate)) {
      continue;
    }
    if (!solved || candidate.squared_error < best.squared_error) {
      best = candidate;
      solved = true;
    }
  }
  if (!solved) return false;
  *solution = best;
  return true;
}

}  // namespace

template <int M, std::uint32_t Capacity>
bool FitRsdOne(const LinkageCodebookAlgebra<M>& algebra, const float* atom_dots,
               float residual_squared_norm, LinkageSolution<M, Capacity>* solution,
               std::uint32_t maximum_rank,
               LeastSquaresCostMode cost_mode) noexcept {
  return cost_mode == LeastSquaresCostMode::kNormalEquation
             ? FitRsdOneWithCost<M, LeastSquaresCostMode::kNormalEquation, Capacity>(
                   algebra, atom_dots, residual_squared_norm, solution,
                   maximum_rank)
             : FitRsdOneWithCost<M, LeastSquaresCostMode::kExplicitQuadratic, Capacity>(
                   algebra, atom_dots, residual_squared_norm, solution,
                   maximum_rank);
}

template <int M, std::uint32_t Capacity>
bool FitRsdOneFixedLayerZero(const LinkageCodebookAlgebra<M>& algebra,
                             const float* atom_dots,
                             float residual_squared_norm,
                             std::uint8_t fixed_layer_zero_code,
                             LinkageSolution<M, Capacity>* solution,
                             std::uint32_t maximum_rank,
                             LeastSquaresCostMode cost_mode) noexcept {
  return cost_mode == LeastSquaresCostMode::kNormalEquation
             ? FitRsdOneFixedLayerZeroWithCost<M, LeastSquaresCostMode::kNormalEquation, Capacity>(
                   algebra, atom_dots, residual_squared_norm,
                   fixed_layer_zero_code, solution, maximum_rank)
             : FitRsdOneFixedLayerZeroWithCost<M, LeastSquaresCostMode::kExplicitQuadratic, Capacity>(
                   algebra, atom_dots, residual_squared_norm,
                   fixed_layer_zero_code, solution, maximum_rank);
}

template <int M, std::uint32_t Capacity>
void RefineRsdOne(const LinkageCodebookAlgebra<M>& algebra,
                  const float* atom_dots, float residual_squared_norm,
                  const std::uint8_t* icm_layer_order,
                  LinkageSolution<M, Capacity>* solution,
                  LeastSquaresCostMode cost_mode) noexcept {
  RunLinkageIcmSweep(algebra, atom_dots, residual_squared_norm, icm_layer_order,
                     solution, cost_mode);
}

template <int M, std::uint32_t Capacity>
bool EncodeRsdOne(const LinkageCodebookAlgebra<M>& algebra,
                  const float* atom_dots, float residual_squared_norm,
                  const std::uint8_t* icm_layer_order,
                  LinkageSolution<M, Capacity>* solution, std::uint32_t maximum_rank,
                  LeastSquaresCostMode cost_mode,
                  std::uint32_t icm_rounds) noexcept {
  if (!FitRsdOne(algebra, atom_dots, residual_squared_norm, solution,
                 maximum_rank, cost_mode)) {
    return false;
  }
  for (std::uint32_t round = 0U; round < icm_rounds; ++round) {
    RefineRsdOne(algebra, atom_dots, residual_squared_norm, icm_layer_order,
                 solution, cost_mode);
  }
  return true;
}

template bool EncodeRsdOne<0>(const LinkageCodebookAlgebra<0>&, const float*,
                              float, const std::uint8_t*, LinkageSolution<0>*,
                              std::uint32_t, LeastSquaresCostMode,
                              std::uint32_t) noexcept;
template bool EncodeRsdOne<5>(const LinkageCodebookAlgebra<5>&, const float*,
                              float, const std::uint8_t*, LinkageSolution<5>*,
                              std::uint32_t, LeastSquaresCostMode,
                              std::uint32_t) noexcept;
template bool EncodeRsdOne<12>(const LinkageCodebookAlgebra<12>&, const float*,
                               float, const std::uint8_t*, LinkageSolution<12>*,
                               std::uint32_t, LeastSquaresCostMode,
                               std::uint32_t) noexcept;
template bool EncodeRsdOne<6>(const LinkageCodebookAlgebra<6>&, const float*,
                              float, const std::uint8_t*, LinkageSolution<6>*,
                              std::uint32_t, LeastSquaresCostMode,
                              std::uint32_t) noexcept;
template bool EncodeRsdOne<10>(const LinkageCodebookAlgebra<10>&, const float*,
                               float, const std::uint8_t*, LinkageSolution<10>*,
                               std::uint32_t, LeastSquaresCostMode,
                               std::uint32_t) noexcept;
template bool FitRsdOne<0>(const LinkageCodebookAlgebra<0>&, const float*,
                           float, LinkageSolution<0>*, std::uint32_t,
                           LeastSquaresCostMode) noexcept;
template bool FitRsdOne<5>(const LinkageCodebookAlgebra<5>&, const float*,
                           float, LinkageSolution<5>*, std::uint32_t,
                           LeastSquaresCostMode) noexcept;
template bool FitRsdOne<12>(const LinkageCodebookAlgebra<12>&, const float*,
                            float, LinkageSolution<12>*, std::uint32_t,
                            LeastSquaresCostMode) noexcept;
template bool FitRsdOne<6>(const LinkageCodebookAlgebra<6>&, const float*,
                           float, LinkageSolution<6>*, std::uint32_t,
                           LeastSquaresCostMode) noexcept;
template bool FitRsdOne<10>(const LinkageCodebookAlgebra<10>&, const float*,
                            float, LinkageSolution<10>*, std::uint32_t,
                            LeastSquaresCostMode) noexcept;
template bool FitRsdOneFixedLayerZero<0>(const LinkageCodebookAlgebra<0>&,
                                         const float*, float, std::uint8_t,
                                         LinkageSolution<0>*, std::uint32_t,
                                         LeastSquaresCostMode) noexcept;
template bool FitRsdOneFixedLayerZero<5>(const LinkageCodebookAlgebra<5>&,
                                         const float*, float, std::uint8_t,
                                         LinkageSolution<5>*, std::uint32_t,
                                         LeastSquaresCostMode) noexcept;
template bool FitRsdOneFixedLayerZero<12>(const LinkageCodebookAlgebra<12>&,
                                          const float*, float, std::uint8_t,
                                          LinkageSolution<12>*, std::uint32_t,
                                          LeastSquaresCostMode) noexcept;
template bool FitRsdOneFixedLayerZero<6>(const LinkageCodebookAlgebra<6>&,
                                         const float*, float, std::uint8_t,
                                         LinkageSolution<6>*, std::uint32_t,
                                         LeastSquaresCostMode) noexcept;
template bool FitRsdOneFixedLayerZero<10>(const LinkageCodebookAlgebra<10>&,
                                          const float*, float, std::uint8_t,
                                          LinkageSolution<10>*, std::uint32_t,
                                          LeastSquaresCostMode) noexcept;
template void RefineRsdOne<0>(const LinkageCodebookAlgebra<0>&, const float*,
                              float, const std::uint8_t*, LinkageSolution<0>*,
                              LeastSquaresCostMode) noexcept;
template void RefineRsdOne<5>(const LinkageCodebookAlgebra<5>&, const float*,
                              float, const std::uint8_t*, LinkageSolution<5>*,
                              LeastSquaresCostMode) noexcept;
template void RefineRsdOne<12>(const LinkageCodebookAlgebra<12>&, const float*,
                               float, const std::uint8_t*, LinkageSolution<12>*,
                               LeastSquaresCostMode) noexcept;
template void RefineRsdOne<6>(const LinkageCodebookAlgebra<6>&, const float*,
                              float, const std::uint8_t*, LinkageSolution<6>*,
                              LeastSquaresCostMode) noexcept;
template void RefineRsdOne<10>(const LinkageCodebookAlgebra<10>&, const float*,
                               float, const std::uint8_t*, LinkageSolution<10>*,
                               LeastSquaresCostMode) noexcept;

template bool EncodeRsdOne<0, kWideRuntimeScratchLayers>(const LinkageCodebookAlgebra<0>&, const float*,
                              float, const std::uint8_t*, LinkageSolution<0, kWideRuntimeScratchLayers>*,
                              std::uint32_t, LeastSquaresCostMode,
                              std::uint32_t) noexcept;
template bool FitRsdOne<0, kWideRuntimeScratchLayers>(const LinkageCodebookAlgebra<0>&, const float*,
                           float, LinkageSolution<0, kWideRuntimeScratchLayers>*, std::uint32_t,
                           LeastSquaresCostMode) noexcept;
template bool FitRsdOneFixedLayerZero<0, kWideRuntimeScratchLayers>(const LinkageCodebookAlgebra<0>&,
                                         const float*, float, std::uint8_t,
                                         LinkageSolution<0, kWideRuntimeScratchLayers>*, std::uint32_t,
                                         LeastSquaresCostMode) noexcept;
template void RefineRsdOne<0, kWideRuntimeScratchLayers>(const LinkageCodebookAlgebra<0>&, const float*,
                              float, const std::uint8_t*, LinkageSolution<0, kWideRuntimeScratchLayers>*,
                              LeastSquaresCostMode) noexcept;

}  // namespace stlqpp::detail
