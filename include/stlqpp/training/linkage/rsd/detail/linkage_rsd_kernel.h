#pragma once

#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/rsd/linkage_support.h"
#include "stlqpp/training/least_squares_cost_mode.h"

#include <cstdint>

namespace stlqpp::detail {

// Fits and selects registered global-greedy RSD rank-2 without consuming an
// ICM permutation. Tiny CPU topology evaluation uses this boundary because
// The cluster RNG advances only after a nonsingular RSD fit.
template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
[[nodiscard]] bool FitRsdOne(
    const LinkageCodebookAlgebra<M>& algebra,
    const float* atom_dots,
    float residual_squared_norm,
    LinkageSolution<M, Capacity>* solution,
    std::uint32_t maximum_rank = 2U,
    LeastSquaresCostMode cost_mode =
        LeastSquaresCostMode::kNormalEquation) noexcept;

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
[[nodiscard]] bool FitRsdOneFixedLayerZero(
    const LinkageCodebookAlgebra<M>& algebra,
    const float* atom_dots,
    float residual_squared_norm,
    std::uint8_t fixed_layer_zero_code,
    LinkageSolution<M, Capacity>* solution,
    std::uint32_t maximum_rank = 2U,
    LeastSquaresCostMode cost_mode =
        LeastSquaresCostMode::kNormalEquation) noexcept;

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
void RefineRsdOne(
    const LinkageCodebookAlgebra<M>& algebra,
    const float* atom_dots,
    float residual_squared_norm,
    const std::uint8_t* icm_layer_order,
    LinkageSolution<M, Capacity>* solution,
    LeastSquaresCostMode cost_mode =
        LeastSquaresCostMode::kNormalEquation) noexcept;

// Exact scalar endpoint shared by the ordinary CPU RSD encoder and
// the causal executor's tiny CPU tail. The caller owns projection and scratch;
// this function owns only rank-2 single-deviation selection and the registered
// ICM sweep.
template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
[[nodiscard]] bool EncodeRsdOne(
    const LinkageCodebookAlgebra<M>& algebra,
    const float* atom_dots,
    float residual_squared_norm,
    const std::uint8_t* icm_layer_order,
    LinkageSolution<M, Capacity>* solution,
    std::uint32_t maximum_rank = 2U,
    LeastSquaresCostMode cost_mode =
        LeastSquaresCostMode::kNormalEquation,
    std::uint32_t icm_rounds = 1U) noexcept;

extern template bool EncodeRsdOne<5>(
    const LinkageCodebookAlgebra<5>&,
    const float*, float, const std::uint8_t*,
    LinkageSolution<5>*, std::uint32_t, LeastSquaresCostMode,
    std::uint32_t) noexcept;
extern template bool EncodeRsdOne<10>(
    const LinkageCodebookAlgebra<10>&,
    const float*, float, const std::uint8_t*,
    LinkageSolution<10>*, std::uint32_t, LeastSquaresCostMode,
    std::uint32_t) noexcept;

extern template bool FitRsdOne<5>(
    const LinkageCodebookAlgebra<5>&,
    const float*, float, LinkageSolution<5>*, std::uint32_t,
    LeastSquaresCostMode) noexcept;
extern template bool FitRsdOne<10>(
    const LinkageCodebookAlgebra<10>&,
    const float*, float, LinkageSolution<10>*, std::uint32_t,
    LeastSquaresCostMode) noexcept;
extern template bool FitRsdOneFixedLayerZero<5>(
    const LinkageCodebookAlgebra<5>&,
    const float*, float, std::uint8_t,
    LinkageSolution<5>*, std::uint32_t, LeastSquaresCostMode) noexcept;
extern template bool FitRsdOneFixedLayerZero<10>(
    const LinkageCodebookAlgebra<10>&,
    const float*, float, std::uint8_t,
    LinkageSolution<10>*, std::uint32_t, LeastSquaresCostMode) noexcept;
extern template void RefineRsdOne<5>(
    const LinkageCodebookAlgebra<5>&,
    const float*, float, const std::uint8_t*,
    LinkageSolution<5>*, LeastSquaresCostMode) noexcept;
extern template void RefineRsdOne<10>(
    const LinkageCodebookAlgebra<10>&,
    const float*, float, const std::uint8_t*,
    LinkageSolution<10>*, LeastSquaresCostMode) noexcept;

}  // namespace stlqpp::detail
