#pragma once

#include <cstdint>

#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/rsd/linkage_support.h"

namespace stlqpp {

inline constexpr std::uint32_t kMinimumRsdRank = 1U;
inline constexpr std::uint32_t kMaximumRsdRank = 3U;

[[nodiscard]] constexpr bool IsSupportedRsdRank(std::uint32_t rank) noexcept {
  return rank >= kMinimumRsdRank && rank <= kMaximumRsdRank;
}

template <int M>
[[nodiscard]] constexpr std::uint32_t RsdCandidateCount(
    std::uint32_t rank,
    std::uint32_t actual_m = FixedModelLayers<M>()) noexcept {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  return 1U + (M == 0 ? actual_m : M) * (rank - 1U);
}

// Ranked Single-Deviation (RSD) over global greedy search. Rank 1 is the
// native path. RSD rank-k cumulatively retains every single-deviation branch
// from rank 2 through rank k at every native step, then resumes rank-1 greedy
// search. Candidate count 1+M*(k-1) is derived from rank and model shape.
template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
class RankedSingleDeviation final {
 public:
  static_assert(M == 0 ||
                (M >= kMinimumModelLayers && M <= kMaximumModelLayers));
  static constexpr std::uint32_t kBaselineRank = 2U;
  static constexpr std::uint32_t kBaselineCandidateCount = M + 1U;

  [[nodiscard]] static RankedSingleDeviationSet<M, Capacity> Build(
      const LinkageCodebookAlgebra<M>& algebra, const float* atom_dots,
      std::uint32_t maximum_rank = kBaselineRank) noexcept;

  [[nodiscard]] static RankedSingleDeviationSet<M, Capacity> BuildFixedLayerZero(
      const LinkageCodebookAlgebra<M>& algebra, const float* atom_dots,
      std::uint8_t fixed_layer_zero_code,
      std::uint32_t maximum_rank = kBaselineRank) noexcept;
};

extern template class RankedSingleDeviation<5>;
extern template class RankedSingleDeviation<10>;

}  // namespace stlqpp
