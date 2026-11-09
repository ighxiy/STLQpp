#include "stlqpp/training/linkage/rsd/ranked_single_deviation.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "stlqpp/training/linkage/rsd/detail/rsd_candidate_selector.h"

namespace stlqpp {
namespace {

template <std::size_t RankCount, int M, std::uint32_t Capacity>
std::array<detail::RsdRankedAtom, RankCount> SelectTop(
    const LinkageCodebookAlgebra<M>& algebra,
    const std::array<float, Capacity * 256>& residual_dots,
    std::uint32_t available_layers) noexcept {
  const std::uint32_t m = M == 0 ? algebra.shape().m : M;
  std::array<detail::RsdRankedAtom, RankCount> best{};
  detail::SelectRsdCandidates(residual_dots.data(),
                              algebra.inverse_norms().data(), m,
                              available_layers, RankCount, best.data());
  return best;
}

template <int M, std::uint32_t Capacity>
void CopyAvailableResidualDots(
    std::uint32_t m,
    const std::array<float, Capacity * 256>& source,
    std::uint32_t available_layers,
    std::array<float, Capacity * 256>* destination) noexcept {
  for (std::uint32_t layer = 0U; layer < m; ++layer) {
    if ((available_layers & (1U << layer)) == 0U) {
      continue;
    }
    const std::uint32_t begin = layer * 256U;
    std::copy_n(source.data() + begin, 256U, destination->data() + begin);
  }
}

template <int M, std::uint32_t Capacity>
void ApplyAtom(
    const LinkageCodebookAlgebra<M>& algebra, std::uint32_t atom,
    std::uint32_t* available_layers, LinkageSupport<M, Capacity>* support,
    std::array<float, Capacity * 256>* residual_dots) noexcept {
  const std::uint32_t m = M == 0 ? algebra.shape().m : M;
  const std::uint32_t layer = atom / 256U;
  support->codes[layer] = static_cast<std::uint8_t>(atom & 255U);
  *available_layers &= ~(1U << layer);
  const float squared_norm = algebra.squared_norm(atom);
  const float coefficient =
      squared_norm > 0.0F ? (*residual_dots)[atom] / squared_norm : 0.0F;
  // Dots for an already selected layer are never read again. Updating only
  // the remaining layers preserves every future arithmetic operation while
  // removing the dominant redundant writes from the scalar CPU path.
  for (std::uint32_t other_layer = 0U; other_layer < m; ++other_layer) {
    if ((*available_layers & (1U << other_layer)) == 0U) {
      continue;
    }
    const std::uint32_t begin = algebra.layer_offset(other_layer);
    for (std::uint32_t code = 0U; code < 256U; ++code) {
      const std::uint32_t other = begin + code;
      (*residual_dots)[other] -= coefficient * algebra.dot(other, atom);
    }
  }
}

template <int M, std::uint32_t Capacity>
void CompleteRankOneSuffix(
    const LinkageCodebookAlgebra<M>& algebra, std::uint32_t start_step,
    std::uint32_t available_layers, LinkageSupport<M, Capacity>* support,
    std::array<float, Capacity * 256>* residual_dots) noexcept {
  const std::uint32_t m = M == 0 ? algebra.shape().m : M;
  for (std::uint32_t step = start_step; step < m; ++step) {
    const detail::RsdRankedAtom best =
        SelectTop<1U, M, Capacity>(algebra, *residual_dots, available_layers)[0];
    ApplyAtom(algebra, best.atom, &available_layers, support, residual_dots);
  }
}

template <std::size_t MaximumRank, int M, std::uint32_t Capacity>
RankedSingleDeviationSet<M, Capacity> BuildThroughRank(
    const LinkageCodebookAlgebra<M>& algebra, const float* atom_dots) noexcept {
  const std::uint32_t m = M == 0 ? algebra.shape().m : M;
  static_assert(MaximumRank >= 1U && MaximumRank <= 3U);
  RankedSingleDeviationSet<M, Capacity> output;
  output.candidate_count = RsdCandidateCount<M>(MaximumRank, m);
  std::array<float, Capacity * 256> native_residual_dots{};
  std::copy_n(atom_dots, algebra.atom_count(), native_residual_dots.data());
  std::uint32_t native_available = (1U << m) - 1U;

  for (std::uint32_t step = 0U; step < m; ++step) {
    const auto ranked =
        SelectTop<MaximumRank, M, Capacity>(algebra, native_residual_dots, native_available);
    for (std::uint32_t rank = 2U;
         rank <= static_cast<std::uint32_t>(MaximumRank); ++rank) {
      const std::uint32_t proposal = 1U + (rank - 2U) * m + step;
      LinkageSupport<M, Capacity>& branch = output.candidates[proposal];
      branch = output.candidates[0];
      std::array<float, Capacity * 256> branch_residual_dots;
      CopyAvailableResidualDots<M, Capacity>(m, native_residual_dots, native_available,
                                   &branch_residual_dots);
      std::uint32_t branch_available = native_available;
      ApplyAtom(algebra, ranked[rank - 1U].atom, &branch_available, &branch,
                &branch_residual_dots);
      CompleteRankOneSuffix(algebra, step + 1U, branch_available, &branch,
                            &branch_residual_dots);
    }

    ApplyAtom(algebra, ranked[0].atom, &native_available, &output.candidates[0],
              &native_residual_dots);
  }
  return output;
}

template <std::size_t MaximumRank, int M, std::uint32_t Capacity>
RankedSingleDeviationSet<M, Capacity> BuildFixedLayerZeroThroughRank(
    const LinkageCodebookAlgebra<M>& algebra, const float* atom_dots,
    std::uint8_t fixed_layer_zero_code) noexcept {
  const std::uint32_t m = M == 0 ? algebra.shape().m : M;
  static_assert(MaximumRank >= 1U && MaximumRank <= 3U);
  RankedSingleDeviationSet<M, Capacity> output;
  output.candidate_count =
      1U + (m - 1U) * (static_cast<std::uint32_t>(MaximumRank) - 1U);
  std::array<float, Capacity * 256> native_residual_dots{};
  std::copy_n(atom_dots, algebra.atom_count(), native_residual_dots.data());
  std::uint32_t native_available = (1U << m) - 1U;
  ApplyAtom(algebra, fixed_layer_zero_code, &native_available,
            &output.candidates[0], &native_residual_dots);

  for (std::uint32_t step = 1U; step < m; ++step) {
    const auto ranked =
        SelectTop<MaximumRank, M, Capacity>(algebra, native_residual_dots, native_available);
    for (std::uint32_t rank = 2U;
         rank <= static_cast<std::uint32_t>(MaximumRank); ++rank) {
      const std::uint32_t proposal = 1U + (rank - 2U) * (m - 1U) + (step - 1U);
      LinkageSupport<M, Capacity>& branch = output.candidates[proposal];
      branch = output.candidates[0];
      std::array<float, Capacity * 256> branch_residual_dots;
      CopyAvailableResidualDots<M, Capacity>(m, native_residual_dots, native_available,
                                   &branch_residual_dots);
      std::uint32_t branch_available = native_available;
      ApplyAtom(algebra, ranked[rank - 1U].atom, &branch_available, &branch,
                &branch_residual_dots);
      CompleteRankOneSuffix(algebra, step + 1U, branch_available, &branch,
                            &branch_residual_dots);
    }
    ApplyAtom(algebra, ranked[0].atom, &native_available, &output.candidates[0],
              &native_residual_dots);
  }
  return output;
}

}  // namespace

template <int M, std::uint32_t Capacity>
RankedSingleDeviationSet<M, Capacity> RankedSingleDeviation<M, Capacity>::Build(
    const LinkageCodebookAlgebra<M>& algebra, const float* atom_dots,
    std::uint32_t maximum_rank) noexcept {
  if (maximum_rank == 1U) {
    return BuildThroughRank<1U, M, Capacity>(algebra, atom_dots);
  }
  return maximum_rank == 3U ? BuildThroughRank<3U, M, Capacity>(algebra, atom_dots)
                            : BuildThroughRank<2U, M, Capacity>(algebra, atom_dots);
}

template <int M, std::uint32_t Capacity>
RankedSingleDeviationSet<M, Capacity> RankedSingleDeviation<M, Capacity>::BuildFixedLayerZero(
    const LinkageCodebookAlgebra<M>& algebra, const float* atom_dots,
    std::uint8_t fixed_layer_zero_code, std::uint32_t maximum_rank) noexcept {
  if (maximum_rank == 1U) {
    return BuildFixedLayerZeroThroughRank<1U, M, Capacity>(algebra, atom_dots,
                                              fixed_layer_zero_code);
  }
  return maximum_rank == 3U ? BuildFixedLayerZeroThroughRank<3U, M, Capacity>(
                                  algebra, atom_dots, fixed_layer_zero_code)
                            : BuildFixedLayerZeroThroughRank<2U, M, Capacity>(
                                  algebra, atom_dots, fixed_layer_zero_code);
}

template class RankedSingleDeviation<0>;
template class RankedSingleDeviation<0, kWideRuntimeScratchLayers>;
template class RankedSingleDeviation<5>;
template class RankedSingleDeviation<12>;
template class RankedSingleDeviation<6>;
template class RankedSingleDeviation<10>;

}  // namespace stlqpp
