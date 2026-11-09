#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "stlqpp/model/model_layers.h"
#include "stlqpp/training/basic/support_least_squares.h"
#include "stlqpp/training/linkage/rsd/detail/rsd_candidate_selector.h"
#include "stlqpp/training/linkage/rsd/ranked_single_deviation.h"

namespace stlqpp::detail {

// One arbitrary-root RSD algorithm for fixed and runtime widths. The caller
// chooses the root and owns output storage; this primitive fits its suffix.
template <int M, std::uint32_t Capacity = kModelScratchLayers<M>, typename Algebra>
bool FitBasicRsdSupport(const Algebra& algebra, std::uint32_t root,
                        float root_dot, const float* small_dots,
                        float sample_norm2, std::uint32_t maximum_rank,
                        LeastSquaresCostMode cost_mode,
                        std::uint8_t* output_codes, float* output_coefficients,
                        float* output_error) noexcept {
  using LeastSquares =
      std::conditional_t<M == 0, RuntimeBasicSupportLeastSquares,
                         BasicSupportLeastSquares<M>>;
  const std::uint32_t m = M == 0 ? algebra.shape().m : M;
  constexpr auto capacity = Capacity;
  const std::uint32_t small_count = (m - 1U) * 256U;
  constexpr std::uint32_t kSmallCount = (capacity - 1U) * 256U;
  constexpr std::uint32_t kMaxCandidates =
      1U + (kMaximumRsdRank - 1U) * (capacity - 1U);
  std::array<std::array<std::uint8_t, capacity - 1>, kMaxCandidates>
      candidates{};
  std::array<float, kSmallCount> native{};
  std::copy_n(small_dots, small_count, native.data());
  const float root_norm2 = algebra.root_norm2(root);
  if (!(root_norm2 > 0.0F)) {
    return false;
  }
  const float root_coefficient = root_dot / root_norm2;
  for (std::uint32_t atom = 0U; atom < small_count; ++atom) {
    native[atom] -= root_coefficient * algebra.root_small_dot(root, atom);
  }
  std::uint32_t available = (1U << (m - 1U)) - 1U;
  const auto apply_atom =
      [&](std::uint32_t atom, std::uint32_t* remaining,
          std::array<float, kSmallCount>* dots,
          std::array<std::uint8_t, capacity - 1>* codes) noexcept {
        const std::uint32_t layer = atom / 256U;
        (*codes)[layer] = static_cast<std::uint8_t>(atom & 255U);
        *remaining &= ~(1U << layer);
        const float norm2 = algebra.small_norm2(atom);
        const float coefficient = norm2 > 0.0F ? (*dots)[atom] / norm2 : 0.0F;
        for (std::uint32_t other_layer = 0U; other_layer < m - 1U;
             ++other_layer) {
          if ((*remaining & (1U << other_layer)) == 0U) continue;
          const std::uint32_t begin = other_layer * 256U;
          for (std::uint32_t code = 0U; code < 256U; ++code) {
            const std::uint32_t other = begin + code;
            (*dots)[other] -= coefficient * algebra.small_dot(other, atom);
          }
        }
      };
  const auto& inverse_norms = algebra.small_inverse_norms();
  for (std::uint32_t step = 0U; step < m - 1U; ++step) {
    std::array<detail::RsdRankedAtom, 3> ranked{};
    detail::SelectRsdCandidates(native.data(), inverse_norms.data(), m - 1U,
                                available, maximum_rank, ranked.data());
    for (std::uint32_t rank = 2U; rank <= maximum_rank; ++rank) {
      const std::uint32_t proposal = 1U + (rank - 2U) * (m - 1U) + step;
      candidates[proposal] = candidates[0];
      auto branch_dots = native;
      std::uint32_t branch_available = available;
      apply_atom(ranked[rank - 1U].atom, &branch_available, &branch_dots,
                 &candidates[proposal]);
      for (std::uint32_t suffix = step + 1U; suffix < m - 1U; ++suffix) {
        detail::RsdRankedAtom best{};
        detail::SelectRsdCandidates(branch_dots.data(), inverse_norms.data(),
                                    m - 1U, branch_available, 1U, &best);
        apply_atom(best.atom, &branch_available, &branch_dots,
                   &candidates[proposal]);
      }
    }
    apply_atom(ranked[0].atom, &available, &native, &candidates[0]);
  }
  const std::uint32_t candidate_count = 1U + (maximum_rank - 1U) * (m - 1U);
  float best_error = std::numeric_limits<float>::infinity();
  std::uint32_t best_candidate = kMaxCandidates;
  std::array<float, capacity> best_coefficients{};
  for (std::uint32_t candidate = 0U; candidate < candidate_count; ++candidate) {
    std::array<float, capacity> coefficients{};
    float bump = 0.0F;
    if (!SolveBasicSupport<M, Capacity>(algebra, root, candidates[candidate].data(),
                             root_dot, small_dots, coefficients.data(), &bump))
      continue;
    const float error =
        cost_mode == LeastSquaresCostMode::kNormalEquation
            ? LeastSquares::NormalEquationSquaredCost(
                  algebra, candidates[candidate].data(), root_dot, small_dots,
                  coefficients.data(), sample_norm2, bump)
            : LeastSquares::SquaredCost(
                  algebra, root, candidates[candidate].data(), root_dot,
                  small_dots, coefficients.data(), sample_norm2);
    if (std::isfinite(error) && error < best_error) {
      best_error = error;
      best_candidate = candidate;
      best_coefficients = coefficients;
    }
  }
  if (best_candidate == kMaxCandidates) {
    return false;
  }
  std::copy_n(candidates[best_candidate].data(), m - 1U, output_codes);
  std::copy_n(best_coefficients.data(), m, output_coefficients);
  *output_error = best_error;
  return true;
}

}  // namespace stlqpp::detail
