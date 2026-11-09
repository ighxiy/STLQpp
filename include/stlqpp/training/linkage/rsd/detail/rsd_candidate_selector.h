#pragma once

#include <cstdint>

namespace stlqpp::detail {

inline constexpr std::uint32_t kMaximumRsdSelectedCandidates = 3U;

struct RsdRankedAtom final {
  float score;
  std::uint32_t atom;
};

// Selects rank_count atoms from the available 256-atom layers. Ordering is
// descending absolute normalized score, then ascending flattened atom ID.
void SelectRsdCandidates(
    const float* residual_dots,
    const float* inverse_norms,
    std::uint32_t layer_count,
    std::uint32_t available_layers,
    std::uint32_t rank_count,
    RsdRankedAtom* output) noexcept;

void SelectRsdCandidatesScalar(
    const float* residual_dots,
    const float* inverse_norms,
    std::uint32_t layer_count,
    std::uint32_t available_layers,
    std::uint32_t rank_count,
    RsdRankedAtom* output) noexcept;

void SelectRsdCandidatesAvx2(
    const float* residual_dots,
    const float* inverse_norms,
    std::uint32_t layer_count,
    std::uint32_t available_layers,
    std::uint32_t rank_count,
    RsdRankedAtom* output) noexcept;

}  // namespace stlqpp::detail
