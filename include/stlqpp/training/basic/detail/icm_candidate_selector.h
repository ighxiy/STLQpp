#pragma once

#include <cstdint>

namespace stlqpp::detail {

struct IcmCandidateSelection final {
  std::uint32_t code = 0U;
  float score = 0.0F;
};

// Selects exactly as the registered scalar loop: candidates are visited in
// ascending code order, only a strict score improvement replaces the winner,
// NaNs are ignored, and old_code survives when no score exceeds -infinity.
[[nodiscard]] IcmCandidateSelection SelectIcmCandidate(
    const float* residual_dots,
    const float* inverse_norms,
    std::uint32_t old_code,
    bool use_absolute_score) noexcept;

[[nodiscard]] IcmCandidateSelection SelectIcmCandidateScalar(
    const float* residual_dots,
    const float* inverse_norms,
    std::uint32_t old_code,
    bool use_absolute_score) noexcept;

[[nodiscard]] IcmCandidateSelection SelectIcmCandidateAvx2(
    const float* residual_dots,
    const float* inverse_norms,
    std::uint32_t old_code,
    bool use_absolute_score) noexcept;

}  // namespace stlqpp::detail
