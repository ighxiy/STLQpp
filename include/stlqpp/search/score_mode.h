#pragma once

#include <cstdint>

namespace stlqpp {

enum class SearchScoreMode : std::uint8_t {
  kClassic = 0U,
  kScaled,
  kCosineRecon,
  kMipsRecon,
  kMipsScaled,
};

[[nodiscard]] constexpr bool IsScaledScoreMode(SearchScoreMode mode) noexcept {
  return mode == SearchScoreMode::kScaled || mode == SearchScoreMode::kMipsScaled;
}

[[nodiscard]] constexpr bool IsMipsScoreMode(SearchScoreMode mode) noexcept {
  return mode == SearchScoreMode::kMipsRecon || mode == SearchScoreMode::kMipsScaled;
}

// Resident tables: -2*gain for scaled L2, -gain for scaled MIPS.
// The L2 loop uses an FMA; the independent MIPS loop only multiplies.
struct ScaledScoreTableView final {
  const float* dot_multipliers = nullptr;
  const float* biases = nullptr;
  std::uint32_t center_count = 0U;

  [[nodiscard]] bool enabled() const noexcept {
    return dot_multipliers != nullptr && biases != nullptr;
  }
};

}  // namespace stlqpp
