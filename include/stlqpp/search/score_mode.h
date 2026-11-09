#pragma once

#include <cstdint>

namespace stlqpp {

enum class SearchScoreMode : std::uint8_t {
  kClassic = 0U,
  kScaled,
  kCosineRecon,
  kMipsRecon,
};

// Resident hot-path table. Dot multipliers are precomputed as -2*scale so
// the AVX2 loop performs one FMA per row and no separate scale multiply.
struct ScaledScoreTableView final {
  const float* dot_multipliers = nullptr;
  const float* biases = nullptr;
  std::uint32_t center_count = 0U;

  [[nodiscard]] bool enabled() const noexcept {
    return dot_multipliers != nullptr && biases != nullptr;
  }
};

}  // namespace stlqpp
