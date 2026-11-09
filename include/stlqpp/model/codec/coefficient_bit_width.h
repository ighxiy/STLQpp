#pragma once

#include "stlqpp/core/status_or.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace stlqpp {

inline constexpr std::uint32_t kDefaultCoefficientBits = 6U;
inline constexpr std::uint32_t kMinimumCoefficientBits = 2U;
inline constexpr std::uint32_t kMaximumCoefficientBits = 8U;

[[nodiscard]] constexpr std::int32_t CoefficientLimitFromBits(
    std::uint32_t bits) noexcept {
  return static_cast<std::int32_t>((1U << (bits - 1U)) - 1U);
}

[[nodiscard]] constexpr std::uint32_t CoefficientAlphabetFromBits(
    std::uint32_t bits) noexcept {
  return 2U * static_cast<std::uint32_t>(CoefficientLimitFromBits(bits)) + 1U;
}

[[nodiscard]] inline StatusOr<std::vector<std::uint32_t>>
NormalizeCoefficientBits(std::vector<std::uint32_t> values,
                         std::uint32_t layer_count,
                         std::string_view field) {
  if (layer_count == 0U) {
    return Status::InvalidArgument("coefficient layer count must be positive");
  }
  if (values.empty()) values.push_back(kDefaultCoefficientBits);
  for (const std::uint32_t bits : values) {
    if (bits < kMinimumCoefficientBits || bits > kMaximumCoefficientBits) {
      return Status::InvalidArgument(std::string(field) +
          " must contain coefficient widths in [2,8]");
    }
  }
  values.resize(layer_count, values.back());
  return values;
}

[[nodiscard]] inline std::uint32_t MaximumCoefficientBits(
    const std::vector<std::uint32_t>& root,
    const std::vector<std::uint32_t>& linkage) noexcept {
  std::uint32_t maximum = kMinimumCoefficientBits;
  for (const std::uint32_t bits : root) maximum = std::max(maximum, bits);
  for (const std::uint32_t bits : linkage) maximum = std::max(maximum, bits);
  return maximum;
}

[[nodiscard]] inline std::uint32_t MaximumCoefficientBits(
    const std::vector<std::uint32_t>& base,
    const std::vector<std::uint32_t>& root,
    const std::vector<std::uint32_t>& linkage) noexcept {
  return std::max(MaximumCoefficientBits(base, root),
                  MaximumCoefficientBits(base, linkage));
}

}  // namespace stlqpp
