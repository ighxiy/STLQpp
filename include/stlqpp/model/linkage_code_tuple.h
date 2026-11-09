#pragma once

#include "stlqpp/model/model_shape.h"
#include "stlqpp/model/model_layers.h"

#include <array>
#include <cstdint>

namespace stlqpp {

// Persistent-neutral support tuple shared by the topology producer and final
// index assembler. The first atom can name the large native root codebook;
// every trailing layer is fixed to 256 entries.
template <int M>
struct LinkageCodeTuple final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t first = 0U;
  std::array<std::uint8_t, (M == 0 ? kMaximumModelLayers : M) - 1U> trailing{};

  [[nodiscard]] std::uint32_t code(std::uint32_t layer) const noexcept {
    return layer == 0U ? first : trailing[layer - 1U];
  }

  friend bool operator==(const LinkageCodeTuple& left,
                         const LinkageCodeTuple& right) noexcept {
    return left.first == right.first && left.trailing == right.trailing;
  }
  friend bool operator!=(const LinkageCodeTuple& left,
                         const LinkageCodeTuple& right) noexcept {
    return !(left == right);
  }
};

}  // namespace stlqpp
