#pragma once

#include <cstdint>

#include "stlqpp/core/status.h"

namespace stlqpp {

inline constexpr std::uint32_t kMinimumModelLayers = 2U;
inline constexpr std::uint32_t kMaximumModelLayers = 24U;

// Only fixed-width instantiations may omit a width argument. Instantiation of
// this default for M=0 is a compile error, not a runtime validation branch.
template <int M>
constexpr std::uint32_t FixedModelLayers() noexcept {
  static_assert(M >= kMinimumModelLayers && M <= kMaximumModelLayers,
                "runtime-M requires an explicit model width");
  return M;
}

struct ModelShape final {
  std::uint32_t dimension = 0;
  std::uint32_t m = 0;
  std::uint32_t nlist = 0;
  std::uint32_t residual_codebook_size = 256;

  [[nodiscard]] bool IsRegisteredProduct() const noexcept {
    return m >= kMinimumModelLayers && m <= kMaximumModelLayers;
  }

  [[nodiscard]] Status Validate() const;
};

}  // namespace stlqpp
