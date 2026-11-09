#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "stlqpp/model/model_shape.h"

namespace stlqpp {

// M=0 selects one runtime-width implementation, not a padded fixed-width model.
// Fixed fast paths retain their original std::array storage.
template <typename T, int M, int Skip = 0>
using ModelLayers =
    std::conditional_t<M == 0, std::vector<T>,
                       std::array<T, (M > Skip ? M - Skip : 0)>>;

template <typename T, int M, int Skip = 0>
ModelLayers<T, M, Skip> MakeModelLayers(std::uint32_t m) {
  if constexpr (M == 0) {
    return ModelLayers<T, M, Skip>(m - Skip);
  } else {
    return {};
  }
}

// Numerical capacity is deliberately independent of the supported model range.
// Existing runtime widths keep the original stack/shared/local allocation.
inline constexpr std::uint32_t kLegacyRuntimeScratchLayers = 20U;
inline constexpr std::uint32_t kWideRuntimeScratchLayers = 24U;

// Temporary bounded support/solver state. Storage capacity is not logical M.
template <int M>
inline constexpr std::uint32_t kModelScratchLayers =
    M == 0 ? kLegacyRuntimeScratchLayers : M;

}  // namespace stlqpp
