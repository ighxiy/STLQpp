#pragma once

#include "stlqpp/model/model_shape.h"
#include <array>
#include <cstdint>

#include "stlqpp/model/model_layers.h"

namespace stlqpp {

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
struct LinkageSupport final {
  static_assert(M == 0 ||
                (M >= kMinimumModelLayers && M <= kMaximumModelLayers));
  std::array<std::uint8_t, Capacity> codes{};
};

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
struct LinkageSolution final {
  static_assert(M == 0 ||
                (M >= kMinimumModelLayers && M <= kMaximumModelLayers));
  LinkageSupport<M, Capacity> support;
  std::array<float, Capacity> coefficients{};
  float squared_error = 0.0F;
};

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
struct RankedSingleDeviationSet final {
  static_assert(M == 0 ||
                (M >= kMinimumModelLayers && M <= kMaximumModelLayers));
  // Storage covers the currently supported cumulative rank range.  The first
  // candidate is native global greedy. Rank-2 branches occupy the next M
  // entries and rank-3 branches the final M entries, so increasing rank never
  // removes or reorders a lower-rank candidate.
  std::array<LinkageSupport<M, Capacity>, 2 * Capacity + 1> candidates{};
  std::uint32_t candidate_count = M + 1U;
};

}  // namespace stlqpp
