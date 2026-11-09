#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/model_layers.h"
#include "stlqpp/model/model_shape.h"

namespace stlqpp {

template <int M>
[[nodiscard]] Status ValidateIcmLayerOrders(
    std::uint32_t sample_count, ArrayView<const std::uint8_t> orders,
    std::uint32_t m = FixedModelLayers<M>()) {
  static_assert(M == 0 ||
                (M >= kMinimumModelLayers && M <= kMaximumModelLayers));
  if (m < kMinimumModelLayers || m > kMaximumModelLayers || (M != 0 && m != M))
    return Status::InvalidArgument("invalid ICM M");
  const std::size_t required = static_cast<std::size_t>(sample_count) * m;
  if (orders.size() != required) {
    return Status::InvalidArgument(
        "one complete ICM layer permutation is required per residual");
  }
  for (std::uint32_t sample = 0U; sample < sample_count; ++sample) {
    std::array<bool, (M == 0 ? kMaximumModelLayers : M)> seen{};
    const std::uint8_t* order =
        orders.data() + static_cast<std::size_t>(sample) * m;
    for (std::uint32_t position = 0U; position < m; ++position) {
      const std::uint32_t layer = order[position];
      if (layer >= m || seen[layer]) {
        return Status::InvalidArgument(
            "ICM layer order must be a permutation of [0,M)");
      }
      seen[layer] = true;
    }
  }
  return Status::Ok();
}

// Builds the one registered per-real-row edge ICM permutation. The stream is
// keyed only by the semantic seed and stable sample ID and matches the frozen
// SplitMix64 Fisher-Yates recurrence used by the CPU/CUDA RSD and
// replay paths. Worker, CID-window and task grouping are deliberately absent.
template <int M>
[[nodiscard]] StatusOr<std::vector<std::uint8_t>> BuildEdgeIcmLayerOrders(
    std::uint32_t seed, ArrayView<const std::uint64_t> stable_sample_ids,
    std::uint32_t m = FixedModelLayers<M>());

// Frozen DEV full-topology evaluation creates one mt19937 permutation from
// the CID task seed and shares it across every residual in that task. Keep
// this separate from the stable-ID permutation stream used by replay.
template <int M>
[[nodiscard]] StatusOr<std::vector<std::uint8_t>> BuildSharedTaskIcmLayerOrders(
    std::uint32_t seed, std::uint32_t sample_count,
    std::uint32_t m = FixedModelLayers<M>());

extern template StatusOr<std::vector<std::uint8_t>> BuildEdgeIcmLayerOrders<5>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
extern template StatusOr<std::vector<std::uint8_t>> BuildEdgeIcmLayerOrders<10>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
extern template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<5>(std::uint32_t, std::uint32_t, std::uint32_t);
extern template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<10>(std::uint32_t, std::uint32_t, std::uint32_t);

}  // namespace stlqpp
