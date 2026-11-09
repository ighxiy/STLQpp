#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace stlqpp {
namespace {

constexpr std::uint64_t kSplitMixIncrement = 0x9e3779b97f4a7c15ULL;
constexpr std::uint64_t kSplitMixFirst = 0xbf58476d1ce4e5b9ULL;
constexpr std::uint64_t kSplitMixSecond = 0x94d049bb133111ebULL;

std::uint64_t SplitMixNext(std::uint64_t* state) noexcept {
  std::uint64_t value = (*state += kSplitMixIncrement);
  value = (value ^ (value >> 30U)) * kSplitMixFirst;
  value = (value ^ (value >> 27U)) * kSplitMixSecond;
  return value ^ (value >> 31U);
}

// Portable high half of an unsigned 64x64 product. This is the exact host
// counterpart of CUDA __umul64hi and avoids a compiler-specific u128 type on
// Windows.
std::uint64_t MultiplyHigh(std::uint64_t left,
                           std::uint64_t right) noexcept {
  const std::uint64_t left_low = static_cast<std::uint32_t>(left);
  const std::uint64_t left_high = left >> 32U;
  const std::uint64_t right_low = static_cast<std::uint32_t>(right);
  const std::uint64_t right_high = right >> 32U;
  const std::uint64_t low_product = left_low * right_low;
  const std::uint64_t middle =
      left_high * right_low + (low_product >> 32U);
  const std::uint64_t middle_low = static_cast<std::uint32_t>(middle);
  const std::uint64_t middle_high = middle >> 32U;
  const std::uint64_t upper_middle =
      middle_low + left_low * right_high;
  return left_high * right_high + middle_high +
      (upper_middle >> 32U);
}

std::uint64_t UniformBelow(std::uint64_t* state,
                           std::uint64_t bound) noexcept {
  return MultiplyHigh(SplitMixNext(state), bound);
}

}  // namespace

template <int M>
StatusOr<std::vector<std::uint8_t>> BuildEdgeIcmLayerOrders(
    std::uint32_t seed,
    ArrayView<const std::uint64_t> stable_sample_ids, std::uint32_t m) {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  if (m < 2U || m > stlqpp::kMaximumModelLayers || (M != 0 && m != M)) return Status::InvalidArgument("invalid ICM M");
  if (stable_sample_ids.size() >
      std::numeric_limits<std::size_t>::max() / m) {
    return Status::ResourceExhausted(
        "edge ICM permutation stream exceeds the host address domain");
  }
  std::vector<std::uint8_t> output(stable_sample_ids.size() * m);
  for (std::size_t sample = 0U; sample < stable_sample_ids.size(); ++sample) {
    std::array<std::uint8_t, (M == 0 ? kMaximumModelLayers : M)> order{};
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      order[layer] = static_cast<std::uint8_t>(layer);
    }
    std::uint64_t state = static_cast<std::uint64_t>(seed);
    state ^= (stable_sample_ids[sample] + 1ULL) * kSplitMixFirst;
    // The production edge codec performs exactly one ICM sweep: phase zero,
    // iteration zero in the frozen recurrence.
    state ^= kSplitMixSecond;
    state ^= kSplitMixIncrement;
    for (std::uint32_t remaining = m;
         remaining > 1U; --remaining) {
      const std::uint32_t last = remaining - 1U;
      const std::uint32_t other = static_cast<std::uint32_t>(
          UniformBelow(&state, remaining));
      const std::uint8_t value = order[last];
      order[last] = order[other];
      order[other] = value;
    }
    for (std::uint32_t position = 0U;
         position < m; ++position) {
      output[sample * m + position] = order[position];
    }
  }
  return output;
}

template <int M>
StatusOr<std::vector<std::uint8_t>> BuildSharedTaskIcmLayerOrders(
    std::uint32_t seed,
    std::uint32_t sample_count, std::uint32_t m) {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  if (m < 2U || m > stlqpp::kMaximumModelLayers || (M != 0 && m != M)) return Status::InvalidArgument("invalid ICM M");
  if (sample_count > std::numeric_limits<std::size_t>::max() / m) {
    return Status::ResourceExhausted(
        "shared task ICM permutation stream exceeds the host address domain");
  }
  std::array<std::uint8_t, (M == 0 ? kMaximumModelLayers : M)> order{};
  std::iota(order.begin(), order.begin() + m, static_cast<std::uint8_t>(0U));
  std::mt19937 random(seed);
  std::shuffle(order.begin(), order.begin() + m, random);

  std::vector<std::uint8_t> output(
      static_cast<std::size_t>(sample_count) * m);
  for (std::uint32_t sample = 0U; sample < sample_count; ++sample) {
    std::copy(order.begin(), order.begin() + m,
              output.begin() + static_cast<std::size_t>(sample) * m);
  }
  return output;
}

template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<0>(std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<0>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<5>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<2>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<3>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<4>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<7>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<8>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<9>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<11>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<12>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<13>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<14>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<15>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<16>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<17>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<18>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<19>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<20>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<6>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildEdgeIcmLayerOrders<10>(
    std::uint32_t, ArrayView<const std::uint64_t>, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<5>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<2>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<3>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<4>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<7>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<8>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<9>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<11>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<12>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<13>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<14>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<15>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<16>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<17>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<18>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<19>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<20>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<6>(std::uint32_t, std::uint32_t, std::uint32_t);
template StatusOr<std::vector<std::uint8_t>>
BuildSharedTaskIcmLayerOrders<10>(std::uint32_t, std::uint32_t, std::uint32_t);

}  // namespace stlqpp
