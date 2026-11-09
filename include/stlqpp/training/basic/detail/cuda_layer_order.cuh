#pragma once

#include <cstdint>
#include <type_traits>

namespace stlqpp::detail {

// Fixed kernels retain their single-register order. Runtime M needs five bits
// per layer and two words to represent all twenty-four layers without truncation.
struct RuntimeCudaLayerOrder {
  std::uint64_t low = 0;
  std::uint64_t high = 0;
};

template <int M>
using CudaLayerOrder =
    std::conditional_t<M == 0, RuntimeCudaLayerOrder, std::uint64_t>;

template <int M>
CudaLayerOrder<M> PackCudaLayerOrder(const std::uint8_t* order,
                                     std::uint32_t count) {
  CudaLayerOrder<M> packed{};
  for (std::uint32_t i = 0; i < count; ++i) {
    if constexpr (M == 0) {
      auto& word = i < 12U ? packed.low : packed.high;
      word |= static_cast<std::uint64_t>(order[i]) << ((i % 12U) * 5U);
    } else {
      packed |= static_cast<std::uint64_t>(order[i]) << (i * 4U);
    }
  }
  return packed;
}

template <int M>
__device__ __forceinline__ std::uint32_t ReadCudaLayerOrder(
    CudaLayerOrder<M> packed, std::uint32_t index) {
  if constexpr (M == 0) {
    const auto word = index < 12U ? packed.low : packed.high;
    return static_cast<std::uint32_t>((word >> ((index % 12U) * 5U)) & 31U);
  } else {
    return static_cast<std::uint32_t>((packed >> (index * 4U)) & 15U);
  }
}

}  // namespace stlqpp::detail
