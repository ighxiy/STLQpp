#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace stlqpp::linkage_candidates {

struct RankedNeighbor final {
  float distance = std::numeric_limits<float>::infinity();
  std::uint32_t local_id = 0U;
};

[[nodiscard]] inline bool RankedNeighborLess(
    const RankedNeighbor& left,
    const RankedNeighbor& right) noexcept {
  return left.distance < right.distance ||
      (left.distance == right.distance &&
       left.local_id < right.local_id);
}

inline void OfferRankedNeighbor(RankedNeighbor candidate,
                                RankedNeighbor* heap,
                                std::uint32_t* count,
                                std::uint32_t capacity) {
  if (*count < capacity) {
    heap[*count] = candidate;
    ++*count;
    std::push_heap(heap, heap + *count, RankedNeighborLess);
    return;
  }
  if (!RankedNeighborLess(candidate, heap[0])) {
    return;
  }
  std::uint32_t hole = 0U;
  while (true) {
    const std::uint32_t left = 2U * hole + 1U;
    if (left >= *count) break;
    const std::uint32_t right = left + 1U;
    const std::uint32_t worse_child =
        right < *count && RankedNeighborLess(heap[left], heap[right])
        ? right : left;
    if (!RankedNeighborLess(candidate, heap[worse_child])) break;
    heap[hole] = heap[worse_child];
    hole = worse_child;
  }
  heap[hole] = candidate;
}

}  // namespace stlqpp::linkage_candidates
