#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace stlqpp {

struct BestFitCacheLimits final {
  std::size_t maximum_buffers = 8U;
  std::uint64_t maximum_bytes = 256ULL * 1024ULL * 1024ULL;
};

struct BestFitCacheStats final {
  std::uint64_t idle_buffers = 0U;
  std::uint64_t idle_bytes = 0U;
  std::uint64_t peak_idle_bytes = 0U;
  std::uint64_t evictions = 0U;
  std::uint64_t allocation_retries = 0U;
};

// Worker-confined ownership pool for expensive, capacity-sized allocations.
// A checked-out resource is absent from this object and therefore cannot be
// evicted. Resource destruction is the release operation for the underlying
// allocator (cudaFree for the CUDA backend).
template <typename Resource, typename CapacityBytes>
class BoundedBestFitCache final {
 public:
  BoundedBestFitCache(BestFitCacheLimits limits,
                      CapacityBytes capacity_bytes)
      : limits_(limits), capacity_bytes_(std::move(capacity_bytes)) {
    assert(limits_.maximum_buffers != 0U);
    assert(limits_.maximum_bytes != 0U);
  }

  BoundedBestFitCache(const BoundedBestFitCache&) = delete;
  BoundedBestFitCache& operator=(const BoundedBestFitCache&) = delete;

  template <typename Allocator>
  [[nodiscard]] StatusOr<Resource> Acquire(
      std::uint64_t required_bytes,
      Allocator&& allocator) {
    auto best = idle_.end();
    for (auto current = idle_.begin(); current != idle_.end(); ++current) {
      const std::uint64_t capacity = capacity_bytes_(*current);
      if (capacity >= required_bytes &&
          (best == idle_.end() || capacity < capacity_bytes_(*best))) {
        best = current;
      }
    }
    if (best != idle_.end()) {
      const std::uint64_t capacity = capacity_bytes_(*best);
      Resource result = std::move(*best);
      idle_.erase(best);
      idle_bytes_ -= capacity;
      return StatusOr<Resource>(std::move(result));
    }

    auto allocated = std::invoke(
        std::forward<Allocator>(allocator), required_bytes);
    if (allocated.ok() ||
        allocated.status().code() != StatusCode::kResourceExhausted) {
      return allocated;
    }
    Clear();
    ++allocation_retries_;
    return std::invoke(std::forward<Allocator>(allocator), required_bytes);
  }

  void Release(Resource resource) {
    const std::uint64_t capacity = capacity_bytes_(resource);
    assert(capacity != 0U);
    if (capacity > limits_.maximum_bytes) {
      ++evictions_;
      return;
    }
    idle_bytes_ += capacity;
    idle_.push_back(std::move(resource));
    EnforceLimits();
    peak_idle_bytes_ = std::max(peak_idle_bytes_, idle_bytes_);
  }

  void Clear() noexcept {
    evictions_ += idle_.size();
    idle_.clear();
    idle_bytes_ = 0U;
  }

  [[nodiscard]] BestFitCacheStats Stats() const noexcept {
    return BestFitCacheStats{
        .idle_buffers = idle_.size(),
        .idle_bytes = idle_bytes_,
        .peak_idle_bytes = peak_idle_bytes_,
        .evictions = evictions_,
        .allocation_retries = allocation_retries_,
    };
  }

 private:
  void EnforceLimits() {
    while (idle_.size() > limits_.maximum_buffers ||
           idle_bytes_ > limits_.maximum_bytes) {
      // Preserve expensive high-capacity allocations for the long tail of
      // large IVF lists. Best-fit acquisition still consumes the smallest
      // sufficient block; under pressure the smallest idle capacity is the
      // least useful survivor and is released first.
      const auto smallest = std::min_element(
          idle_.begin(), idle_.end(),
          [this](const Resource& left, const Resource& right) {
            return capacity_bytes_(left) < capacity_bytes_(right);
          });
      idle_bytes_ -= capacity_bytes_(*smallest);
      idle_.erase(smallest);
      ++evictions_;
    }
  }

  BestFitCacheLimits limits_;
  CapacityBytes capacity_bytes_;
  std::vector<Resource> idle_;
  std::uint64_t idle_bytes_ = 0U;
  std::uint64_t peak_idle_bytes_ = 0U;
  std::uint64_t evictions_ = 0U;
  std::uint64_t allocation_retries_ = 0U;
};

}  // namespace stlqpp
