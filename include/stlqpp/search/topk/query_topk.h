#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

namespace stlqpp {

struct SearchCandidate final {
  float squared_distance = 0.0F;
  std::uint32_t id = 0U;
};

class QueryTopK final {
 public:
  explicit QueryTopK(std::uint32_t top_k)
      : capacity_(top_k), entries_(top_k) {
    assert(top_k != 0U);
  }

  [[nodiscard]] bool full() const noexcept { return size_ == capacity_; }
  [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::uint32_t size() const noexcept { return size_; }
  [[nodiscard]] float cutoff_distance() const noexcept {
    return entries_[0].squared_distance;
  }
  [[nodiscard]] std::uint32_t cutoff_id() const noexcept {
    return entries_[0].id;
  }

  void Consider(std::uint32_t id, float squared_distance) noexcept {
    const SearchCandidate candidate = {squared_distance, id};
    if (size_ != capacity_) {
      entries_[size_] = candidate;
      ++size_;
      if (size_ == capacity_) {
        BuildHeap();
      }
      return;
    }
    if (Better(candidate, entries_[0])) {
      entries_[0] = candidate;
      SiftDown(0U);
    }
  }

  // The dense score sink has already applied the exact (distance, id)
  // boundary check. Keeping that check outside the heap avoids repeating it
  // for the sparse set of SIMD survivors.
  void ConsiderAssumeBetter(std::uint32_t id,
                            float squared_distance) noexcept {
    assert(size_ == capacity_);
    entries_[0] = {squared_distance, id};
    SiftDown(0U);
  }

  [[nodiscard]] const std::vector<SearchCandidate>& Finalize() noexcept;
  void FinalizeTo(std::uint64_t* ids, float* squared_distances) noexcept;
  void Reset() noexcept;

  [[nodiscard]] const std::vector<SearchCandidate>& results() const noexcept {
    return entries_;
  }

 private:
  [[nodiscard]] static bool Better(const SearchCandidate& lhs,
                                   const SearchCandidate& rhs) noexcept {
    return lhs.squared_distance < rhs.squared_distance ||
           (lhs.squared_distance == rhs.squared_distance && lhs.id < rhs.id);
  }
  [[nodiscard]] static bool Worse(const SearchCandidate& lhs,
                                  const SearchCandidate& rhs) noexcept {
    return Better(rhs, lhs);
  }

  void BuildHeap() noexcept {
    for (std::uint32_t parent = capacity_ / 2U; parent != 0U; --parent) {
      SiftDown(parent - 1U);
    }
  }

  void SiftDown(std::uint32_t parent) noexcept {
    const SearchCandidate value = entries_[parent];
    while (true) {
      const std::uint32_t left = 2U * parent + 1U;
      if (left >= size_) {
        break;
      }
      const std::uint32_t right = left + 1U;
      std::uint32_t worse_child = left;
      if (right < size_ && Worse(entries_[right], entries_[left])) {
        worse_child = right;
      }
      if (!Worse(entries_[worse_child], value)) {
        break;
      }
      entries_[parent] = entries_[worse_child];
      parent = worse_child;
    }
    entries_[parent] = value;
  }

  std::uint32_t capacity_ = 0U;
  std::uint32_t size_ = 0U;
  std::vector<SearchCandidate> entries_;
};

}  // namespace stlqpp
