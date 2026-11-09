#pragma once

#include "stlqpp/core/status_or.h"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace stlqpp {

// Shared bounded lease table for several frontends that address the same
// physical worker lanes. A lane can host multiple operation types without
// allowing their CUDA stream/workspace lifetimes to overlap.
class ExclusiveIndexPool final {
 public:
  [[nodiscard]] static StatusOr<std::shared_ptr<ExclusiveIndexPool>> Create(
      std::size_t size) {
    if (size == 0U) {
      return Status::InvalidArgument(
          "exclusive index pool requires at least one lane");
    }
    return std::shared_ptr<ExclusiveIndexPool>(
        new ExclusiveIndexPool(size));
  }

  [[nodiscard]] StatusOr<std::size_t> Acquire() const {
    std::unique_lock lock(mutex_);
    available_.wait(lock, [this]() {
      return !free_.empty() || live_ == 0U;
    });
    if (free_.empty()) {
      return Status::FailedPrecondition(
          "all shared execution lanes have been retired");
    }
    const std::size_t index = free_.back();
    free_.pop_back();
    return index;
  }

  void Release(std::size_t index) const {
    {
      std::lock_guard lock(mutex_);
      if (index >= retired_.size() || retired_[index]) return;
      free_.push_back(index);
    }
    available_.notify_one();
  }

  // Permanently removes a failed physical lane from every frontend sharing
  // this lease table. The caller may retire either a leased or idle lane.
  void Retire(std::size_t index) const {
    {
      std::lock_guard lock(mutex_);
      if (index >= retired_.size() || retired_[index]) return;
      retired_[index] = true;
      --live_;
      const auto found = std::find(free_.begin(), free_.end(), index);
      if (found != free_.end()) free_.erase(found);
    }
    available_.notify_all();
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  explicit ExclusiveIndexPool(std::size_t size) : size_(size) {
    free_.reserve(size);
    retired_.assign(size, false);
    live_ = size;
    for (std::size_t index = size; index != 0U; --index) {
      free_.push_back(index - 1U);
    }
  }

  std::size_t size_ = 0U;
  mutable std::mutex mutex_;
  mutable std::condition_variable available_;
  mutable std::vector<std::size_t> free_;
  mutable std::vector<bool> retired_;
  mutable std::size_t live_ = 0U;
};

}  // namespace stlqpp
