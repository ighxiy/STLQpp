#pragma once

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace stlqpp {

template <typename T>
class BoundedQueue final {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {
    assert(capacity_ != 0U);
  }

  BoundedQueue(const BoundedQueue&) = delete;
  BoundedQueue& operator=(const BoundedQueue&) = delete;

  [[nodiscard]] bool Push(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [this] {
      return closed_ || queue_.size() < capacity_;
    });
    if (closed_) {
      return false;
    }
    queue_.push_back(std::move(value));
    not_empty_.notify_one();
    return true;
  }

  // Attempts one admission without waiting for capacity. `value` is moved
  // only after a slot has been reserved, so a caller can retain move-only
  // work and make progress elsewhere when the queue is full.
  [[nodiscard]] bool TryPush(T* value) {
    assert(value != nullptr);
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || queue_.size() >= capacity_) {
      return false;
    }
    queue_.push_back(std::move(*value));
    not_empty_.notify_one();
    return true;
  }

  [[nodiscard]] std::optional<T> Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [this] {
      return closed_ || !queue_.empty();
    });
    if (queue_.empty()) {
      return std::nullopt;
    }
    std::optional<T> value(std::move(queue_.front()));
    queue_.pop_front();
    not_full_.notify_one();
    return value;
  }

  [[nodiscard]] std::optional<T> TryPop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return std::nullopt;
    }
    std::optional<T> value(std::move(queue_.front()));
    queue_.pop_front();
    not_full_.notify_one();
    return value;
  }

  template <typename Clock, typename Duration>
  [[nodiscard]] std::optional<T> PopUntil(
      const std::chrono::time_point<Clock, Duration>& deadline) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait_until(lock, deadline, [this] {
      return closed_ || !queue_.empty();
    });
    if (queue_.empty()) {
      return std::nullopt;
    }
    std::optional<T> value(std::move(queue_.front()));
    queue_.pop_front();
    not_full_.notify_one();
    return value;
  }

  void Close() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
  }

 private:
  const std::size_t capacity_;
  std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::deque<T> queue_;
  bool closed_ = false;
};

}  // namespace stlqpp
