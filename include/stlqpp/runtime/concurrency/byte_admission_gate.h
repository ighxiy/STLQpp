#pragma once

#include "stlqpp/core/status.h"

#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace stlqpp {

struct ByteAdmissionStats final {
  std::uint64_t capacity_bytes = 0U;
  std::uint64_t live_bytes = 0U;
  std::uint64_t peak_live_bytes = 0U;
};

// Blocking outer-work admission. GPU worker threads never wait on this gate;
// registrations acquire credits before entering a worker queue, while ordered
// unregistration returns them after the backend releases live ownership.
class ByteAdmissionGate final {
 public:
  explicit ByteAdmissionGate(std::uint64_t capacity_bytes)
      : capacity_bytes_(capacity_bytes) {
    assert(capacity_bytes_ != 0U);
  }

  ByteAdmissionGate(const ByteAdmissionGate&) = delete;
  ByteAdmissionGate& operator=(const ByteAdmissionGate&) = delete;

  [[nodiscard]] Status Acquire(std::uint64_t bytes) {
    if (bytes == 0U || bytes > capacity_bytes_) {
      return Status::ResourceExhausted(
          "registration exceeds the live byte admission capacity");
    }
    std::unique_lock<std::mutex> lock(mutex_);
    available_.wait(lock, [this, bytes] {
      return live_bytes_ <= capacity_bytes_ - bytes;
    });
    live_bytes_ += bytes;
    peak_live_bytes_ =
        live_bytes_ > peak_live_bytes_ ? live_bytes_ : peak_live_bytes_;
    return Status::Ok();
  }

  void Release(std::uint64_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(bytes <= live_bytes_);
    live_bytes_ -= bytes;
    available_.notify_all();
  }

  [[nodiscard]] ByteAdmissionStats Stats() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return {
        .capacity_bytes = capacity_bytes_,
        .live_bytes = live_bytes_,
        .peak_live_bytes = peak_live_bytes_,
    };
  }

 private:
  const std::uint64_t capacity_bytes_;
  mutable std::mutex mutex_;
  std::condition_variable available_;
  std::uint64_t live_bytes_ = 0U;
  std::uint64_t peak_live_bytes_ = 0U;
};

}  // namespace stlqpp
