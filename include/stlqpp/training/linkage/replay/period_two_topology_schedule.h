#pragma once

#include "stlqpp/core/status_or.h"

#include <cstdint>

namespace stlqpp {

enum class TopologyRoundKind : std::uint8_t {
  kFull = 0U,
  kReplay = 1U,
};

// Registered training schedule: full, replay, full, replay, ... with the
// first and final rounds unconditionally full. Rounds are zero-based here.
class PeriodTwoTopologySchedule final {
 public:
  [[nodiscard]] static StatusOr<PeriodTwoTopologySchedule> Create(
      std::uint32_t round_count);

  [[nodiscard]] std::uint32_t round_count() const noexcept {
    return round_count_;
  }
  [[nodiscard]] TopologyRoundKind kind(
      std::uint32_t round) const noexcept;

 private:
  explicit PeriodTwoTopologySchedule(std::uint32_t round_count) noexcept
      : round_count_(round_count) {}

  std::uint32_t round_count_ = 0U;
};

}  // namespace stlqpp
