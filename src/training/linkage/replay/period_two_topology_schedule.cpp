#include "stlqpp/training/linkage/replay/period_two_topology_schedule.h"

#include <cassert>
#include <cstdint>

namespace stlqpp {

StatusOr<PeriodTwoTopologySchedule> PeriodTwoTopologySchedule::Create(
    std::uint32_t round_count) {
  if (round_count == 0U) {
    return Status::InvalidArgument(
        "period-two topology schedule requires at least one round");
  }
  return PeriodTwoTopologySchedule(round_count);
}

TopologyRoundKind PeriodTwoTopologySchedule::kind(
    std::uint32_t round) const noexcept {
  assert(round < round_count_);
  return round == 0U || round + 1U == round_count_ || (round % 2U) == 0U
      ? TopologyRoundKind::kFull
      : TopologyRoundKind::kReplay;
}

}  // namespace stlqpp
