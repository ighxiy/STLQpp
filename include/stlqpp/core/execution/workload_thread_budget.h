#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace stlqpp {

struct WorkloadThreadBudget final {
  std::uint64_t work_units = 0U;
  std::uint64_t units_per_worker = 1U;
  std::uint32_t available_workers = 1U;
  std::uint32_t concurrent_jobs = 1U;
  std::uint32_t maximum_workers = 1U;
};

// A shared capacity rule. Callers supply stage-specific units and calibration;
// the result is bounded by the available CPU capacity and concurrent jobs.
[[nodiscard]] constexpr std::uint32_t SelectWorkloadThreadCount(
    WorkloadThreadBudget budget) noexcept {
  const std::uint32_t available = std::max(1U, budget.available_workers);
  const std::uint32_t concurrent = std::max(1U, budget.concurrent_jobs);
  const std::uint32_t limit = std::max(1U, std::min(
      budget.maximum_workers, std::max(1U, available / concurrent)));
  const std::uint64_t unit = std::max<std::uint64_t>(
      1U, budget.units_per_worker);
  const std::uint64_t desired = budget.work_units / unit +
      static_cast<std::uint64_t>(budget.work_units % unit != 0U);
  std::uint32_t selected = 1U;
  while (selected < desired && selected < limit) {
    selected = selected > limit / 2U ? limit : selected * 2U;
  }
  return selected;
}

[[nodiscard]] constexpr std::uint64_t SaturatingWorkMultiply(
    std::uint64_t left, std::uint64_t right) noexcept {
  return right != 0U && left >
      std::numeric_limits<std::uint64_t>::max() / right
      ? std::numeric_limits<std::uint64_t>::max() : left * right;
}

// Approximate Cholesky plus triangular solves. The 400M-unit calibration
// selects 2/4 workers for M5 and 16/16 for M10 at 128 dimensions, with a
// 32-worker outer capacity; it is a capacity policy, not a numerical rule.
[[nodiscard]] constexpr std::uint32_t SelectLargeSpdThreadCount(
    std::uint32_t order, std::uint32_t rhs_count,
    std::uint32_t available_workers) noexcept {
  if (order < 1024U) return 1U;
  const std::uint64_t square = SaturatingWorkMultiply(order, order);
  const std::uint64_t linear = static_cast<std::uint64_t>(order / 3U) +
      rhs_count;
  return SelectWorkloadThreadCount({
      .work_units = SaturatingWorkMultiply(square, linear),
      .units_per_worker = 400'000'000U,
      .available_workers = available_workers,
      .concurrent_jobs = 1U,
      .maximum_workers = 16U,
  });
}

}  // namespace stlqpp
