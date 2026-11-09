#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"

#include <cassert>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

namespace stlqpp {

namespace detail {
struct PreparedKnnRowsAccess;
}

struct PreparedKnnRow final {
  ArrayView<const std::uint32_t> local_ids;
  ArrayView<const float> squared_distances;
};

class PreparedKnnRows final {
 public:
  PreparedKnnRows() = default;
  PreparedKnnRows(PreparedKnnRows&&) noexcept = default;
  PreparedKnnRows& operator=(PreparedKnnRows&&) noexcept = default;
  PreparedKnnRows(const PreparedKnnRows&) = delete;
  PreparedKnnRows& operator=(const PreparedKnnRows&) = delete;

  [[nodiscard]] std::uint32_t row_count() const noexcept {
    return row_count_;
  }
  [[nodiscard]] std::uint32_t neighbors_per_row() const noexcept {
    return neighbors_per_row_;
  }
  // Only built-in exact/HNSW producers set this after producing unique,
  // in-range labels. Arbitrary prepared rows retain full consumer validation.
  [[nodiscard]] bool producer_guaranteed_unique() const noexcept {
    return producer_guaranteed_unique_;
  }
  [[nodiscard]] PreparedKnnRow row(std::uint32_t index) const noexcept {
    assert(index < row_count_);
    const std::size_t offset =
        static_cast<std::size_t>(index) * neighbors_per_row_;
    const std::uint32_t* ids = local_ids_.empty()
        ? nullptr
        : local_ids_.data() + offset;
    const float* distances = squared_distances_.empty()
        ? nullptr
        : squared_distances_.data() + offset;
    return {
        .local_ids = {ids, neighbors_per_row_},
        .squared_distances = {distances, neighbors_per_row_},
    };
  }

 private:
  friend class CpuExactKnnProvider;
  friend struct HnswKnnRowsAccess;
  friend struct detail::PreparedKnnRowsAccess;

  std::uint32_t row_count_ = 0U;
  std::uint32_t neighbors_per_row_ = 0U;
  bool producer_guaranteed_unique_ = false;
  std::vector<std::uint32_t> local_ids_;
  std::vector<float> squared_distances_;
};

// A pipeline worker contributes one CPU slot only after permanently exiting.
// Tail KNN calls lease these slots; active cluster owners keep their own slot.
// No credits are available while every outer worker is still alive.
class ExactKnnCpuBudget final {
 public:
  explicit ExactKnnCpuBudget(std::uint32_t workers) noexcept : workers_(workers) {
    assert(workers != 0U);
  }
  [[nodiscard]] bool RetireWorker() noexcept {
    return retired_.fetch_add(1U, std::memory_order_acq_rel) + 1U == workers_;
  }
  void Cancel() noexcept { cancelled_.store(true, std::memory_order_release); }
  [[nodiscard]] bool cancelled() const noexcept {
    return cancelled_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::uint32_t workers() const noexcept { return workers_; }
  [[nodiscard]] std::uint32_t AcquireHelpers(std::uint32_t held,
                                            std::uint32_t maximum) noexcept {
    const auto retired = retired_.load(std::memory_order_acquire);
    const auto alive = workers_ - retired;
    const auto fair_share = workers_ / std::max(1U, alive) - 1U;
    const auto limit = std::min(maximum, fair_share);
    if (held >= limit || cancelled()) return 0U;
    auto borrowed = borrowed_.load(std::memory_order_relaxed);
    while (borrowed < retired) {
      const auto count = std::min(limit - held, retired - borrowed);
      if (borrowed_.compare_exchange_weak(borrowed, borrowed + count,
                                        std::memory_order_acq_rel,
                                        std::memory_order_relaxed)) return count;
    }
    return 0U;
  }
  void ReleaseHelpers(std::uint32_t count) noexcept {
    if (count != 0U) {
      const auto previous = borrowed_.fetch_sub(count, std::memory_order_release);
      assert(previous >= count);
      (void)previous;
    }
  }
 private:
  const std::uint32_t workers_;
  std::atomic<std::uint32_t> retired_{0U};
  std::atomic<std::uint32_t> borrowed_{0U};
  std::atomic<bool> cancelled_{false};
};

class CpuExactKnnProvider final {
 public:
  explicit CpuExactKnnProvider(const F32GemmBackend& gemm) noexcept
      : gemm_(&gemm) {}

  // Evaluates each unordered pair once and offers it to both endpoint heaps.
  [[nodiscard]] StatusOr<PreparedKnnRows> BuildSymmetric(
      ColumnMajorMatrixF32View points,
      std::uint32_t requested_k,
      ExactKnnCpuBudget* cpu_budget = nullptr) const;

  // Each output label is a column-local ID in `centers`.
  [[nodiscard]] StatusOr<PreparedKnnRows> BuildBipartite(
      ColumnMajorMatrixF32View centers,
      ColumnMajorMatrixF32View queries,
      std::uint32_t requested_k,
      ExactKnnCpuBudget* cpu_budget = nullptr) const;

 private:
  const F32GemmBackend* gemm_ = nullptr;
};

}  // namespace stlqpp
