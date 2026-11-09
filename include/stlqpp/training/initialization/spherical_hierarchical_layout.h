#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/parallel_for.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

// Frozen DEV large-K layout. Fine center IDs stay flat and are partitioned
// into contiguous ranges [coarse * fine_capacity, ...). The final ranges may
// be short or empty because fine_capacity is ceil(K / K1), exactly as in DEV.
struct SphericalHierarchicalLayout final {
  std::uint32_t center_count = 0U;
  std::uint32_t coarse_count = 0U;
  std::uint32_t fine_capacity = 0U;

  [[nodiscard]] std::uint32_t group_begin(
      std::uint32_t coarse) const noexcept {
    return coarse * fine_capacity;
  }

  [[nodiscard]] std::uint32_t group_size(
      std::uint32_t coarse) const noexcept;

  [[nodiscard]] std::uint32_t non_empty_coarse_count() const noexcept {
    return fine_capacity == 0U ? 0U : static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(center_count) + fine_capacity - 1U) /
        fine_capacity);
  }
};

struct SphericalHierarchicalSplit final {
  std::uint32_t fixed_coarse_count = 0U;
  std::uint32_t minimum_coarse_count = 128U;
  std::uint32_t maximum_coarse_count = 1'024U;
  bool round_to_power_of_two = true;
};

[[nodiscard]] SphericalHierarchicalLayout MakeSphericalHierarchicalLayout(
    std::uint32_t center_count,
    SphericalHierarchicalSplit split) noexcept;

// Registered DEV split:
// round(sqrt(K)) -> clamp [128, 1024] -> nearest power of two (ties upward)
// -> cap K1 by K; K2 = ceil(K / K1).
[[nodiscard]] SphericalHierarchicalLayout
MakeRegisteredSphericalHierarchicalLayout(
    std::uint32_t center_count) noexcept;

// Rebuilds the normalized coarse representatives from the current fine
// centers. This is performed on every Lloyd iteration before assignment.
[[nodiscard]] StatusOr<std::vector<float>> BuildSphericalCoarseCenters(
    ColumnMajorMatrixF32View fine,
    const SphericalHierarchicalLayout& layout,
    const ParallelForExecutor& parallel);

}  // namespace stlqpp
