#pragma once

#include "stlqpp/core/array_view.h"

#include <cassert>
#include <cstdint>
#include <vector>

namespace stlqpp {

struct PreparedParentCandidateRow final {
  std::uint32_t query_local_id = 0U;
  ArrayView<const std::uint32_t> parent_local_ids;
};

// CSR rows are stored in the requested current-layer processing order. Parent
// labels are cluster-local and sorted by exact (distance, local_id) order.
class PreparedParentCandidates final {
 public:
  PreparedParentCandidates() = default;
  PreparedParentCandidates(PreparedParentCandidates&&) noexcept = default;
  PreparedParentCandidates& operator=(PreparedParentCandidates&&) noexcept =
      default;
  PreparedParentCandidates(const PreparedParentCandidates&) = delete;
  PreparedParentCandidates& operator=(const PreparedParentCandidates&) =
      delete;

  [[nodiscard]] std::uint32_t row_count() const noexcept {
    return static_cast<std::uint32_t>(query_local_ids_.size());
  }

  [[nodiscard]] PreparedParentCandidateRow row(
      std::uint32_t index) const noexcept {
    assert(index < row_count());
    const std::uint32_t begin = offsets_[index];
    const std::uint32_t end = offsets_[index + 1U];
    const std::uint32_t* parents = parent_local_ids_.empty()
        ? nullptr
        : parent_local_ids_.data() + begin;
    return {
        .query_local_id = query_local_ids_[index],
        .parent_local_ids = {
            parents,
            static_cast<std::size_t>(end - begin),
        },
    };
  }

 private:
  friend class CpuExactParentCandidateProvider;

  std::vector<std::uint32_t> query_local_ids_;
  std::vector<std::uint32_t> offsets_;
  std::vector<std::uint32_t> parent_local_ids_;
};

}  // namespace stlqpp
