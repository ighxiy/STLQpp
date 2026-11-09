#include "stlqpp/search/topk/query_topk.h"

#include <algorithm>

namespace stlqpp {

const std::vector<SearchCandidate>& QueryTopK::Finalize() noexcept {
  entries_.resize(size_);
  std::sort(entries_.begin(), entries_.end(),
            [](const SearchCandidate& lhs, const SearchCandidate& rhs) {
              return Better(lhs, rhs);
            });
  return entries_;
}

void QueryTopK::FinalizeTo(std::uint64_t* ids,
                           float* squared_distances) noexcept {
  const std::vector<SearchCandidate>& ordered = Finalize();
  for (std::uint32_t rank = 0U; rank < size_; ++rank) {
    ids[rank] = ordered[rank].id;
    squared_distances[rank] = ordered[rank].squared_distance;
  }
}

void QueryTopK::Reset() noexcept {
  entries_.resize(capacity_);
  size_ = 0U;
}

}  // namespace stlqpp
