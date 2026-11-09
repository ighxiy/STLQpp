#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/search/query_batch.h"
#include "stlqpp/search/search_options.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

struct SearchResult final {
  std::uint32_t query_count = 0;
  std::uint32_t top_k = 0;
  std::vector<std::uint64_t> ids;
  // Reconstruction norm minus 2*q dot reconstruction. The omitted ||q||^2
  // is query-constant and preserves the exact DEV cutoff/ranking contract.
  // ExactL2Reranker instead returns complete squared L2, including ||q||^2.
  std::vector<float> adc_distances;
};

class Searcher {
 public:
  virtual ~Searcher() = default;

  Searcher(const Searcher&) = delete;
  Searcher& operator=(const Searcher&) = delete;
  Searcher(Searcher&&) = delete;
  Searcher& operator=(Searcher&&) = delete;

  [[nodiscard]] virtual Status Search(const QueryBatchView& queries,
                                      const SearchOptions& options,
                                      SearchResult* result) const = 0;

 protected:
  Searcher() = default;
};

}  // namespace stlqpp
