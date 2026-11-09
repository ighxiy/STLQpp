#pragma once

#include "stlqpp/search/query_completion.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/search/searcher.h"

#include <filesystem>
#include <memory>
#include <string>

namespace stlqpp {

struct ExactL2RerankSource final {
  std::filesystem::path path;
  std::string format = "auto";  // bvecs, fvecs, fbin, or float16 npy
  std::string storage = "memory";  // resident source slice, or disk (O_DIRECT)
  std::uint32_t dimension = 0U;
  std::uint64_t first_row = 0U;
  std::uint64_t row_count = 0U;  // stable IDs are local to this slice
};

// Immutable original vectors and reusable per-worker rerank scratch. One call
// at a time per instance. No dependency on M, coefficient format, or ADC kernel.
// Source loading/opening is outside query timing. Disk mode has no page cache:
// each query owner reads its own candidates immediately after ADC completion.
class ExactL2Reranker final {
 public:
  [[nodiscard]] static StatusOr<std::unique_ptr<ExactL2Reranker>> Open(
      const ExactL2RerankSource& source, std::uint32_t workers);
  ~ExactL2Reranker();

  template <typename SearchPlan>
  [[nodiscard]] Status SearchTrusted(
      const SearchPlan& searcher, const TrustedQueryBatchView& queries,
      const SearchOptions& candidate_options, std::uint32_t top_k,
      SearchResult* result) {
    if (const auto status = Prepare(queries, candidate_options.top_k, top_k, result);
        !status.ok()) return status;
    const auto status = searcher.SearchStreamingTrusted(
        queries, candidate_options, nullptr, Complete, this);
    if (!status.ok()) return status;
    Publish(result);
    return Status::Ok();
  }

 private:
  struct State;
  explicit ExactL2Reranker(std::unique_ptr<State> state) noexcept;
  [[nodiscard]] Status Prepare(const TrustedQueryBatchView&, std::uint32_t,
                               std::uint32_t, SearchResult*);
  static Status Complete(void*, std::uint32_t, std::uint32_t,
                         ArrayView<const std::uint64_t>) noexcept;
  void Publish(SearchResult*);
  std::unique_ptr<State> state_;
};

}  // namespace stlqpp
