#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/search/index/resident_index.h"
#include "stlqpp/search/plans/dense_cpu_plan.h"
#include "stlqpp/search/query_tables.h"
#include "stlqpp/search/routing/exact_ivf_router.h"
#include "stlqpp/search/search_options.h"
#include "stlqpp/search/tables/scaled_cluster_tables.h"
#include "stlqpp/search/topk/query_topk.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

struct DenseIntQueryProfile final {
  double route_worker_seconds = 0.0;
  double scan_worker_seconds = 0.0;
  double online_topk_worker_seconds = 0.0;
  double finalize_worker_seconds = 0.0;
};

template <int M>
class DenseQueryExecutor;
template <int M>
class ExactDenseSearcher;

// One instance belongs to one query worker. It reuses route, dot, table and
// heap storage across that worker's independent queries; no state is shared
// between queries or workers.
template <int M>
class DenseQueryWorkspace final {
 public:
  static_assert(M >= 2 && M <= 20);

  [[nodiscard]] static StatusOr<DenseQueryWorkspace<M>> Create(
      const ResidentIndex<M>& index,
      std::uint32_t top_k);

  DenseQueryWorkspace(DenseQueryWorkspace&&) noexcept = default;
  DenseQueryWorkspace& operator=(DenseQueryWorkspace&&) noexcept = default;
  DenseQueryWorkspace(const DenseQueryWorkspace&) = delete;
  DenseQueryWorkspace& operator=(const DenseQueryWorkspace&) = delete;

  [[nodiscard]] const std::vector<SearchCandidate>& results() const noexcept {
    return top_k_.results();
  }
  [[nodiscard]] ArrayView<const std::uint32_t> selected_routes() const noexcept {
    return routes_.selected();
  }

  // Allocates the profile-only distance staging buffer before a timed region.
  // The ordinary workspace retains no second max-cluster float buffer.
  [[nodiscard]] Status PrepareIntProfiling();

 private:
  friend class DenseQueryExecutor<M>;
  friend class ExactDenseSearcher<M>;

  DenseQueryWorkspace(const ResidentIndex<M>* owner,
                      std::uint32_t top_k)
      : owner_(owner),
        routes_(owner->nlist()),
        dot_scratch_(owner->max_node_count()),
        top_k_(top_k) {}

  const ResidentIndex<M>* owner_ = nullptr;
  ExactIvfRouteWorkspace routes_;
  std::vector<float> dot_scratch_;
  std::vector<float> profile_distance_scratch_;
  ScaledClusterQueryTables<M> scaled_tables_;
  QueryTopK top_k_;
};

// Exact single-query execution plan. Batch schedulers assign one workspace to
// each owner thread and call Search independently; this method finalizes the
// same query before returning.
template <int M>
class DenseQueryExecutor final {
 public:
  static_assert(M >= 2 && M <= 20);

  [[nodiscard]] static StatusOr<DenseQueryExecutor<M>> Create(
      const ResidentIndex<M>& index,
      const DenseCpuPlan& plan);

  [[nodiscard]] Status Search(const TrustedQueryTables<M>& query,
                              const SearchOptions& options,
                              DenseQueryWorkspace<M>* workspace) const;

  // The public Searcher calls this after request/index/workspace validation.
  // It is the complete query hot path and cannot report recoverable errors.
  void SearchTrusted(const TrustedQueryTables<M>& query,
                     std::uint32_t nprobe,
                     DenseQueryWorkspace<M>* workspace) const noexcept;

  // Query-owner batch execution finalizes directly into the caller's output
  // row. It is otherwise the identical single-query plan.
  void SearchTrustedTo(const TrustedQueryTables<M>& query,
                       std::uint32_t nprobe,
                       DenseQueryWorkspace<M>* workspace,
                       std::uint64_t* ids,
                       float* squared_distances) const noexcept;

  // Detailed int-only path. Scan and finalization are separate so a batch
  // scheduler can create additive global phase walls while retaining one
  // small TopK state per query and one large scratch set per worker.
  void ScanTrustedProfiled(const TrustedQueryTables<M>& query,
                           std::uint32_t nprobe,
                           DenseQueryWorkspace<M>* workspace,
                           QueryTopK* top_k,
                           DenseIntQueryProfile* profile) const noexcept;
  void FinalizeProfiled(QueryTopK* top_k,
                        std::uint64_t* ids,
                        float* squared_distances,
                        DenseIntQueryProfile* profile) const noexcept;

 private:
  template <bool DirectOutput>
  void SearchTrustedImpl(const TrustedQueryTables<M>& query,
                         std::uint32_t nprobe,
                         DenseQueryWorkspace<M>* workspace,
                         std::uint64_t* ids,
                         float* squared_distances) const noexcept;

  explicit DenseQueryExecutor(const ResidentIndex<M>* index) noexcept
      : index_(index) {}

  const ResidentIndex<M>* index_ = nullptr;
};

extern template class DenseQueryWorkspace<5>;
extern template class DenseQueryWorkspace<10>;
extern template class DenseQueryExecutor<5>;
extern template class DenseQueryExecutor<10>;

}  // namespace stlqpp
