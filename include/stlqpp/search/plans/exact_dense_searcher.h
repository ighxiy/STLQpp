#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/search/index/resident_index.h"
#include "stlqpp/search/plans/dense_cpu_plan.h"
#include "stlqpp/search/plans/dense_query_executor.h"
#include "stlqpp/search/query_batch.h"
#include "stlqpp/search/routing/ivf_hnsw_router.h"
#include "stlqpp/search/searcher.h"
#include "stlqpp/search/query_completion.h"
#include "stlqpp/search/tables/query_table_builder.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace stlqpp {

struct ExactDenseSearchRuntime final {
  std::uint32_t query_block_size = 64U;
  // Precompute R^T C once while opening the searcher and build query tables
  // directly from source-space queries. The scan representation is unchanged.
  bool pretransform_query_codebooks = false;
  // Diagnostic request only. A separately compiled detailed path is selected
  // for parallel materialized-parent quantized-scaled scans. Other legal eval
  // representations keep their ordinary path without detailed accounting.
  bool profile_timing = false;
  IvfHnswOptions ivf_hnsw = {};
};

struct ExactDenseSearchStatistics final {
  std::uint64_t query_blocks = 0U;
  bool detailed_profile = false;
  double qt_rotate_seconds = 0.0;
  double table_build_seconds = 0.0;
  double probe_selection_seconds = 0.0;
  // Measured scan and TopK phase, excluding the probe selection phase.
  double scan_topk_seconds = 0.0;
  // Detailed-profile measured phase walls. route_scan_phase_seconds contains
  // score generation and online heap maintenance with routes already selected;
  // finalization has a separate global phase.
  double route_scan_phase_seconds = 0.0;
  double finalize_phase_seconds = 0.0;
  // Active-worker sums used only to apportion the scan/online-heap phase.
  // route_worker_seconds measures consumption of the preselected route view.
  // They are diagnostic CPU work, not wall times.
  double route_worker_seconds = 0.0;
  double scan_worker_seconds = 0.0;
  double online_topk_worker_seconds = 0.0;
  double finalize_worker_seconds = 0.0;
  // Additive normalized-wall interpretation. TopK includes online heap work
  // plus the directly measured finalize phase.
  double logical_route_seconds = 0.0;
  double logical_scan_seconds = 0.0;
  double logical_topk_seconds = 0.0;
  // DEV-matched query work: table construction plus route/scan/top-k.
  double core_seconds = 0.0;
  // Complete SearchProfiled call, including query validation, result and
  // reusable-query-state allocation. SearchProfiledTrusted starts after the
  // caller's one-time query validation. Both values are exposed deliberately.
  double wall_seconds = 0.0;
};

// Public exact search composition. Query blocks amortize rotation/table GEMMs;
// each query is then wholly owned by one runtime worker through finalize.
template <int M>
class ExactDenseSearcher final : public Searcher {
 public:
  static_assert(M >= 2 && M <= 20);

  [[nodiscard]] static StatusOr<std::unique_ptr<ExactDenseSearcher<M>>> Create(
      std::shared_ptr<const ResidentIndex<M>> index,
      std::shared_ptr<const F32GemmBackend> gemm,
      std::shared_ptr<const ParallelForExecutor> parallel,
      const DenseCpuPlan& plan,
      ExactDenseSearchRuntime runtime);

  [[nodiscard]] std::filesystem::path routing_index_path() const {
    return hnsw_router_ ? hnsw_router_->index_path() : std::filesystem::path{};
  }

  [[nodiscard]] Status Search(const QueryBatchView& queries,
                              const SearchOptions& options,
                              SearchResult* result) const override;

  // Validate a borrowed query batch once when the caller will reuse the same
  // immutable storage across warmups, repeats or probe settings.
  [[nodiscard]] StatusOr<TrustedQueryBatchView> ValidateQueries(
      const QueryBatchView& queries) const;
  [[nodiscard]] Status SearchTrusted(
      const TrustedQueryBatchView& queries,
      const SearchOptions& options,
      SearchResult* result) const;

  // Query owners immediately consume finalized candidates (e.g. exact L2
  // reranking). This separate entry does not instrument ordinary Search.
  // A null result retains candidates for one query block only; the callback
  // still receives global query numbers and runs before the next owner task.
  [[nodiscard]] Status SearchStreamingTrusted(
      const TrustedQueryBatchView& queries, const SearchOptions& options,
      SearchResult* result, DenseQueryCompletion completion,
      void* completion_context) const;

  // Benchmark boundary for the same exact implementation. Wall time covers
  // the complete call. Core time is the matched DEV boundary containing query
  // transform/table construction, routing, dense ADC, norm, top-k and
  // finalize. The default runtime instruments only query-block boundaries.
  // runtime.profile_timing selects the detailed specialization only for its
  // supported representation; it never changes whether eval itself is legal.
  [[nodiscard]] Status SearchProfiled(
      const QueryBatchView& queries,
      const SearchOptions& options,
      SearchResult* result,
      ExactDenseSearchStatistics* statistics) const;
  [[nodiscard]] Status SearchProfiledTrusted(
      const TrustedQueryBatchView& queries,
      const SearchOptions& options,
      SearchResult* result,
      ExactDenseSearchStatistics* statistics) const;

 private:
  struct QueryBlockExecution;
  struct CompletedQueryExecution;

  ExactDenseSearcher(std::shared_ptr<const ResidentIndex<M>> index,
                     std::shared_ptr<const F32GemmBackend> gemm,
                     std::shared_ptr<const ParallelForExecutor> parallel,
                     QueryTableBuilder<M> table_builder,
                     DenseQueryExecutor<M> executor,
                     ExactDenseSearchRuntime runtime) noexcept;

  static void ExecuteQuery(void* context,
                           std::uint32_t worker_index,
                           std::uint32_t query_index) noexcept;
  static void ExecuteCompletedQuery(void*, std::uint32_t, std::uint32_t) noexcept;
  static void ExecuteProbeSelection(void* context,
                                    std::uint32_t worker_index,
                                    std::uint32_t query_index) noexcept;
  static void ExecuteProfiledScan(void* context,
                                  std::uint32_t worker_index,
                                  std::uint32_t query_index) noexcept;
  static void ExecuteProfiledFinalize(void* context,
                                      std::uint32_t worker_index,
                                      std::uint32_t query_index) noexcept;
  template <bool DetailedTiming, bool Streaming = false>
  [[nodiscard]] Status SearchImpl(
      const TrustedQueryBatchView& queries,
      const SearchOptions& options,
      SearchResult* result,
      ExactDenseSearchStatistics* statistics,
      [[maybe_unused]] DenseQueryCompletion completion = nullptr,
      [[maybe_unused]] void* completion_context = nullptr) const;

  std::shared_ptr<const ResidentIndex<M>> index_;
  std::shared_ptr<const F32GemmBackend> gemm_;
  std::shared_ptr<const ParallelForExecutor> parallel_;
  QueryTableBuilder<M> table_builder_;
  DenseQueryExecutor<M> executor_;
  ExactDenseSearchRuntime runtime_;
  std::unique_ptr<IvfHnswRouter> hnsw_router_;
};

extern template class ExactDenseSearcher<5>;
extern template class ExactDenseSearcher<10>;

}  // namespace stlqpp
