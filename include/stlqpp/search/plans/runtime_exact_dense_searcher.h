#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "stlqpp/search/kernels/runtime_dense.h"
#include "stlqpp/search/layout/resident_cluster.h"
#include "stlqpp/search/plans/exact_dense_searcher.h"
#include "stlqpp/search/searcher.h"
#include "stlqpp/storage/canonical_cluster_codec.h"
#include "stlqpp/storage/canonical_model_codec.h"

namespace stlqpp {

// General-M search. The fixed-M SIMD searcher remains the
// fast path for M=5/6/10/12; this one does not instantiate a new template.
class RuntimeExactDenseSearcher final : public Searcher {
 public:
  [[nodiscard]] static StatusOr<std::unique_ptr<RuntimeExactDenseSearcher>>
  Open(const std::filesystem::path& directory,
       ParentAccessMode parent_access_mode = ParentAccessMode::kNativeLouds,
       SearchCoefficientMode coefficient_mode = SearchCoefficientMode::kInt8,
       SearchScoreMode score_mode = SearchScoreMode::kClassic,
       std::shared_ptr<const F32GemmBackend> gemm = {},
       std::shared_ptr<const ParallelForExecutor> parallel = {},
       ExactDenseSearchRuntime runtime = {},
       const IndexRowScores* row_scores = nullptr);

  [[nodiscard]] static StatusOr<std::unique_ptr<RuntimeExactDenseSearcher>>
  Create(RuntimeCanonicalModel model,
         std::vector<RuntimeDecodedCanonicalCluster> clusters,
         std::vector<float> norm8_centers,
         ParentAccessMode parent_access_mode = ParentAccessMode::kNativeLouds,
         SearchCoefficientMode coefficient_mode = SearchCoefficientMode::kInt8,
         SearchScoreMode score_mode = SearchScoreMode::kClassic,
         std::vector<RuntimeFloatCluster> native_float = {},
         ScaledScore8Codebook scaled = {},
         std::shared_ptr<const F32GemmBackend> gemm = {},
         std::shared_ptr<const ParallelForExecutor> parallel = {},
         ExactDenseSearchRuntime runtime = {},
         IndexRowScores row_scores = {});

  [[nodiscard]] bool uses_source_queries() const noexcept { return model_.source_space(); }
  [[nodiscard]] std::filesystem::path routing_index_path() const {
    return hnsw_router_ ? hnsw_router_->index_path() : std::filesystem::path{};
  }

  [[nodiscard]] Status Search(const QueryBatchView& queries,
                              const SearchOptions& options,
                              SearchResult* result) const override;
  [[nodiscard]] StatusOr<TrustedQueryBatchView> ValidateQueries(
      const QueryBatchView& queries) const;
  [[nodiscard]] Status SearchTrusted(const TrustedQueryBatchView& queries,
                                     const SearchOptions& options,
                                     SearchResult* result) const;
  // A null result retains candidates for one query block only; the callback
  // still receives global query numbers and runs before the next owner task.
  [[nodiscard]] Status SearchStreamingTrusted(
      const TrustedQueryBatchView& queries, const SearchOptions& options,
      SearchResult* result, DenseQueryCompletion completion, void* context) const;
  [[nodiscard]] Status SearchProfiledTrusted(
      const TrustedQueryBatchView& queries, const SearchOptions& options,
      SearchResult* result, ExactDenseSearchStatistics* statistics) const;

 private:
  RuntimeExactDenseSearcher(
      RuntimeCanonicalModel model,
      std::vector<RuntimeDecodedCanonicalCluster> clusters,
      std::vector<float> norm8_centers, std::vector<float> root_inverse_norms,
      std::vector<std::vector<std::uint32_t>> materialized_parents) noexcept;

  template <bool Streaming = false>
  [[nodiscard]] Status SearchImpl(const TrustedQueryBatchView& queries,
                                  const SearchOptions& options,
                                  SearchResult* result,
                                  ExactDenseSearchStatistics* statistics,
                                  [[maybe_unused]] DenseQueryCompletion completion = nullptr,
                                  [[maybe_unused]] void* context = nullptr) const;
  struct QueryExecution;
  struct CompletedQueryExecution;
  static void ExecuteCompletedQuery(void*, std::uint32_t, std::uint32_t) noexcept;
  static void ExecuteQuery(void*, std::uint32_t, std::uint32_t) noexcept;
  static void ExecuteProbeSelection(void*, std::uint32_t,
                                    std::uint32_t) noexcept;

  RuntimeCanonicalModel model_;
  std::vector<RuntimeDecodedCanonicalCluster> clusters_;
  std::vector<float> norm8_centers_;
  std::vector<float> norm8_inverse_norms_;
  std::vector<float> root_inverse_norms_;
  std::vector<std::vector<std::uint32_t>> materialized_parents_;
  SearchCoefficientMode coefficient_mode_ = SearchCoefficientMode::kInt8;
  SearchScoreMode score_mode_ = SearchScoreMode::kClassic;
  std::vector<RuntimeFloatCluster> native_float_;
  ScaledScore8Codebook scaled_;
  IndexRowScores row_scores_;
  std::shared_ptr<const F32GemmBackend> gemm_;
  std::shared_ptr<const ParallelForExecutor> parallel_;
  ExactDenseSearchRuntime runtime_;
  std::uint32_t max_node_count_ = 0U;
  std::unique_ptr<IvfHnswRouter> hnsw_router_;
};

}  // namespace stlqpp
