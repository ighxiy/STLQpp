#include "stlqpp/search/plans/exact_dense_searcher.h"

#include "stlqpp/search/tables/query_table_batch.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace stlqpp {

template <int M>
struct ExactDenseSearcher<M>::QueryBlockExecution final {
  const ExactDenseSearcher<M>* searcher = nullptr;
  QueryTableBatch<M>* tables = nullptr;
  const SearchOptions* options = nullptr;
  std::vector<DenseQueryWorkspace<M>>* workspaces = nullptr;
  std::vector<QueryTopK>* profile_top_k = nullptr;
  std::vector<DenseIntQueryProfile>* worker_profiles = nullptr;
  SearchResult* result = nullptr;
  std::uint32_t first_query = 0U;
};

template <int M>
struct ExactDenseSearcher<M>::CompletedQueryExecution final {
  QueryBlockExecution* block;
  const TrustedQueryBatchView* route_queries;
  DenseQueryCompletion completion;
  void* context;
  std::vector<Status>* failures;
  std::atomic<bool>* failed;
  std::uint32_t first_query;
};

template <int M>
void ExactDenseSearcher<M>::ExecuteCompletedQuery(
    void* opaque, std::uint32_t worker, std::uint32_t query) noexcept {
  auto& e = *static_cast<CompletedQueryExecution*>(opaque);
  if (e.failed->load(std::memory_order_relaxed)) return;
  auto& block = *e.block;
  auto& failure = (*e.failures)[worker];
  if (e.route_queries != nullptr) {
    const auto& queries = *e.route_queries;
    failure = block.searcher->hnsw_router_->RouteQueryTrusted(
        queries.data() + static_cast<std::size_t>(query) * queries.stride_floats(),
        block.options->nprobe,
        block.tables->routes_.data() + static_cast<std::size_t>(query) * block.options->nprobe,
        block.tables->root0_.data() + static_cast<std::size_t>(query) * block.tables->nlist());
    if (!failure.ok()) { e.failed->store(true, std::memory_order_relaxed); return; }
  }
  ExecuteQuery(e.block, worker, query);
  const auto row = e.first_query + query;
  const auto offset = static_cast<std::size_t>(block.first_query + query) * block.options->top_k;
  failure = e.completion(e.context, worker, row,
      {block.result->ids.data() + offset,
       block.options->top_k});
  if (!failure.ok()) e.failed->store(true, std::memory_order_relaxed);
}

template <int M>
ExactDenseSearcher<M>::ExactDenseSearcher(
    std::shared_ptr<const ResidentIndex<M>> index,
    std::shared_ptr<const F32GemmBackend> gemm,
    std::shared_ptr<const ParallelForExecutor> parallel,
    QueryTableBuilder<M> table_builder,
    DenseQueryExecutor<M> executor,
    ExactDenseSearchRuntime runtime) noexcept
    : index_(std::move(index)),
      gemm_(std::move(gemm)),
      parallel_(std::move(parallel)),
      table_builder_(std::move(table_builder)),
      executor_(std::move(executor)),
      runtime_(runtime) {}

template <int M>
StatusOr<std::unique_ptr<ExactDenseSearcher<M>>>
ExactDenseSearcher<M>::Create(
    std::shared_ptr<const ResidentIndex<M>> index,
    std::shared_ptr<const F32GemmBackend> gemm,
    std::shared_ptr<const ParallelForExecutor> parallel,
    const DenseCpuPlan& plan,
    ExactDenseSearchRuntime runtime) {
  if (index == nullptr || gemm == nullptr || parallel == nullptr) {
    return Status::InvalidArgument(
        "exact dense searcher requires owned index and runtime backends");
  }
  if (runtime.query_block_size == 0U || parallel->worker_count() == 0U) {
    return Status::InvalidArgument(
        "exact dense search runtime dimensions must be positive");
  }
  if (runtime.pretransform_query_codebooks !=
      index->query_codebooks_pretransformed()) {
    return Status::FailedPrecondition(
        "exact dense runtime and resident query-codebook space disagree");
  }
  auto executor = DenseQueryExecutor<M>::Create(*index, plan);
  if (!executor.ok()) {
    return executor.status();
  }
  QueryTableBuilder<M> table_builder(*index, *gemm);
  auto output = std::unique_ptr<ExactDenseSearcher<M>>(
      new ExactDenseSearcher<M>(
          std::move(index), std::move(gemm), std::move(parallel),
          std::move(table_builder), std::move(executor).value(), runtime));
  if (runtime.ivf_hnsw.enabled) {
    auto router = IvfHnswRouter::Open(output->index_->codebooks().root0(), runtime.ivf_hnsw);
    if (!router.ok()) return router.status();
    output->hnsw_router_ = std::move(router).value();
  }
  return output;
}

template <int M>
void ExactDenseSearcher<M>::ExecuteQuery(
    void* context,
    std::uint32_t worker_index,
    std::uint32_t query_index) noexcept {
  auto* block = static_cast<QueryBlockExecution*>(context);
  DenseQueryWorkspace<M>& workspace = (*block->workspaces)[worker_index];
  const std::size_t output_offset =
      static_cast<std::size_t>(block->first_query + query_index) *
      block->options->top_k;
  block->searcher->executor_.SearchTrustedTo(
      block->tables->query(query_index), block->options->nprobe, &workspace,
      block->result->ids.data() + output_offset,
      block->result->adc_distances.data() + output_offset);
}

template <int M>
void ExactDenseSearcher<M>::ExecuteProbeSelection(
    void* context, std::uint32_t worker_index,
    std::uint32_t query_index) noexcept {
  auto& block = *static_cast<QueryBlockExecution*>(context);
  auto& workspace = (*block.workspaces)[worker_index];
  const auto query = block.tables->query(query_index);
  SelectExactIvfRoutes(block.searcher->index_->native_ivf(),
                      query.root0_values(), block.options->nprobe,
                      &workspace.routes_);
  // Routes are small query metadata; dot/scan/TopK storage remains per worker.
  auto& tables = *block.tables;
  std::copy_n(workspace.routes_.selected().data(), block.options->nprobe,
              tables.routes_.data() +
                  static_cast<std::size_t>(query_index) * block.options->nprobe);
}

template <int M>
void ExactDenseSearcher<M>::ExecuteProfiledScan(
    void* context,
    std::uint32_t worker_index,
    std::uint32_t query_index) noexcept {
  auto* block = static_cast<QueryBlockExecution*>(context);
  block->searcher->executor_.ScanTrustedProfiled(
      block->tables->query(query_index), block->options->nprobe,
      &(*block->workspaces)[worker_index],
      &(*block->profile_top_k)[query_index],
      &(*block->worker_profiles)[worker_index]);
}

template <int M>
void ExactDenseSearcher<M>::ExecuteProfiledFinalize(
    void* context,
    std::uint32_t worker_index,
    std::uint32_t query_index) noexcept {
  auto* block = static_cast<QueryBlockExecution*>(context);
  const std::size_t output_offset =
      static_cast<std::size_t>(block->first_query + query_index) *
      block->options->top_k;
  block->searcher->executor_.FinalizeProfiled(
      &(*block->profile_top_k)[query_index],
      block->result->ids.data() + output_offset,
      block->result->adc_distances.data() + output_offset,
      &(*block->worker_profiles)[worker_index]);
}

template <int M>
Status ExactDenseSearcher<M>::Search(
    const QueryBatchView& queries,
    const SearchOptions& options,
    SearchResult* result) const {
  if (result == nullptr) {
    return Status::InvalidArgument("search result pointer must not be null");
  }
  auto trusted = ValidateQueries(queries);
  if (!trusted.ok()) return trusted.status();
  return SearchTrusted(trusted.value(), options, result);
}

template <int M>
StatusOr<TrustedQueryBatchView> ExactDenseSearcher<M>::ValidateQueries(
    const QueryBatchView& queries) const {
  return TrustedQueryBatchView::Create(queries, index_->shape().dimension);
}

template <int M>
Status ExactDenseSearcher<M>::SearchTrusted(
    const TrustedQueryBatchView& queries,
    const SearchOptions& options,
    SearchResult* result) const {
  return SearchImpl<false>(queries, options, result, nullptr);
}

template <int M>
Status ExactDenseSearcher<M>::SearchStreamingTrusted(
    const TrustedQueryBatchView& queries, const SearchOptions& options,
    SearchResult* result, DenseQueryCompletion completion, void* context) const {
  if (completion == nullptr) return Status::InvalidArgument("query completion is required");
  return SearchImpl<false, true>(queries, options, result, nullptr, completion, context);
}

template <int M>
Status ExactDenseSearcher<M>::SearchProfiled(
    const QueryBatchView& queries,
    const SearchOptions& options,
    SearchResult* result,
    ExactDenseSearchStatistics* statistics) const {
  if (statistics == nullptr) {
    return Status::InvalidArgument(
        "profiled exact search requires a statistics sink");
  }
  const auto begin = std::chrono::steady_clock::now();
  auto trusted = ValidateQueries(queries);
  if (!trusted.ok()) {
    *statistics = {};
    statistics->wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    return trusted.status();
  }
  const Status status = SearchProfiledTrusted(
      trusted.value(), options, result, statistics);
  // Preserve the public boundary: its wall time includes query validation.
  statistics->wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
  return status;
}

template <int M>
Status ExactDenseSearcher<M>::SearchProfiledTrusted(
    const TrustedQueryBatchView& queries,
    const SearchOptions& options,
    SearchResult* result,
    ExactDenseSearchStatistics* statistics) const {
  if (statistics == nullptr) {
    return Status::InvalidArgument(
        "profiled exact search requires a statistics sink");
  }
  *statistics = {};
  const auto begin = std::chrono::steady_clock::now();
  const bool detailed_profile = runtime_.profile_timing &&
      parallel_->worker_count() > 1U &&
      index_->coefficient_mode() == SearchCoefficientMode::kInt8 &&
      index_->score_mode() == SearchScoreMode::kScaled &&
      index_->parent_access_mode() == ParentAccessMode::kMaterialized;
  const Status status = detailed_profile
      ? SearchImpl<true>(queries, options, result, statistics)
      : SearchImpl<false>(queries, options, result, statistics);
  statistics->wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
  statistics->core_seconds = statistics->qt_rotate_seconds +
      statistics->table_build_seconds + statistics->probe_selection_seconds +
      statistics->scan_topk_seconds;
  return status;
}

template <int M>
template <bool DetailedTiming, bool Streaming>
Status ExactDenseSearcher<M>::SearchImpl(
    const TrustedQueryBatchView& queries,
    const SearchOptions& options,
    SearchResult* result,
    ExactDenseSearchStatistics* statistics, [[maybe_unused]] DenseQueryCompletion completion,
    [[maybe_unused]] void* completion_context) const {
  if (result == nullptr && !Streaming) {
    return Status::InvalidArgument("search result pointer must not be null");
  }
  if constexpr (DetailedTiming) {
    assert(statistics != nullptr);
    assert(parallel_->worker_count() > 1U &&
           index_->coefficient_mode() == SearchCoefficientMode::kInt8 &&
           index_->score_mode() == SearchScoreMode::kScaled &&
           index_->parent_access_mode() == ParentAccessMode::kMaterialized);
    statistics->detailed_profile = true;
  }
  if (queries.dimension() != index_->shape().dimension) {
    return Status::InvalidArgument(
        "query dimension does not match index dimension");
  }
  if (const Status option_status = options.Validate(index_->shape());
      !option_status.ok()) {
    return option_status;
  }

  const std::size_t query_count = queries.count();
  const std::size_t top_k = options.top_k;
  if (query_count != 0U &&
      top_k > std::numeric_limits<std::size_t>::max() / query_count) {
    return Status::ResourceExhausted("search result extent overflows size_t");
  }

  try {
    SearchResult completed;
    completed.query_count = queries.count();
    completed.top_k = options.top_k;
    const std::size_t result_count = (Streaming && result == nullptr
        ? std::min<std::size_t>(runtime_.query_block_size, query_count) : query_count) * top_k;
    completed.ids.assign(result_count,
                         std::numeric_limits<std::uint64_t>::max());
    completed.adc_distances.assign(
        result_count, std::numeric_limits<float>::infinity());

    const std::uint32_t block_capacity =
        std::min(runtime_.query_block_size, queries.count());
    auto tables = QueryTableBatch<M>::Create(index_->shape(), block_capacity);
    if (!tables.ok()) {
      return tables.status();
    }
    std::vector<float> model_queries;
    TrustedQueryBatchView table_queries = queries;
    if (table_builder_.requires_query_transform()) {
      model_queries.resize(
          static_cast<std::size_t>(index_->shape().dimension) *
          queries.count());
      std::chrono::steady_clock::time_point transform_begin;
      if (statistics != nullptr) {
        transform_begin = std::chrono::steady_clock::now();
      }
      if (const Status transform_status = table_builder_.Transform(
              queries, model_queries.data());
          !transform_status.ok()) {
        return transform_status;
      }
      if (statistics != nullptr) {
        statistics->qt_rotate_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - transform_begin).count();
      }
      table_queries = TrustedQueryBatchView::FromTrustedTransform(
          model_queries.data(), queries.count(), index_->shape().dimension);
    }

    std::vector<Status> completion_failures;
    std::atomic<bool> completion_failed{false};
    if constexpr (Streaming) completion_failures.resize(parallel_->worker_count());
    std::vector<Status> hnsw_failures;
    if (statistics != nullptr) {
      tables.value().routes_.resize(
          static_cast<std::size_t>(block_capacity) * options.nprobe);
    }
    std::vector<DenseQueryWorkspace<M>> workspaces;
    workspaces.reserve(parallel_->worker_count());
    for (std::uint32_t worker = 0U; worker < parallel_->worker_count();
         ++worker) {
      auto workspace = DenseQueryWorkspace<M>::Create(*index_, options.top_k);
      if (!workspace.ok()) {
        return workspace.status();
      }
      if constexpr (DetailedTiming) {
        if (const Status profile_status =
                workspace.value().PrepareIntProfiling();
            !profile_status.ok()) {
          return profile_status;
        }
      }
      workspaces.push_back(std::move(workspace).value());
    }
    std::vector<DenseIntQueryProfile> worker_profiles;
    if constexpr (DetailedTiming) {
      worker_profiles.resize(parallel_->worker_count());
    }
    for (std::uint32_t first = 0U; first < queries.count();
         first += block_capacity) {
      const std::uint32_t block_count =
          std::min(block_capacity, queries.count() - first);
      const TrustedQueryBatchView block =
          table_queries.Slice(first, block_count);
      std::chrono::steady_clock::time_point table_begin;
      if (statistics != nullptr) {
        table_begin = std::chrono::steady_clock::now();
      }
      if (const Status table_status =
              table_builder_.BuildModelSpaceTrusted(block, &tables.value(),
                  hnsw_router_ && options.nprobe < index_->nlist());
          !table_status.ok()) {
        return table_status;
      }
      if (statistics != nullptr) {
        statistics->table_build_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - table_begin).count();
      }
      std::chrono::steady_clock::time_point probe_begin;
      if (statistics != nullptr) probe_begin = std::chrono::steady_clock::now();
      if (hnsw_router_ && options.nprobe < index_->nlist()) {
        auto& batch = tables.value();
        batch.route_count_ = options.nprobe;
        if constexpr (Streaming) {
          batch.routes_.resize(static_cast<std::size_t>(block_count) * options.nprobe);
        } else if (const auto status = hnsw_router_->RouteBlock(
                block, options.nprobe, *parallel_, batch.routes_, batch.root0_.data(), hnsw_failures);
            !status.ok()) return status;
      } else {
        tables.value().route_count_ = 0U;
      }
      QueryBlockExecution execution = {
          .searcher = this,
          .tables = &tables.value(),
          .options = &options,
          .workspaces = &workspaces,
          .profile_top_k = nullptr,
          .worker_profiles = nullptr,
          .result = &completed,
          .first_query = Streaming && result == nullptr ? 0U : first,
      };
      if (statistics != nullptr) {
        if (tables.value().route_count_ == 0U) {
          parallel_->Run(block_count, &execution, &ExecuteProbeSelection);
          tables.value().route_count_ = options.nprobe;
        }
        statistics->probe_selection_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - probe_begin).count();
      }
      if constexpr (DetailedTiming) {
        std::vector<QueryTopK> profile_top_k;
        profile_top_k.reserve(block_count);
        for (std::uint32_t query = 0U; query < block_count; ++query) {
          profile_top_k.emplace_back(options.top_k);
        }
        execution.profile_top_k = &profile_top_k;
        execution.worker_profiles = &worker_profiles;

        const auto scan_begin = std::chrono::steady_clock::now();
        parallel_->Run(block_count, &execution, &ExecuteProfiledScan);
        const double scan_phase = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - scan_begin).count();
        const auto finalize_begin = std::chrono::steady_clock::now();
        parallel_->Run(block_count, &execution, &ExecuteProfiledFinalize);
        const double finalize_phase = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - finalize_begin).count();
        statistics->route_scan_phase_seconds += scan_phase;
        statistics->finalize_phase_seconds += finalize_phase;
        statistics->scan_topk_seconds += scan_phase + finalize_phase;
      } else {
        std::chrono::steady_clock::time_point scan_begin;
        if (statistics != nullptr) {
          scan_begin = std::chrono::steady_clock::now();
        }
        if constexpr (Streaming) {
          CompletedQueryExecution completed_execution{
              &execution, hnsw_router_ && options.nprobe < index_->nlist() ? &block : nullptr,
              completion, completion_context, &completion_failures, &completion_failed, first};
          parallel_->Run(block_count, &completed_execution, &ExecuteCompletedQuery);
          for (const auto& failure : completion_failures)
            if (!failure.ok()) return failure;
        } else {
          parallel_->Run(block_count, &execution, &ExecuteQuery);
        }
        if (statistics != nullptr) {
          statistics->scan_topk_seconds += std::chrono::duration<double>(
              std::chrono::steady_clock::now() - scan_begin).count();
        }
      }
      if (statistics != nullptr) {
        ++statistics->query_blocks;
      }
    }
    if constexpr (DetailedTiming) {
      for (const DenseIntQueryProfile& profile : worker_profiles) {
        statistics->route_worker_seconds += profile.route_worker_seconds;
        statistics->scan_worker_seconds += profile.scan_worker_seconds;
        statistics->online_topk_worker_seconds +=
            profile.online_topk_worker_seconds;
        statistics->finalize_worker_seconds +=
            profile.finalize_worker_seconds;
      }
      const double fused_worker = statistics->route_worker_seconds +
          statistics->scan_worker_seconds +
          statistics->online_topk_worker_seconds;
      if (fused_worker > 0.0) {
        const double phase = statistics->route_scan_phase_seconds;
        statistics->logical_route_seconds =
            statistics->probe_selection_seconds +
            phase * statistics->route_worker_seconds / fused_worker;
        statistics->logical_scan_seconds =
            phase * statistics->scan_worker_seconds / fused_worker;
        const double online_topk =
            phase * statistics->online_topk_worker_seconds / fused_worker;
        statistics->logical_topk_seconds =
            online_topk + statistics->finalize_phase_seconds;
      }
    }
    if constexpr (Streaming) {
      if (result) *result = std::move(completed);
    } else {
      *result = std::move(completed);
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted(
        "exact dense search scratch/result allocation failed");
  }
}

template class ExactDenseSearcher<5>;
template class ExactDenseSearcher<12>;
template class ExactDenseSearcher<6>;
template class ExactDenseSearcher<10>;

}  // namespace stlqpp
