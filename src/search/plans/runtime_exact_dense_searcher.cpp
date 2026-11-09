#include "stlqpp/search/plans/runtime_exact_dense_searcher.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <utility>

#include "stlqpp/model/topology/parent_louds.h"
#include "stlqpp/search/topk/query_topk.h"
#include "stlqpp/storage/canonical_index_files.h"

namespace stlqpp {
namespace {
using Clock = std::chrono::steady_clock;
double Elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

StatusOr<RuntimeCanonicalModel> Pretransform(const RuntimeCanonicalModel& model,
                                             const F32GemmBackend& gemm) {
  const auto shape = model.shape();
  const std::size_t rotation_size =
      static_cast<std::size_t>(shape.dimension) * shape.dimension;
  const std::uint32_t columns = shape.nlist + (2U * shape.m - 1U) * 256U;
  std::vector<float> values(
      rotation_size + static_cast<std::size_t>(shape.dimension) * columns,
      0.0F);
  for (std::uint32_t axis = 0; axis < shape.dimension; ++axis)
    values[static_cast<std::size_t>(axis) * shape.dimension + axis] = 1.0F;
  if (const auto status = gemm.MultiplyTransposedLeft(
          {.left = model.rotation(),
           .right = model.root0().values,
           .query_count = columns,
           .right_leading_dimension = shape.dimension,
           .output = values.data() + rotation_size,
           .output_leading_dimension = shape.dimension});
      !status.ok())
    return status;
  return RuntimeCanonicalModel(shape, std::move(values));
}
}  // namespace

RuntimeExactDenseSearcher::RuntimeExactDenseSearcher(
    RuntimeCanonicalModel model,
    std::vector<RuntimeDecodedCanonicalCluster> clusters,
    std::vector<float> norms, std::vector<float> inverse_norms,
    std::vector<std::vector<std::uint32_t>> parents) noexcept
    : model_(std::move(model)),
      clusters_(std::move(clusters)),
      norm8_centers_(std::move(norms)),
      root_inverse_norms_(std::move(inverse_norms)),
      materialized_parents_(std::move(parents)) {}

StatusOr<std::unique_ptr<RuntimeExactDenseSearcher>>
RuntimeExactDenseSearcher::Create(
    RuntimeCanonicalModel model,
    std::vector<RuntimeDecodedCanonicalCluster> clusters,
    std::vector<float> norms, ParentAccessMode parent_mode,
    SearchCoefficientMode coefficient_mode, SearchScoreMode score_mode,
    std::vector<RuntimeFloatCluster> native, ScaledScore8Codebook scaled,
    std::shared_ptr<const F32GemmBackend> gemm,
    std::shared_ptr<const ParallelForExecutor> parallel,
    ExactDenseSearchRuntime runtime, IndexRowScores row_scores) {
  try {
    if (const auto status = model.Validate(); !status.ok()) return status;
    const auto shape = model.shape();
    const bool floating = coefficient_mode == SearchCoefficientMode::kFloat;
    if (!row_scores.clusters.empty() && row_scores.clusters.size() != shape.nlist)
      return Status::DataLoss("evaluation norm CID count disagrees with index");
    if (clusters.size() != shape.nlist || runtime.query_block_size == 0 ||
        (floating && native.size() != shape.nlist) ||
        (!floating && (norms.empty() || norms.size() > 256)) ||
        (parent_mode != ParentAccessMode::kNativeLouds &&
         parent_mode != ParentAccessMode::kMaterialized) ||
        (coefficient_mode != SearchCoefficientMode::kInt8 && !floating) ||
        (score_mode != SearchScoreMode::kClassic &&
         score_mode != SearchScoreMode::kScaled &&
         score_mode != SearchScoreMode::kCosineRecon &&
         score_mode != SearchScoreMode::kMipsRecon))
      return Status::InvalidArgument(
          "runtime index options or record counts are invalid");
    if (!floating && score_mode == SearchScoreMode::kScaled &&
        !scaled.enabled() && row_scores.clusters.empty())
      return Status::FailedPrecondition(
          "runtime scaled search requires a scale/bias table");
    for (float value : norms) {
      if (!std::isfinite(value) || value < 0)
        return Status::DataLoss("invalid norm8 center");
    }
    if (!parallel) parallel = std::make_shared<InlineParallelFor>();
    if (parallel->worker_count() == 0)
      return Status::InvalidArgument(
          "runtime search requires at least one worker");
    if (runtime.pretransform_query_codebooks) {
      if (!gemm)
        return Status::InvalidArgument(
            "query pretransform requires a GEMM backend");
      auto transformed = Pretransform(model, *gemm);
      if (!transformed.ok()) return transformed.status();
      model = std::move(transformed).value();
    }
    std::vector<float> inverse_norms(shape.nlist);
    std::vector<std::vector<std::uint32_t>> parents;
    if (parent_mode == ParentAccessMode::kMaterialized)
      parents.resize(shape.nlist);
    std::uint32_t max_nodes = 0;
    for (std::uint32_t cid = 0; cid < shape.nlist; ++cid) {
      if (!row_scores.clusters.empty()) {
        if (const auto status = ValidateClusterRowScores(
                row_scores.clusters[cid], clusters[cid].topology.real_ids.size());
            !status.ok()) return status;
        if (floating && row_scores.clusters[cid].has_reference_correction)
          std::fill(native[cid].norm2.begin(), native[cid].norm2.end(), 0.0F);
      }
      if (const auto status = PrepareRuntimeDenseCluster(
              clusters[cid], shape, cid,
              {norms.data(), static_cast<std::uint32_t>(norms.size())},
              floating ? &native[cid] : nullptr, score_mode);
          !status.ok())
        return status;
      const auto& t = clusters[cid].topology;
      const auto nodes =
          t.virtual_count + static_cast<std::uint32_t>(t.real_ids.size());
      max_nodes = std::max(max_nodes, nodes);
      if (!parents.empty()) {
        if (const auto status =
                DecodeParentLouds(t.parent_louds_words, nodes, parents[cid]);
            !status.ok())
          return status;
      }
      double norm = 0;
      const float* root = model.root0().column(cid);
      for (std::uint32_t axis = 0; axis < shape.dimension; ++axis)
        norm += static_cast<double>(root[axis]) * root[axis];
      inverse_norms[cid] =
          1.0F / std::sqrt(std::max(static_cast<float>(norm), 1.0e-20F));
    }
    auto output = std::unique_ptr<RuntimeExactDenseSearcher>(
        new RuntimeExactDenseSearcher(
            std::move(model), std::move(clusters), std::move(norms),
            std::move(inverse_norms), std::move(parents)));
    if (score_mode == SearchScoreMode::kCosineRecon && !floating && row_scores.clusters.empty()) {
      output->norm8_inverse_norms_.resize(output->norm8_centers_.size());
      std::transform(output->norm8_centers_.begin(), output->norm8_centers_.end(),
          output->norm8_inverse_norms_.begin(),
          [](float norm) { return 1.0F / std::sqrt(std::max(norm, 1.0e-20F)); });
    }
    output->coefficient_mode_ = coefficient_mode;
    output->score_mode_ = score_mode;
    output->native_float_ = std::move(native);
    output->scaled_ = std::move(scaled);
    output->row_scores_ = std::move(row_scores);
    output->gemm_ = std::move(gemm);
    output->parallel_ = std::move(parallel);
    output->runtime_ = runtime;
    output->max_node_count_ = max_nodes;
    if (runtime.ivf_hnsw.enabled) {
      auto router = IvfHnswRouter::Open(output->model_.root0(), runtime.ivf_hnsw);
      if (!router.ok()) return router.status();
      output->hnsw_router_ = std::move(router).value();
    }
    return output;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted(
        "runtime index materialization exhausted memory");
  }
}

StatusOr<std::unique_ptr<RuntimeExactDenseSearcher>>
RuntimeExactDenseSearcher::Open(
    const std::filesystem::path& directory, ParentAccessMode parent_mode,
    SearchCoefficientMode coefficient_mode, SearchScoreMode score_mode,
    std::shared_ptr<const F32GemmBackend> gemm,
    std::shared_ptr<const ParallelForExecutor> parallel,
    ExactDenseSearchRuntime runtime, const IndexRowScores* row_scores) {
  try {
    const bool floating = coefficient_mode == SearchCoefficientMode::kFloat;
    auto opened = row_scores != nullptr
        ? CanonicalIndexFiles::OpenCoefficients(directory, floating)
        : floating ? (score_mode == SearchScoreMode::kScaled
                        ? CanonicalIndexFiles::OpenNativeFloatScaled(directory)
                        : CanonicalIndexFiles::OpenNativeFloat(directory))
                 : CanonicalIndexFiles::OpenQuantized(directory);
    if (!opened.ok()) return opened.status();
    CanonicalIndexFiles files = std::move(opened).value();
    const auto shape = files.manifest().shape;
    if (row_scores != nullptr && row_scores->clusters.size() != shape.nlist)
      return Status::DataLoss("evaluation norm CID count disagrees with index");
    if (row_scores == nullptr && !floating && score_mode == SearchScoreMode::kScaled &&
        !files.manifest().has_scaled_score8())
      return Status::FailedPrecondition(
          "runtime scaled search requires scaled score8");
    if (row_scores == nullptr && !floating && score_mode != SearchScoreMode::kScaled &&
        !files.manifest().has_classic_norm8())
      return Status::FailedPrecondition(
          "runtime quantized search requires classic norm8");
    auto model = CanonicalModelCodec::DecodeRuntime(files.model_bytes().data(),
                                                    files.model_bytes().size());
    if (!model.ok()) return model.status();
    const auto& decoded = model.value().shape();
    if (decoded.dimension != shape.dimension || decoded.m != shape.m ||
        decoded.nlist != shape.nlist ||
        decoded.residual_codebook_size != shape.residual_codebook_size)
      return Status::DataLoss(
          "runtime canonical model shape disagrees with manifest");
    std::vector<RuntimeDecodedCanonicalCluster> clusters;
    std::vector<RuntimeFloatCluster> native;
    clusters.resize(shape.nlist);
    if (floating) native.resize(shape.nlist);
    std::uint64_t real_count = 0;
    for (auto cid : files.physical_cid_order()) {
      const auto& entry = files.entry(cid);
      if (floating) {
        const auto records = files.topology_records(cid);
        auto topology = DecodeCanonicalNativeBfsTopologyRuntime(
            shape.m, cid, entry.real_count, entry.virtual_count,
            entry.real_root_count, records.codes, records.parents);
        if (!topology.ok()) return topology.status();
        RuntimeDecodedCanonicalCluster cluster;
        cluster.topology = std::move(topology).value();
        clusters[cid] = std::move(cluster);
        auto& v = native[cid];
        v.coefficients_layer_major.resize(
            (static_cast<std::size_t>(entry.real_count) + entry.virtual_count) *
            shape.m);
        v.norm2.resize(entry.real_count);
        if (const auto s =
                files.ReadFloatCoefficients(cid, v.coefficients_layer_major);
            !s.ok())
          return s;
        if (row_scores != nullptr) v.norm2 = row_scores->clusters[cid].norm2;
        else if (const auto s = files.ReadFloatNorm2(cid, v.norm2); !s.ok()) return s;
        if (row_scores != nullptr && score_mode == SearchScoreMode::kScaled) {
          v.source_norm2 = row_scores->clusters[cid].biases;
          v.dot_multipliers = row_scores->clusters[cid].dot_multipliers;
        } else if (score_mode == SearchScoreMode::kScaled) {
          v.source_norm2.resize(entry.real_count);
          if (const auto s = files.ReadFloatSourceNorm2(cid, v.source_norm2);
              !s.ok())
            return s;
        }
      } else {
        auto cluster = DecodeCanonicalNativeBfsClusterRuntime(
            shape.m, cid, entry.real_count, entry.virtual_count,
            entry.real_root_count, files.cluster_records(cid), files.norm8(), row_scores == nullptr);
        if (!cluster.ok()) return cluster.status();
        clusters[cid] = std::move(cluster).value();
      }
      real_count += entry.real_count;
    }
    if (real_count != files.manifest().counts.real)
      return Status::DataLoss(
          "runtime resident real-row count disagrees with manifest");
    std::vector<float> centers;
    ScaledScore8Codebook scaled;
    if (!floating) {
      const auto norm8 = files.norm8();
      if (row_scores != nullptr) centers.assign(256U, 0.0F);
      else centers.assign(norm8.centers, norm8.centers + norm8.center_count);
      if (row_scores == nullptr && score_mode == SearchScoreMode::kScaled) {
        const auto view = files.scaled_score8();
        auto table = ScaledScore8Codebook::Create(
            {view.scales, view.center_count}, {view.biases, view.center_count});
        if (!table.ok()) return table.status();
        scaled = std::move(table).value();
      }
    }
    return Create(std::move(model).value(), std::move(clusters),
                  std::move(centers), parent_mode, coefficient_mode, score_mode,
                  std::move(native), std::move(scaled), std::move(gemm),
                  std::move(parallel), runtime, row_scores ? *row_scores : IndexRowScores{});
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted(
        "runtime canonical index materialization exhausted memory");
  }
}

Status RuntimeExactDenseSearcher::Search(const QueryBatchView& queries,
                                         const SearchOptions& options,
                                         SearchResult* result) const {
  auto trusted = ValidateQueries(queries);
  if (!trusted.ok()) return trusted.status();
  return SearchTrusted(trusted.value(), options, result);
}
StatusOr<TrustedQueryBatchView> RuntimeExactDenseSearcher::ValidateQueries(
    const QueryBatchView& queries) const {
  return TrustedQueryBatchView::Create(queries, model_.shape().dimension);
}
Status RuntimeExactDenseSearcher::SearchProfiledTrusted(
    const TrustedQueryBatchView& queries, const SearchOptions& options,
    SearchResult* result, ExactDenseSearchStatistics* statistics) const {
  if (!statistics)
    return Status::InvalidArgument("runtime statistics are null");
  return SearchImpl(queries, options, result, statistics);
}
Status RuntimeExactDenseSearcher::SearchTrusted(
    const TrustedQueryBatchView& queries, const SearchOptions& options,
    SearchResult* result) const {
  return SearchImpl(queries, options, result, nullptr);
}

Status RuntimeExactDenseSearcher::SearchStreamingTrusted(
    const TrustedQueryBatchView& queries, const SearchOptions& options,
    SearchResult* result, DenseQueryCompletion completion, void* context) const {
  if (completion == nullptr) return Status::InvalidArgument("query completion is required");
  return SearchImpl<true>(queries, options, result, nullptr, completion, context);
}

struct RuntimeExactDenseSearcher::QueryExecution final {
  struct Worker final {
    Worker(const ModelShape& shape, std::uint32_t max_nodes, std::uint32_t k)
        : routes(shape.nlist), top_k(k) {
      dots.reserve(max_nodes);
      scores.reserve(max_nodes);
    }
    std::vector<std::uint32_t> routes;
    std::vector<float> dots, scores;
    QueryTopK top_k;
  };
  const RuntimeExactDenseSearcher* owner;
  const SearchOptions* options;
  SearchResult* output;
  std::vector<Worker>* workers;
  std::vector<float>* roots;
  const std::vector<float>* small;
  const std::vector<float>* edges;
  std::uint32_t first;
  std::vector<std::uint32_t>* selected_routes;
};

struct RuntimeExactDenseSearcher::CompletedQueryExecution final {
  QueryExecution* block;
  const TrustedQueryBatchView* route_queries;
  DenseQueryCompletion completion;
  void* context;
  std::vector<Status>* failures;
  std::atomic<bool>* failed;
  std::uint32_t first;
};

void RuntimeExactDenseSearcher::ExecuteCompletedQuery(
    void* opaque, std::uint32_t worker, std::uint32_t query) noexcept {
  auto& e = *static_cast<CompletedQueryExecution*>(opaque);
  if (e.failed->load(std::memory_order_relaxed)) return;
  auto& block = *e.block;
  auto& failure = (*e.failures)[worker];
  if (e.route_queries) {
    const auto& queries = *e.route_queries;
    failure = block.owner->hnsw_router_->RouteQueryTrusted(
        queries.data() + static_cast<std::size_t>(query) * queries.stride_floats(),
        block.options->nprobe,
        block.selected_routes->data() + static_cast<std::size_t>(query) * block.options->nprobe,
        block.roots->data() + static_cast<std::size_t>(query) * block.owner->model_.shape().nlist);
    if (!failure.ok()) { e.failed->store(true, std::memory_order_relaxed); return; }
  }
  ExecuteQuery(e.block, worker, query);
  const auto row = e.first + query;
  const auto offset = static_cast<std::size_t>(block.first + query) * block.options->top_k;
  failure = e.completion(e.context, worker, row,
      {block.output->ids.data() + offset,
       block.options->top_k});
  if (!failure.ok()) e.failed->store(true, std::memory_order_relaxed);
}

void RuntimeExactDenseSearcher::ExecuteProbeSelection(
    void* context, std::uint32_t worker_index,
    std::uint32_t query_index) noexcept {
  auto& e = *static_cast<QueryExecution*>(context);
  const auto& self = *e.owner;
  auto& routes = (*e.workers)[worker_index].routes;
  const auto nlist = self.model_.shape().nlist;
  const float* roots = e.roots->data() +
      static_cast<std::size_t>(query_index) * nlist;
  for (std::uint32_t cid = 0; cid < nlist; ++cid) routes[cid] = cid;
  const auto better = [&](std::uint32_t lhs, std::uint32_t rhs) {
    const float left = roots[lhs] * self.root_inverse_norms_[lhs];
    const float right = roots[rhs] * self.root_inverse_norms_[rhs];
    return left > right || (left == right && lhs < rhs);
  };
  std::partial_sort(routes.begin(), routes.begin() + e.options->nprobe,
                    routes.end(), better);
  std::copy_n(routes.data(), e.options->nprobe,
              e.selected_routes->data() +
                  static_cast<std::size_t>(query_index) * e.options->nprobe);
}

void RuntimeExactDenseSearcher::ExecuteQuery(
    void* context, std::uint32_t worker_index,
    std::uint32_t query_index) noexcept {
  const auto& e = *static_cast<QueryExecution*>(context);
  const auto& self = *e.owner;
  auto& w = (*e.workers)[worker_index];
  const auto m = self.model_.shape().m;
  const auto nlist = self.model_.shape().nlist;
  const auto offset_for = [query_index](std::uint32_t stride) {
    return static_cast<std::size_t>(query_index) * stride;
  };
  const RuntimeCanonicalQueryTableView tables{
      m, nlist,
      {e.roots->data() + offset_for(nlist), nlist},
      {e.small->data() + offset_for((m - 1U) * 256U), (m - 1U) * 256U},
      {e.edges->data() + offset_for(m * 256U), m * 256U}};
  if (e.selected_routes != nullptr) {
    std::copy_n(e.selected_routes->data() + static_cast<std::size_t>(query_index) * e.options->nprobe,
                e.options->nprobe, w.routes.data());
  } else {
  for (std::uint32_t cid = 0; cid < tables.nlist; ++cid) w.routes[cid] = cid;
  const auto better = [&](std::uint32_t lhs, std::uint32_t rhs) {
    const float left = tables.root0[lhs] * self.root_inverse_norms_[lhs];
    const float right = tables.root0[rhs] * self.root_inverse_norms_[rhs];
    return left > right || (left == right && lhs < rhs);
  };
  std::partial_sort(w.routes.begin(), w.routes.begin() + e.options->nprobe,
                    w.routes.end(), better);
  }
  w.top_k.Reset();
  for (std::uint32_t rank = 0; rank < e.options->nprobe; ++rank) {
    const auto cid = w.routes[rank];
    const auto& cluster = self.clusters_[cid];
    ScoreRuntimeCanonicalClusterTrusted(
        tables, cluster,
        {self.norm8_centers_.data(),
         static_cast<std::uint32_t>(self.norm8_centers_.size())},
        self.coefficient_mode_ == SearchCoefficientMode::kFloat
            ? &self.native_float_[cid]
            : nullptr,
        self.score_mode_, self.scaled_.view(),
        self.materialized_parents_.empty()
            ? ArrayView<const std::uint32_t>{}
            : ArrayView<const std::uint32_t>{self.materialized_parents_[cid]},
        w.dots, w.scores, self.row_scores_.clusters.empty()
            ? nullptr : &self.row_scores_.clusters[cid],
        self.norm8_inverse_norms_.empty() ? nullptr : self.norm8_inverse_norms_.data());
    for (std::size_t real = 0; real < w.scores.size(); ++real)
      w.top_k.Consider(cluster.topology.real_ids[real], w.scores[real]);
  }
  const auto offset =
      static_cast<std::size_t>(e.first + query_index) * e.options->top_k;
  w.top_k.FinalizeTo(e.output->ids.data() + offset,
                     e.output->adc_distances.data() + offset);
}

template <bool Streaming>
Status RuntimeExactDenseSearcher::SearchImpl(
    const TrustedQueryBatchView& queries, const SearchOptions& options,
    SearchResult* result, ExactDenseSearchStatistics* statistics,
    [[maybe_unused]] DenseQueryCompletion completion, [[maybe_unused]] void* context) const {
  const auto wall_start = Clock::now();
  if ((!result && !Streaming) || queries.dimension() != model_.shape().dimension)
    return Status::InvalidArgument(
        "runtime search result or query dimension is invalid");
  if (const auto status = options.Validate(model_.shape()); !status.ok())
    return status;
  try {
    ExactDenseSearchStatistics timing;
    SearchResult output;
    output.query_count = queries.count();
    output.top_k = options.top_k;
    const auto size =
        static_cast<std::uint64_t>(Streaming && !result
            ? std::min(runtime_.query_block_size, queries.count()) : queries.count()) * options.top_k;
    if (size > std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t))
      return Status::ResourceExhausted("runtime search result is too large");
    output.ids.assign(static_cast<std::size_t>(size),
                      std::numeric_limits<std::uint64_t>::max());
    output.adc_distances.assign(static_cast<std::size_t>(size),
                                std::numeric_limits<float>::infinity());
    const auto& shape = model_.shape();
    const auto capacity = std::min(runtime_.query_block_size, queries.count());
    const auto small_columns = (shape.m - 1U) * 256U;
    const auto edge_columns = shape.m * 256U;
    std::vector<float> transformed(static_cast<std::size_t>(capacity) *
                                   shape.dimension);
    std::vector<float> roots(static_cast<std::size_t>(capacity) * shape.nlist);
    std::vector<float> small(static_cast<std::size_t>(capacity) *
                             small_columns);
    std::vector<float> edges(static_cast<std::size_t>(capacity) * edge_columns);
    const bool hnsw = hnsw_router_ && options.nprobe < shape.nlist;
    std::vector<std::uint32_t> hnsw_routes;
    if (statistics != nullptr || (Streaming && hnsw)) {
      hnsw_routes.resize(static_cast<std::size_t>(capacity) * options.nprobe);
    }
    std::vector<Status> completion_failures;
    std::atomic<bool> completion_failed{false};
    if constexpr (Streaming) completion_failures.resize(parallel_->worker_count());
    std::vector<Status> hnsw_failures;
    std::vector<QueryExecution::Worker> workers;
    workers.reserve(parallel_->worker_count());
    for (std::uint32_t w = 0; w < parallel_->worker_count(); ++w)
      workers.emplace_back(shape, max_node_count_, options.top_k);
    for (std::uint32_t first = 0; first < queries.count();) {
      const auto count = std::min(capacity, queries.count() - first);
      const float* input = queries.data() + static_cast<std::size_t>(first) *
                                                queries.stride_floats();
      if (gemm_) {
        auto start = Clock::now();
        std::size_t stride = queries.stride_floats();
        if (!runtime_.pretransform_query_codebooks) {
          if (const auto s = gemm_->MultiplyLeft(
                  {.left = model_.rotation(),
                   .right = input,
                   .query_count = count,
                   .right_leading_dimension = stride,
                   .output = transformed.data(),
                   .output_leading_dimension = shape.dimension});
              !s.ok())
            return s;
          input = transformed.data();
          stride = shape.dimension;
        }
        timing.qt_rotate_seconds += Elapsed(start);
        start = Clock::now();
        const auto build = [&](ColumnMajorMatrixF32View book,
                               std::vector<float>& values) {
          return gemm_->MultiplyTransposedLeft(
              {.left = book,
               .right = input,
               .query_count = count,
               .right_leading_dimension = stride,
               .output = values.data(),
               .output_leading_dimension = book.columns});
        };
        if (hnsw) {
          [[maybe_unused]] const auto block = runtime_.pretransform_query_codebooks
              ? queries.Slice(first, count)
              : TrustedQueryBatchView::FromTrustedTransform(input, count, shape.dimension);
          timing.table_build_seconds += Elapsed(start);
          start = Clock::now();
          if constexpr (!Streaming)
          if (const auto s = hnsw_router_->RouteBlock(block, options.nprobe,
                  *parallel_, hnsw_routes, roots.data(), hnsw_failures); !s.ok()) return s;
          timing.probe_selection_seconds += Elapsed(start);
          start = Clock::now();
        } else if (const auto s = build(model_.root0(), roots); !s.ok()) return s;
        if (const auto s = build(
                {model_.root_small(1).values, shape.dimension, small_columns},
                small);
            !s.ok())
          return s;
        if (const auto s = build(
                {model_.edge(0).values, shape.dimension, edge_columns}, edges);
            !s.ok())
          return s;
        timing.table_build_seconds += Elapsed(start);
      } else {
        auto start = Clock::now();
        for (std::uint32_t q = 0; q < count; ++q) {
          BuildRuntimeCanonicalQueryTablesTrusted(
              model_,
              input + static_cast<std::size_t>(q) * queries.stride_floats(),
              transformed.data() + static_cast<std::size_t>(q) * shape.dimension,
              roots.data() + static_cast<std::size_t>(q) * shape.nlist,
              small.data() + static_cast<std::size_t>(q) * small_columns,
              edges.data() + static_cast<std::size_t>(q) * edge_columns, hnsw);
        }
        if (hnsw) {
          [[maybe_unused]] const auto block = TrustedQueryBatchView::FromTrustedTransform(
              transformed.data(), count, shape.dimension);
          timing.table_build_seconds += Elapsed(start);
          start = Clock::now();
          if constexpr (!Streaming)
          if (const auto s = hnsw_router_->RouteBlock(block, options.nprobe,
                  *parallel_, hnsw_routes, roots.data(), hnsw_failures); !s.ok()) return s;
          timing.probe_selection_seconds += Elapsed(start);
          start = Clock::now();
        }
        timing.table_build_seconds += Elapsed(start);
      }
      QueryExecution execution{this,   &options, &output, &workers,
                               &roots, &small,   &edges,  Streaming && !result ? 0U : first,
                               (hnsw || statistics != nullptr)
                                   ? &hnsw_routes : nullptr};
      if (!hnsw && statistics != nullptr) {
        const auto start = Clock::now();
        parallel_->Run(count, &execution, ExecuteProbeSelection);
        timing.probe_selection_seconds += Elapsed(start);
      }
      const auto start = Clock::now();
      if constexpr (Streaming) {
        const auto route_queries = runtime_.pretransform_query_codebooks
            ? queries.Slice(first, count)
            : TrustedQueryBatchView::FromTrustedTransform(transformed.data(), count, shape.dimension);
        CompletedQueryExecution completed_execution{&execution, hnsw ? &route_queries : nullptr,
            completion, context, &completion_failures, &completion_failed, first};
        parallel_->Run(count, &completed_execution, ExecuteCompletedQuery);
        for (const auto& failure : completion_failures)
          if (!failure.ok()) return failure;
      } else {
        parallel_->Run(count, &execution, ExecuteQuery);
      }
      timing.scan_topk_seconds += Elapsed(start);
      ++timing.query_blocks;
      first += count;
    }
    timing.core_seconds = timing.qt_rotate_seconds +
                          timing.table_build_seconds + timing.probe_selection_seconds +
                          timing.scan_topk_seconds;
    timing.wall_seconds = Elapsed(wall_start);
    // A detailed AVX2 profile is inapplicable to this scalar kernel. Like
    // other fixed-M representations, the request retains coarse timing.
    if (statistics) *statistics = timing;
    if constexpr (Streaming) {
      if (result) *result = std::move(output);
    } else {
      *result = std::move(output);
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("runtime search exhausted memory");
  }
}
}  // namespace stlqpp
