#include "stlqpp/pipeline/training_initializer.h"

#include "stlqpp/index/emit/base_list_reader.h"
#include "stlqpp/training/analysis/round_linkage_analysis.h"
#include "stlqpp/pipeline/root_only_initialization_moment_replay.h"
#include "stlqpp/pipeline/training_list_root_initialization_source.h"
#include "stlqpp/runtime/telemetry/logger.h"
#include "stlqpp/training/linkage/initialization/spherical_root_code_partitioner.h"
#include "stlqpp/training/update/initialization_model_updater.h"
#include "stlqpp/training/update/root_only_initialization_update_sink.h"
#include "stlqpp/training/update/round_update_transcript.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace stlqpp {

TrainingInitializationSummary::TrainingInitializationSummary() = default;
TrainingInitializationSummary::~TrainingInitializationSummary() = default;
TrainingInitializationSummary::TrainingInitializationSummary(
    TrainingInitializationSummary&&) noexcept = default;
TrainingInitializationSummary& TrainingInitializationSummary::operator=(
    TrainingInitializationSummary&&) noexcept = default;

namespace {

using Clock = std::chrono::steady_clock;

double ElapsedSeconds(Clock::time_point begin) noexcept {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

std::string FormatFixed(double value, int precision) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(precision) << value;
  return output.str();
}

std::string FormatSecondsReport(const char* label, double seconds) {
  std::ostringstream output;
  output << label << ": " << seconds << "s";
  return output.str();
}

void ReportRootUpdate(const TrainingInitializationSummary& summary) {
  if (summary.post_root_analysis.has_value()) {
    const double mse = summary.post_root_analysis->squared_error_sum /
        static_cast<double>(summary.basic->rows);
    LogInfo("Train basic MSE after updating C_root: " +
            FormatFixed(static_cast<float>(mse), 6));
    LogInfo(FormatSecondsReport("Train update C_root MSE finalization wall time(s)",
                                summary.timing.root_mse_wall_seconds));
    LogInfo(FormatSecondsReport("Train update C_root solve wall time(s)",
                                summary.timing.root_update_wall_seconds));
  } else {
    LogInfo(FormatSecondsReport("Train update C_root wall time(s)",
                                summary.timing.root_update_wall_seconds));
  }
}

void ReportRootOnlyBadClusters(
    const TrainingInitializationSummary& summary, bool log_metrics) {
  if (log_metrics && summary.bad_clusters->real_rows != 0U) {
    LogInfo("Train init-linkage MSE (C_root only): Max=" +
            FormatFixed(summary.bad_clusters->maximum_squared_error, 4) +
            ", Min=" +
            FormatFixed(summary.bad_clusters->minimum_squared_error, 4) +
            ", Mean=" +
            FormatFixed(summary.bad_clusters->mean_squared_error, 4));
  }
  const std::uint64_t bad_count = static_cast<std::uint64_t>(
      std::count(summary.bad_clusters->mask.begin(),
                 summary.bad_clusters->mask.end(), 1U));
  LogInfo("Virtual init (init linkage): num bad clusters = " +
          std::to_string(bad_count) + " / " +
          std::to_string(summary.bad_clusters->mask.size()) +
          " (baseline=" +
          std::to_string(summary.bad_clusters->quantile_baseline) + ")");
  LogInfo(FormatSecondsReport(
      "Init-linkage bad-mask+MSE (C_root only) wall time(s)",
      summary.timing.bad_mask_wall_seconds));
}

void ReportRootOnlyEdgeAnalysis(
    const TrainingInitializationSummary& summary) {
  LogInfo(FormatSecondsReport(
      "Init-linkage linkage-summary+MSE analysis wall time(s)",
      summary.timing.post_c_one_analysis_wall_seconds));
  LogInfo(
      "Train init-linkage encode analysis after updating C_one "
      "(note: init-linkage stores layer0 as C_root codes; if h0_one<h0_root "
      "this analysis maps root->one, so treat as a reference metric):");
  const double rows =
      static_cast<double>(summary.post_edge_analysis->real_rows);
  LogInfo("  Linkage Ratio: " + FormatFixed(
      100.0 * static_cast<double>(
          summary.post_edge_analysis->linked_real_rows) / rows,
      2) + "%");
  LogInfo("  Max depth: " +
          std::to_string(summary.post_edge_analysis->maximum_depth) +
          " Mean depth: " + FormatFixed(
              static_cast<double>(summary.post_edge_analysis->depth_sum) / rows, 2));
  LogInfo("  Linkage Error: Max=" +
          FormatFixed(summary.post_edge_analysis->maximum_squared_error, 4) +
          ", Min=" +
          FormatFixed(summary.post_edge_analysis->minimum_squared_error, 4) +
          ", Mean=" + FormatFixed(
              summary.post_edge_analysis->squared_error_sum / rows, 4));
}

template <int M, typename Scalar>
Status ValidateRequest(const TrainingInitializationRequest<M>& request,
                       const TrainingInitializationRuntime<M, Scalar>& runtime) {
  if (request.checkpoint_directory.empty() ||
      request.identity.training_source.empty() ||
      request.identity.training_recipe.empty() ||
      request.resume_checkpoint.has_value() ==
          (request.initial_model != nullptr)) {
    return Status::InvalidArgument(
        "training initialization requires one fresh or resume authority");
  }
  if (request.resume_checkpoint.has_value()) {
    return Status::Ok();
  }
  if (runtime.basic_source == nullptr || runtime.basic_runtime == nullptr ||
      (request.mode == TrainingInitializationMode::kLegacyRootOnly &&
       runtime.topology == nullptr) ||
      runtime.parallel == nullptr ||
      runtime.root_moment_parallel == nullptr ||
      runtime.list_parallel == nullptr || runtime.spd_solver == nullptr ||
      (request.mode == TrainingInitializationMode::kFastInit &&
       (runtime.full_topology == nullptr || runtime.metric_gemm == nullptr)) ||
      (request.log_metrics && runtime.metric_gemm == nullptr) ||
      !std::isfinite(request.good_cluster_fraction) ||
      request.good_cluster_fraction < 0.0 ||
      request.good_cluster_fraction > 1.0 ||
      !std::isfinite(request.bad_cluster_training_weight) ||
      request.bad_cluster_training_weight <= 0.0F ||
      request.edge_moment_shard_count == 0U ||
      request.topology_pipeline.maximum_inflight_clusters == 0U ||
      (M != 0 && request.initial_model->codebooks().shape().m !=
          static_cast<std::uint32_t>(M))) {
    return Status::InvalidArgument(
        "fresh training initialization request or runtime is incomplete");
  }
  return Status::Ok();
}

template <int M>
class FastInitializationSink final : public FullTopologyClusterSink<M> {
 public:
  FastInitializationSink(RoundUpdateTranscriptWriter<M>& transcript,
                         InitializationModelUpdater<M>& updater,
                         const ProductCodebooks<M>& codebooks,
                         std::uint32_t nlist) noexcept
      : transcript_(&transcript), updater_(&updater),
        codebooks_(&codebooks), sums_(nlist, 0.0), counts_(nlist, 0U),
        minima_(nlist, 0.0), maxima_(nlist, 0.0) {}

  [[nodiscard]] bool RequiresRotatedExactReal() const noexcept override {
    return true;
  }

  [[nodiscard]] bool RequiresQualityBeforeRelease()
      const noexcept override { return true; }

  [[nodiscard]] Status ObserveQualityBeforeRelease(
      const ClusterTopologyResult<M>& topology) override {
    const std::uint32_t cid = topology.cid;
    if (cid >= sums_.size() ||
        topology.squared_errors.size() != topology.roles.size()) {
      return Status::DataLoss(
          "fast initialization quality stream has invalid extent");
    }
    double sum = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    bool first = true;
    for (const float error : topology.squared_errors) {
      if (!std::isfinite(error) || error < 0.0F) {
        return Status::DataLoss("fast initialization has invalid error");
      }
      sum += error;
      if (first) {
        minimum = maximum = error;
        first = false;
      } else {
        minimum = std::min(minimum, static_cast<double>(error));
        maximum = std::max(maximum, static_cast<double>(error));
      }
    }
    sums_[cid] = sum;
    counts_[cid] = static_cast<std::uint32_t>(topology.roles.size());
    minima_[cid] = minimum;
    maxima_[cid] = maximum;
    return Status::Ok();
  }

  [[nodiscard]] Status Commit(
      std::uint64_t sequence,
      FullTopologyClusterCommitPayload payload,
      FullTopologyClusterProduct<M> product) override {
    const auto& topology = product.topology;
    if (sequence >= sums_.size() || sequence != topology.cid ||
        sequence != payload.cid || product.virtual_roots.has_value() ||
        topology.roles.size() != payload.real_count()) {
      return Status::InvalidArgument(
          "fast initialization received an invalid ordered forest");
    }
    if (const Status status = transcript_->Append(
            topology, payload.real_sample_ids, {}, {});
        !status.ok()) {
      return status;
    }
    if (payload.real_count() == 0U) return Status::Ok();
    return updater_->AccumulateFrozenEdgeCluster(
        *codebooks_, {
            .exact_real = {payload.rotated_exact_real.data(),
                           payload.dimension, payload.real_count()},
            .topology = &topology,
            .virtual_small_codes = {},
            .virtual_coefficients = {},
        });
  }

  [[nodiscard]] InitializationBadClusterResult BadClusters(
      double good_fraction) const {
    std::vector<double> averages;
    averages.reserve(sums_.size());
    double total = 0.0;
    std::uint64_t rows = 0U;
    for (std::size_t cid = 0U; cid < sums_.size(); ++cid) {
      if (counts_[cid] == 0U) continue;
      averages.push_back(sums_[cid] /
                         static_cast<double>(counts_[cid]));
      total += sums_[cid];
      rows += counts_[cid];
    }
    std::sort(averages.begin(), averages.end());
    double baseline = 0.0;
    if (!averages.empty()) {
      const auto rounded = std::llround(
          good_fraction * static_cast<double>(averages.size()));
      const auto position = static_cast<std::size_t>(
          std::max<long long>(1LL, std::min<long long>(
              rounded, static_cast<long long>(averages.size()))));
      baseline = averages[position - 1U];
    }
    InitializationBadClusterResult result;
    result.mask.assign(sums_.size(), 0U);
    result.quantile_baseline = baseline;
    result.real_rows = rows;
    bool has_error = false;
    for (std::size_t cid = 0U; cid < sums_.size(); ++cid) {
      if (counts_[cid] == 0U) continue;
      if (!has_error) {
        result.minimum_squared_error = minima_[cid];
        result.maximum_squared_error = maxima_[cid];
        has_error = true;
      } else {
        result.minimum_squared_error = std::min(
            result.minimum_squared_error, minima_[cid]);
        result.maximum_squared_error = std::max(
            result.maximum_squared_error, maxima_[cid]);
      }
    }
    result.mean_squared_error = rows == 0U
        ? 0.0 : total / static_cast<double>(rows);
    for (std::size_t cid = 0U; cid < sums_.size(); ++cid) {
      if (counts_[cid] != 0U &&
          sums_[cid] / static_cast<double>(counts_[cid]) > baseline) {
        result.mask[cid] = 1U;
      }
    }
    return result;
  }

 private:
  RoundUpdateTranscriptWriter<M>* transcript_;
  InitializationModelUpdater<M>* updater_;
  const ProductCodebooks<M>* codebooks_;
  std::vector<double> sums_;
  std::vector<std::uint32_t> counts_;
  std::vector<double> minima_;
  std::vector<double> maxima_;
};

// Once basic has emitted model-space lists and root moments, neither the
// model update nor topology construction depends on the input scalar format.
template <int M>
struct InitializationModelRuntime final {
  RootOnlyInitializationTopologyContextFactory<M>* topology;
  TrainingTopologyContextFactory<M>* full_topology;
  const ParallelForExecutor* parallel;
  const F32GemmBackend* metric_gemm;
};

template <int M>
StatusOr<TrainingInitializationResult<M>> InitializeFastLinkage(
    const TrainingInitializationRequest<M>& request,
    const InitializationModelRuntime<M>& runtime,
    const RootTrainingModel<M>& root_model,
    InitializationModelUpdater<M>& updater,
    TrainingInitializationSummary summary) {
  const auto& basic = *summary.basic;
  const auto& root = root_model.codebooks();
  SphericalRootCodePartitioner partitioner(*runtime.parallel);
  // The separate fast-init centroid fit uses Master's seed salt. BuildEdgeSeed
  // applies its own internal salt, so cancel that one here.
  auto edge_zero = partitioner.BuildEdgeSeed(
      root.root0(), request.seed ^ 0xC001D00DU ^ 0xF4171C01U);
  if (!edge_zero.ok()) return edge_zero.status();
  ProductCodebookView<M> seeded_view;
  seeded_view.shape = root.shape();
  seeded_view.root0 = root.root0();
  if constexpr (M == 0) {
    seeded_view.root_small.resize(root.m() - 1U);
    seeded_view.edge.resize(root.m());
  }
  seeded_view.edge[0U] = {
      edge_zero.value().data(), root.shape().dimension, 256U};
  for (std::uint32_t layer = 1U;
       layer < root.m(); ++layer) {
    seeded_view.root_small[layer - 1U] = root.root_small(layer);
    seeded_view.edge[layer] = root.root_small(layer);
  }
  auto seeded = TrainingModel<M>::Create(
      root_model.rotation(), seeded_view);
  if (!seeded.ok()) return seeded.status();
  const Clock::time_point init_linkage_begin = Clock::now();
  const std::vector<std::uint8_t> no_bad(root.shape().nlist, 0U);
  const TrainingBasicRoundArtifact artifact{
      .list_directory = request.basic.list_directory,
      .rows = basic.rows,
      .lifetime = {},
      .metric_vectors = 0U,
      .initial_squared_error_sum = 0.0,
      .final_squared_error_sum = 0.0,
      .encode_wall_seconds = 0.0,
  };
  auto context = runtime.full_topology->CreateFull(
      seeded.value(), no_bad, artifact, 0U, request.fast_topology);
  if (!context.ok()) return context.status();
  std::filesystem::path transcript_destination =
      request.checkpoint_directory;
  transcript_destination += ".fast-init-transcript";
  auto transcript_writer = RoundUpdateTranscriptWriter<M>::Create(
      transcript_destination, root.shape().dimension, root.shape().nlist, root.m());
  if (!transcript_writer.ok()) return transcript_writer.status();
  FastInitializationSink<M> sink(
      transcript_writer.value(), updater, seeded.value().codebooks(),
      root.shape().nlist);
  FullTopologyPipelineOptions pipeline = {
      .maximum_inflight_clusters =
          request.topology_pipeline.maximum_inflight_clusters,
      .maximum_prefetched_clusters =
          request.topology_pipeline.maximum_prefetched_clusters,
      .progress_total_clusters = root.shape().nlist,
      .collect_quality_statistics = request.log_metrics,
  };
  auto topology = FullTopologyPipeline<M>::Run(
      context.value()->source(), sink, context.value()->executors(),
      request.fast_topology, pipeline);
  if (!topology.ok()) return topology.status();
  summary.timing.init_linkage_wall_seconds = ElapsedSeconds(init_linkage_begin);
  LogInfo("Init-linkage fast two-codebook wall time(s): " +
          FormatFixed(summary.timing.init_linkage_wall_seconds, 3));
  summary.bad_clusters.emplace(sink.BadClusters(request.good_cluster_fraction));
  auto& bad_clusters = *summary.bad_clusters;
  if (!request.virtual_enabled) {
    std::fill(bad_clusters.mask.begin(), bad_clusters.mask.end(), std::uint8_t{0});
  }
  const auto resources = context.value()->resource_statistics();
  context.value().reset();
  auto transcript = transcript_writer.value().Finish();
  if (!transcript.ok()) return transcript.status();
  const Clock::time_point edge_begin = Clock::now();
  auto update = updater.Finish();
  if (!update.ok()) return update.status();
  summary.timing.c_one_update_wall_seconds = ElapsedSeconds(edge_begin);
  if (request.log_metrics) {
    const Clock::time_point analysis_begin = Clock::now();
    auto analyzed = AnalyzeRoundLinkage<M>(
        request.basic.list_directory, transcript.value(),
        update.value().model, *runtime.metric_gemm, *runtime.parallel);
    if (!analyzed.ok()) return analyzed.status();
    summary.post_edge_analysis.emplace(std::move(analyzed).value());
    summary.timing.post_c_one_analysis_wall_seconds =
        ElapsedSeconds(analysis_begin);
    LogInfo("Train init-linkage fast two-codebook MSE: " +
            FormatFixed(summary.post_edge_analysis->squared_error_sum /
                static_cast<double>(summary.post_edge_analysis->real_rows), 6));
  }
  const Clock::time_point checkpoint_begin = Clock::now();
  if (const Status status = InitializedTrainingCheckpoint<M>::Write(
          request.checkpoint_directory, request.identity,
          update.value().model, bad_clusters.mask);
      !status.ok()) {
    return status;
  }
  auto checkpoint = InitializedTrainingCheckpoint<M>::Open(
      request.checkpoint_directory, request.identity);
  if (!checkpoint.ok()) return checkpoint.status();
  summary.timing.checkpoint_wall_seconds = ElapsedSeconds(checkpoint_begin);
  summary.checkpoint_directory = request.checkpoint_directory;
  summary.fast_topology.emplace(std::move(topology).value());
  summary.update = update.value().statistics;
  summary.linkage_resources = resources;
  return TrainingInitializationResult<M>(
      std::move(checkpoint).value(), std::move(summary));
}

template <int M>
StatusOr<TrainingInitializationResult<M>> FinishInitialization(
    const TrainingInitializationRequest<M>& request,
    const InitializationModelRuntime<M>& runtime,
    InitializationModelUpdater<M>& updater,
    TrainingInitializationSummary summary) {
  const auto& basic = *summary.basic;
  const Clock::time_point root_update_begin = Clock::now();
  auto root_model = updater.FinishRootUpdate();
  if (!root_model.ok()) {
    return root_model.status();
  }
  summary.timing.root_update_wall_seconds = ElapsedSeconds(root_update_begin);
  if (request.log_metrics) {
    const Clock::time_point root_mse_begin = Clock::now();
    const auto squared_error =
        updater.post_root_squared_error_sum();
    if (!squared_error.has_value()) {
      return Status::Internal("post-root moment metric is unavailable");
    }
    summary.timing.root_mse_wall_seconds = ElapsedSeconds(root_mse_begin);
    summary.post_root_analysis.emplace(BasicPostRootAnalysis{
        .rows = basic.rows,
        .squared_error_sum = *squared_error,
    });
  }
  ReportRootUpdate(summary);

  if (request.mode == TrainingInitializationMode::kFastInit) {
    return InitializeFastLinkage(
        request, runtime, root_model.value(), updater, std::move(summary));
  }

  const Clock::time_point init_linkage_begin = Clock::now();
  SphericalRootCodePartitioner partitioner(*runtime.parallel);
  auto mapping = partitioner.Build(
      root_model.value().codebooks().root0(), request.seed);
  if (!mapping.ok()) {
    return mapping.status();
  }
  auto lists = BaseListReader<M>::Open(request.basic.list_directory);
  if (!lists.ok()) {
    return lists.status();
  }
  auto source = TrainingListRootInitializationSource<M>::Create(
      lists.value(), root_model.value(), request.topology.seed);
  if (!source.ok()) {
    return source.status();
  }
  auto context = runtime.topology->Create(root_model.value().codebooks());
  if (!context.ok()) {
    return context.status();
  }
  if (const auto resources = context.value()->resource_statistics(); resources) {
    LogInfo("Init-linkage execution: scheduler_backends=" +
            std::to_string(resources->scheduler_backend_count) +
            " direct_backends=" + std::to_string(resources->direct_backend_count));
  }
  std::filesystem::path transcript_destination =
      request.checkpoint_directory;
  transcript_destination += ".root-only-transcript";
  auto transcript_writer =
      RootOnlyInitializationTranscriptWriter<M>::Create(
          transcript_destination,
          root_model.value().codebooks().shape().dimension,
          root_model.value().codebooks().shape().nlist,
          root_model.value().codebooks().m());
  if (!transcript_writer.ok()) {
    return transcript_writer.status();
  }
  auto sink = RootOnlyInitializationUpdateSink<M>::Create(
      transcript_writer.value(),
      root_model.value().codebooks().shape().nlist,
      request.good_cluster_fraction);
  if (!sink.ok()) {
    return sink.status();
  }
  RootOnlyInitializationTopologyPolicy topology_policy = request.topology;
  auto topology = RootOnlyInitializationPipeline<M>::Run(
      source.value(), sink.value(), context.value()->executors(),
      topology_policy, request.topology_pipeline);
  if (!topology.ok()) {
    return topology.status();
  }
  summary.timing.init_linkage_wall_seconds = ElapsedSeconds(init_linkage_begin);
  LogInfo("Init-linkage wall time(s): " +
          FormatFixed(summary.timing.init_linkage_wall_seconds, 3));
  const Clock::time_point bad_mask_begin = Clock::now();
  auto bad_clusters = sink.value().Finish();
  if (!bad_clusters.ok()) {
    return bad_clusters.status();
  }
  summary.bad_clusters.emplace(std::move(bad_clusters).value());
  if (!request.virtual_enabled) {
    std::fill(summary.bad_clusters->mask.begin(),
              summary.bad_clusters->mask.end(), std::uint8_t{0});
  }
  summary.timing.bad_mask_wall_seconds = ElapsedSeconds(bad_mask_begin);
  ReportRootOnlyBadClusters(summary, request.log_metrics);
  auto transcript = transcript_writer.value().Finish();
  if (!transcript.ok()) {
    return transcript.status();
  }
  const std::optional<LinkageResourceStatistics> linkage_resources =
      context.value()->resource_statistics();
  context.value().reset();
  const Clock::time_point c_one_update_begin = Clock::now();
  auto replay = RootOnlyInitializationMomentReplay<M>::Run(
      lists.value(), root_model.value(), transcript.value(),
      summary.bad_clusters->mask,
      request.bad_cluster_training_weight,
      mapping.value(), updater);
  if (!replay.ok()) {
    return replay.status();
  }
  auto model_update = updater.Finish();
  if (!model_update.ok()) {
    return model_update.status();
  }
  summary.timing.c_one_update_wall_seconds = ElapsedSeconds(c_one_update_begin);
  LogInfo(FormatSecondsReport(
      "Init-linkage update C_one wall time(s)",
      summary.timing.c_one_update_wall_seconds));
  if (request.log_metrics) {
    const Clock::time_point analysis_begin = Clock::now();
    auto analyzed = AnalyzeInitializationAfterEdgeUpdate<M>(
        lists.value(), transcript.value(), model_update.value().model,
        mapping.value(), *runtime.parallel);
    if (!analyzed.ok()) return analyzed.status();
    summary.timing.post_c_one_analysis_wall_seconds =
        ElapsedSeconds(analysis_begin);
    summary.post_edge_analysis.emplace(analyzed.value());
#ifndef NDEBUG
    LogDebug(
        "Train linkage-summary MSE coverage: clusters=" +
        std::to_string(analyzed.value().clusters) + "/" +
        std::to_string(model_update.value().model.codebooks().shape().nlist) +
        " rows=" + std::to_string(analyzed.value().real_rows) + "/" +
        std::to_string(lists.value().store().total_vectors()));
#endif
    ReportRootOnlyEdgeAnalysis(summary);
  }
  if (request.log_checkpoint_publication) {
    LogInfo("Saving train model to: " +
            request.checkpoint_directory.string() + " (R_iters=0)");
  }
  const Clock::time_point checkpoint_begin = Clock::now();
  if (const Status status = InitializedTrainingCheckpoint<M>::Write(
          request.checkpoint_directory, request.identity,
          model_update.value().model, summary.bad_clusters->mask);
      !status.ok()) {
    return status;
  }
  auto checkpoint = InitializedTrainingCheckpoint<M>::Open(
      request.checkpoint_directory, request.identity);
  if (!checkpoint.ok()) {
    return checkpoint.status();
  }
  summary.timing.checkpoint_wall_seconds = ElapsedSeconds(checkpoint_begin);
  summary.checkpoint_directory = request.checkpoint_directory;
  summary.topology = topology.value();
  summary.update = model_update.value().statistics;
  summary.linkage_resources = linkage_resources;
  return TrainingInitializationResult<M>(
      std::move(checkpoint).value(), std::move(summary));
}

}  // namespace

template <int M, typename Scalar>
StatusOr<TrainingInitializationResult<M>> InitializeTraining(
    const TrainingInitializationRequest<M>& request,
    TrainingInitializationRuntime<M, Scalar>& runtime) {
  if (const Status status = ValidateRequest(request, runtime); !status.ok()) {
    return status;
  }
  if (request.resume_checkpoint.has_value()) {
    auto checkpoint = InitializedTrainingCheckpoint<M>::Open(
        *request.resume_checkpoint, request.identity);
    if (!checkpoint.ok()) {
      return checkpoint.status();
    }
    TrainingInitializationSummary summary;
    summary.checkpoint_directory = *request.resume_checkpoint;
    return TrainingInitializationResult<M>(
        std::move(checkpoint).value(), std::move(summary));
  }

  if (const Status status =
          runtime.basic_source->ReleaseRepeatedScanCache();
      !status.ok()) {
    return status;
  }
  auto updater = InitializationModelUpdater<M>::Create(
      *request.initial_model, *runtime.spd_solver,
      runtime.root_accumulation_parallel == nullptr
          ? *runtime.root_moment_parallel
          : *runtime.root_accumulation_parallel,
      *runtime.root_moment_parallel, request.edge_moment_shard_count);
  if (!updater.ok()) {
    return updater.status();
  }
  if (request.log_metrics) {
    if (const Status status = updater.value().EnablePostRootMetric();
        !status.ok()) {
      return status;
    }
  }
  RootInitializationBasicStageOptions basic_options = request.basic;
  basic_options.seed = request.seed;
  TrainingInitializationSummary summary;
  const Clock::time_point basic_begin = Clock::now();
  auto basic = ScalarRootInitializationBasicStage<M, Scalar>::Run(
      *request.initial_model, *runtime.basic_source,
      *runtime.basic_runtime, updater.value(), *runtime.list_parallel,
      basic_options);
  if (!basic.ok()) {
    return basic.status();
  }
  summary.timing.basic_wall_seconds = ElapsedSeconds(basic_begin);
  summary.basic.emplace(std::move(basic).value());
  const InitializationModelRuntime<M> model_runtime{
      runtime.topology, runtime.full_topology, runtime.parallel,
      runtime.metric_gemm};
  return FinishInitialization(request, model_runtime, updater.value(),
                              std::move(summary));
}

template StatusOr<TrainingInitializationResult<0>> InitializeTraining<0>(
    const TrainingInitializationRequest<0>&,
    TrainingInitializationRuntime<0, std::uint8_t>&);
template StatusOr<TrainingInitializationResult<0>> InitializeTraining<0>(
    const TrainingInitializationRequest<0>&,
    TrainingInitializationRuntime<0, float>&);
template StatusOr<TrainingInitializationResult<5>> InitializeTraining<5>(
    const TrainingInitializationRequest<5>&,
    TrainingInitializationRuntime<5, std::uint8_t>&);
template StatusOr<TrainingInitializationResult<12>> InitializeTraining<12>(
    const TrainingInitializationRequest<12>&,
    TrainingInitializationRuntime<12, std::uint8_t>&);
template StatusOr<TrainingInitializationResult<6>> InitializeTraining<6>(
    const TrainingInitializationRequest<6>&,
    TrainingInitializationRuntime<6, std::uint8_t>&);
template StatusOr<TrainingInitializationResult<10>> InitializeTraining<10>(
    const TrainingInitializationRequest<10>&,
    TrainingInitializationRuntime<10, std::uint8_t>&);
template StatusOr<TrainingInitializationResult<5>> InitializeTraining<5>(
    const TrainingInitializationRequest<5>&,
    TrainingInitializationRuntime<5, float>&);
template StatusOr<TrainingInitializationResult<12>> InitializeTraining<12>(
    const TrainingInitializationRequest<12>&,
    TrainingInitializationRuntime<12, float>&);
template StatusOr<TrainingInitializationResult<6>> InitializeTraining<6>(
    const TrainingInitializationRequest<6>&,
    TrainingInitializationRuntime<6, float>&);
template StatusOr<TrainingInitializationResult<10>> InitializeTraining<10>(
    const TrainingInitializationRequest<10>&,
    TrainingInitializationRuntime<10, float>&);

}  // namespace stlqpp
