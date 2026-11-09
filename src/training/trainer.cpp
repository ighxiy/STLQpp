#include "stlqpp/training/trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "stlqpp/runtime/telemetry/logger.h"
#include "stlqpp/storage/trained_model_file.h"
#include "stlqpp/training/analysis/basic_post_root_analysis.h"
#include "stlqpp/training/analysis/round_linkage_analysis.h"
#include "stlqpp/training/checkpoint/checkpoint_path.h"

namespace stlqpp {
namespace {

using Clock = std::chrono::steady_clock;

double ElapsedSeconds(Clock::time_point begin) noexcept {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

std::filesystem::path RoundCheckpointDirectory(
    const std::filesystem::path& root, std::uint32_t round) {
  std::ostringstream name;
  name << "round_" << std::setw(8) << std::setfill('0') << round;
  return root / name.str();
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

std::string FormatLinkageTimeReport(double wall_seconds, double sink_seconds) {
  std::ostringstream output;
  output << "Iteration-linkage wall time(s): " << wall_seconds
         << "s (sink: " << sink_seconds << "s)";
  return output.str();
}

template <typename Statistics>
Status LogLinkageAnalysis(const char* heading, const Statistics& statistics) {
  if (statistics.real_rows == 0U || !statistics.has_squared_error) {
    return Status::DataLoss(
        "linkage analysis requested without complete real-row errors");
  }
  if (statistics.linked_real_rows > statistics.real_rows) {
    return Status::DataLoss(
        "linkage analysis linked-row count exceeds real rows");
  }
  const double rows = static_cast<double>(statistics.real_rows);
  LogInfo(heading);
  LogInfo(
      "  Linkage Ratio: " +
      FormatFixed(
          100.0 * static_cast<double>(statistics.linked_real_rows) / rows, 2) +
      "%");
  LogInfo("  Max depth: " + std::to_string(statistics.maximum_depth) +
          " Mean depth: " +
          FormatFixed(static_cast<double>(statistics.depth_sum) / rows, 2));
  LogInfo("  Linkage Error: Max=" +
          FormatFixed(statistics.maximum_squared_error, 4) +
          ", Min=" + FormatFixed(statistics.minimum_squared_error, 4) +
          ", Mean=" + FormatFixed(statistics.squared_error_sum / rows, 4));
  return Status::Ok();
}

Status ValidateRequest(const TrainRequest& request, bool execution_ready,
                       bool metrics_ready, bool rotation_ready) {
  if (request.checkpoint_root.empty() ||
      request.identity.training_source.empty() ||
      request.identity.training_recipe.empty() ||
      !std::isfinite(request.bad_cluster_training_weight) ||
      request.bad_cluster_training_weight <= 0.0F) {
    return Status::InvalidArgument("training request or runtime is incomplete");
  }
  if (request.resume_checkpoint.has_value() ==
      request.initialization_checkpoint.has_value()) {
    return Status::InvalidArgument(
        "training requires exactly one initialization or round checkpoint");
  }
  if (request.periodic_checkpoint_enabled &&
      request.periodic_checkpoint_every_rounds != 0U &&
      request.periodic_checkpoint_base_path.empty()) {
    return Status::InvalidArgument(
        "periodic training checkpoint path is empty");
  }
  if (request.periodic_checkpoint_enabled &&
      (request.periodic_checkpoint_every_rounds & 1U) != 0U) {
    return Status::InvalidArgument(
        "periodic training checkpoint interval must be even");
  }
  if (!request.round_seeds.empty() &&
      (!execution_ready ||
       request.topology_pipeline.maximum_inflight_clusters == 0U ||
       request.edge_moment_shard_count == 0U)) {
    return Status::InvalidArgument("training execution runtime is incomplete");
  }
  if (!request.round_seeds.empty() && request.log_metrics && !metrics_ready) {
    return Status::InvalidArgument(
        "training metric scan has no model-space GEMM backend");
  }
  if (!request.round_seeds.empty() && request.update_opq_rotation &&
      !rotation_ready) {
    return Status::InvalidArgument(
        "training rotation lacks GEMM or numerical partitions");
  }
  return Status::Ok();
}

// Type erasure stops at coarse numerical stages. No per-row operation goes
// through this boundary; policy, logging, timing and publication run once.
class TrainingRoundDriver {
 public:
  virtual ~TrainingRoundDriver() = default;
  [[nodiscard]] virtual Status BeginRound(TopologyRoundKind) = 0;
  [[nodiscard]] virtual Status AccumulateBasic(std::uint32_t, std::uint32_t,
                                               TopologyRoundKind) = 0;
  [[nodiscard]] virtual const TrainingBasicRoundArtifact& basic() const = 0;
  [[nodiscard]] virtual Status FinishRoot() = 0;
  [[nodiscard]] virtual std::optional<double> root_error() const = 0;
  [[nodiscard]] virtual Status PrepareCheckpoint(std::uint32_t,
                                                 TopologyRoundKind) = 0;
  [[nodiscard]] virtual Status RunTopology(TrainingRoundStatistics&) = 0;
  [[nodiscard]] virtual StatusOr<InitializationPostEdgeAnalysis> Analyze(
      bool post_edge, bool metrics) = 0;
  [[nodiscard]] virtual Status FinishEdge() = 0;
  [[nodiscard]] virtual StatusOr<RoundModelUpdateStatistics> FinishRound(
      bool rotation) = 0;
  [[nodiscard]] virtual Status FinishCheckpoint() = 0;
  [[nodiscard]] virtual Status Publish(const std::filesystem::path&,
                                       bool model_only) = 0;
  virtual void Advance() = 0;
  virtual void EndRound() = 0;
  [[nodiscard]] virtual bool can_fuse_rotation() const = 0;
  [[nodiscard]] virtual std::uint32_t nlist() const = 0;
};

StatusOr<std::vector<TrainingRoundStatistics>> RunTrainingRounds(
    const TrainRequest& request, std::uint32_t first_global_round,
    std::filesystem::path& current_directory, TrainingRoundDriver& driver) {
  auto schedule = PeriodTwoTopologySchedule::Create(
      static_cast<std::uint32_t>(request.round_seeds.size()));
  if (!schedule.ok()) {
    return schedule.status();
  }

  std::vector<TrainingRoundStatistics> statistics;
  statistics.reserve(schedule.value().round_count());
  for (std::uint32_t local_round = 0U;
       local_round < schedule.value().round_count(); ++local_round) {
    const std::uint32_t round = first_global_round + local_round;
    LogInfo(
        "============== Global R iteration " + std::to_string(round + 1U) +
        " / " +
        std::to_string(first_global_round + schedule.value().round_count()) +
        " ==============");
    const Clock::time_point round_begin = Clock::now();
    const TrainingRoundSeeds seeds = request.round_seeds[local_round];
    const TopologyRoundKind topology_kind = schedule.value().kind(local_round);
    if (topology_kind == TopologyRoundKind::kFull) {
      LogInfo("Train topology schedule: full round " +
              std::to_string(local_round + 1U) + " (period=2)");
    } else {
      LogInfo("Train topology schedule: replay round " +
              std::to_string(local_round + 1U) + " (full rebuild period=2)");
    }
    if (const auto status = driver.BeginRound(topology_kind); !status.ok())
      return status;
    const Clock::time_point basic_begin = Clock::now();
    if (const auto status =
            driver.AccumulateBasic(round, seeds.basic, topology_kind);
        !status.ok())
      return status;
    const auto& basic = driver.basic();
    if (basic.list_directory.empty() || basic.rows == 0U) {
      return Status::FailedPrecondition(
          "training basic pass did not publish its round artifact");
    }
    const double basic_wall_seconds = ElapsedSeconds(basic_begin);
    const Clock::time_point root_update_begin = Clock::now();
    if (const auto status = driver.FinishRoot(); !status.ok()) return status;

    const double root_update_wall_seconds = ElapsedSeconds(root_update_begin);
    BasicPostRootAnalysis root_analysis;
    double root_mse_wall_seconds = 0.0;
    if (request.log_metrics) {
      const Clock::time_point root_mse_begin = Clock::now();
      const auto squared_error = driver.root_error();
      if (!squared_error.has_value()) {
        return Status::Internal("post-root moment metric is unavailable");
      }
      root_analysis = {.rows = basic.rows, .squared_error_sum = *squared_error};
      root_mse_wall_seconds = ElapsedSeconds(root_mse_begin);
      const double mse = root_analysis.squared_error_sum /
                         static_cast<double>(root_analysis.rows);
      LogInfo("Train basic MSE after updating C_root: " +
              FormatFixed(static_cast<float>(mse), 6));
      // Source-norm accumulation occurs in the basic pass and the RHS
      // contraction in the root solve; this timer covers final retrieval.
      LogInfo(FormatSecondsReport("Train update C_root MSE finalization wall time(s)",
                                  root_mse_wall_seconds));
      LogInfo(FormatSecondsReport("Train update C_root solve wall time(s)",
                                  root_update_wall_seconds));
    } else {
      LogInfo(FormatSecondsReport("Train update C_root wall time(s)",
                                  root_update_wall_seconds));
    }
    const Clock::time_point topology_checkpoint_begin = Clock::now();
    // Each local schedule starts with full topology, then alternates with
    // replay. Save on the last full round before an even-round boundary.
    const bool periodic_checkpoint_due =
        request.periodic_checkpoint_enabled &&
        request.periodic_checkpoint_every_rounds != 0U &&
        (local_round % request.periodic_checkpoint_every_rounds ==
         request.periodic_checkpoint_every_rounds - 2U);
    if (const auto status = driver.PrepareCheckpoint(round, topology_kind);
        !status.ok())
      return status;

    TrainingRoundStatistics round_statistics;
    round_statistics.round = round;
    round_statistics.seeds = seeds;
    round_statistics.topology_kind = topology_kind;
    round_statistics.basic_wall_seconds = basic_wall_seconds;
    round_statistics.root_update_wall_seconds = root_update_wall_seconds;
    round_statistics.root_mse_wall_seconds = root_mse_wall_seconds;
    round_statistics.root_metric_rows = root_analysis.rows;
    round_statistics.root_post_update_mse =
        root_analysis.rows == 0U ? 0.0
                                 : root_analysis.squared_error_sum /
                                       static_cast<double>(root_analysis.rows);
    if (const auto status = driver.RunTopology(round_statistics); !status.ok())
      return status;
    if (request.log_metrics && request.log_linkage_pre_c1) {
      const Clock::time_point analysis_begin = Clock::now();
      auto analyzed = driver.Analyze(false, true);
      if (!analyzed.ok()) return analyzed.status();
      round_statistics.pre_c_one_analysis_wall_seconds =
          ElapsedSeconds(analysis_begin);
      round_statistics.pre_c_one_analysis.emplace(analyzed.value());
      LogInfo(FormatSecondsReport(
          "Iteration-linkage linkage-summary+MSE analysis "
          "(pre C_one update) wall time(s)",
          round_statistics.pre_c_one_analysis_wall_seconds));
      if (const Status status = LogLinkageAnalysis(
              "Train Linkage encode analysis before updating C_one:",
              analyzed.value());
          !status.ok()) {
        return status;
      }
    }
    const Clock::time_point edge_update_begin = Clock::now();
    if (const auto status = driver.FinishEdge(); !status.ok()) return status;
    round_statistics.edge_update_wall_seconds =
        ElapsedSeconds(edge_update_begin);
    LogInfo(FormatSecondsReport("Iteration-linkage update C_one wall time(s)",
                                round_statistics.edge_update_wall_seconds));
    if (request.log_metrics) {
      const Clock::time_point analysis_begin = Clock::now();
      auto analyzed = driver.Analyze(true, true);
      if (!analyzed.ok()) return analyzed.status();
      round_statistics.post_edge_analysis_wall_seconds =
          ElapsedSeconds(analysis_begin);
      round_statistics.post_edge_analysis.emplace(analyzed.value());
#ifndef NDEBUG
      LogDebug("Train linkage-summary MSE coverage: clusters=" +
               std::to_string(analyzed.value().clusters) + "/" +
               std::to_string(driver.nlist()) +
               " rows=" + std::to_string(analyzed.value().real_rows) + "/" +
               std::to_string(basic.rows));
#endif
      LogInfo(FormatSecondsReport(
          "Iteration-linkage linkage-summary+MSE analysis "
          "(post C_one update) wall time(s)",
          round_statistics.post_edge_analysis_wall_seconds));
      if (const Status status = LogLinkageAnalysis(
              "Train Linkage encode analysis after updating C_one:",
              analyzed.value());
          !status.ok()) {
        return status;
      }
    }
    if (!request.log_metrics && request.update_opq_rotation &&
        driver.can_fuse_rotation()) {
      auto covariance_only = driver.Analyze(true, false);
      if (!covariance_only.ok()) return covariance_only.status();
    }
    std::optional<double> rotation_update_wall_seconds;
    StatusOr<RoundModelUpdateStatistics> update =
        [&]() -> StatusOr<RoundModelUpdateStatistics> {
      if (!request.update_opq_rotation) {
        LogInfo(
            "Train streaming: use_opq_rotation=false; keeping R unchanged.");
        return driver.FinishRound(false);
      }
      const auto rotation_begin = Clock::now();
      auto finished = driver.FinishRound(true);
      if (finished.ok())
        rotation_update_wall_seconds = ElapsedSeconds(rotation_begin);
      return finished;
    }();
    if (!update.ok()) {
      return update.status();
    }
    if (rotation_update_wall_seconds.has_value()) {
      LogInfo("Iteration-tail wall time(s): opq_update=" +
              FormatFixed(*rotation_update_wall_seconds, 3));
    }
    round_statistics.update = update.value();
    if (const auto status = driver.FinishCheckpoint(); !status.ok())
      return status;
    const bool final_round = local_round + 1U == schedule.value().round_count();
    if (periodic_checkpoint_due || final_round) {
      if (request.periodic_checkpoint_enabled &&
          !request.periodic_checkpoint_base_path.empty()) {
        const std::filesystem::path recovery_directory =
            RoundCheckpointDirectory(request.checkpoint_root, round);
        if (const Status status = driver.Publish(recovery_directory, false);
            !status.ok())
          return status;
      }
      if (!request.periodic_checkpoint_base_path.empty()) {
        auto destination =
            NextNumberedTrainedModelPath(request.periodic_checkpoint_base_path);
        if (!destination.ok()) return destination.status();
        current_directory = std::move(destination).value();
      } else {
        current_directory =
            RoundCheckpointDirectory(request.checkpoint_root, round);
      }
      if (request.periodic_checkpoint_base_path.empty()) {
        LogInfo("Saving train checkpoint to: " + current_directory.string() +
                " (R_iters=" + std::to_string(round + 1U) + ")");
        if (const Status status = driver.Publish(current_directory, false);
            !status.ok())
          return status;
      } else {
        LogInfo("Saving train model to: " + current_directory.string() +
                " (R_iters=" + std::to_string(round + 1U) + ")");
        if (const Status status = driver.Publish(current_directory, true);
            !status.ok())
          return status;
      }
    }
    driver.Advance();
    round_statistics.topology_checkpoint_wall_seconds =
        ElapsedSeconds(topology_checkpoint_begin);
    round_statistics.wall_seconds = ElapsedSeconds(round_begin);
    {
      std::ostringstream timing;
      timing << std::fixed << std::setprecision(3)
             << "Global R iteration wall time(s): "
             << round_statistics.wall_seconds;
      LogInfo(timing.str());
    }
    statistics.push_back(std::move(round_statistics));
    driver.EndRound();
  }

  return statistics;
}

template <int M>
class TypedTrainingRoundDriver final : public TrainingRoundDriver {
 public:
  TypedTrainingRoundDriver(
      const TrainRequest& request, TrainingRuntime<M>& runtime,
      std::optional<TrainingCheckpoint<M>>& current,
      const std::optional<InitializedTrainingCheckpoint<M>>& initialized)
      : request_(request),
        runtime_(runtime),
        current_(current),
        initialized_(initialized) {}

  Status BeginRound(TopologyRoundKind topology_kind) override {
    const auto& request = request_;
    auto& runtime = runtime_;
    const auto& previous = current_ ? current_->model() : initialized_->model();
    const auto bad_cluster_mask = current_ ? current_->bad_cluster_mask()
                                           : initialized_->bad_cluster_mask();
    auto updater = RoundModelUpdater<M>::Create(
        previous, *runtime.spd_solver, *runtime.svd,
        {
            .bad_cluster_mask = bad_cluster_mask,
            .bad_cluster_edge_weight = request.bad_cluster_training_weight,
        },
        runtime.root_accumulation_parallel == nullptr
            ? *runtime.root_moment_parallel
            : *runtime.root_accumulation_parallel,
        *runtime.root_moment_parallel, request.edge_moment_shard_count,
        topology_kind == TopologyRoundKind::kReplay ? runtime.update_parallel
                                                    : nullptr);
    if (!updater.ok()) {
      return updater.status();
    }
    state_.emplace(std::move(updater).value());
    if (request.log_metrics) {
      if (const Status status = state_->updater.EnablePostRootMetric();
          !status.ok()) {
        return status;
      }
    }
    return Status::Ok();
  }

  Status AccumulateBasic(std::uint32_t round, std::uint32_t seed,
                         TopologyRoundKind kind) override {
    const auto& previous = current_ ? current_->model() : initialized_->model();
    auto result = kind == TopologyRoundKind::kReplay
                      ? runtime_.basic->AccumulateFixedRoots(
                            previous, *current_, round, seed, state_->updater)
                      : runtime_.basic->Accumulate(previous, round, seed,
                                                   state_->updater);
    if (!result.ok()) return result.status();
    state_->basic = std::move(result).value();
    return Status::Ok();
  }
  const TrainingBasicRoundArtifact& basic() const override {
    return state_->basic;
  }
  Status FinishRoot() override {
    auto model = state_->updater.FinishRootUpdate();
    if (!model.ok()) return model.status();
    state_->topology_model.emplace(std::move(model).value());
    return Status::Ok();
  }
  std::optional<double> root_error() const override {
    return state_->updater.post_root_squared_error_sum();
  }
  Status PrepareCheckpoint(std::uint32_t round,
                           TopologyRoundKind kind) override {
    const auto mask = current_ ? current_->bad_cluster_mask()
                               : initialized_->bad_cluster_mask();
    auto writer = TrainingCheckpointWriter<M>::Create(
        {.completed_round = round,
         .identity = request_.identity,
         .shape = state_->topology_model->codebooks().shape(),
         .bad_cluster_mask = mask,
         .expected_real_rows = state_->basic.rows,
         .reuse_forest_from =
             kind == TopologyRoundKind::kReplay ? &*current_ : nullptr});
    if (!writer.ok()) return writer.status();
    state_->writer.emplace(std::move(writer).value());
    return Status::Ok();
  }
  Status RunTopology(TrainingRoundStatistics& round_statistics) override {
    const auto& request = request_;
    auto& runtime = runtime_;
    auto& current_checkpoint = current_;
    const auto round = round_statistics.round;
    const auto seeds = round_statistics.seeds;
    const auto bad_cluster_mask = current_ ? current_->bad_cluster_mask()
                                           : initialized_->bad_cluster_mask();
    auto& topology_model = state_->topology_model;
    auto& writer = state_->writer;
    auto& update_transcript = state_->update_transcript;
    const auto& basic = state_->basic;
    const auto transcript_destination =
        basic.list_directory.parent_path() / "round_update";
    if (round_statistics.topology_kind == TopologyRoundKind::kFull) {
      FullTopologyPolicy policy = request.topology_policy;
      policy.seed = seeds.topology;
      policy.virtual_root_seed = seeds.basic;
      auto context = runtime.topology->CreateFull(
          topology_model.value(), bad_cluster_mask, basic, round, policy);
      if (!context.ok()) {
        return context.status();
      }
      FullTopologyPipelineOptions topology_options = request.topology_pipeline;
      topology_options.collect_quality_statistics = false;
      auto result = FullTopologyCheckpointRound<M>::Run(
          context.value()->source(), context.value()->executors(), policy,
          topology_options, topology_model.value(), transcript_destination,
          state_->updater, writer.value());
      if (!result.ok()) {
        return result.status();
      }
      round_statistics.full.emplace(std::move(result.value().topology));
      LogInfo(FormatLinkageTimeReport(round_statistics.full->wall_seconds,
                                      round_statistics.full->sink_seconds));
      update_transcript.emplace(std::move(result.value().transcript));
      round_statistics.linkage_resources =
          context.value()->resource_statistics();
    } else {
      if (!current_checkpoint.has_value()) {
        return Status::Internal(
            "period-two replay has no preceding full checkpoint");
      }
      auto context = runtime.topology->CreateReplay(
          *current_checkpoint, topology_model.value(), basic, round,
          request.topology_policy.rsd_rank, request.topology_policy.icm_rounds,
          seeds.topology);
      if (!context.ok()) {
        return context.status();
      }
      auto result = ReplayCheckpointRound<M>::Run(
          context.value()->loader(), context.value()->executor(),
          topology_model.value(), transcript_destination, state_->updater,
          writer.value(), false);
      if (!result.ok()) {
        return result.status();
      }
      round_statistics.replay.emplace(result.value().replay);
      LogInfo(FormatLinkageTimeReport(round_statistics.replay->wall_seconds,
                                      round_statistics.replay->sink_seconds));
      update_transcript.emplace(std::move(result.value().transcript));
    }

    return Status::Ok();
  }
  StatusOr<InitializationPostEdgeAnalysis> Analyze(bool post_edge,
                                                   bool metrics) override {
    const bool fused = post_edge && request_.update_opq_rotation &&
                       runtime_.covariance_backend;
    return AnalyzeRoundLinkage<M>(
        state_->basic.list_directory, *state_->update_transcript,
        post_edge ? *state_->post_edge_model : *state_->topology_model,
        *runtime_.metric_gemm, *runtime_.update_parallel,
        fused ? runtime_.covariance_backend : nullptr,
        fused ? &state_->fused_covariance : nullptr, metrics,
        fused ? runtime_.rotation_moment_parallel->worker_count() : 0U);
  }
  Status FinishEdge() override {
    auto model = state_->updater.FinishEdgeUpdate();
    if (!model.ok()) return model.status();
    state_->post_edge_model.emplace(std::move(model).value());
    return Status::Ok();
  }
  StatusOr<RoundModelUpdateStatistics> FinishRound(bool rotation) override {
    auto result = [&]() -> StatusOr<RoundModelUpdate<M>> {
      if (!rotation) return state_->updater.FinishRoundKeepingRotation();
      auto covariance = state_->fused_covariance
                            ? StatusOr<RotationCrossCovariance>(
                                  std::move(*state_->fused_covariance))
                            : runtime_.rotation->Accumulate(
                                  state_->basic, *state_->update_transcript,
                                  *state_->post_edge_model);
      if (!covariance.ok()) return covariance.status();
      return state_->updater.FinishRound(std::move(covariance).value(), *runtime_.metric_gemm);
    }();
    if (!result.ok()) return result.status();
    state_->update.emplace(std::move(result).value());
    return state_->update->statistics;
  }
  Status FinishCheckpoint() override {
    auto checkpoint =
        state_->writer->FinishMemory(std::move(state_->update->model));
    if (!checkpoint.ok()) return checkpoint.status();
    state_->next.emplace(std::move(checkpoint).value());
    return Status::Ok();
  }
  Status Publish(const std::filesystem::path& path, bool model_only) override {
    return model_only ? state_->next->PublishModelOnly(path)
                      : state_->next->Publish(path);
  }
  void Advance() override { current_.emplace(std::move(*state_->next)); }
  void EndRound() override { state_.reset(); }
  bool can_fuse_rotation() const override {
    return runtime_.covariance_backend != nullptr;
  }
  std::uint32_t nlist() const override {
    return state_->topology_model->codebooks().shape().nlist;
  }

 private:
  // Preserve round resource lifetime and destruction order. In particular,
  // transcript/model/writer owners die before the basic artifact and updater.
  struct State {
    explicit State(RoundModelUpdater<M> value) : updater(std::move(value)) {}
    RoundModelUpdater<M> updater;
    TrainingBasicRoundArtifact basic;
    std::optional<TrainingModel<M>> topology_model;
    std::optional<TrainingCheckpointWriter<M>> writer;
    std::optional<TrainingModel<M>> post_edge_model;
    std::optional<RoundUpdateTranscript<M>> update_transcript;
    std::optional<RotationCrossCovariance> fused_covariance;
    std::optional<RoundModelUpdate<M>> update;
    std::optional<TrainingCheckpoint<M>> next;
  };
  const TrainRequest& request_;
  TrainingRuntime<M>& runtime_;
  std::optional<TrainingCheckpoint<M>>& current_;
  const std::optional<InitializedTrainingCheckpoint<M>>& initialized_;
  std::optional<State> state_;
};

}  // namespace

template <int M>
StatusOr<TrainResult<M>> Train(const TrainRequest& request,
                               TrainingRuntime<M>& runtime) {
  if (const Status status = ValidateRequest(
          request,
          runtime.basic && runtime.topology && runtime.rotation &&
              runtime.spd_solver && runtime.svd &&
              runtime.root_moment_parallel && runtime.update_parallel,
          runtime.metric_gemm != nullptr,
          runtime.metric_gemm &&
              (!runtime.covariance_backend || runtime.rotation_moment_parallel));
      !status.ok()) {
    return status;
  }

  std::optional<TrainingCheckpoint<M>> current_checkpoint;
  std::optional<InitializedTrainingCheckpoint<M>> initialized;
  std::filesystem::path current_directory;
  std::uint32_t first_global_round = 0U;
  if (request.resume_checkpoint.has_value()) {
    const std::filesystem::path initialized_manifest =
        *request.resume_checkpoint / "initialized.stlqpp";
    std::error_code error;
    bool initialization_only =
        std::filesystem::exists(initialized_manifest, error);
    if (error) {
      return Status::Internal("cannot inspect trained model artifact: " +
                              error.message());
    }
    if (std::filesystem::is_regular_file(*request.resume_checkpoint, error)) {
      auto artifact = TrainedModelFile::Open(*request.resume_checkpoint);
      if (!artifact.ok()) return artifact.status();
      initialization_only =
          artifact.value().header().kind == TrainedModelKind::kInitialized;
    }
    if (error) {
      return Status::Internal("cannot inspect resumed model path: " +
                              error.message());
    }
    if (initialization_only) {
      auto reopened = InitializedTrainingCheckpoint<M>::Open(
          *request.resume_checkpoint, request.identity);
      if (!reopened.ok()) return reopened.status();
      current_directory = *request.resume_checkpoint;
      initialized.emplace(std::move(reopened).value());
    } else {
      auto reopened = TrainingCheckpoint<M>::Open({
          .directory = *request.resume_checkpoint,
          .expected_identity = request.identity,
      });
      if (!reopened.ok()) return reopened.status();
      // The local schedule starts with Full: only model, mask and round
      // metadata are inputs. Replay consumes the new Full checkpoint after
      // Advance(), never the resumed artifact's forest.
      current_directory = *request.resume_checkpoint;
      first_global_round = reopened.value().completed_round() + 1U;
      current_checkpoint.emplace(std::move(reopened).value());
    }
  } else {
    auto reopened = InitializedTrainingCheckpoint<M>::Open(
        *request.initialization_checkpoint, request.identity);
    if (!reopened.ok()) {
      return reopened.status();
    }
    initialized.emplace(std::move(reopened).value());
    current_directory = *request.initialization_checkpoint;
  }

  if (request.round_seeds.empty()) {
    if (current_checkpoint.has_value()) {
      return TrainResult<M>{
          .checkpoint_directory = std::move(current_directory),
          .checkpoint =
              TrainingFinalCheckpoint<M>(std::move(*current_checkpoint)),
          .executed_rounds = {},
      };
    }
    return TrainResult<M>{
        .checkpoint_directory = std::move(current_directory),
        .checkpoint = TrainingFinalCheckpoint<M>(std::move(*initialized)),
        .executed_rounds = {},
    };
  }

  TypedTrainingRoundDriver<M> driver(request, runtime, current_checkpoint,
                                     initialized);
  auto statistics =
      RunTrainingRounds(request, first_global_round, current_directory, driver);
  if (!statistics.ok()) return statistics.status();

  if (!current_checkpoint.has_value()) {
    return Status::Internal("training produced no final checkpoint");
  }
  return TrainResult<M>{
      .checkpoint_directory = std::move(current_directory),
      .checkpoint = TrainingFinalCheckpoint<M>(std::move(*current_checkpoint)),
      .executed_rounds = std::move(statistics).value(),
  };
}

template StatusOr<TrainResult<0>> Train<0>(const TrainRequest&,
                                           TrainingRuntime<0>&);

template StatusOr<TrainResult<5>> Train<5>(const TrainRequest&,
                                           TrainingRuntime<5>&);
template StatusOr<TrainResult<12>> Train<12>(const TrainRequest&,
                                             TrainingRuntime<12>&);
template StatusOr<TrainResult<6>> Train<6>(const TrainRequest&,
                                           TrainingRuntime<6>&);
template StatusOr<TrainResult<10>> Train<10>(const TrainRequest&,
                                             TrainingRuntime<10>&);

}  // namespace stlqpp
