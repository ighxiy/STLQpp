#include "build_completion_log.h"

#include "control_evidence.h"
#include "effective_build_recipe.h"

#include "stlqpp/application/config_text.h"
#include "stlqpp/runtime/telemetry/logger.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>

namespace {

using stlqpp::Status;
using stlqpp::app::BuildCommand;
using stlqpp::app::BuildExecutionReport;
using stlqpp::app::BuildResourcePlan;
using stlqpp::app::EffectiveBuildRecipe;

Status WriteRequestedConfig(
    const BuildCommand& command,
    const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return Status::Internal(
        "failed to create requested build config: " + path.string());
  }
  for (const auto& assignment : command.effective_assignments) {
    output << assignment.key << " = "
           << stlqpp::app::QuoteConfigValue(assignment.value)
           << "  # source: " << assignment.source << '\n';
  }
  output.close();
  return output.fail()
      ? Status::Internal(
            "failed to close requested build config: " + path.string())
      : Status::Ok();
}

BuildCommand MakeEffectiveCommand(
    const BuildCommand& requested,
    const EffectiveBuildRecipe* recipe) {
  BuildCommand effective = requested;
  const stlqpp::app::EffectiveBuildExecutionConfig execution =
      stlqpp::app::ResolveBuildExecutionConfig(requested);
  effective.initialization_linkage_direct_pool_size =
      static_cast<std::int32_t>(execution.initialization_direct_pool_size);
  effective.linkage_init_max_batch_pairs = static_cast<std::int32_t>(
      execution.initialization_maximum_batch_pairs);
  effective.linkage_eval_async_pinned_mebibytes =
      static_cast<std::int32_t>(
          stlqpp::app::EffectiveLinkageAsyncPinnedMebibytes(requested));
  effective.initialization_linkage_eval_async_pinned_mebibytes =
      static_cast<std::int32_t>(
          execution.initialization_async_pinned_mebibytes);
  effective.linkage_scheduler_workers = static_cast<std::int32_t>(
      stlqpp::app::EffectiveLinkageSchedulerWorkers(
          requested, requested.linkage_direct_pool_size));
  effective.init_linkage_large_root_xc0_chunk_mebibytes =
      static_cast<std::int32_t>(
          execution.initialization_large_root_xc0_chunk_mebibytes);
  effective.init_linkage_large_root_workspace_mebibytes =
      static_cast<std::int32_t>(
          execution.initialization_large_root_workspace_mebibytes);
  effective.cpu_threads = static_cast<std::int32_t>(
      requested.effective_cpu_threads);
  effective.base_list_threads = static_cast<std::int32_t>(
      requested.effective_base_list_threads);
  effective.root_moment_partitions =
      requested.effective_root_moment_partitions;
  effective.rotation_moment_partitions =
      requested.effective_rotation_moment_partitions;
  effective.initialization_edge_moment_shards = static_cast<std::int32_t>(
      requested.effective_initialization_edge_moment_shards);
  effective.training_edge_moment_shards = static_cast<std::int32_t>(
      requested.effective_training_edge_moment_shards);
  effective.training_block_rows = static_cast<std::int32_t>(
      stlqpp::app::EffectiveTrainingBlockRows(requested));
  effective.base_block_rows = static_cast<std::int32_t>(
      stlqpp::app::EffectiveBaseBlockRows(requested));
  effective.cids_per_bucket = static_cast<std::int32_t>(
      stlqpp::app::EffectiveCidsPerBucket(requested));
  if (recipe != nullptr) {
    const auto& batching = recipe->base_topology.batching;
    effective.linkage_inner_batch_rows = static_cast<std::int32_t>(
        batching.inner_batch_rows);
    effective.linkage_many_nodes_preflush_target_pairs =
        static_cast<std::int32_t>(
            batching.many_nodes_preflush_target_pairs);
    effective.linkage_same_layer_block_rows = static_cast<std::int32_t>(
        batching.same_layer_block_rows);
    effective.linkage_same_layer_minimum_pairs =
        static_cast<std::int32_t>(
            batching.same_layer_minimum_batch_pairs);
    effective.linkage_same_layer_window_maximum_pairs =
        static_cast<std::int32_t>(
            batching.same_layer_window_maximum_pairs);
    effective.linkage_same_layer_maximum_pending_nodes =
        static_cast<std::int32_t>(
            batching.same_layer_maximum_pending_nodes);
    effective.linkage_same_layer_preflush_target_nodes =
        static_cast<std::int32_t>(
            batching.same_layer_preflush_target_nodes);
    effective.linkage_tiny_forced_cpu_enabled =
        batching.tiny_forced_cpu_enabled;
    effective.linkage_tiny_forced_pair_limit =
        static_cast<std::int32_t>(batching.tiny_forced_pair_limit);
    effective.linkage_tiny_forced_node_limit =
        static_cast<std::int32_t>(batching.tiny_forced_node_limit);
    effective.linkage_bad_window_rows = static_cast<std::int32_t>(
        batching.bad_frozen_window_rows);
    effective.linkage_bad_window_maximum_pairs =
        static_cast<std::int32_t>(
            batching.bad_frozen_maximum_pairs);
    effective.initialization_linkage_tiny_forced_cpu_enabled =
        recipe->initialization_topology.batching.tiny_forced_cpu_enabled;
    effective.initialization_samples =
        recipe->root_initialization.sample_vectors;
    effective.root_initialization.maximum_iterations =
        static_cast<std::int32_t>(
            recipe->root_initialization.maximum_iterations);
    effective.root_initialization.tolerance =
        recipe->root_initialization.tolerance;
    effective.root_initialization.kmeans_parallel_rounds =
        static_cast<std::int32_t>(
            recipe->root_initialization.kmeans_parallel_rounds);
    effective.root_initialization.kmeans_parallel_oversample =
        recipe->root_initialization.kmeans_parallel_oversample;
    effective.root_initialization.candidate_cap =
        static_cast<std::int32_t>(recipe->root_initialization.candidate_cap);
    effective.root_initialization.initial_weight =
        recipe->root_initialization.initial_weight;
    effective.root_initialization.minimum_weight =
        recipe->root_initialization.minimum_weight;
    effective.root_initialization.outlier_quantile =
        recipe->root_initialization.outlier_quantile;
    effective.root_initialization.cost_threshold =
        recipe->root_initialization.cost_threshold;
    effective.root_initialization.annealing_factor =
        recipe->root_initialization.annealing_factor;
    effective.root_initialization.warmup_iterations =
        static_cast<std::int32_t>(
            recipe->root_initialization.warmup_iterations);
    effective.root_initialization.quantile_bins =
        static_cast<std::int32_t>(recipe->root_initialization.quantile_bins);
    effective.root_initialization.large_layer_threshold =
        static_cast<std::int32_t>(
            recipe->root_initialization.large_layer_threshold);
    effective.root_initialization.large_layer_top_coarse =
        static_cast<std::int32_t>(
            recipe->root_initialization.large_layer_top_coarse);
    effective.root_initialization.large_layer_hierarchy_enabled =
        recipe->root_initialization.large_layer_hierarchy_enabled;
    effective.root_initialization.large_layer_fixed_coarse_count =
        static_cast<std::int32_t>(
            recipe->root_initialization.large_layer_fixed_coarse_count);
    effective.root_initialization.large_layer_minimum_coarse_count =
        static_cast<std::int32_t>(
            recipe->root_initialization.large_layer_minimum_coarse_count);
    effective.root_initialization.large_layer_maximum_coarse_count =
        static_cast<std::int32_t>(
            recipe->root_initialization.large_layer_maximum_coarse_count);
    effective.root_initialization.large_layer_round_coarse_to_power_of_two =
        recipe->root_initialization.large_layer_round_coarse_to_power_of_two;
  }
  return effective;
}

Status WriteEffectiveConfig(
    const BuildCommand& command,
    const EffectiveBuildRecipe* recipe,
    const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return Status::Internal(
        "failed to create effective build config: " + path.string());
  }
  stlqpp::app::WriteRoundTripBuildConfig(
      MakeEffectiveCommand(command, recipe), output);
  output.close();
  return output.fail()
      ? Status::Internal(
            "failed to close effective build config: " + path.string())
      : Status::Ok();
}

Status FreezeOriginalConfigs(
    const BuildCommand& command,
    const std::filesystem::path& run_root,
    const std::string& run_id) {
  if (command.config_files.size() != command.frozen_config_inputs.size()) {
    return Status::FailedPrecondition(
        "build config provenance was not frozen by the production parser");
  }
  for (std::size_t index = 0U;
       index < command.frozen_config_inputs.size(); ++index) {
    const auto& frozen = command.frozen_config_inputs[index];
    if (frozen.path != command.config_files[index]) {
      return Status::FailedPrecondition(
          "frozen build config order disagrees with parser provenance");
    }
    std::ostringstream prefix;
    prefix << "original_config_" << run_id << '_' << std::setw(2)
           << std::setfill('0') << index + 1U << '_';
    const std::filesystem::path destination =
        run_root / (prefix.str() + frozen.path.filename().string());
    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    if (!output) {
      return Status::Internal(
          "failed to create frozen original config " +
          destination.string());
    }
    output.write(frozen.bytes.data(),
                 static_cast<std::streamsize>(frozen.bytes.size()));
    output.close();
    if (output.fail()) {
      return Status::Internal(
          "failed to write frozen original config " +
          destination.string());
    }
  }
  return Status::Ok();
}

Status TransferArchivedFile(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) {
  std::error_code error;
  std::filesystem::rename(source, destination, error);
  if (error == std::errc::cross_device_link) {
    error.clear();
    std::filesystem::copy_file(
        source, destination,
        std::filesystem::copy_options::overwrite_existing, error);
    if (!error) {
      std::error_code remove_error;
      if (!std::filesystem::remove(source, remove_error) || remove_error) {
        return Status::Internal(
            "cannot remove transferred config: " + source.string());
      }
    }
  }
  return error
      ? Status::Internal("cannot transfer archived config: " +
                         source.string() + ": " + error.message())
      : Status::Ok();
}

Status FreezeControlEvidence(
    const BuildCommand& command,
    const std::filesystem::path& run_root,
    const std::string& run_id) {
  if (command.control_evidence.empty()) return Status::Ok();
  std::ifstream input(command.control_evidence, std::ios::binary);
  if (!input) {
    return Status::FailedPrecondition(
        "cannot open guarded control evidence " +
        command.control_evidence.string());
  }
  const std::string bytes{
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>()};
  if (input.bad() || bytes.empty()) {
    return Status::FailedPrecondition(
        "cannot read guarded control evidence " +
        command.control_evidence.string());
  }
  const std::filesystem::path destination =
      run_root / ("control_evidence_" + run_id + ".tsv");
  std::ofstream output(destination, std::ios::binary | std::ios::trunc);
  if (!output) {
    return Status::Internal(
        "cannot freeze guarded control evidence " + destination.string());
  }
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  output.close();
  return output.fail()
      ? Status::Internal(
            "cannot close guarded control evidence " + destination.string())
      : Status::Ok();
}

void WritePlannedResources(
    const BuildResourcePlan& resources,
    const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    stlqpp::LogWarn("failed to create planned resource record: " +
                    path.string());
    return;
  }
  output << "record.kind = planned\n"
         << "basic.cuda_instances = " << resources.basic_cuda_instances
         << '\n'
         << "basic.cpu_instances = " << resources.basic_cpu_instances
         << '\n'
         << "initialization.scheduler_backends = "
         << resources.initialization_scheduler_backends << '\n'
         << "initialization.direct_backends = "
         << resources.initialization_direct_backends << '\n'
         << "initialization.cpu_direct_backends = "
         << resources.initialization_cpu_direct_backends << '\n'
         << "ordinary.scheduler_backends = "
         << resources.ordinary_scheduler_backends << '\n'
         << "ordinary.direct_lanes = "
         << resources.ordinary_direct_backends << '\n'
         << "ordinary.cpu_direct_lanes = "
         << resources.ordinary_cpu_direct_backends << '\n'
         << "replay.cuda_instances = " << resources.replay_cuda_instances
         << '\n'
         << "replay.cpu_instances = " << resources.replay_cpu_instances
         << '\n';
}

void WriteObservedResource(
    std::ostream& output,
    const char* stage,
    const stlqpp::LinkageResourceStatistics& resources) {
  output << stage << ".scheduler_backends = "
         << resources.scheduler_backend_count << '\n'
         << stage << ".direct_backends = "
         << resources.direct_backend_count << '\n'
         << stage << ".ordinary_direct_lanes = "
         << resources.ordinary_direct_lane_count << '\n'
         << stage << ".fixed_workspace_bytes = "
         << resources.fixed_workspace_resident_bytes << '\n'
         << stage << ".deferred_workspace_bytes = "
         << resources.deferred_workspace_reserve_bytes << '\n'
         << stage << ".live_registration_capacity_bytes = "
         << resources.live_registration_capacity_bytes << '\n'
         << stage << ".live_registration_peak_bytes = "
         << resources.live_registration_peak_bytes << '\n'
         << stage << ".evaluation_source_tasks = "
         << resources.evaluation_source_tasks << '\n'
         << stage << ".evaluation_transport_batches = "
         << resources.evaluation_transport_batches << '\n'
         << stage << ".evaluation_cross_cluster_transport_batches = "
         << resources.evaluation_cross_cluster_transport_batches << '\n'
         << stage << ".evaluation_numerical_batches = "
         << resources.evaluation_numerical_batches << '\n'
         << stage << ".evaluation_cross_cluster_numerical_batches = "
         << resources.evaluation_cross_cluster_numerical_batches << '\n'
         << stage << ".merged_numerical_source_tasks = "
         << resources.merged_numerical_source_tasks << '\n'
         << stage << ".maximum_source_tasks_per_transport_batch = "
         << resources.maximum_source_tasks_per_transport_batch << '\n'
         << stage << ".maximum_source_tasks_per_numerical_batch = "
         << resources.maximum_source_tasks_per_numerical_batch << '\n';
}

Status WriteObservedResources(
    const BuildExecutionReport& report,
    const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return Status::Internal(
        "failed to create observed resource record: " + path.string());
  }
  output << "record.kind = constructed-observed\n";
  if (report.initialization_resources.has_value()) {
    WriteObservedResource(
        output, "initialization", *report.initialization_resources);
  }
  for (const auto& round : report.training_rounds) {
    if (round.linkage_resources.has_value()) {
      const std::string stage =
          "training_round_" + std::to_string(round.round);
      WriteObservedResource(
          output, stage.c_str(), *round.linkage_resources);
    }
  }
  if (report.final_resources.has_value()) {
    WriteObservedResource(output, "final", *report.final_resources);
  }
  output.close();
  return output.fail()
      ? Status::Internal(
            "failed to close observed resource record: " + path.string())
      : Status::Ok();
}

Status ValidateResource(
    const stlqpp::LinkageResourceStatistics& observed,
    std::uint32_t scheduler_backends,
    std::uint32_t direct_backends,
    std::uint32_t ordinary_direct_lanes,
    const char* stage) {
  if (observed.scheduler_backend_count != scheduler_backends ||
      observed.direct_backend_count != direct_backends ||
      observed.ordinary_direct_lane_count != ordinary_direct_lanes) {
    return Status::FailedPrecondition(
        std::string("constructed linkage resources disagree with plan for ") +
        stage);
  }
  return Status::Ok();
}

Status ValidateObservedResources(
    const BuildExecutionReport& report,
    const BuildResourcePlan& plan) {
  const std::uint32_t initialization_direct_backends =
      plan.initialization_direct_backends +
      plan.initialization_cpu_direct_backends;
  const std::uint32_t ordinary_direct_backends =
      plan.ordinary_direct_backends +
      plan.ordinary_cpu_direct_backends;
  if (report.initialization_resources.has_value()) {
    if (const Status status = ValidateResource(
            *report.initialization_resources,
            plan.initialization_scheduler_backends,
            // Fast init borrows ordinary root lanes; scheduler-backed
            // candidate evaluation does not also create direct evaluators.
            report.fast_initialization &&
                    plan.initialization_scheduler_backends != 0U
                ? 0U : initialization_direct_backends,
            report.fast_initialization ? initialization_direct_backends : 0U,
            "initialization");
        !status.ok()) {
      return status;
    }
  }
  for (const auto& round : report.training_rounds) {
    if (!round.linkage_resources.has_value()) {
      continue;
    }
    if (const Status status = ValidateResource(
            *round.linkage_resources,
            plan.ordinary_scheduler_backends,
            plan.ordinary_scheduler_backends == 0U
                ? ordinary_direct_backends : 0U,
            ordinary_direct_backends, "training");
        !status.ok()) {
      return status;
    }
  }
  if (report.final_resources.has_value()) {
    return ValidateResource(
        *report.final_resources, plan.ordinary_scheduler_backends,
        plan.ordinary_scheduler_backends == 0U
            ? ordinary_direct_backends : 0U,
        ordinary_direct_backends, "final");
  }
  return Status::Ok();
}

}  // namespace

namespace stlqpp::app {

BuildRunSession::BuildRunSession(
    ApplicationRunSession owner,
    BuildResourcePlan planned_resources,
    bool active) noexcept
    : owner_(std::move(owner)), planned_resources_(planned_resources),
      active_(active) {}

StatusOr<BuildRunSession> BuildRunSession::StartImpl(
  const BuildCommand& command,
  const EffectiveBuildRecipe* recipe,
  const std::filesystem::path& provisional_root) {
  if (!command.control_evidence.empty()) {
    if (recipe == nullptr) {
      return Status::FailedPrecondition(
          "guarded build requires an effective recipe before startup");
    }
    if (const Status status = ValidateControlEvidence(command, *recipe);
        !status.ok()) {
      return status;
    }
  }
  auto owner = ApplicationRunSession::Start({
      .durable_root = provisional_root.empty()
          ? command.output_directory : provisional_root,
      .archive_log = command.archive_log,
      .require_existing_root = false,
  });
  if (!owner.ok()) return owner.status();
  const std::filesystem::path run_root = owner.value().root();
  const std::string run_id = owner.value().run_id();

  const BuildResourcePlan resources = ResolveBuildResourcePlan(command);
  BuildRunSession session(
      std::move(owner).value(), resources, true);
  session.deferred_archive_ = !provisional_root.empty();
  if (!provisional_root.empty()) {
    if (const Status status = WriteEffectiveConfig(
            command, recipe, run_root / "config_snapshot.txt");
        !status.ok()) return status;
    if (const Status status = FreezeOriginalConfigs(
            command, run_root, run_id); !status.ok()) return status;
    if (const Status status = session.WriteState("running", "");
        !status.ok()) return status;
    if (session.owner_.log_attached()) {
      LogInfo("Training log (pending artifact identity): " +
              session.owner_.log_path().string());
    }
    if (command.protect_existing_outputs) {
      LogInfo("Protected linkage target will be checked as soon as the "
              "trained model hash is known, before base encoding.");
    }
    return session;
  }
  const std::filesystem::path requested =
      run_root / ("requested_config_" + run_id + ".cfg");
  const std::filesystem::path effective =
      run_root / ("effective_config_" + run_id + ".cfg");
  const std::filesystem::path snapshot = run_root / "config_snapshot.txt";
  for (const Status& status : {
           WriteRequestedConfig(command, requested),
           WriteEffectiveConfig(command, recipe, effective),
           WriteEffectiveConfig(command, recipe, snapshot),
           FreezeOriginalConfigs(command, run_root, run_id)}) {
    if (!status.ok()) {
      LogWarn(status.message());
    }
  }
  if (const Status status =
          FreezeControlEvidence(command, run_root, run_id);
      !status.ok()) {
    return status;
  }
  WritePlannedResources(
      resources, run_root / ("planned_resources_" + run_id + ".txt"));
  if (const Status status = session.WriteState("running", ""); !status.ok()) {
    LogWarn(status.message());
  }
  if (session.owner_.log_attached()) {
    LogInfo("Archiving runtime log to: " +
            session.owner_.log_path().string());
  }
  LogInfo("Requested build config: " + requested.string());
  LogInfo("Effective build config: " + effective.string());
  if (recipe != nullptr) {
    LogInfo(
        "CPU workload budget: root_accumulators=" +
        std::to_string(recipe->update_reduction.root_moment_partitions) +
        " root_spd_threads=" +
        std::to_string(recipe->update_reduction.root_spd_threads) +
        " edge_spd_threads=" +
        std::to_string(recipe->update_reduction.edge_spd_threads));
  }
  if (command.use_cuda) {
    LogInfo(
        "CUDA phase plan: basic_instances=1, init_linkage=direct" +
        std::string(", scheduler_stable=") +
        (command.linkage_scheduler_stable ? "true" : "false") +
        ", init_contexts=" + std::to_string(
            resources.initialization_scheduler_backends +
            resources.initialization_direct_backends) +
        ", ordinary_direct_contexts=" +
        std::to_string(resources.ordinary_direct_backends) +
        ", ordinary_scheduler_contexts=" +
        std::to_string(resources.ordinary_scheduler_backends));
  } else {
    LogInfo(
        "CPU phase plan: omp_threads=" +
        std::to_string(command.effective_cpu_threads) +
        ", basic_threads=" +
        std::to_string(command.effective_cpu_threads) +
        ", base_list_threads=" +
        std::to_string(command.effective_base_list_threads) +
        ", init_linkage_lanes=" +
        std::to_string(resources.initialization_cpu_direct_backends) +
        ", ordinary_linkage_lanes=" +
        std::to_string(resources.ordinary_cpu_direct_backends));
  }
  return session;
}

StatusOr<BuildRunSession> BuildRunSession::Start(
    const BuildCommand& command) {
  return StartImpl(command, nullptr);
}

StatusOr<BuildRunSession> BuildRunSession::StartProvisional(
    const BuildCommand& command,
    const EffectiveBuildRecipe& recipe,
    const std::filesystem::path& provisional_root) {
  return StartImpl(command, &recipe, provisional_root);
}

Status BuildRunSession::AttachArtifactRoot(
    const BuildCommand& command,
    const EffectiveBuildRecipe& recipe,
    const BuildExecutionReport& report) {
  (void)recipe;
  if (!deferred_archive_ || report.run_root.empty()) {
    return Status::FailedPrecondition(
        "artifact root is unavailable for build archive");
  }
  const std::filesystem::path provisional_root = owner_.root();
  const std::string provisional_id = owner_.run_id();
  if (const Status status = owner_.MoveTo(report.run_root); !status.ok()) {
    return status;
  }
  if (const Status status = TransferArchivedFile(
          provisional_root / "config_snapshot.txt",
          owner_.root() / "config_snapshot.txt"); !status.ok()) return status;
  if (command.config_files.size() != command.frozen_config_inputs.size()) {
    return Status::FailedPrecondition("build config provenance is incomplete");
  }
  for (std::size_t i = 0; i < command.frozen_config_inputs.size(); ++i) {
    const auto& input = command.frozen_config_inputs[i];
    std::filesystem::path basename = input.path.filename();
    if (basename.empty()) basename = "config.cfg";
    const auto archived_name = [&](const std::string& id) {
      std::ostringstream prefix;
      prefix << "original_config_" << id << '_' << std::setw(2)
             << std::setfill('0') << i + 1U << '_';
      return prefix.str() + basename.string();
    };
    std::filesystem::path name = basename;
    name = "original_" + name.string();
    if (command.frozen_config_inputs.size() > 1U) {
      name = std::to_string(i + 1U) + "_" + name.string();
    }
    if (const Status status = TransferArchivedFile(
            provisional_root / archived_name(provisional_id),
            owner_.root() / name);
        !status.ok()) return status;
  }
  if (const Status status = FreezeControlEvidence(
          command, owner_.root(), owner_.run_id()); !status.ok()) {
    return status;
  }
  std::ofstream hashes(owner_.root() / "store_hashes.txt",
                       std::ios::binary | std::ios::trunc);
  if (!hashes) return Status::Internal("cannot create store hash archive");
  hashes << "base_basic = " << report.base_basic_hash.ToHex() << '\n'
         << "base_list = " << report.base_list_hash.ToHex() << '\n'
         << "linkage_list = " << report.linkage_hash.ToHex() << '\n';
  hashes.close();
  if (hashes.fail()) return Status::Internal("cannot close store hash archive");
  std::error_code error;
  std::filesystem::remove(provisional_root / "run_state.txt", error);
  return WriteState("running", "");
}

StatusOr<BuildRunSession> BuildRunSession::Start(
    const BuildCommand& command,
    const EffectiveBuildRecipe& recipe) {
  return StartImpl(command, &recipe);
}

Status BuildRunSession::WriteState(
    const char* state, const std::string& detail) {
  const std::filesystem::path path = owner_.root() / "run_state.txt";
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return Status::Internal("failed to create build run state");
  }
  output << "state = " << state << '\n'
         << "log = " << owner_.log_path().string() << '\n';
  if (!detail.empty()) {
    output << "detail = " << detail << '\n';
  }
  output.close();
  return output.fail()
      ? Status::Internal("failed to close build run state")
      : Status::Ok();
}

Status BuildRunSession::Complete(const BuildExecutionReport& report) {
  if (!deferred_archive_) {
    const std::filesystem::path observed =
        owner_.root() /
        ("observed_resources_" + owner_.run_id() + ".txt");
    if (const Status status = WriteObservedResources(report, observed);
        !status.ok()) {
      LogWarn(status.message());
    }
  }
  if (const Status status =
          ValidateObservedResources(report, planned_resources_);
      !status.ok()) {
    if (const Status state = WriteState("failed", status.message());
        !state.ok()) {
      LogWarn(state.message());
    }
    owner_.Close();
    return status;
  }
  const std::filesystem::path state_path = owner_.root() / "run_state.txt";
  std::ofstream state(state_path, std::ios::binary | std::ios::trunc);
  if (!state) {
    return Status::Internal("failed to create completed build run state");
  }
  state << "state = complete\n"
        << "log = " << owner_.log_path().string() << '\n'
        << "index = " << report.index_directory.string() << '\n'
        << "base_list = " << report.base_list_directory.string() << '\n'
        << "work = " << report.work_root.string() << '\n'
        << "trained_model = "
        << report.trained_model_artifact.string() << '\n'
        << "executed_rounds = " << report.training_rounds.size() << '\n';
  if (!report.training_rounds.empty()) {
    state << "final_round = " << report.training_rounds.back().round << '\n';
  }
  state.close();
  const Status result = state.fail()
      ? Status::Internal("failed to close completed build run state")
      : Status::Ok();
  owner_.Close();
  return result;
}

Status BuildRunSession::Fail(const Status& failure) {
  const Status result = WriteState("failed", failure.message());
  owner_.Close();
  return result;
}

void LogCompletedIndexBuild(const BuildExecutionReport& report) {
  // Final linkage logs its path at the stage owner, in DEV order.  The
  // completion hook remains for application-level vocabulary that may be
  // added independently of stage output.
  (void)report;
}

}  // namespace stlqpp::app
