#include "run_index_build.h"

#include "build_output_preparation.h"
#include "stlqpp/application/cpu_index_build_runtime.h"
#include "stlqpp/application/source_norm_provider.h"
#if STLQPP_HAS_CUDA_RUNTIME
#include "stlqpp/application/cuda_index_build_runtime.h"
#endif
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "stlqpp/core/artifact_hash.h"
#include "stlqpp/index/emit/base_list_emitter.h"
#include "stlqpp/model/model_dispatch.h"
#include "stlqpp/runtime/cpu/cblas_f32_gemm.h"
#include "stlqpp/runtime/cpu/openmp_parallel_for.h"
#include "stlqpp/runtime/telemetry/logger.h"
#include "stlqpp/storage/artifact_hash_factory.h"
#include "stlqpp/storage/canonical_index_files.h"
#include "stlqpp/storage/trained_model_file.h"
#include "stlqpp/training/checkpoint/checkpoint_path.h"
#include "stlqpp/training/checkpoint/runtime_model_checkpoint.h"

namespace stlqpp::app {
namespace {

#if STLQPP_HAS_CUDA_RUNTIME
constexpr std::uint64_t kMebibyte = 1024ULL * 1024ULL;

std::uint64_t Mebibytes(std::uint32_t value) noexcept {
  return static_cast<std::uint64_t>(value) * kMebibyte;
}
#endif

std::uint64_t BlockCount(std::uint64_t rows,
                         std::uint32_t block_rows) noexcept {
  return rows / block_rows +
         static_cast<std::uint64_t>(rows % block_rows != 0U);
}

BasicSchedule TrainingBasicSchedule(const BuildCommand& command) noexcept {
  return BasicSchedule::Configured(
      BasicScoreRule::kSigned, command.train_basic_icm_iterations,
      command.train_basic_ils_iterations, command.train_basic_perturbations,
      command.least_squares_cost_mode);
}

BasicSchedule TrainingVirtualRootSchedule(
    const BuildCommand& command) noexcept {
  return BasicSchedule::Configured(
      BasicScoreRule::kAbsolute, command.train_basic_icm_iterations,
      command.train_basic_ils_iterations, command.train_basic_perturbations,
      command.least_squares_cost_mode);
}

BasicSchedule BaseBasicSchedule(const BuildCommand& command) noexcept {
  return BasicSchedule::Configured(
      BasicScoreRule::kAbsolute, command.base_basic_icm_iterations,
      command.base_basic_ils_iterations, command.base_basic_perturbations,
      command.least_squares_cost_mode);
}

void RemoveAllBestEffort(const std::filesystem::path& path, const char* role) {
  std::error_code error;
  if (!std::filesystem::exists(path, error)) return;
  if (error) {
    LogWarn("AutoCleanup: failed to inspect " + path.string() + " : " +
            error.message());
    return;
  }
  std::filesystem::remove_all(path, error);
  if (error) {
    LogWarn("AutoCleanup: failed to remove " + path.string() + " : " +
            error.message());
    return;
  }
  LogInfo("Cleanup: deleted " + std::string(role) + " " + path.string());
}

Status PersistBuildOriginalNorms(const BuildCommand& command,
                                const std::filesystem::path& index) {
  const bool scaled = std::any_of(
      command.eval_representations.begin(), command.eval_representations.end(),
      [](EvalRepresentation mode) {
        return mode == EvalRepresentation::kFloatScaled ||
               mode == EvalRepresentation::kQuantizedScaled;
      });
  if (index.empty() || !scaled) return Status::Ok();
  auto files = CanonicalIndexFiles::OpenCoefficients(
      index, !command.coefficient_codec_enabled);
  if (!files.ok()) return files.status();
  auto parallel = OpenMpParallelFor::Create(command.effective_cpu_threads);
  if (!parallel.ok()) return parallel.status();
  const char* format = "auto";
  switch (command.effective_base_format) {
    case SourceFormat::kAuto: break;
    case SourceFormat::kBvecsU8: format = "bvecs"; break;
    case SourceFormat::kFvecsF32: format = "fvecs"; break;
    case SourceFormat::kFbinF32: format = "fbin"; break;
    case SourceFormat::kNpyFloat16: format = "npy"; break;
  }
  auto norms = SourceNormProvider::OpenForIndex(
      index, files.value().manifest(),
      command.effective_base_file.empty() ? command.base_file
                                          : command.effective_base_file,
      format, command.base_first_row, command.base_rows, parallel.value());
  return norms.ok() ? Status::Ok() : norms.status();
}

template <int M>
void ApplySuccessfulBuildCleanup(const BuildCommand& command,
                                 const IndexBuildResult<M>& built) {
  if (!command.cleanup.enabled) return;
  const ResolvedCleanupPolicy policy = ResolveCleanupPolicy(command.cleanup);

  // PP's bucket store is the transient/checkpointed equivalent of DEV's
  // base_basic payload. Stable-ID/CID and IVF order survive independently in
  // the list store, so deleting this directory never removes those roles.
  if (built.stages.encode_base && !policy.keep_base_basic) {
    RemoveAllBestEffort(built.resolved_paths.has_value()
                            ? built.resolved_paths->work_root / "base-buckets"
                            : command.work_directory / "base-buckets",
                        "base_basic directory");
  }

  const std::filesystem::path base_lists =
      built.base_basic.list_directory.empty()
          ? command.output_directory / "base-lists"
          : built.base_basic.list_directory;
  // An existing io.base_file is an input, even when it is physically inside
  // an output tree. Only this invocation's base encoder may authorize cleanup.
  if (built.stages.build_linkage && built.owns_base_list) {
    // In PP, one canonical list store jointly owns DEV's base_list plus the
    // replaced cluster-id/IVF-CSR roles. It is safe to remove only when no
    // retained role needs it; otherwise retain the shared representation.
    const bool retain_list_store =
        policy.keep_base_basic || policy.keep_base_list ||
        policy.keep_base_basic_cluster_id || policy.keep_base_basic_ivf_lists;
    if (!retain_list_store) {
      RemoveAllBestEffort(base_lists, "base_list directory");
    } else if (!policy.keep_base_list_raw) {
      const Status status = BaseListStore<M>::DiscardRawPayload(base_lists);
      if (!status.ok()) {
        LogWarn("AutoCleanup: failed to remove base_list raw payload in " +
                base_lists.string() + " : " + status.message());
      } else {
        LogInfo("Cleanup: deleted base_list raw vector payload in " +
                base_lists.string());
      }
    }
  } else if (built.owns_base_list && !policy.keep_base_list_raw) {
    // Without final linkage, raw vectors have no later in-run consumer.
    const Status status = BaseListStore<M>::DiscardRawPayload(base_lists);
    if (!status.ok()) {
      LogWarn("AutoCleanup: failed to remove base_list raw payload in " +
              base_lists.string() + " : " + status.message());
    } else {
      LogInfo("Cleanup: deleted base_list raw vector payload in " +
              base_lists.string());
    }
  }

  if (built.stages.initialize_training || built.stages.train) {
    // Round-local training-basic worksets have scoped lifetimes and normally
    // remove themselves. These exact paths cover interrupted or retained
    // initialization work only after the complete requested build succeeded.
    RemoveAllBestEffort(command.work_directory / "training-basic",
                        "train_basic directory");
    RemoveAllBestEffort(command.work_directory / "initial-buckets",
                        "train initial buckets");
    RemoveAllBestEffort(command.work_directory / "initial-lists",
                        "train initial lists");
  }
}

#if STLQPP_HAS_CUDA_RUNTIME
CudaLinkageSchedulerFactoryOptions MakeLinkageResources(
    const BuildCommand& command, const IndexBuildExecutionControls& controls,
    std::uint32_t phase_direct_pool_size,
    std::uint32_t direct_maximum_task_pairs,
    std::uint32_t direct_admitted_maximum_task_pairs,
    std::uint32_t direct_async_pinned_mebibytes) {
  CudaLinkageSchedulerFactoryOptions output;
  output.device = command.cuda_device;
  output.allow_tf32 = command.cuda_allow_tf32;
  output.scheduler_enabled =
      controls.scheduler_enabled.value_or(command.linkage_scheduler_enabled);
  output.transport_batching_enabled =
      controls.linkage_scheduler_transport_batching_enabled.value_or(
          command.linkage_scheduler_transport_batching_enabled);
  output.stable = controls.linkage_scheduler_stable.value_or(
      command.linkage_scheduler_stable);
  output.least_squares_cost_mode = command.least_squares_cost_mode;
  output.worker_count =
      EffectiveLinkageSchedulerWorkers(command, phase_direct_pool_size);
  output.direct_worker_count = phase_direct_pool_size;
  output.direct_maximum_task_pairs = direct_maximum_task_pairs;
  output.direct_admitted_maximum_task_pairs =
      direct_admitted_maximum_task_pairs;
  output.direct_async_pinned_mebibytes = direct_async_pinned_mebibytes;
  output.internal_chunk_max_pairs = command.linkage_chunk_max_pairs;
  output.numerical_workspace_budget_mebibytes =
      command.linkage_large_root_workspace_mebibytes;
  output.queue_capacity = command.linkage_scheduler_queue_capacity;
  output.worker_rfull_cache_limits = {
      .maximum_buffers = command.linkage_rfull_cache_buffers,
      .maximum_bytes = Mebibytes(command.linkage_rfull_cache_mebibytes),
  };
  output.minimum_safety_reserve_bytes =
      Mebibytes(command.linkage_safety_reserve_mebibytes);
  return output;
}

CudaLinkageSchedulerFactoryOptions MakeInitializationLinkageResources(
    const BuildCommand& command, const IndexBuildExecutionControls& controls,
    std::uint32_t phase_direct_pool_size,
    std::uint32_t direct_maximum_task_pairs,
    std::uint32_t direct_admitted_maximum_task_pairs,
    std::uint32_t direct_async_pinned_mebibytes) {
  CudaLinkageSchedulerFactoryOptions output = MakeLinkageResources(
      command, controls, phase_direct_pool_size, direct_maximum_task_pairs,
      direct_admitted_maximum_task_pairs, direct_async_pinned_mebibytes);
  output.internal_chunk_max_pairs = command.init_linkage_chunk_max_pairs == 0U
                                        ? command.linkage_chunk_max_pairs
                                        : command.init_linkage_chunk_max_pairs;
  return output;
}
#endif

IndexSourceSpec MakeIndexSourceSpec(const std::filesystem::path& path,
                                    SourceFormat format,
                                    std::uint32_t dimension,
                                    std::uint64_t first_row,
                                    std::uint64_t row_count,
                                    std::uint32_t block_rows) {
  if (format == SourceFormat::kBvecsU8)
    return BvecsU8SourceSpec{path, dimension, first_row, row_count, block_rows};
  const F32SourceLayout layout =
      format == SourceFormat::kNpyFloat16 ? F32SourceLayout::kNpyFloat16
      : format == SourceFormat::kFbinF32  ? F32SourceLayout::kFbin
                                          : F32SourceLayout::kFvecs;
  return FvecsF32SourceSpec{path,      dimension,  first_row,
                            row_count, block_rows, layout};
}

// One complete product recipe is prepared before numerical M is selected.
// The callback borrows RunIndexBuild's command/recipe, which outlive execution.
struct PreparedIndexBuild final : IndexBuildOptions {
  std::optional<TrainingInitializationOptions> initialization;
  CpuIndexBuildRuntimeOptions cpu;
#if STLQPP_HAS_CUDA_RUNTIME
  CudaIndexBuildRuntimeOptions cuda;
#endif
};

StatusOr<PreparedIndexBuild> PrepareIndexBuild(
    const BuildCommand& command, EffectiveBuildRecipe& recipe,
    const IndexBuildExecutionControls& controls,
    const std::filesystem::path& namespace_work_base) {
  PreparedIndexBuild prepared;
  const bool effective_scheduler_enabled =
      controls.scheduler_enabled.value_or(command.linkage_scheduler_enabled);
  recipe.training_topology.batching.single_gpu_scheduler_backed =
      effective_scheduler_enabled;
  recipe.base_topology.batching.single_gpu_scheduler_backed =
      effective_scheduler_enabled;
  const bool effective_transport_batching =
      controls.linkage_scheduler_transport_batching_enabled.value_or(
          command.linkage_scheduler_transport_batching_enabled);
  const bool effective_stable = controls.linkage_scheduler_stable.value_or(
      command.linkage_scheduler_stable);
  const LinkageNumericalPolicy linkage_numerical_policy =
      command.use_cuda && effective_scheduler_enabled &&
              effective_transport_batching && !effective_stable
          ? LinkageNumericalPolicy::kMergedThroughput
          : LinkageNumericalPolicy::kSourceTask;
#if STLQPP_HAS_CUDA_RUNTIME
  const bool ordinary_training_basic_runs =
      command.train_enabled && recipe.additional_training_round_count != 0U;
  const EffectiveBuildExecutionConfig execution =
      ResolveBuildExecutionConfig(command);
#endif
  const BuildResourcePlan resource_plan =
      ResolveBuildResourcePlan(command, controls.scheduler_enabled);
  const std::filesystem::path& training_file =
      command.effective_training_file.empty() ? command.training_file
                                              : command.effective_training_file;
  const std::filesystem::path& base_file = command.effective_base_file.empty()
                                               ? command.base_file
                                               : command.effective_base_file;
  const std::uint32_t training_block_rows = EffectiveTrainingBlockRows(command);
  const std::uint32_t base_block_rows = EffectiveBaseBlockRows(command);
  const std::uint32_t cids_per_bucket = EffectiveCidsPerBucket(command);
  const IndexSourceSpec training_source = MakeIndexSourceSpec(
      training_file, command.effective_training_format, command.dimension,
      command.training_first_row, command.training_rows, training_block_rows);
  const IndexSourceSpec base_source = MakeIndexSourceSpec(
      base_file, command.effective_base_format, command.dimension,
      command.base_first_row, command.base_rows, base_block_rows);
  const std::uint32_t training_scalar_bytes =
      std::holds_alternative<BvecsU8SourceSpec>(training_source)
          ? sizeof(std::uint8_t)
          : sizeof(float);
  const std::uint32_t base_scalar_bytes =
      std::holds_alternative<BvecsU8SourceSpec>(base_source)
          ? sizeof(std::uint8_t)
          : sizeof(float);
#if STLQPP_HAS_CUDA_RUNTIME
  auto& runtime_options = prepared.cuda;
  runtime_options.create_training_initialization =
      command.train_enabled && command.train_initialization_enabled &&
      command.resume_round_checkpoint.empty();
  runtime_options.create_fast_initialization =
      runtime_options.create_training_initialization &&
      command.resume_initialized_checkpoint.empty() &&
      command.init_linkage_mode == "fast_init";
  runtime_options.create_root_initialization =
      runtime_options.create_training_initialization &&
      command.resume_initialized_checkpoint.empty();
  runtime_options.create_training = ordinary_training_basic_runs;
  runtime_options.create_base_encode = command.base_encode_enabled;
  runtime_options.create_final_linkage = command.base_linkage_enabled;
  runtime_options.training_basic_rsd_rank = command.training_basic_rsd_rank;
  runtime_options.base_basic_rsd_rank = command.base_basic_rsd_rank;
  runtime_options.initialization_source = training_source;
  runtime_options.training_source = training_source;
  runtime_options.base_source = base_source;
  runtime_options.cpu_worker_count = command.effective_cpu_threads;
  runtime_options.base_list_worker_count = command.effective_base_list_threads;
  runtime_options.root_moment_partition_count =
      recipe.update_reduction.root_moment_partitions;
  runtime_options.rotation_moment_partition_count =
      recipe.update_reduction.rotation_moment_partitions;
  runtime_options.cuda_device = command.cuda_device;
  runtime_options.cuda_allow_tf32 = command.cuda_allow_tf32;
  runtime_options.root_initialization = {
      .seeding =
          {
              .maximum_score_bytes = static_cast<std::size_t>(
                  Mebibytes(command.root_score_buffer_mebibytes)),
          },
      .lloyd =
          {
              .maximum_score_bytes = static_cast<std::size_t>(
                  Mebibytes(command.root_score_buffer_mebibytes)),
              .maximum_normalized_cache_bytes = static_cast<std::size_t>(
                  Mebibytes(command.root_normalized_cache_mebibytes)),
              .collect_metrics = command.train_log_metrics,
          },
  };
  runtime_options.training_basic = {
      .artifact_root = command.work_directory / "training-basic",
      .cids_per_bucket = cids_per_bucket,
      .bucket_flush_bytes = EffectiveBucketFlushBytes(command),
      .source_rows = command.training_rows,
      .source_block_rows = training_block_rows,
      .cpu_threads = command.effective_cpu_threads,
      .schedule = TrainingBasicSchedule(command),
      .pipeline =
          {
              .maximum_inflight_blocks = ResolveBasicInflightBlockCount(
                  command, training_scalar_bytes, training_block_rows),
              .maximum_prefetched_blocks = ResolveBasicPrefetchBlockCount(
                  command, training_scalar_bytes, training_block_rows),
              .maximum_pending_writes = ResolveBasicPendingWriteCount(
                  command, training_scalar_bytes, training_block_rows),
              .progress_total_blocks =
                  BlockCount(command.training_rows, training_block_rows),
              .log_progress = true,
              .log_metrics = command.train_log_metrics,
          },
  };
  runtime_options.initialization_topology = {
      .icm_rounds = recipe.initialization_topology.icm_rounds,
      .outer_worker_count = command.effective_cpu_threads,
      .precompute_thread_count = command.effective_cpu_threads,
      .model =
          {
              .root_dot_workspace_bytes =
                  std::min(Mebibytes(recipe.initialization_workspace
                                         .large_root_xc0_chunk_mebibytes),
                           Mebibytes(recipe.initialization_workspace
                                         .large_root_workspace_mebibytes) /
                               16U),
              .full_root_dot_workspace_bytes =
                  Mebibytes(recipe.initialization_workspace
                                .large_root_xc0_chunk_mebibytes),
              .encoder_workspace_bytes =
                  Mebibytes(recipe.initialization_workspace
                                .large_root_workspace_mebibytes),
          },
      .linkage_scheduler = MakeInitializationLinkageResources(
          command, controls, execution.initialization_direct_pool_size,
          execution.initialization_maximum_batch_pairs,
          MaximumProducedEvaluationPairs(recipe.initialization_topology),
          execution.initialization_async_pinned_mebibytes),
  };
  runtime_options.training_topology =
      ResolveTrainingTopologyContextOptions(command, recipe, controls);
  runtime_options.fast_initialization_topology =
      runtime_options.training_topology;
  runtime_options.fast_initialization_topology.linkage_scheduler =
      runtime_options.initialization_topology.linkage_scheduler;
  runtime_options.fast_initialization_topology.ordinary_direct_lane_count =
      execution.initialization_direct_pool_size;
  runtime_options.final_topology =
      ResolveFinalTopologyContextOptions(command, recipe, controls);
#endif

  auto computed_identity = ComputeTrainingIdentity(
      command, recipe, effective_scheduler_enabled, linkage_numerical_policy);
  if (!computed_identity.ok()) return computed_identity.status();
  const TrainingCheckpointIdentity identity = computed_identity.value();
  const TrainingRoundSeeds repeated_train_seeds = {
      .basic = recipe.train_seed,
      .topology = recipe.training_topology.seed,
  };
  std::vector<TrainingRoundSeeds> round_seeds(
      recipe.additional_training_round_count, repeated_train_seeds);
  if (!command.train_enabled) round_seeds.clear();
  const std::filesystem::path formal_checkpoint_base =
      TrainingModelPublicationBase(command, identity.training_source,
                                   identity.training_recipe);
  if (command.train_enabled) {
    std::error_code error;
    std::filesystem::create_directories(formal_checkpoint_base.parent_path(),
                                        error);
    if (error) {
      return Status::Internal("cannot create trained-model directory: " +
                              error.message());
    }
  }
  auto& request = prepared;
  request.stages = {
      .initialize_training =
          command.train_enabled && command.train_initialization_enabled,
      .train = command.train_enabled,
      .encode_base = command.base_encode_enabled,
      .build_linkage = command.base_linkage_enabled,
      .reused_base_basic_directory = command.reused_base_list_directory,
  };
  if (command.train_enabled && command.train_initialization_enabled &&
      command.resume_round_checkpoint.empty()) {
    request.initialization.emplace();
    if (round_seeds.empty()) {
      auto checkpoint = NextNumberedTrainedModelPath(formal_checkpoint_base);
      if (!checkpoint.ok()) return checkpoint.status();
      request.initialization->checkpoint_directory =
          std::move(checkpoint).value();
      request.initialization->log_checkpoint_publication = true;
    } else {
      request.initialization->checkpoint_directory =
          command.work_directory / "initialized";
    }
    request.initialization->identity = identity;
    request.initialization->seed = command.train_seed;
    request.initialization->mode =
        command.init_linkage_mode == "fast_init"
            ? TrainingInitializationMode::kFastInit
            : TrainingInitializationMode::kLegacyRootOnly;
    request.initialization->virtual_enabled = command.virtual_roots.enabled;
    request.initialization->good_cluster_fraction =
        command.good_cluster_fraction;
    request.initialization->bad_cluster_training_weight =
        recipe.bad_cluster_training_weight;
    request.initialization->edge_moment_shard_count =
        recipe.update_reduction.initialization_edge_moment_shards;
    request.initialization->basic = {
        .bucket_directory = command.work_directory / "initial-buckets",
        .list_directory = command.work_directory / "initial-lists",
        .cids_per_bucket = cids_per_bucket,
        .bucket_flush_bytes = EffectiveBucketFlushBytes(command),
        .source_rows = command.training_rows,
        .source_block_rows = training_block_rows,
        .cpu_threads = command.effective_cpu_threads,
        .schedule = TrainingBasicSchedule(command),
        .pipeline =
            {
                .maximum_inflight_blocks = ResolveBasicInflightBlockCount(
                    command, training_scalar_bytes, training_block_rows),
                .maximum_prefetched_blocks = ResolveBasicPrefetchBlockCount(
                    command, training_scalar_bytes, training_block_rows),
                .maximum_pending_writes = ResolveBasicPendingWriteCount(
                    command, training_scalar_bytes, training_block_rows),
                .progress_total_blocks =
                    BlockCount(command.training_rows, training_block_rows),
                .log_progress = true,
                .log_metrics = command.train_log_metrics,
            },
    };
    request.initialization->topology = recipe.initialization_topology;
    request.initialization->fast_topology = recipe.training_topology;
    request.initialization->fast_topology.radial_layer_count =
        recipe.initialization_topology.radial_layer_count;
    request.initialization->fast_topology.root_fraction =
        recipe.initialization_topology.root_fraction;
    request.initialization->fast_topology.depth_k =
        recipe.initialization_topology.depth_k;
    request.initialization->fast_topology.knn_k =
        recipe.initialization_topology.knn_k;
    request.initialization->fast_topology.maximum_depth =
        recipe.initialization_topology.maximum_depth;
    request.initialization->fast_topology.seed =
        recipe.initialization_topology.seed;
    request.initialization->fast_topology.icm_rounds =
        recipe.initialization_topology.icm_rounds;
    request.initialization->fast_topology.batching =
        recipe.initialization_topology.batching;
    request.initialization->fast_topology.virtual_enabled = false;
    request.initialization->topology_pipeline = {
        .maximum_inflight_clusters = command.topology_inflight_clusters,
        .maximum_prefetched_clusters =
            ResolveLinkagePrefetchClusterCount(command),
        .progress_total_clusters = command.nlist};
    request.initialization->log_metrics = command.train_log_metrics;
    if (command.resume_initialized_checkpoint.empty()) {
      request.root_model.emplace();
      request.root_model->shape = recipe.shape;
      request.root_model->source_rows = command.training_rows;
      request.root_model->source_block_rows = training_block_rows;
      request.root_model->seed = command.train_seed;
      request.root_model->recipe = recipe.root_initialization;
    } else {
      request.initialization->resume_checkpoint =
          command.resume_initialized_checkpoint;
    }
  }

  request.training = {
      .checkpoint_root = command.work_directory / "rounds",
      .initialization_checkpoint = std::nullopt,
      .resume_checkpoint =
          command.resume_round_checkpoint.empty()
              ? std::optional<std::filesystem::path>{}
              : std::optional<
                    std::filesystem::path>{command.resume_round_checkpoint},
      .identity = identity,
      .round_seeds = std::move(round_seeds),
      .periodic_checkpoint_enabled = command.train_checkpoint_enabled,
      .periodic_checkpoint_every_rounds = command.train_checkpoint_every_r,
      .periodic_checkpoint_base_path = formal_checkpoint_base,
      .update_opq_rotation = recipe.train_use_opq_rotation,
      .log_metrics = command.train_log_metrics,
      .log_linkage_pre_c1 = command.train_log_linkage_pre_c1,
      .bad_cluster_training_weight = recipe.bad_cluster_training_weight,
      .edge_moment_shard_count =
          recipe.update_reduction.training_edge_moment_shards,
      .topology_policy = recipe.training_topology,
      .topology_pipeline = {.maximum_inflight_clusters =
                                command.topology_inflight_clusters,
                            .maximum_prefetched_clusters =
                                ResolveLinkagePrefetchClusterCount(command),
                            .progress_total_clusters = command.nlist},
  };
  request.base_basic = {
      .bucket_directory = command.work_directory / "base-buckets",
      .list_directory = command.output_directory / "base-lists",
      .cids_per_bucket = cids_per_bucket,
      .bucket_flush_bytes = EffectiveBucketFlushBytes(command),
      .seed = command.base_seed,
      .checkpoint_enabled = command.base_basic_checkpoint_enabled,
      .checkpoint_identity = {},
      .list_identity = {},
      .retain_bucket_directory =
          ResolveCleanupPolicy(command.cleanup).keep_base_basic,
      .source_rows = command.base_rows,
      .source_block_rows = base_block_rows,
      .cpu_threads = command.effective_cpu_threads,
      .schedule = BaseBasicSchedule(command),
      .pipeline =
          {
              .maximum_inflight_blocks = ResolveBasicInflightBlockCount(
                  command, base_scalar_bytes, base_block_rows),
              .maximum_prefetched_blocks = ResolveBasicPrefetchBlockCount(
                  command, base_scalar_bytes, base_block_rows),
              .maximum_pending_writes = ResolveBasicPendingWriteCount(
                  command, base_scalar_bytes, base_block_rows),
              .progress_total_blocks =
                  BlockCount(command.base_rows, base_block_rows),
              .log_progress = true,
              .log_metrics = command.train_log_metrics,
          },
  };
  request.artifacts = {
      .dataset_name = command.dataset_name,
      .base_source_first_row = command.base_first_row,
      .base_source_count = command.base_rows,
      .base_source_block_rows = base_block_rows,
      .base_basic_rsd_rank = command.base_basic_rsd_rank,
      .basic_executor = !command.use_cuda
                            ? BasicExecutorKind::kCpuOnly
                            : BasicExecutorKind::kCudaOnly,
      .cuda_allow_tf32 = command.use_cuda && command.cuda_allow_tf32,
      .candidate_search = recipe.base_candidate_search,
      .linkage_numerical_policy = linkage_numerical_policy,
  };
  request.final_index = {
      .prepared_spool = command.work_directory / "final-spool",
      .destination = command.output_directory / "linkage_list",
      .protect_existing_outputs = command.protect_existing_outputs,
      .checkpoint_enabled = command.linkage_checkpoint_enabled,
      .coefficient_codec = recipe.coefficient_codec,
      .norm8_mode = recipe.norm8_mode,
      .norm8_lloyd_iterations = recipe.norm8_lloyd_iterations,
      .use_cuda_norm8_fit = command.gpu_norm_enable,
      .cuda_norm_device = command.cuda_device,
      .retain_float_forest =
          !command.coefficient_codec_enabled ||
          std::any_of(command.eval_representations.begin(),
                      command.eval_representations.end(),
                      [](EvalRepresentation representation) {
                        return representation == EvalRepresentation::kFloat ||
                               representation ==
                                   EvalRepresentation::kFloatScaled;
                      }) ||
          (command.base_linkage_store_coefficients_f32 &&
           ResolveCleanupPolicy(command.cleanup).keep_linkage_list_f32),
      .retain_float_norm2 = false,
      .publish_quantized_forest = command.coefficient_codec_enabled,
      .publish_scaled_score8 =
          command.coefficient_codec_enabled &&
          std::find(command.eval_representations.begin(),
                    command.eval_representations.end(),
                    EvalRepresentation::kQuantizedScaled) !=
              command.eval_representations.end(),
      .publish_float_scaled = std::find(command.eval_representations.begin(),
                                        command.eval_representations.end(),
                                        EvalRepresentation::kFloatScaled) !=
                              command.eval_representations.end(),
      .topology = recipe.base_topology,
      .pipeline = {.maximum_inflight_clusters =
                       command.topology_inflight_clusters,
                   .maximum_prefetched_clusters =
                       ResolveLinkagePrefetchClusterCount(command),
                   .progress_total_clusters = command.nlist},
      .spool_window =
          {
              .maximum_clusters = command.final_spool_window_clusters,
              .target_real_rows = command.final_spool_window_real_rows,
          },
  };
  request.final_index.topology.collect_source_norm2 =
      request.final_index.publish_scaled_score8 ||
      request.final_index.publish_float_scaled;
  request.artifact_namespace = IndexBuildNamespace{
      .output_base = command.output_directory,
      .work_base = namespace_work_base.empty() ? command.work_directory
                                               : namespace_work_base,
      .prefix = command.output_prefix,
      .on_resolved = [&](const ResolvedIndexBuildPaths& paths) -> Status {
        BuildCommand resolved = command;
        resolved.work_directory = paths.work_root;
        resolved.output_directory = paths.run_root;
        resolved.train_enabled = false;
        if (const Status status =
                ValidateBuildPathIsolation(resolved, paths.index_directory);
            !status.ok())
          return status;
        if (command.base_linkage_enabled && command.protect_existing_outputs) {
          std::error_code error;
          const bool exists =
              std::filesystem::exists(paths.index_directory, error);
          if (error)
            return Status::Internal(
                "cannot inspect protected linkage target: " + error.message());
          if (exists)
            return Status::FailedPrecondition(
                "linkage target exists and "
                "large.protect_existing_outputs=true: " +
                paths.index_directory.string());
        }
        if (const Status status =
                PrepareBuildOutputDestinations(resolved, recipe);
            !status.ok())
          return status;
        std::error_code error;
        std::filesystem::create_directories(paths.work_root, error);
        if (error)
          return Status::Internal("cannot create artifact work root: " +
                                  error.message());
        std::filesystem::create_directories(paths.run_root, error);
        if (error)
          return Status::Internal("cannot create artifact run root: " +
                                  error.message());
        return Status::Ok();
      },
  };

  if (!command.use_cuda) {
    const bool create_initialization = command.train_enabled &&
                                       command.train_initialization_enabled &&
                                       command.resume_round_checkpoint.empty();
    const bool create_root =
        create_initialization && command.resume_initialized_checkpoint.empty();
    const bool create_training =
        command.train_enabled && recipe.additional_training_round_count != 0U;
    auto& cpu_options = prepared.cpu;
    cpu_options.create_training_initialization = create_initialization;
    cpu_options.create_fast_initialization =
        create_root && command.init_linkage_mode == "fast_init";
    cpu_options.create_root_initialization = create_root;
    cpu_options.create_training = create_training;
    cpu_options.create_base_encode = command.base_encode_enabled;
    cpu_options.create_final_linkage = command.base_linkage_enabled;
    cpu_options.training_basic_rsd_rank = command.training_basic_rsd_rank;
    cpu_options.base_basic_rsd_rank = command.base_basic_rsd_rank;
    cpu_options.initialization_source = training_source;
    cpu_options.training_source = training_source;
    cpu_options.base_source = base_source;
    cpu_options.cpu_worker_count = command.effective_cpu_threads;
    cpu_options.base_list_worker_count = command.effective_base_list_threads;
    cpu_options.root_moment_partition_count =
        recipe.update_reduction.root_moment_partitions;
    cpu_options.rotation_moment_partition_count =
        recipe.update_reduction.rotation_moment_partitions;
    cpu_options.root_initialization = {
        .collect_metrics = command.train_log_metrics,
    };
    cpu_options.training_basic = {
        .artifact_root = command.work_directory / "training-basic",
        .cids_per_bucket = cids_per_bucket,
        .bucket_flush_bytes = EffectiveBucketFlushBytes(command),
        .source_rows = command.training_rows,
        .source_block_rows = training_block_rows,
        .cpu_threads = command.effective_cpu_threads,
        .schedule = TrainingBasicSchedule(command),
        .pipeline =
            {
                .maximum_inflight_blocks = ResolveBasicInflightBlockCount(
                    command, training_scalar_bytes, training_block_rows),
                .maximum_prefetched_blocks = ResolveBasicPrefetchBlockCount(
                    command, training_scalar_bytes, training_block_rows),
                .maximum_pending_writes = ResolveBasicPendingWriteCount(
                    command, training_scalar_bytes, training_block_rows),
                .progress_total_blocks =
                    BlockCount(command.training_rows, training_block_rows),
                .log_progress = true,
                .log_metrics = command.train_log_metrics,
            },
    };
    cpu_options.initialization_topology = {
        .icm_rounds = recipe.initialization_topology.icm_rounds,
        .outer_worker_count = command.effective_cpu_threads,
        .direct_lane_count = resource_plan.initialization_cpu_direct_backends,
        .maximum_task_pairs =
            MaximumProducedEvaluationPairs(recipe.initialization_topology),
    };
    cpu_options.training_topology = {
        .full_topology_worker_count = command.effective_cpu_threads,
        .direct_lane_count = resource_plan.ordinary_cpu_direct_backends,
        .maximum_task_pairs =
            MaximumProducedEvaluationPairs(recipe.training_topology),
        .virtual_root_schedule = TrainingVirtualRootSchedule(command),
        .least_squares_cost_mode = command.least_squares_cost_mode,
    };
    cpu_options.fast_initialization_topology = cpu_options.training_topology;
    cpu_options.fast_initialization_topology.direct_lane_count =
        resource_plan.initialization_cpu_direct_backends;
    cpu_options.fast_initialization_topology.maximum_task_pairs =
        MaximumProducedEvaluationPairs(recipe.initialization_topology);
    cpu_options.final_topology = {
        .full_topology_worker_count = command.effective_cpu_threads,
        .direct_lane_count = resource_plan.ordinary_cpu_direct_backends,
        .maximum_task_pairs =
            MaximumProducedEvaluationPairs(recipe.base_topology),
        .virtual_root_schedule = BaseBasicSchedule(command),
        .least_squares_cost_mode = command.least_squares_cost_mode,
    };
  }
  return prepared;
}

template <int M>
StatusOr<BuildExecutionReport> Run(const BuildCommand& command,
                                    PreparedIndexBuild&& prepared) {
  IndexBuildRequest<M> request;
  static_cast<IndexBuildOptions&>(request) =
      std::move(static_cast<IndexBuildOptions&>(prepared));
  if (prepared.initialization.has_value()) {
    request.initialization.emplace(std::move(*prepared.initialization));
  }
  auto result = [&]() -> StatusOr<IndexBuildResult<M>> {
#if STLQPP_HAS_CUDA_RUNTIME
    if (command.use_cuda) {
      auto runtime =
          CudaIndexBuildRuntime<M>::Create(std::move(prepared.cuda));
      if (!runtime.ok()) return runtime.status();
      return runtime.value()->Build(request);
    }
#else
    if (command.use_cuda) {
      return Status::FailedPrecondition(
          "runtime.use_cuda=true but this binary has no CUDA backend");
    }
#endif
    auto runtime = CpuIndexBuildRuntime<M>::Create(std::move(prepared.cpu));
    if (!runtime.ok()) return runtime.status();
    return runtime.value()->Build(request);
  }();
  if (!result.ok()) {
    return result.status();
  }
  const auto& built = result.value();
  if (auto status = PersistBuildOriginalNorms(command, built.index_directory);
      !status.ok()) return status;
  ApplySuccessfulBuildCleanup<M>(command, built);
  BuildExecutionReport report;
  report.fast_initialization = command.init_linkage_mode == "fast_init" &&
                               built.initialization.has_value();
  report.wall_times = {
      .root_model = built.statistics.root_model_wall_seconds,
      .initialization = built.statistics.initialization_wall_seconds,
      .training = built.statistics.training_wall_seconds,
      .base_basic = built.statistics.base_basic_wall_seconds,
      .final_linkage_publication =
          built.statistics.final_linkage_publication_wall_seconds,
      .total = built.statistics.wall_seconds,
  };
  if (built.root_model.has_value()) {
    report.root_layers.assign(built.root_model->begin(),
                              built.root_model->end());
  }
  if (built.initialization.has_value()) {
    report.initialization_resources = built.initialization->linkage_resources;
  }
  report.training_rounds = built.training.executed_rounds;
  report.final_topology = built.final_topology;
  report.final_resources = built.final_linkage_resources;
  report.trained_model_artifact = built.training.checkpoint_directory;
  report.index_directory = built.index_directory;
  report.base_list_directory = built.base_list_directory;
  report.owns_base_list = built.owns_base_list;
  if (built.resolved_paths.has_value()) {
    report.run_root = built.resolved_paths->run_root;
    report.work_root = built.resolved_paths->work_root;
    report.base_basic_hash = built.resolved_paths->base_basic_hash;
    report.base_list_hash = built.resolved_paths->base_list_hash;
    report.linkage_hash = built.resolved_paths->linkage_hash;
  }
  return report;
}

StatusOr<BuildExecutionReport> DispatchM(
    const BuildCommand& command, PreparedIndexBuild&& prepared) {
  if (command.m != 5U && command.m != 6U && command.m != 10U &&
      command.m != 12U)
    return Run<0>(command, std::move(prepared));
  return DispatchModelM(command.m, [&]<std::uint32_t M>() {
    return Run<M>(command, std::move(prepared));
  });
}

}  // namespace

#if STLQPP_HAS_CUDA_RUNTIME
CudaTrainingTopologyContextOptions ResolveTrainingTopologyContextOptions(
    const BuildCommand& command, const EffectiveBuildRecipe& recipe,
    const IndexBuildExecutionControls& controls) {
  const BuildResourcePlan resource_plan =
      ResolveBuildResourcePlan(command, controls.scheduler_enabled);
  return {
      .full_topology_worker_count = command.effective_cpu_threads,
      .ordinary_direct_lane_count = resource_plan.ordinary_direct_backends,
      .virtual_root_schedule = TrainingVirtualRootSchedule(command),
      .least_squares_cost_mode = command.least_squares_cost_mode,
      .linkage_scheduler = MakeLinkageResources(
          command, controls, command.linkage_direct_pool_size,
          command.linkage_max_batch_pairs,
          MaximumProducedEvaluationPairs(recipe.training_topology),
          EffectiveLinkageAsyncPinnedMebibytes(command)),
  };
}

CudaTrainingTopologyContextOptions ResolveFinalTopologyContextOptions(
    const BuildCommand& command, const EffectiveBuildRecipe& recipe,
    const IndexBuildExecutionControls& controls) {
  const BuildResourcePlan resource_plan =
      ResolveBuildResourcePlan(command, controls.scheduler_enabled);
  return {
      .full_topology_worker_count = command.effective_cpu_threads,
      .ordinary_direct_lane_count = resource_plan.ordinary_direct_backends,
      .virtual_root_schedule = BaseBasicSchedule(command),
      .least_squares_cost_mode = command.least_squares_cost_mode,
      .linkage_scheduler = MakeLinkageResources(
          command, controls, command.linkage_direct_pool_size,
          command.linkage_max_batch_pairs,
          MaximumProducedEvaluationPairs(recipe.base_topology),
          EffectiveLinkageAsyncPinnedMebibytes(command)),
  };
}
#endif

StatusOr<BuildExecutionReport> RunIndexBuild(
    const BuildCommand& command, EffectiveBuildRecipe recipe,
    IndexBuildExecutionControls controls,
    std::filesystem::path namespace_work_base) {
  if (command.base_linkage_enabled && command.coefficient_codec_enabled &&
      command.gpu_norm_enable && !command.use_cuda) {
    return Status::FailedPrecondition(
        "eval.linkage.gpu_norm_enable=true requires runtime.use_cuda=true "
        "for norm publication");
  }
  if (command.base_linkage_enabled && command.coefficient_codec_enabled &&
      command.gpu_norm_enable &&
      stlqpp::PublishedNorm8Mode(command.norm8_mode) != ReconstructionNorm8Mode::kIdentityLloyd) {
    return Status::Unimplemented(
        "GPU norm8 tail-weighted mode is not supported; set "
        "eval.linkage.gpu_norm_enable=false or use lut");
  }
  if (const Status status = ValidateBuildPathIsolation(command); !status.ok()) {
    return status;
  }
  if (!command.base_encode_enabled && command.base_linkage_enabled &&
      !command.reused_base_list_directory.empty()) {
    std::error_code error;
    if (!std::filesystem::is_directory(command.reused_base_list_directory,
                                       error) ||
        error) {
      return Status::FailedPrecondition(
          "io.base_file does not name an existing base-list directory: " +
          command.reused_base_list_directory.string());
    }
  }
  if (command.effective_training_format == SourceFormat::kAuto ||
      command.effective_base_format == SourceFormat::kAuto) {
    return Status::FailedPrecondition(
        "build source formats were not resolved with input metadata");
  }
  BuildCommand training_only = command;
  training_only.base_encode_enabled = false;
  if (const Status status =
          PrepareBuildOutputDestinations(training_only, recipe);
      !status.ok()) {
    return status;
  }
  auto prepared = PrepareIndexBuild(command, recipe, controls,
                                    namespace_work_base);
  if (!prepared.ok()) return prepared.status();
  return DispatchM(command, std::move(prepared).value());
}

}  // namespace stlqpp::app
