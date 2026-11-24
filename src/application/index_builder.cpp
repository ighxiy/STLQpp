#include "stlqpp/application/index_builder.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "stlqpp/application/final_index_log.h"
#include "stlqpp/application/registered_artifact_recipes.h"
#include "stlqpp/index/codec/selected_atom_gram.h"
#include "stlqpp/index/emit/base_list_reader.h"
#include "stlqpp/runtime/telemetry/logger.h"
#include "stlqpp/storage/artifact_hash_factory.h"
#include "stlqpp/training/basic/basic_schedule.h"

namespace stlqpp {

LinkageTopologyRecipe MakeTopologyHashRecipe(
    const FullTopologyPolicy& topology,
    bool cuda_allow_tf32,
    LinkageNumericalPolicy numerical_policy) noexcept {
  return {
      .root_percentile = topology.root_fraction,
      .topology_layers = topology.radial_layer_count,
      .max_depth = topology.maximum_depth,
      .knn_k = topology.knn_k,
      .depth_k = topology.depth_k,
      .rsd_rank = topology.rsd_rank,
      .rsd_icm_rounds = topology.icm_rounds,
      .inner_batch_rows = topology.batching.inner_batch_rows,
      .maximum_batch_pairs = topology.batching.maximum_batch_pairs,
      .many_nodes_preflush_target_pairs =
          topology.batching.many_nodes_preflush_target_pairs,
      .same_layer_block_rows =
          topology.batching.same_layer_block_rows,
      .same_layer_minimum_batch_pairs =
          topology.batching.same_layer_minimum_batch_pairs,
      .same_layer_window_maximum_pairs =
          topology.batching.same_layer_window_maximum_pairs,
      .same_layer_maximum_pending_nodes =
          topology.batching.same_layer_maximum_pending_nodes,
      .same_layer_preflush_target_pairs =
          topology.batching.same_layer_preflush_target_pairs,
      .same_layer_preflush_target_nodes =
          topology.batching.same_layer_preflush_target_nodes,
      .tiny_forced_cpu_enabled =
          topology.batching.tiny_forced_cpu_enabled,
      .tiny_forced_pair_limit =
          topology.batching.tiny_forced_pair_limit,
      .tiny_forced_node_limit =
          topology.batching.tiny_forced_node_limit,
      .single_gpu_route_enabled =
          topology.batching.single_gpu_route_enabled,
      .single_gpu_scheduler_backed =
          topology.batching.single_gpu_scheduler_backed,
      .single_gpu_min_candidates =
          topology.batching.single_gpu_min_candidates,
      .single_gpu_max_candidates =
          topology.batching.single_gpu_max_candidates,
      .same_layer_single_gpu_min_candidates =
          topology.batching.same_layer_single_gpu_min_candidates,
      .bad_frozen_window_rows =
          topology.batching.bad_frozen_window_rows,
      .bad_frozen_maximum_pairs =
          topology.batching.bad_frozen_maximum_pairs,
      .cuda_allow_tf32 = cuda_allow_tf32,
      .seed = topology.seed,
      .least_squares_cost_mode = topology.least_squares_cost_mode,
      .numerical_policy = numerical_policy,
  };
}


namespace {

using Clock = std::chrono::steady_clock;

double ElapsedSeconds(Clock::time_point begin) noexcept {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

std::string FormatSeconds(double seconds) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(3) << seconds;
  return output.str();
}

std::string FormatMetric(double value, int precision) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(precision) << value;
  return output.str();
}

bool SameIdentity(const TrainingCheckpointIdentity& left,
                  const TrainingCheckpointIdentity& right) noexcept {
  return left.training_source == right.training_source &&
         left.training_recipe == right.training_recipe;
}

template <int M>
TrainedModelHashView HashView(const TrainingModel<M>& model) {
  TrainedModelHashView output;
  output.rotation = model.rotation();
  output.root_codebooks.reserve(model.codebooks().shape().m);
  output.root_codebooks.push_back(model.codebooks().root0());
  for (std::uint32_t layer = 1U; layer < model.codebooks().shape().m; ++layer) {
    output.root_codebooks.push_back(model.codebooks().root_small(layer));
  }
  output.edge_codebooks.reserve(model.codebooks().shape().m);
  for (std::uint32_t layer = 0U; layer < model.codebooks().shape().m; ++layer) {
    output.edge_codebooks.push_back(model.codebooks().edge(layer));
  }
  return output;
}

}  // namespace

StatusOr<ResolvedIndexBuildPaths> ResolveIndexBuildPaths(
    const IndexBuildNamespace& ns, const std::string& dataset_name,
    const IndexBuildArtifactHashes& hashes) {
  const auto safe_component = [](const std::string& component) {
    return !component.empty() && component != "." && component != ".." &&
           component.find('/') == std::string::npos &&
           component.find('\\') == std::string::npos;
  };
  if (!safe_component(dataset_name) || !safe_component(ns.prefix) ||
      ns.output_base.empty() || ns.work_base.empty()) {
    return Status::InvalidArgument("invalid artifact namespace");
  }
  const std::filesystem::path suffix =
      std::filesystem::path(dataset_name) / [&] {
        std::ostringstream tag;
        tag << ns.prefix << "__0x" << std::hex
            << hashes.index.linkage_list_directory.value();
        return tag.str();
      }();
  return ResolvedIndexBuildPaths{
      .run_root = ns.output_base / suffix,
      .work_root = ns.work_base / suffix,
      .base_list_directory = ns.output_base / suffix / "base-lists",
      .index_directory = ns.output_base / suffix / "linkage_list",
      .base_basic_hash = hashes.base_basic,
      .base_list_hash = hashes.base_list,
      .linkage_hash = hashes.index.linkage_list_directory,
  };
}

IndexBuildArtifactHashes ComputeIndexBuildArtifactHashesRuntime(
    const ModelShape& shape, const TrainedModelHashView& model,
    ArrayView<const std::uint8_t> bad_cluster_mask, bool base_source_is_u8,
    const BaseBasicStageOptions& base_basic_options,
    const IndexArtifactRecipe& artifacts,
    const FinalIndexBuildOptions& final_index) {
  const std::uint64_t real_count = artifacts.base_source_count;
  std::vector<std::uint32_t> root_sizes(shape.m, 256U);
  root_sizes[0] = shape.nlist;
  const BasicSchedule base_schedule = base_basic_options.schedule;
  const ArtifactHash64 base_basic = ArtifactHashFactory::BaseBasicDirectory({
      .trained_model = model,
      .dataset_name = artifacts.dataset_name,
      .source_is_u8 = base_source_is_u8,
      .source_first_vector = artifacts.base_source_first_row,
      .effective_base_count = real_count,
      .shape = shape,
      .codebook_sizes = root_sizes,
      .store_layout =
          {
              .source_block_rows = artifacts.base_source_block_rows,
              .cids_per_bucket = base_basic_options.cids_per_bucket,
          },
      .encoder =
          {
              .rsd_rank = artifacts.base_basic_rsd_rank,
              .ils_iterations = base_schedule.ils_rounds(),
              .icm_iterations = base_schedule.icm_sweeps(),
              .perturb_layers = base_schedule.perturbations(),
              .norm_bins = 0U,
              .seed = base_basic_options.seed,
              .least_squares_cost_mode =
                  base_schedule.least_squares_cost_mode(),
          },
      .execution =
          {
              .executor = artifacts.basic_executor,
              .cuda_allow_tf32 = artifacts.cuda_allow_tf32,
          },
  });
  const ArtifactHash64 base_list = ArtifactHashFactory::BaseListDirectory({
      .base_basic_directory = base_basic,
      .shape = shape,
      .real_count = real_count,
  });
  const FullTopologyPolicy& topology = final_index.topology;
  const ArtifactHash64 linkage = ArtifactHashFactory::LinkageListDirectory({
      .trained_model = model,
      .base_list_directory = base_list,
      .shape = shape,
      .codebook_sizes = root_sizes,
      .edge_root_codebook_size = shape.residual_codebook_size,
      .bad_cluster_mask = bad_cluster_mask,
      .topology = MakeTopologyHashRecipe(topology, artifacts.cuda_allow_tf32,
                                         artifacts.linkage_numerical_policy),
      .virtual_roots =
          {
              .enabled = topology.virtual_enabled,
              .target_ratio = topology.virtual_ratio,
              .minimum_per_cluster = topology.minimum_virtual_roots,
              .maximum_per_cluster = topology.maximum_virtual_roots,
              .use_fixed_per_cluster = topology.use_fixed_virtual_roots,
              .fixed_per_cluster = topology.fixed_virtual_roots,
              .umap_knn_k = topology.virtual_policy.neighbor_count,
              .local_connectivity = topology.virtual_policy.local_connectivity,
              .overlap_threshold = topology.virtual_policy.overlap_threshold,
              .prefer_density_peaks =
                  topology.virtual_policy.prefer_density_peaks,
              .anchor_neighbor_k = topology.virtual_policy.anchor_neighbor_k,
          },
      .candidate_search = artifacts.candidate_search,
  });
  const ArtifactHash64 codec = ArtifactHashFactory::CoefficientCodec({
      .linkage_list_directory = linkage,
      .codec = final_index.coefficient_codec,
  });
  return {
      .base_basic = base_basic,
      .base_list = base_list,
      .index =
          {
              .linkage_list_directory = linkage,
              .coefficient_codec = final_index.publish_quantized_forest
                                       ? codec
                                       : ArtifactHash64{},
              .norm8_cache =
                  final_index.publish_quantized_forest
                      ? ArtifactHashFactory::Norm8Cache({
                            .linkage_list_directory = linkage,
                            .coefficient_codec = codec,
                            .center_count = kRegisteredNorm8CenterCount,
                            .mode = final_index.norm8_mode,
                            .kmeans_iterations =
                                final_index.norm8_lloyd_iterations,
                        })
                      : ArtifactHash64{},
          },
  };
}

template <int M>
IndexBuildArtifactHashes ComputeIndexBuildArtifactHashes(
    const TrainingModel<M>& model, ArrayView<const std::uint8_t> mask,
    bool base_source_is_u8, const BaseBasicStageOptions& basic,
    const IndexArtifactRecipe& artifacts,
    const FinalIndexBuildOptions& final_index) {
  return ComputeIndexBuildArtifactHashesRuntime(
      model.codebooks().shape(), HashView(model), mask, base_source_is_u8,
      basic, artifacts, final_index);
}

namespace {

template <int M>
Status ValidateRequest(const IndexBuildRequest<M>& request,
                       const IndexBuildRuntime<M>& runtime) {
  if (!request.stages.train && !request.training.round_seeds.empty()) {
    return Status::InvalidArgument(
        "disabled training cannot contain executable rounds");
  }
  if (!request.stages.initialize_training &&
      !request.training.resume_checkpoint.has_value()) {
    return Status::InvalidArgument(
        "disabled training initialization requires a train checkpoint");
  }
  if (request.stages.train && !request.training.round_seeds.empty() &&
      (runtime.training.basic == nullptr ||
       runtime.training.topology == nullptr ||
       runtime.training.rotation == nullptr ||
       runtime.training.spd_solver == nullptr ||
       runtime.training.svd == nullptr ||
       runtime.training.root_moment_parallel == nullptr ||
       runtime.training.update_parallel == nullptr)) {
    return Status::InvalidArgument("training runtime is incomplete");
  }
  if (request.stages.encode_base && (std::visit(
                                         [](const auto& inputs) {
                                           return inputs.source == nullptr ||
                                                  inputs.runtime == nullptr;
                                         },
                                         runtime.base) ||
                                     runtime.base_list_parallel == nullptr)) {
    return Status::InvalidArgument("base encode runtime is incomplete");
  }
  if (request.stages.build_linkage &&
      (runtime.final_topology == nullptr || runtime.parallel == nullptr ||
       runtime.gemm == nullptr || runtime.coefficient_solver == nullptr)) {
    return Status::InvalidArgument("final linkage runtime is incomplete");
  }
  if (request.stages.build_linkage && !request.stages.encode_base &&
      request.stages.reused_base_basic_directory.empty() &&
      !request.artifact_namespace.has_value()) {
    return Status::InvalidArgument(
        "linkage-only build requires a base basic checkpoint");
  }
  if ((request.stages.encode_base || request.stages.build_linkage) &&
      (request.artifacts.dataset_name.empty() ||
       request.artifacts.base_source_block_rows == 0U)) {
    return Status::InvalidArgument("index build artifact recipe is incomplete");
  }
  if (request.training.resume_checkpoint.has_value()) {
    if (request.root_model.has_value() || request.initialization.has_value() ||
        request.training.initialization_checkpoint.has_value()) {
      return Status::InvalidArgument(
          "round-checkpoint restart must skip initialization");
    }
    return Status::Ok();
  }
  if (!request.stages.initialize_training) {
    return Status::InvalidArgument(
        "fresh training requires initialization or a train checkpoint");
  }
  if (!request.initialization.has_value() ||
      request.training.initialization_checkpoint.has_value() ||
      !SameIdentity(request.initialization->identity,
                    request.training.identity)) {
    return Status::InvalidArgument(
        "fresh build requires one identity-matched initialization");
  }
  if (request.initialization->resume_checkpoint.has_value()) {
    if (request.root_model.has_value() ||
        request.initialization->initial_model != nullptr) {
      return Status::InvalidArgument(
          "initialized-checkpoint reopen must skip raw root initialization");
    }
    return Status::Ok();
  }
  if (!request.root_model.has_value() ||
      request.initialization->initial_model != nullptr ||
      std::visit(
          [&](const auto& inputs) {
            return inputs.root_backend == nullptr ||
                   inputs.runtime.basic_source == nullptr ||
                   inputs.runtime.basic_runtime == nullptr ||
                   (request.initialization->mode ==
                        TrainingInitializationMode::kLegacyRootOnly &&
                    inputs.runtime.topology == nullptr) ||
                   (request.initialization->mode ==
                        TrainingInitializationMode::kFastInit &&
                    inputs.runtime.full_topology == nullptr) ||
                   inputs.runtime.parallel == nullptr ||
                   inputs.runtime.root_moment_parallel == nullptr ||
                   inputs.runtime.list_parallel == nullptr ||
                   inputs.runtime.spd_solver == nullptr;
          },
          runtime.initialization)) {
    return Status::InvalidArgument(
        "fresh build requires raw root initialization authority");
  }
  return Status::Ok();
}

struct ResolvedBuildStages final {
  std::optional<ResolvedIndexBuildPaths> paths;
  BaseBasicStageOptions base;
  FinalIndexBuildOptions final_index;
  std::filesystem::path reused_base_directory;
};

StatusOr<ResolvedBuildStages> ResolveBuildStages(
    const std::optional<IndexBuildNamespace>& artifact_namespace,
    const std::string& dataset_name,
    const IndexBuildArtifactHashes& artifact_hashes,
    const BaseBasicStageOptions& base,
    const FinalIndexBuildOptions& final_index,
    std::filesystem::path reused_base_directory) {
  ResolvedBuildStages result;
  result.base = base;
  result.final_index = final_index;
  result.reused_base_directory = std::move(reused_base_directory);
  if (!artifact_namespace.has_value()) return result;
  const auto& ns = *artifact_namespace;
  auto paths = ResolveIndexBuildPaths(ns, dataset_name, artifact_hashes);
  if (!paths.ok()) return paths.status();
  result.paths = std::move(paths).value();
  result.base.bucket_directory = result.paths->work_root / "base-buckets";
  result.base.list_directory = result.paths->base_list_directory;
  result.final_index.prepared_spool = result.paths->work_root / "final-spool";
  result.final_index.destination = result.paths->index_directory;
  if (result.reused_base_directory.empty()) {
    result.reused_base_directory = result.paths->base_list_directory;
  }
  if (ns.on_resolved) {
    if (const Status status = ns.on_resolved(*result.paths); !status.ok())
      return status;
  }
  return result;
}

FinalFullTopologyIndexBuildRequest MakeFinalTopologyBuildRequest(
    const FinalIndexBuildOptions& options,
    const IndexBuildArtifactHashes& hashes) {
  return {
      .index = {
          .prepared_spool = options.prepared_spool,
          .destination = options.destination,
          .protect_existing_outputs = options.protect_existing_outputs,
          .checkpoint_enabled = options.checkpoint_enabled,
          .hashes = hashes.index,
          .coefficient_bits_per_layer =
              options.coefficient_codec.bits_per_layer,
          .coefficient_bits_per_layer_root =
              options.coefficient_codec.bits_per_layer_root,
          .coefficient_bits_per_layer_linkage =
              options.coefficient_codec.bits_per_layer_linkage,
          .coefficient_first_magnitude_quantiles =
              options.coefficient_codec.first_magnitude_quantiles,
          .coefficient_rest_magnitude_quantiles =
              options.coefficient_codec.rest_magnitude_quantiles,
          .coefficient_codec_options = {
              .weighted_quantile = options.coefficient_codec.weighted_quantile,
              .allow_clipping = options.coefficient_codec.allow_clipping,
              .fit_reconstruction_scales =
                  options.coefficient_codec.fit_reconstruction_scales,
              .refinement_sweeps = options.coefficient_codec.refinement_sweeps,
              .refinement_max_layer = options.coefficient_codec.refinement_max_layer,
              .refinement_step_limit =
                  options.coefficient_codec.refinement_step_limit,
          },
          .norm8_mode = options.norm8_mode,
          .norm8_lloyd_iterations = options.norm8_lloyd_iterations,
          .use_cuda_norm8_fit = options.use_cuda_norm8_fit,
          .cuda_norm_device = options.cuda_norm_device,
          .retain_float_forest = options.retain_float_forest,
          .retain_float_norm2 = options.retain_float_norm2,
          .publish_quantized_forest = options.publish_quantized_forest,
          .publish_scaled_score8 = options.publish_scaled_score8,
          .publish_float_scaled = options.publish_float_scaled,
      },
      .topology = options.topology,
      .pipeline = options.pipeline,
      .spool_window = options.spool_window,
  };
}

}  // namespace

namespace {

struct RootBuildSummary final {
  std::optional<SphericalRvqCudaDiagnostics> diagnostics;
  double last_mse = 0.0;
  double wall_seconds = 0.0;
};

struct TrainedBuildSummary final {
  ModelShape shape;
  IndexBuildArtifactHashes hashes;
};

struct FinalBuildSummary final {
  FullTopologyPipelineStatistics topology;
  std::optional<LinkageResourceStatistics> resources;
  double wall_seconds = 0.0;
};

// One binding per complete build, on the caller's stack. These calls occur
// only at stage barriers; CID queues, workers and numerical calls stay typed.
class IndexBuildStages {
 public:
  virtual StatusOr<RootBuildSummary> InitializeRoot() = 0;
  virtual StatusOr<std::filesystem::path> InitializeTraining() = 0;
  virtual Status Train(const TrainRequest& request) = 0;
  virtual TrainedBuildSummary TrainedSummary() const = 0;
  virtual StatusOr<BaseBasicStageResult> EncodeBase(
      const BaseBasicStageOptions& options) = 0;
  virtual StatusOr<FinalBuildSummary> PublishLinkage(
      const BaseBasicStageResult& base,
      const FinalIndexBuildOptions& options,
      const IndexBuildArtifactHashes& hashes,
      Clock::time_point begin) = 0;

 protected:
  ~IndexBuildStages() = default;
};

struct IndexBuildCompletion final {
  BaseBasicStageResult base;
  FinalBuildSummary final;
  ResolvedBuildStages stages;
  IndexBuildStatistics statistics;
};

StatusOr<IndexBuildCompletion> RunBuildStages(
    const IndexBuildOptions& request,
    const TrainingInitializationOptions* initialization,
    bool initialization_source_is_u8, std::uint32_t expected_m,
    IndexBuildStages& operations) {
  const Clock::time_point build_begin = Clock::now();
  IndexBuildStatistics statistics;
  TrainRequest training_request = request.training;
  const Clock::time_point training_begin = Clock::now();
  if (!training_request.resume_checkpoint.has_value()) {
    LogInfo("Training quantizer...");
    if (!initialization->resume_checkpoint.has_value()) {
      LogInfo(
          "============== Baseline training with R = I (no global rotation) "
          "==============");
      LogInfo("(1)RVQ-like init by Spherical kmeans with annealing factor...");
      LogInfo("Train streaming init: using dataset reader (out-of-core).");
      const SphericalRvqInitializationRequest& root_request =
          *request.root_model;
      LogInfo("Train streaming init: dataset-reader RVQ ntrain=" +
              std::to_string(root_request.source_rows) +
              " m=" + std::to_string(root_request.shape.m) + " block_cols=" +
              std::to_string(root_request.source_block_rows) + " bvecs_u8=" +
              std::to_string(initialization_source_is_u8 ? 1 : 0));
      auto root = operations.InitializeRoot();
      if (!root.ok()) return root.status();
      statistics.root_model_wall_seconds = root.value().wall_seconds;
      LogInfo("Train streaming init wall time(s): " +
              FormatSeconds(statistics.root_model_wall_seconds));
      if (root.value().diagnostics.has_value()) {
        const auto& diagnostics = *root.value().diagnostics;
        LogInfo("RVQ kmeans diag: overlap_p1=" +
                std::to_string(diagnostics.overlapped_pass1 ? 1 : 0) +
                " overlap_p2=" +
                std::to_string(diagnostics.overlapped_pass2 ? 1 : 0) +
                " xnorm_dev_used=" +
                std::to_string(diagnostics.full_source_norm2_cache ? 1 : 0) +
                " x_pin_used=" +
                std::to_string(diagnostics.used_pinned_host_source ? 1 : 0));
      }
      if (initialization->basic.pipeline.log_metrics) {
        const double mse = root.value().last_mse;
        LogInfo(std::isfinite(mse) && mse >= 0.0
                    ? "RVQ-like Init mse error: " + FormatMetric(mse, 6)
                    : "RVQ-like Init mse error: unavailable (missing GPU "
                      "metric totals)");
      }
    }
    const Clock::time_point initialization_begin = Clock::now();
    if (!initialization->resume_checkpoint.has_value()) {
      LogInfo("(2)Train basic + init C_one (R=I)...");
    }
    auto initialized = operations.InitializeTraining();
    if (!initialized.ok()) return initialized.status();
    training_request.initialization_checkpoint = std::move(initialized).value();
    statistics.initialization_wall_seconds = ElapsedSeconds(initialization_begin);
  }

  LogInfo(request.stages.train ? "Training global R iterations..."
                               : "Loading trained model...");
  if (const Status status = operations.Train(training_request); !status.ok())
    return status;
  statistics.training_wall_seconds = ElapsedSeconds(training_begin);
  LogInfo(std::string(request.stages.train ? "Training wall time(s): "
                                         : "Trained model load wall time(s): ") +
          FormatSeconds(statistics.training_wall_seconds));
  const TrainedBuildSummary trained = operations.TrainedSummary();
  const auto& artifact_hashes = trained.hashes;
  auto resolved = ResolveBuildStages(
      request.artifact_namespace, request.artifacts.dataset_name,
      artifact_hashes, request.base_basic, request.final_index,
      request.stages.reused_base_basic_directory);
  if (!resolved.ok()) return resolved.status();
  ResolvedBuildStages stages = std::move(resolved).value();
  const Clock::time_point base_basic_begin = Clock::now();
  auto base = [&]() -> StatusOr<BaseBasicStageResult> {
    if (request.stages.encode_base) {
      LogInfo("Using PrecompLargeRoot: h0=" + std::to_string(trained.shape.nlist) +
              " H_small=" + std::to_string((trained.shape.m - 1U) * 256U));
      LogInfo("============== BaseSet Encoding ==============");
      stages.base.checkpoint_identity = artifact_hashes.base_basic;
      stages.base.list_identity = artifact_hashes.base_list;
      return operations.EncodeBase(stages.base);
    }
    if (!request.stages.build_linkage) return BaseBasicStageResult{};
    LogInfo("Reopening base basic checkpoint...");
    auto lists = detail::BaseListMetadata::Open(
        stages.reused_base_directory, expected_m);
    if (!lists.ok()) return lists.status();
    auto mappings = detail::BaseListReaderStorage::Open(lists.value());
    if (!mappings.ok()) return mappings.status();
    if (lists.value().identity != artifact_hashes.base_list) {
      return Status::FailedPrecondition(
          "reused base-list identity does not match the current model and "
          "basic encoding recipe; re-encode the base set");
    }
    return BaseBasicStageResult{
        .pipeline = {}, .rows = lists.value().offsets.back(),
        .list_directory = stages.reused_base_directory};
  }();
  if (!base.ok()) return base.status();
  if ((request.stages.encode_base || request.stages.build_linkage) &&
      base.value().rows != request.artifacts.base_source_count) {
    return Status::FailedPrecondition(
        "base basic artifact row count changed from the registered source");
  }
  statistics.base_basic_wall_seconds = ElapsedSeconds(base_basic_begin);
  if (!request.stages.encode_base && request.stages.build_linkage) {
    LogInfo(
        "Base basic checkpoint reopen wall time(s) [begin: open reused "
        "base-list directory; end: row-count validation; no encoding]: " +
        FormatSeconds(statistics.base_basic_wall_seconds));
  }
  FinalBuildSummary final;
  const Clock::time_point final_begin = Clock::now();
  if (request.stages.build_linkage) {
    LogInfo("Building virtual-mode linkage_list (streaming by cluster)...");
    auto published = operations.PublishLinkage(
        base.value(), stages.final_index, artifact_hashes, final_begin);
    if (!published.ok()) return published.status();
    final = std::move(published).value();
    statistics.final_linkage_publication_wall_seconds = final.wall_seconds;
  } else {
    LogInfo("Final linkage disabled; no index publication performed.");
  }
  statistics.wall_seconds = ElapsedSeconds(build_begin);
  LogInfo(
      "Total enabled build-stage wall time(s) [begin: BuildIndex entry; "
      "end: final enabled stage completion; includes enabled root model, "
      "initialization, training, base encode and linkage/publication; excludes "
      "application config/session setup and post-build archival]: " +
      FormatSeconds(statistics.wall_seconds));
  return IndexBuildCompletion{std::move(base).value(), std::move(final),
                              std::move(stages), statistics};
}

template <int M>
class TypedIndexBuildStages final : public IndexBuildStages {
 public:
  TypedIndexBuildStages(const IndexBuildRequest<M>& request,
                       IndexBuildRuntime<M>& runtime)
      : request_(request), runtime_(runtime) {}

  StatusOr<RootBuildSummary> InitializeRoot() override {
    const auto begin = Clock::now();
    auto root = std::visit([&](auto& inputs) {
      return InitializeRootTrainingModel<M>(*request_.root_model,
          *inputs.runtime.basic_source, *inputs.root_backend);
    }, runtime_.initialization);
    if (!root.ok()) return root.status();
    root_.emplace(std::move(root).value());
    const double wall_seconds = ElapsedSeconds(begin);
    root_statistics.emplace(root_->layers);
    return RootBuildSummary{root_->cuda_diagnostics,
                            root_->layers.back().final_mse_proxy, wall_seconds};
  }

  StatusOr<std::filesystem::path> InitializeTraining() override {
    auto request = *request_.initialization;
    if (root_) request.initial_model = &root_->model;
    auto initialized = std::visit([&](auto& inputs) {
      return stlqpp::InitializeTraining<M>(request, inputs.runtime);
    }, runtime_.initialization);
    if (!initialized.ok()) return initialized.status();
    initialization.emplace(std::move(initialized).value());
    std::filesystem::path directory = initialization->checkpoint_directory;
    root_.reset();
    return directory;
  }

  Status Train(const TrainRequest& request) override {
    auto result = stlqpp::Train<M>(request, runtime_.training);
    if (!result.ok()) return result.status();
    training.emplace(std::move(result).value());
    return Status::Ok();
  }

  TrainedBuildSummary TrainedSummary() const override {
    const auto& model = training->checkpoint.model();
    return {model.codebooks().shape(), ComputeIndexBuildArtifactHashes<M>(
        model, training->checkpoint.bad_cluster_mask(),
        std::holds_alternative<IndexBaseInputs<M, std::uint8_t>>(runtime_.base),
        request_.base_basic, request_.artifacts, request_.final_index)};
  }

  Status PrepareBaseModel() {
    if (base_model_) return Status::Ok();
    const auto& trained = training->checkpoint.model();
    auto books = trained.codebooks().InSourceSpace(trained.rotation(), *runtime_.gemm);
    if (!books.ok()) return books.status();
    base_model_.emplace(TrainingModel<M>::ForBase(
        std::move(books).value(), ArtifactHashFactory::BaseModel(HashView(trained))));
    LogInfo("Base codebooks prepared in original vector coordinates.");
    return Status::Ok();
  }

  StatusOr<BaseBasicStageResult> EncodeBase(
      const BaseBasicStageOptions& options) override {
    if (auto s = PrepareBaseModel(); !s.ok()) return s;
    return std::visit([&](auto& inputs) {
      using Scalar = typename std::decay_t<decltype(inputs)>::Scalar;
      return ScalarBaseBasicStage<M, Scalar>::Run(
          *base_model_, *inputs.source, *inputs.runtime,
          *runtime_.base_list_parallel, options);
    }, runtime_.base);
  }

  StatusOr<FinalBuildSummary> PublishLinkage(
      const BaseBasicStageResult& base, const FinalIndexBuildOptions& options,
      const IndexBuildArtifactHashes& hashes, Clock::time_point begin) override {
    const auto& checkpoint = training->checkpoint;
    if (auto s = PrepareBaseModel(); !s.ok()) return s;
    const auto& model = *base_model_;
    auto grams = SelectedAtomGramTables<M>::Create(
        model.codebooks(), *runtime_.gemm,
        (options.retain_float_norm2 || options.publish_quantized_forest) ? SelectedAtomGramDemand::kReconstructionNorm
                                   : SelectedAtomGramDemand::kCoefficientCodec);
    if (!grams.ok()) return grams.status();
    const TrainingBasicRoundArtifact basic{
        .list_directory = base.list_directory, .rows = base.rows, .lifetime = {}};
    auto topology = runtime_.final_topology->CreateFull(
        model, checkpoint.bad_cluster_mask(), basic,
        checkpoint.completed_round_count(), request_.final_index.topology);
    if (!topology.ok()) return topology.status();
    FinalIndexPublicationStatistics publication;
    auto final_request = MakeFinalTopologyBuildRequest(options, hashes);
    auto published = BuildFinalFullTopologyIndexFromSource<M>(
        final_request, model,
        topology.value()->source(), topology.value()->executors(), grams.value(),
        *runtime_.coefficient_solver, *runtime_.parallel, &publication);
    if (!published.ok()) return published.status();
    FinalBuildSummary result;
    result.topology = std::move(published).value();
    if (const Status status = LogFinalPublicationSummary(publication); !status.ok())
      return status;
    if (const Status status = LogFinalTopologySummary(
            result.topology, publication.canonical.counts, options.destination);
        !status.ok()) return status;
    result.resources = topology.value()->resource_statistics();
    result.wall_seconds = ElapsedSeconds(begin);
    LogInfo(
        "Final linkage, codec and norm8 publication wall time(s) "
        "[begin: selected-atom Gram and topology-context construction; "
        "end: successful canonical publication and DEV-compatible linkage "
        "summaries; excludes final basic encode]: " + FormatSeconds(result.wall_seconds));
    return result;
  }

  std::optional<ModelLayers<SphericalRvqLayerStatistics, M>> root_statistics;
  std::optional<TrainingInitializationResult<M>> initialization;
  std::optional<TrainResult<M>> training;

 private:
  const IndexBuildRequest<M>& request_;
  IndexBuildRuntime<M>& runtime_;
  std::optional<RootModelInitializationResult<M>> root_;
  std::optional<TrainingModel<M>> base_model_;
};

}  // namespace

template <int M>
StatusOr<IndexBuildResult<M>> BuildIndex(const IndexBuildRequest<M>& request,
                                       IndexBuildRuntime<M>& runtime) {
  if (const Status status = ValidateRequest(request, runtime); !status.ok())
    return status;
  TypedIndexBuildStages<M> operations(request, runtime);
  auto completed = RunBuildStages(
      request, request.initialization ? &*request.initialization : nullptr,
      std::holds_alternative<IndexInitializationInputs<M, std::uint8_t>>(
          runtime.initialization), M, operations);
  if (!completed.ok()) return completed.status();
  auto& output = completed.value();
  const std::filesystem::path built_base_list_directory = output.base.list_directory;
  return IndexBuildResult<M>{
      .root_model = std::move(operations.root_statistics),
      .initialization = std::move(operations.initialization),
      .training = std::move(*operations.training),
      .base_basic = std::move(output.base),
      .final_topology = std::move(output.final.topology),
      .final_linkage_resources = std::move(output.final.resources),
      .index_directory = request.stages.build_linkage
                             ? output.stages.final_index.destination
                             : std::filesystem::path{},
      .base_list_directory = built_base_list_directory,
      .owns_base_list = request.stages.encode_base,
      .resolved_paths = std::move(output.stages.paths),
      .stages = request.stages,
      .statistics = output.statistics,
  };
}

template StatusOr<IndexBuildResult<0>> BuildIndex<0>(
    const IndexBuildRequest<0>&, IndexBuildRuntime<0>&);

template StatusOr<IndexBuildResult<5>> BuildIndex<5>(
    const IndexBuildRequest<5>&, IndexBuildRuntime<5>&);

template StatusOr<IndexBuildResult<6>> BuildIndex<6>(
    const IndexBuildRequest<6>&, IndexBuildRuntime<6>&);

template StatusOr<IndexBuildResult<10>> BuildIndex<10>(
    const IndexBuildRequest<10>&, IndexBuildRuntime<10>&);

template StatusOr<IndexBuildResult<12>> BuildIndex<12>(
    const IndexBuildRequest<12>&, IndexBuildRuntime<12>&);

#define STLQPP_INSTANTIATE_ARTIFACT_HASHES(M)                           \
  template IndexBuildArtifactHashes ComputeIndexBuildArtifactHashes<M>( \
      const TrainingModel<M>&, ArrayView<const std::uint8_t>, bool,     \
      const BaseBasicStageOptions&, const IndexArtifactRecipe&,         \
      const FinalIndexBuildOptions&);
STLQPP_INSTANTIATE_ARTIFACT_HASHES(0)
STLQPP_INSTANTIATE_ARTIFACT_HASHES(5)
STLQPP_INSTANTIATE_ARTIFACT_HASHES(6)
STLQPP_INSTANTIATE_ARTIFACT_HASHES(10)
STLQPP_INSTANTIATE_ARTIFACT_HASHES(12)
#undef STLQPP_INSTANTIATE_ARTIFACT_HASHES

}  // namespace stlqpp
