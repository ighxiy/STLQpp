#include "effective_build_recipe.h"
#include "stlqpp/application/index_builder.h"

#include "stlqpp/application/registered_artifact_recipes.h"
#include "stlqpp/core/execution/workload_thread_budget.h"
#include "stlqpp/storage/trained_model_file.h"
#include "stlqpp/storage/canonical_model_codec.h"
#include "stlqpp/storage/index_manifest_codec.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/application/config_text.h"
#include "stlqpp/storage/legacy_rsd_training_identity.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <fstream>
#include <optional>

namespace stlqpp::app {
namespace {

constexpr std::uint32_t kMaximumEvaluationPairs = 32768U;

std::uint32_t PositiveOrOne(std::int32_t value) noexcept {
  return static_cast<std::uint32_t>(std::max(1, value));
}

std::uint32_t NonNegative(std::int32_t value) noexcept {
  return static_cast<std::uint32_t>(std::max(0, value));
}

CausalTopologyBatching ResolveTopologyBatching(
    const BuildCommand& command,
    std::uint32_t maximum_batch_pairs,
    bool initialization) noexcept {
  const std::uint32_t same_layer_block_rows =
      PositiveOrOne(command.linkage_same_layer_block_rows);
  const std::uint32_t same_layer_minimum_pairs =
      NonNegative(command.linkage_same_layer_minimum_pairs);
  const std::uint32_t same_layer_window_maximum_pairs =
      command.linkage_same_layer_window_maximum_pairs > 0
      ? static_cast<std::uint32_t>(
            command.linkage_same_layer_window_maximum_pairs)
      // DEV ordinary falls back to the ordinary many-node cap.  The
      // root-only implementation uses an unbounded int; PP's admitted
      // evaluator contract has an equivalent 32,768-pair hard ceiling.
      : initialization ? kMaximumEvaluationPairs : maximum_batch_pairs;
  const std::uint32_t tiny_pair_limit =
      NonNegative(command.linkage_tiny_forced_pair_limit);
  return {
      .inner_batch_rows = PositiveOrOne(command.linkage_inner_batch_rows),
      .maximum_batch_pairs = maximum_batch_pairs,
      .many_nodes_preflush_target_pairs =
          NonNegative(command.linkage_many_nodes_preflush_target_pairs),
      .same_layer_block_rows = same_layer_block_rows,
      .same_layer_minimum_batch_pairs = same_layer_minimum_pairs,
      .same_layer_window_maximum_pairs =
          same_layer_window_maximum_pairs,
      .same_layer_maximum_pending_nodes =
          command.linkage_same_layer_maximum_pending_nodes > 0
          ? static_cast<std::uint32_t>(
                command.linkage_same_layer_maximum_pending_nodes)
          : same_layer_block_rows,
      .same_layer_preflush_target_pairs =
          command.linkage_same_layer_preflush_target_pairs,
      .same_layer_preflush_target_nodes =
          NonNegative(command.linkage_same_layer_preflush_target_nodes),
      .tiny_forced_cpu_enabled =
          (initialization
               ? command.initialization_linkage_tiny_forced_cpu_enabled
               : command.linkage_tiny_forced_cpu_enabled) &&
          tiny_pair_limit != 0U,
      .tiny_forced_pair_limit = tiny_pair_limit,
      .tiny_forced_node_limit =
          NonNegative(command.linkage_tiny_forced_node_limit),
      .single_gpu_route_enabled = command.use_cuda && !initialization,
      .single_gpu_scheduler_backed = command.linkage_scheduler_enabled,
      .single_gpu_min_candidates =
          command.linkage_single_gpu_min_candidates,
      .single_gpu_max_candidates =
          command.linkage_single_gpu_max_candidates,
      .same_layer_single_gpu_min_candidates =
          command.linkage_same_layer_single_gpu_min_candidates,
      .bad_frozen_window_rows =
          PositiveOrOne(command.linkage_bad_window_rows),
      .bad_frozen_maximum_pairs =
          PositiveOrOne(command.linkage_bad_window_maximum_pairs),
  };
}

std::uint32_t CompleteRowBound(
    std::uint32_t flush_target,
    std::uint64_t maximum_row_pairs) noexcept {
  if (flush_target == 0U || maximum_row_pairs == 0U) return 0U;
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(
      kMaximumEvaluationPairs,
      static_cast<std::uint64_t>(flush_target - 1U) +
          maximum_row_pairs));
}

std::uint32_t RadialEvaluationBound(
    std::uint32_t depth_k,
    std::uint32_t knn_k,
    const CausalTopologyBatching& batching) noexcept {
  const std::uint64_t maximum_inner_row_pairs =
      static_cast<std::uint64_t>(depth_k) * knn_k;
  return std::max(
      CompleteRowBound(
          batching.maximum_batch_pairs, maximum_inner_row_pairs),
      batching.same_layer_window_maximum_pairs);
}

FullTopologyPolicy ComposeTopology(
    const TopologyCommand& topology,
    const VirtualRootCommand& virtual_roots,
    std::uint32_t virtual_root_seed,
    LeastSquaresCostMode least_squares_cost_mode,
    CausalTopologyBatching batching,
    const BuildCommand& command) noexcept {
  return {
      .radial_layer_count = topology.radial_layer_count,
      .root_fraction = topology.root_fraction,
      .depth_k = topology.depth_k,
      .knn_k = topology.knn_k,
      .maximum_depth = topology.maximum_depth,
      .seed = topology.seed,
      .rsd_rank = topology.rsd_rank,
      .icm_rounds = topology.icm_rounds,
      .least_squares_cost_mode = least_squares_cost_mode,
      .virtual_root_seed = virtual_root_seed,
      .batching = batching,
      .virtual_enabled = virtual_roots.enabled,
      .virtual_ratio = virtual_roots.ratio,
      .minimum_virtual_roots = virtual_roots.minimum,
      .maximum_virtual_roots = virtual_roots.maximum,
      .use_fixed_virtual_roots = virtual_roots.use_fixed_per_cluster,
      .fixed_virtual_roots = virtual_roots.fixed_per_cluster,
      .cluster_knn = {
          .backend = topology.bad_cluster_knn_backend == "auto" ? BadClusterKnnBackend::kAuto
              : topology.bad_cluster_knn_backend == "hnsw" ? BadClusterKnnBackend::kHnsw
              : BadClusterKnnBackend::kCpuExact,
          .cpu_exact_max_points = topology.bad_cluster_cpu_exact_max_points,
          .hnsw_m = command.hnsw_m,
          .hnsw_ef_construction_cap = command.hnsw_ef_construction_cap,
          .hnsw_ef_search = command.hnsw_ef_search,
      },
      .virtual_policy = {
          .neighbor_count = virtual_roots.neighbor_count,
          .local_connectivity = virtual_roots.local_connectivity,
          .overlap_threshold = virtual_roots.overlap_threshold,
          .prefer_density_peaks = virtual_roots.prefer_density_peaks,
          .anchor_neighbor_k = virtual_roots.anchor_neighbor_k,
      },
  };
}


VirtualRootRecipe HashVirtualRoots(
    const FullTopologyPolicy& topology) noexcept {
  return {
      .enabled = topology.virtual_enabled,
      .target_ratio = topology.virtual_ratio,
      .minimum_per_cluster = topology.minimum_virtual_roots,
      .maximum_per_cluster = topology.maximum_virtual_roots,
      .use_fixed_per_cluster = topology.use_fixed_virtual_roots,
      .fixed_per_cluster = topology.fixed_virtual_roots,
      .umap_knn_k = topology.virtual_policy.neighbor_count,
      .local_connectivity = topology.virtual_policy.local_connectivity,
      .overlap_threshold = topology.virtual_policy.overlap_threshold,
      .prefer_density_peaks = topology.virtual_policy.prefer_density_peaks,
      .anchor_neighbor_k = topology.virtual_policy.anchor_neighbor_k,
  };
}

}  // namespace

std::uint32_t MaximumProducedEvaluationPairs(
    const RootOnlyInitializationTopologyPolicy& topology) noexcept {
  return RadialEvaluationBound(
      topology.depth_k, topology.knn_k, topology.batching);
}

std::uint32_t MaximumProducedEvaluationPairs(
    const FullTopologyPolicy& topology) noexcept {
  const std::uint32_t radial = RadialEvaluationBound(
      topology.depth_k, topology.knn_k, topology.batching);
  const std::uint64_t maximum_bad_row_pairs =
      2ULL * topology.knn_k;
  const std::uint32_t bad = CompleteRowBound(
      topology.batching.bad_frozen_maximum_pairs,
      maximum_bad_row_pairs);
  return std::max(radial, bad);
}

StatusOr<IndexBuildArtifactHashes> ComputeEvalArtifactHashes(
    BuildCommand command, std::uint32_t default_cpu_threads) {
  // Reuse the build's identity calculation, but perform no training or base
  // encoding. The explicit train input can name its manifest or directory.
  if (command.train_checkpoint_file.empty()) {
    return stlqpp::Status::InvalidArgument(
        "eval-only without io.linkage_file requires io.train_file");
  }
  if (command.train_checkpoint_file.filename() == "checkpoint.stlqpp" ||
      command.train_checkpoint_file.filename() == "train_manifest.stlqpp" ||
      command.train_checkpoint_file.filename() == "trained_model.stlqpp" ||
      command.train_checkpoint_file.filename() == "model.bin") {
    command.train_checkpoint_file =
        command.train_checkpoint_file.parent_path();
  }
  command.eval_only = false;
  command.train_enabled = false;
  command.train_initialization_enabled = false;
  command.base_encode_enabled = false;
  command.base_linkage_enabled = false;
  if (const stlqpp::Status status =
          stlqpp::app::ResolveBuildInputMetadata(&command);
      !status.ok()) return status;
  stlqpp::app::ResolveBuildRuntimeThreadCounts(
      &command, default_cpu_threads);
  auto recipe = stlqpp::app::ComposeEffectiveBuildRecipe(command);
  if (!recipe.ok()) return recipe.status();
  const bool scheduler = command.linkage_scheduler_enabled;
  recipe.value().base_topology.batching.single_gpu_scheduler_backed =
      scheduler;
  const bool merged = command.use_cuda && scheduler &&
      command.linkage_scheduler_transport_batching_enabled &&
      !command.linkage_scheduler_stable;
  const auto numerical_policy = merged
      ? stlqpp::LinkageNumericalPolicy::kMergedThroughput
      : stlqpp::LinkageNumericalPolicy::kSourceTask;
  return [&]()
      -> stlqpp::StatusOr<stlqpp::IndexBuildArtifactHashes> {
    std::optional<stlqpp::TrainedModelFile> artifact;
    stlqpp::ReadOnlyMappedFile model_file;
    stlqpp::ReadOnlyMappedFile mask_file;
    stlqpp::ArrayView<const std::uint8_t> model_bytes;
    stlqpp::ArrayView<const std::uint8_t> mask_bytes;
    if (command.train_checkpoint_file.extension() == ".stlqmodel") {
      auto opened = stlqpp::TrainedModelFile::Open(
          command.train_checkpoint_file);
      if (!opened.ok()) return opened.status();
      artifact.emplace(std::move(opened).value());
      model_bytes = artifact->model_bytes();
      mask_bytes = artifact->bad_cluster_mask();
    } else {
      auto opened_model = stlqpp::ReadOnlyMappedFile::Open(
          command.train_checkpoint_file / "model.bin");
      if (!opened_model.ok()) return opened_model.status();
      auto opened_mask = stlqpp::ReadOnlyMappedFile::Open(
          command.train_checkpoint_file / "bad_clusters.bin");
      if (!opened_mask.ok()) return opened_mask.status();
      model_file = std::move(opened_model).value();
      mask_file = std::move(opened_mask).value();
      model_bytes = {model_file.data(), model_file.size()};
      mask_bytes = {mask_file.data(), mask_file.size()};
    }
    auto canonical = stlqpp::CanonicalModelCodec::DecodeRuntime(
        model_bytes.data(), model_bytes.size());
    if (!canonical.ok()) return canonical.status();
    const auto& model = canonical.value();
    if (model.source_space())
      return stlqpp::Status::InvalidArgument("io.train_file requires a training model with its rotation");
    if (model.shape().m != command.m ||
        model.shape().dimension != command.dimension ||
        model.shape().nlist != command.nlist ||
        mask_bytes.size() != command.nlist ||
        !std::all_of(mask_bytes.begin(), mask_bytes.end(),
                     [](std::uint8_t value) { return value <= 1U; })) {
      return stlqpp::Status::DataLoss(
          "io.train_file model or bad-cluster mask disagrees with its "
          "checkpoint header");
    }
    const stlqpp::BasicSchedule schedule =
        stlqpp::BasicSchedule::Configured(
            stlqpp::BasicScoreRule::kAbsolute,
            command.base_basic_icm_iterations,
            command.base_basic_ils_iterations,
            command.base_basic_perturbations,
            command.least_squares_cost_mode);
    stlqpp::BaseBasicStageOptions basic;
    basic.cids_per_bucket = stlqpp::app::EffectiveCidsPerBucket(command);
    basic.seed = command.base_seed;
    basic.schedule = schedule;
    stlqpp::IndexArtifactRecipe artifacts;
    artifacts.dataset_name = command.dataset_name;
    artifacts.base_source_first_row = command.base_first_row;
    artifacts.base_source_count = command.base_rows;
    artifacts.base_source_block_rows =
        stlqpp::app::EffectiveBaseBlockRows(command);
    artifacts.base_basic_rsd_rank = command.base_basic_rsd_rank;
    artifacts.basic_executor = !command.use_cuda
        ? stlqpp::BasicExecutorKind::kCpuOnly
        : stlqpp::BasicExecutorKind::kCudaOnly;
    artifacts.cuda_allow_tf32 = command.use_cuda &&
        command.cuda_allow_tf32;
    artifacts.candidate_search = recipe.value().base_candidate_search;
    artifacts.linkage_numerical_policy = numerical_policy;
    stlqpp::FinalIndexBuildOptions final;
    final.topology = recipe.value().base_topology;
    final.coefficient_codec = recipe.value().coefficient_codec;
    final.norm8_mode = recipe.value().norm8_mode;
    final.norm8_lloyd_iterations = recipe.value().norm8_lloyd_iterations;
    final.publish_quantized_forest = command.coefficient_codec_enabled;
    stlqpp::TrainedModelHashView hash_view;
    hash_view.rotation = model.rotation();
    hash_view.root_codebooks.push_back(model.root0());
    for (std::uint32_t layer = 1U; layer < model.shape().m; ++layer)
      hash_view.root_codebooks.push_back(model.root_small(layer));
    for (std::uint32_t layer = 0U; layer < model.shape().m; ++layer)
      hash_view.edge_codebooks.push_back(model.edge(layer));
    return stlqpp::ComputeIndexBuildArtifactHashesRuntime(
        model.shape(), hash_view, mask_bytes,
        command.effective_base_format ==
            stlqpp::app::SourceFormat::kBvecsU8,
        basic, artifacts, final);
  }();
}


StatusOr<EffectiveBuildRecipe> ComposeEffectiveBuildRecipe(
    const BuildCommand& command) {
  EffectiveBuildRecipe output;
  const EffectiveBuildExecutionConfig execution =
      ResolveBuildExecutionConfig(command);
  output.shape = {
      command.dimension, command.m, command.nlist, 256U};
  if (const Status status = output.shape.Validate(); !status.ok()) {
    return status;
  }
  const std::uint64_t effective_initialization_samples =
      std::min(command.initialization_samples, command.training_rows);
  if (effective_initialization_samples < command.nlist ||
      effective_initialization_samples >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Status::InvalidArgument(
        "effective root-initialization sample count is outside the "
        "registered CUDA k-means domain");
  }
  output.root_initialization.sample_vectors =
      effective_initialization_samples;
  output.root_initialization.seed_method =
      command.root_initialization.seed_method == "random"
      ? SphericalSeedMethod::kRandom
      : command.root_initialization.seed_method == "kmeans++"
          ? SphericalSeedMethod::kKmeansPlusPlus
          : SphericalSeedMethod::kKmeansParallel;
  output.root_initialization.maximum_iterations =
      static_cast<std::uint32_t>(
          std::max(1, command.root_initialization.maximum_iterations));
  output.root_initialization.tolerance =
      static_cast<float>(command.root_initialization.tolerance);
  output.root_initialization.kmeans_parallel_rounds =
      static_cast<std::uint32_t>(std::clamp(
          command.root_initialization.kmeans_parallel_rounds, 1, 16));
  output.root_initialization.kmeans_parallel_oversample =
      static_cast<float>(std::clamp(
          command.root_initialization.kmeans_parallel_oversample,
          1.0, 16.0));
  output.root_initialization.kmeans_parallel_seed =
      command.root_initialization.kmeans_parallel_seed;
  // Auto is resolved per RVQ layer, not from the IVF/root cardinality.
  output.root_initialization.candidate_cap =
      command.root_initialization.candidate_cap <= 0
          ? (command.nlist == 256U
                 ? static_cast<std::uint32_t>(std::min<std::uint64_t>(
                       effective_initialization_samples, 4096U))
                 : 0U)
          : static_cast<std::uint32_t>(std::min<std::uint64_t>(
                effective_initialization_samples,
                command.root_initialization.candidate_cap));
  output.root_initialization.initial_weight =
      static_cast<float>(command.root_initialization.initial_weight);
  output.root_initialization.minimum_weight =
      static_cast<float>(command.root_initialization.minimum_weight);
  output.root_initialization.outlier_quantile =
      static_cast<float>(command.root_initialization.outlier_quantile);
  output.root_initialization.cost_threshold =
      static_cast<float>(command.root_initialization.cost_threshold);
  output.root_initialization.annealing_factor =
      static_cast<float>(command.root_initialization.annealing_factor);
  output.root_initialization.warmup_iterations =
      static_cast<std::uint32_t>(
          std::max(0, command.root_initialization.warmup_iterations));
  output.root_initialization.quantile_bins =
      static_cast<std::uint32_t>(
          std::max(1, command.root_initialization.quantile_bins));
  output.root_initialization.large_layer_threshold =
      static_cast<std::uint32_t>(
          std::max(1, command.root_initialization.large_layer_threshold));
  output.root_initialization.large_layer_top_coarse =
      static_cast<std::uint32_t>(std::clamp(
          command.root_initialization.large_layer_top_coarse, 1, 2));
  output.root_initialization.large_layer_hierarchy_enabled =
      command.root_initialization.large_layer_hierarchy_enabled;
  output.root_initialization.large_layer_fixed_coarse_count =
      static_cast<std::uint32_t>(
          command.root_initialization.large_layer_fixed_coarse_count);
  output.root_initialization.large_layer_minimum_coarse_count =
      static_cast<std::uint32_t>(
          command.root_initialization.large_layer_minimum_coarse_count);
  output.root_initialization.large_layer_maximum_coarse_count =
      static_cast<std::uint32_t>(
          command.root_initialization.large_layer_maximum_coarse_count);
  output.root_initialization.large_layer_round_coarse_to_power_of_two =
      command.root_initialization.large_layer_round_coarse_to_power_of_two;
  if (const Status status = output.root_initialization.Validate();
      !status.ok()) {
    return status;
  }
  output.train_seed = command.train_seed;
  output.additional_training_round_count = command.training_rounds;
  output.train_use_opq_rotation = command.train_use_opq_rotation;
  output.bad_cluster_training_weight = static_cast<float>(
      command.bad_cluster_training_weight);
  const auto candidate_recipe = [&](const TopologyCommand& topology) {
    auto recipe = RegisteredCandidateSearchRecipe();
    recipe.bad_graph = topology.bad_cluster_knn_backend == "auto" ? BadClusterGraphRecipe::kAutoHnsw
        : topology.bad_cluster_knn_backend == "hnsw" ? BadClusterGraphRecipe::kHnsw : BadClusterGraphRecipe::kExactOnly;
    recipe.cpu_exact_max_points = topology.bad_cluster_cpu_exact_max_points;
    recipe.hnsw_m = command.hnsw_m;
    recipe.hnsw_ef_construction_cap = command.hnsw_ef_construction_cap;
    recipe.hnsw_ef_search = command.hnsw_ef_search;
    return recipe;
  };
  output.training_candidate_search = candidate_recipe(command.training_topology);
  output.base_candidate_search = candidate_recipe(command.base_topology);
  output.coefficient_codec = RegisteredCoefficientCodecRecipe(
      command.coefficient_first_magnitude_quantiles,
      command.coefficient_rest_magnitude_quantiles);
  output.coefficient_codec.bits_per_layer =
      command.coefficient_bits_per_layer;
  output.coefficient_codec.bits_per_layer_root =
      command.coefficient_bits_per_layer_root;
  output.coefficient_codec.bits_per_layer_linkage =
      command.coefficient_bits_per_layer_linkage;
  output.coefficient_codec.weighted_quantile =
      command.coefficient_weighted_quantile;
  output.coefficient_codec.allow_clipping =
      command.coefficient_allow_clipping;
  output.coefficient_codec.fit_reconstruction_scales =
      command.coefficient_fit_scale;
  output.coefficient_codec.refinement_sweeps =
      command.coefficient_refinement_sweeps;
  output.coefficient_codec.refine_all_layers =
      command.coefficient_refinement_max_layer == 0U;
  output.coefficient_codec.refinement_max_layer =
      command.coefficient_refinement_max_layer;
  output.coefficient_codec.refinement_step_limit =
      command.coefficient_refinement_step_limit;
  output.norm8_mode = PublishedNorm8Mode(command.norm8_mode);
  output.norm8_lloyd_iterations = command.norm8_lloyd_iterations;
  output.least_squares_cost_mode = command.least_squares_cost_mode;
  output.initialization_workspace = {
      .large_root_xc0_chunk_mebibytes =
          execution.initialization_large_root_xc0_chunk_mebibytes,
      .large_root_workspace_mebibytes =
          execution.initialization_large_root_workspace_mebibytes,
  };
  output.update_reduction = {
      .root_moment_partitions = command.effective_root_moment_partitions,
      .rotation_moment_partitions =
          command.effective_rotation_moment_partitions,
      .initialization_edge_moment_shards =
          command.effective_initialization_edge_moment_shards,
      .training_edge_moment_shards =
          command.effective_training_edge_moment_shards,
      .root_spd_threads = SelectLargeSpdThreadCount(
          (command.m - 1U) * 256U, command.dimension,
          command.effective_cpu_threads),
      .edge_spd_threads = SelectLargeSpdThreadCount(
          command.m * 256U, command.dimension,
          command.effective_cpu_threads),
  };
  output.initialization_topology = {
      .radial_layer_count =
          command.training_topology.radial_layer_count,
      .root_fraction = command.training_topology.root_fraction,
      .depth_k = command.training_topology.depth_k,
      .knn_k = command.training_topology.knn_k,
      .maximum_depth = command.training_topology.maximum_depth,
      .seed = command.training_topology.seed,
      .icm_rounds = command.init_linkage_icm_rounds >= 0
          ? static_cast<std::uint32_t>(command.init_linkage_icm_rounds)
          : command.base_topology.icm_rounds,
      .shared_cluster_seed = false,
      .batching = ResolveTopologyBatching(
          command, execution.initialization_maximum_batch_pairs, true),
  };
  const CausalTopologyBatching batching = ResolveTopologyBatching(
      command, command.linkage_max_batch_pairs, false);
  output.training_topology = ComposeTopology(
      command.training_topology, command.virtual_roots,
      command.train_seed, command.least_squares_cost_mode, batching, command);
  output.base_topology = ComposeTopology(
      command.base_topology, command.virtual_roots,
      command.base_seed, command.least_squares_cost_mode, batching, command);
  return output;
}

TrainResumeHashInput MakeTrainResumeHashInput(
    const BuildCommand& command,
    const EffectiveBuildRecipe& recipe,
    std::uint32_t training_basic_rsd_rank,
    LinkageNumericalPolicy linkage_numerical_policy) {
  return {
      .shape = recipe.shape,
      .train_seed = recipe.train_seed,
      .fast_initialization = command.init_linkage_mode == "fast_init",
      .root_initialization = {
          .seed_method = static_cast<std::uint8_t>(
              recipe.root_initialization.seed_method),
          .sample_vectors = recipe.root_initialization.sample_vectors,
          .maximum_iterations =
              recipe.root_initialization.maximum_iterations,
          .tolerance = recipe.root_initialization.tolerance,
          .kmeans_parallel_rounds =
              recipe.root_initialization.kmeans_parallel_rounds,
          .kmeans_parallel_oversample =
              recipe.root_initialization.kmeans_parallel_oversample,
          .kmeans_parallel_seed =
              recipe.root_initialization.kmeans_parallel_seed,
          .candidate_cap = recipe.root_initialization.candidate_cap,
          .initial_weight = recipe.root_initialization.initial_weight,
          .minimum_weight = recipe.root_initialization.minimum_weight,
          .outlier_quantile = recipe.root_initialization.outlier_quantile,
          .cost_threshold = recipe.root_initialization.cost_threshold,
          .annealing_factor = recipe.root_initialization.annealing_factor,
          .warmup_iterations = recipe.root_initialization.warmup_iterations,
          .quantile_bins = recipe.root_initialization.quantile_bins,
          .large_layer_threshold =
              recipe.root_initialization.large_layer_threshold,
          .large_layer_top_coarse =
              recipe.root_initialization.large_layer_top_coarse,
          .large_layer_hierarchy_enabled =
              recipe.root_initialization.large_layer_hierarchy_enabled,
          .large_layer_fixed_coarse_count =
              recipe.root_initialization.large_layer_fixed_coarse_count,
          .large_layer_minimum_coarse_count =
              recipe.root_initialization.large_layer_minimum_coarse_count,
          .large_layer_maximum_coarse_count =
              recipe.root_initialization.large_layer_maximum_coarse_count,
          .large_layer_round_coarse_to_power_of_two =
              recipe.root_initialization.large_layer_round_coarse_to_power_of_two,
      },
      .good_cluster_fraction = command.good_cluster_fraction,
      .initialization_topology = {
          .radial_layer_count =
              recipe.initialization_topology.radial_layer_count,
          .root_fraction = recipe.initialization_topology.root_fraction,
          .depth_k = recipe.initialization_topology.depth_k,
          .knn_k = recipe.initialization_topology.knn_k,
          .maximum_depth = recipe.initialization_topology.maximum_depth,
          .seed = recipe.initialization_topology.seed,
          .icm_rounds = recipe.initialization_topology.icm_rounds,
          .shared_cluster_seed =
              recipe.initialization_topology.shared_cluster_seed,
          .inner_batch_rows =
              recipe.initialization_topology.batching.inner_batch_rows,
          .maximum_batch_pairs =
              recipe.initialization_topology.batching.maximum_batch_pairs,
          .many_nodes_preflush_target_pairs =
              recipe.initialization_topology.batching
                  .many_nodes_preflush_target_pairs,
          .same_layer_block_rows =
              recipe.initialization_topology.batching
                  .same_layer_block_rows,
          .same_layer_minimum_batch_pairs =
              recipe.initialization_topology.batching
                  .same_layer_minimum_batch_pairs,
          .same_layer_window_maximum_pairs =
              recipe.initialization_topology.batching
                  .same_layer_window_maximum_pairs,
          .same_layer_maximum_pending_nodes =
              recipe.initialization_topology.batching
                  .same_layer_maximum_pending_nodes,
          .same_layer_preflush_target_pairs =
              recipe.initialization_topology.batching
                  .same_layer_preflush_target_pairs,
          .same_layer_preflush_target_nodes =
              recipe.initialization_topology.batching
                  .same_layer_preflush_target_nodes,
          .tiny_forced_cpu_enabled =
              recipe.initialization_topology.batching
                  .tiny_forced_cpu_enabled,
          .tiny_forced_pair_limit =
              recipe.initialization_topology.batching
                  .tiny_forced_pair_limit,
          .tiny_forced_node_limit =
              recipe.initialization_topology.batching
                  .tiny_forced_node_limit,
          .large_root_xc0_chunk_mebibytes =
              recipe.initialization_workspace
                  .large_root_xc0_chunk_mebibytes,
          .large_root_workspace_mebibytes =
              recipe.initialization_workspace
                  .large_root_workspace_mebibytes,
      },
      .root_moment_partitions =
          recipe.update_reduction.root_moment_partitions,
      .rotation_moment_partitions =
          recipe.update_reduction.rotation_moment_partitions,
      .initialization_edge_moment_shards =
          recipe.update_reduction.initialization_edge_moment_shards,
      .training_edge_moment_shards =
          recipe.update_reduction.training_edge_moment_shards,
      .root_spd_threads = recipe.update_reduction.root_spd_threads,
      .edge_spd_threads = recipe.update_reduction.edge_spd_threads,
      .source_block_rows = EffectiveTrainingBlockRows(command),
      .basic_encoder = {
          .rsd_rank = training_basic_rsd_rank,
          .ils_iterations = command.train_basic_ils_iterations,
          .icm_iterations = command.train_basic_icm_iterations,
          .perturb_layers = command.train_basic_perturbations,
          .norm_bins = 256U,
          .seed = command.train_seed,
          .least_squares_cost_mode = command.least_squares_cost_mode,
      },
      .basic_execution = {
          .executor = !command.use_cuda
              ? BasicExecutorKind::kCpuOnly
              : BasicExecutorKind::kCudaOnly,
          .cuda_allow_tf32 = command.cuda_allow_tf32,
      },
      // max_R_iters is an invocation control. It cannot invalidate the
      // checkpoint identity that it is intended to extend.
      .training_round_count = 0U,
      .update_opq_rotation = recipe.train_use_opq_rotation,
      .topology = MakeTopologyHashRecipe(
          recipe.training_topology, command.cuda_allow_tf32,
          linkage_numerical_policy),
      .virtual_roots = HashVirtualRoots(recipe.training_topology),
      .bad_cluster_training_weight = recipe.bad_cluster_training_weight,
      .candidate_search = recipe.training_candidate_search,
  };
}

StatusOr<TrainingCheckpointIdentity> ComputeTrainingIdentity(
    const BuildCommand& command, EffectiveBuildRecipe recipe,
    bool scheduler_enabled,
    LinkageNumericalPolicy linkage_numerical_policy) {
  // An explicit, read-only model is an input, not a claim that the current
  // training recipe produced it. Continuation and implicit lookup stay strict.
  const bool executes_training = command.train_enabled &&
      (command.train_initialization_enabled ||
       recipe.additional_training_round_count != 0U);
  if (!executes_training &&
      command.train_checkpoint_file.extension() == ".stlqmodel") {
    auto artifact = TrainedModelFile::Open(command.train_checkpoint_file);
    if (!artifact.ok()) return artifact.status();
    return TrainingCheckpointIdentity{
        .training_source = ArtifactHash64::FromValue(
            artifact.value().header().training_source),
        .training_recipe = ArtifactHash64::FromValue(
            artifact.value().header().training_recipe),
    };
  }
  if (command.dimension == 0U ||
      command.effective_training_format == SourceFormat::kAuto) {
    return Status::InvalidArgument(
        "training model identity requires resolved training source metadata");
  }
  recipe.training_topology.batching.single_gpu_scheduler_backed =
      scheduler_enabled;
  const auto hash_input = MakeTrainResumeHashInput(
      command, recipe, command.training_basic_rsd_rank, linkage_numerical_policy);
  TrainingCheckpointIdentity identity{
      .training_source = ArtifactHashFactory::TrainingSourceSlice({
          .registered_source = ArtifactHashFactory::RegisteredTrainingSource(
              command.dataset_name),
          .scalar = command.effective_training_format == SourceFormat::kBvecsU8
              ? SourceScalarKind::kU8 : SourceScalarKind::kF32,
          .dimension = command.dimension,
          .first_vector = command.training_first_row,
          .vector_count = command.training_rows,
      }),
      .training_recipe = ArtifactHashFactory::TrainResume(hash_input),
  };
  return identity;
}

struct ReadOnlyTrainingModelIdentities final {
  TrainingCheckpointIdentity current;
  std::array<ArtifactHash64, 6U> recipes;

  bool Matches(const TrainedModelFileHeader& header) const noexcept {
    return header.training_source == current.training_source.value() &&
        std::any_of(recipes.begin(), recipes.end(), [&](ArtifactHash64 recipe) {
          return header.training_recipe == recipe.value();
        });
  }
};

static StatusOr<ReadOnlyTrainingModelIdentities>
ImplicitTrainingIdentities(const BuildCommand& command,
                           std::uint32_t default_cpu_threads) {
  BuildCommand identity_command = command;
  identity_command.eval_only = false;
  identity_command.train_enabled = true;
  identity_command.train_initialization_enabled = true;
  identity_command.base_encode_enabled = false;
  identity_command.base_linkage_enabled = false;
  identity_command.resume_initialized_checkpoint.clear();
  identity_command.resume_round_checkpoint.clear();
  if (const Status status = ResolveBuildInputMetadata(&identity_command);
      !status.ok()) return status;
  ResolveBuildRuntimeThreadCounts(
      &identity_command, default_cpu_threads);
  auto recipe = ComposeEffectiveBuildRecipe(identity_command);
  if (!recipe.ok()) return recipe.status();
  const bool scheduler = identity_command.linkage_scheduler_enabled;
  const bool merged = identity_command.use_cuda && scheduler &&
      identity_command.linkage_scheduler_transport_batching_enabled &&
      !identity_command.linkage_scheduler_stable;
  const auto numerical_policy = merged
      ? LinkageNumericalPolicy::kMergedThroughput
      : LinkageNumericalPolicy::kSourceTask;
  recipe.value().training_topology.batching.single_gpu_scheduler_backed =
      scheduler;
  const auto hash_input = MakeTrainResumeHashInput(
      identity_command, recipe.value(),
      identity_command.training_basic_rsd_rank, numerical_policy);
  auto identity = ComputeTrainingIdentity(
      identity_command, std::move(recipe).value(), scheduler, numerical_policy);
  if (!identity.ok()) return identity.status();
  return ReadOnlyTrainingModelIdentities{
      identity.value(),
      {identity.value().training_recipe,
       SharedCpuTailIcmTrainingModelIdentity(hash_input),
       PrePositivePivotTrainingModelIdentity(hash_input),
       PreCpuTailRsdTrainingModelIdentity(hash_input),
       PreviousRsdTrainingModelIdentity(hash_input),
       LegacyRsdTrainingModelIdentity(hash_input)}};
}

StatusOr<std::filesystem::path> ResolveImplicitTrainModel(
    const BuildCommand& command, std::uint32_t default_cpu_threads) {
  if (!command.train_checkpoint_file.empty()) {
    return command.train_checkpoint_file;
  }
  if (command.training_file.empty()) {
    return Status::InvalidArgument(
        "automatic io.train_file lookup requires dataset.train_path in the "
        "complete build configuration");
  }
  auto identities = ImplicitTrainingIdentities(command, default_cpu_threads);
  if (!identities.ok()) return identities.status();
  const auto source = identities.value().current.training_source;
  // Preserve current/previous/legacy priority and first-publication ordering.
  // These identities only locate read-only models, never recovery checkpoints.
  auto selected = StatusOr<std::filesystem::path>(Status::FailedPrecondition(
      "no matching trained model"));
  for (const auto recipe : identities.value().recipes) {
    selected = FindFirstTrainedModelPath(
        TrainingModelPublicationBase(command, source, recipe),
        source.value(), recipe.value());
    if (selected.ok() || selected.status().code() != StatusCode::kFailedPrecondition)
      return selected;
  }
  return selected;
}

StatusOr<std::optional<CompletedEvalRun>> FindCompletedEvalRun(
    const BuildCommand& command, std::uint32_t default_cpu_threads) {
  std::optional<CompletedEvalRun> selected;
  if (!command.train_checkpoint_file.empty()) return selected;
  const auto parent = command.output_directory / command.dataset_name;
  std::error_code error;
  if (!std::filesystem::exists(parent, error)) {
    if (error) return Status::Internal("cannot inspect eval output: " + error.message());
    return selected;
  }
  const std::string prefix = command.output_prefix + "__0x";
  bool completed_linkage_found = false;
  std::optional<ReadOnlyTrainingModelIdentities> identities;
  for (std::filesystem::directory_iterator it(parent, error), end;
       !error && it != end; it.increment(error)) {
    const auto root = it->path();
    if (!root.filename().string().starts_with(prefix)) continue;
    const auto state_path = root / "run_state.txt";
    if (!std::filesystem::exists(state_path, error)) {
      if (error) break;
      continue;
    }
    std::ifstream state(state_path);
    if (!state) return Status::Internal("cannot read " + state_path.string());
    std::string phase;
    std::filesystem::path recorded_index;
    std::filesystem::path recorded_model;
    std::string line;
    while (std::getline(state, line)) {
      const auto equals = line.find('=');
      if (equals == std::string::npos) continue;
      const auto key = TrimConfigText(std::string_view(line).substr(0U, equals));
      const auto value = TrimConfigText(std::string_view(line).substr(equals + 1U));
      if (key == "state") phase = value;
      else if (key == "index") recorded_index = value;
      else if (key == "trained_model") recorded_model = value;
    }
    if (state.bad()) return Status::Internal("cannot read " + state_path.string());
    // Running/failed builds and training-only runs are not eval authorities.
    if (phase != "complete" || recorded_index.empty()) continue;
    completed_linkage_found = true;
    if (recorded_model.empty()) {
      return Status::DataLoss("completed eval run lacks trained_model: " + state_path.string());
    }
    // Rebase the index within the discovered run, even if its old absolute or
    // cwd-relative location still exists after an archive was copied/moved.
    const auto index = root / recorded_index.filename();
    auto mapped = ReadOnlyMappedFile::Open(index / "manifest.stlqpp");
    if (!mapped.ok()) return mapped.status();
    IndexManifest manifest;
    if (const Status status = IndexManifestCodec::Decode(
            mapped.value().data(), mapped.value().size(), manifest);
        !status.ok()) return status;
    if (manifest.shape.m != command.m || manifest.shape.nlist != command.nlist ||
        (command.dimension != 0U && manifest.shape.dimension != command.dimension)) {
      continue;
    }
    std::filesystem::path model;
    for (const auto& shard : manifest.shards) {
      if (shard.kind == ShardKind::kCodebooks && !shard.reference_path.empty()) {
        model = (index / shard.reference_path).lexically_normal();
        break;
      }
    }
    if (model.empty()) {
      model = parent / "models" / recorded_model.filename();
      if (!std::filesystem::exists(model, error)) {
        if (error) break;
        model = recorded_model;
      }
    }
    if (model.filename() != recorded_model.filename()) {
      return Status::DataLoss("run_state trained_model differs from index model reference: " +
                              state_path.string());
    }
    if (!identities) {
      auto resolved = ImplicitTrainingIdentities(command, default_cpu_threads);
      if (!resolved.ok()) return resolved.status();
      identities = resolved.value();
    }
    auto artifact = TrainedModelFile::Open(model);
    if (!artifact.ok()) return artifact.status();
    const auto& header = artifact.value().header();
    if (!identities->Matches(header)) continue;
    BuildCommand candidate = command;
    candidate.train_checkpoint_file = model;
    auto hashes = ComputeEvalArtifactHashes(std::move(candidate), default_cpu_threads);
    if (!hashes.ok()) return hashes.status();
    if (hashes.value().index.linkage_list_directory !=
        manifest.hashes.linkage_list_directory) continue;
    if (selected.has_value()) {
      return Status::FailedPrecondition(
          "multiple completed runs match this eval recipe: " + selected->root.string() +
          " and " + root.string() + "; set io.linkage_file or io.train_file explicitly");
    }
    selected = CompletedEvalRun{root, model, hashes.value()};
  }
  if (error) return Status::Internal("cannot enumerate eval runs: " + error.message());
  if (!selected.has_value() && completed_linkage_found) {
    return Status::FailedPrecondition(
        "no completed run matches the current training identity and base/topology recipe under " + parent.string() +
        " (prefix " + command.output_prefix + "); set io.linkage_file explicitly");
  }
  return selected;
}

}  // namespace stlqpp::app
