#pragma once

#include "build_command.h"
#include "stlqpp/application/index_builder.h"

#include "stlqpp/storage/artifact_hash_factory.h"
#include "stlqpp/training/checkpoint/training_checkpoint.h"
#include "stlqpp/training/initialization/root_model_initializer.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_cluster_builder.h"
#include "stlqpp/training/linkage/topology/full_topology_cluster_builder.h"

#include <cstdint>
#include <optional>

namespace stlqpp::app {

struct RootOnlyInitializationWorkspaceRecipe final {
  std::uint32_t large_root_xc0_chunk_mebibytes = 0U;
  std::uint32_t large_root_workspace_mebibytes = 0U;
};

struct ModelUpdateReductionRecipe final {
  std::uint32_t root_moment_partitions = 1U;
  std::uint32_t rotation_moment_partitions = 0U;
  std::uint32_t initialization_edge_moment_shards = 0U;
  std::uint32_t training_edge_moment_shards = 0U;
  std::uint32_t root_spd_threads = 1U;
  std::uint32_t edge_spd_threads = 1U;
};

// Immutable semantic result of parsing one production build command. Paths,
// worker counts, device selection and queue capacities remain in BuildCommand
// as execution/location policy and cannot enter this type accidentally.
struct EffectiveBuildRecipe final {
  ModelShape shape;
  SphericalRvqInitializationRecipe root_initialization;
  std::uint32_t train_seed = 0U;
  // DEV train.max_R_iters: additional rounds requested by this invocation.
  std::uint32_t additional_training_round_count = 0U;
  bool train_use_opq_rotation = true;
  float bad_cluster_training_weight = 1.0F;
  CandidateSearchRecipe training_candidate_search;
  CandidateSearchRecipe base_candidate_search;
  CoefficientCodecRecipe coefficient_codec;
  ReconstructionNorm8Mode norm8_mode =
      ReconstructionNorm8Mode::kIdentityLloyd;
  std::uint32_t norm8_lloyd_iterations = 0U;
  LeastSquaresCostMode least_squares_cost_mode =
      LeastSquaresCostMode::kNormalEquation;
  RootOnlyInitializationWorkspaceRecipe initialization_workspace;
  ModelUpdateReductionRecipe update_reduction;
  RootOnlyInitializationTopologyPolicy initialization_topology;
  FullTopologyPolicy training_topology;
  FullTopologyPolicy base_topology;
};

[[nodiscard]] StatusOr<EffectiveBuildRecipe> ComposeEffectiveBuildRecipe(
    const BuildCommand& command);

// Exact upper bound implied by the current producer's append-then-flush
// semantics. It is an execution resource bound, not an algorithm parameter.
[[nodiscard]] std::uint32_t MaximumProducedEvaluationPairs(
    const RootOnlyInitializationTopologyPolicy& topology) noexcept;
[[nodiscard]] std::uint32_t MaximumProducedEvaluationPairs(
    const FullTopologyPolicy& topology) noexcept;

[[nodiscard]] TrainResumeHashInput MakeTrainResumeHashInput(
    const BuildCommand& command,
    const EffectiveBuildRecipe& recipe,
    std::uint32_t training_basic_rsd_rank,
    LinkageNumericalPolicy linkage_numerical_policy);

// Shared authority for publication and automatic model discovery. The source
// slice and training recipe are known before training; model contents are not.
[[nodiscard]] StatusOr<TrainingCheckpointIdentity> ComputeTrainingIdentity(
    const BuildCommand& command, EffectiveBuildRecipe recipe,
    bool scheduler_enabled,
    LinkageNumericalPolicy linkage_numerical_policy);

// Only for base-only and full-config eval-only flows. Explicit io.train_file
// always takes precedence; additional training rounds accept an explicit
// trained model or checkpoint and start by rebuilding Full topology.
[[nodiscard]] StatusOr<std::filesystem::path> ResolveImplicitTrainModel(
    const BuildCommand& command, std::uint32_t default_cpu_threads);

[[nodiscard]] StatusOr<IndexBuildArtifactHashes> ComputeEvalArtifactHashes(
    BuildCommand command, std::uint32_t default_cpu_threads);

struct CompletedEvalRun final {
  std::filesystem::path root;
  std::filesystem::path trained_model;
  IndexBuildArtifactHashes hashes;
};

// Eval-only discovery filters the current training identity among completed
// producer associations, then verifies the current base/topology recipe. An empty result means no completed linkage run
// exists for this prefix; only that case may use legacy model discovery.
[[nodiscard]] StatusOr<std::optional<CompletedEvalRun>> FindCompletedEvalRun(
    const BuildCommand& command, std::uint32_t default_cpu_threads);

[[nodiscard]] inline TrainResumeHashInput MakeTrainResumeHashInput(
    const BuildCommand& command,
    const EffectiveBuildRecipe& recipe,
    std::uint32_t training_basic_rsd_rank) {
  const bool merged = command.use_cuda &&
      command.linkage_scheduler_enabled &&
      command.linkage_scheduler_transport_batching_enabled &&
      !command.linkage_scheduler_stable;
  return MakeTrainResumeHashInput(
      command, recipe, training_basic_rsd_rank,
      merged ? LinkageNumericalPolicy::kMergedThroughput
             : LinkageNumericalPolicy::kSourceTask);
}

[[nodiscard]] inline TrainResumeHashInput MakeTrainResumeHashInput(
    const BuildCommand& command,
    const EffectiveBuildRecipe& recipe) {
  return MakeTrainResumeHashInput(
      command, recipe,
      command.training_basic_rsd_rank);
}

}  // namespace stlqpp::app
