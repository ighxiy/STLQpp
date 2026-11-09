#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <variant>

#include "stlqpp/pipeline/base_basic_stage.h"
#include "stlqpp/pipeline/final_full_topology_index_builder.h"
#include "stlqpp/pipeline/training_initializer.h"
#include "stlqpp/storage/artifact_hash_factory.h"
#include "stlqpp/training/initialization/root_model_initializer.h"
#include "stlqpp/training/trainer.h"

namespace stlqpp {

[[nodiscard]] LinkageTopologyRecipe MakeTopologyHashRecipe(
    const FullTopologyPolicy& topology, bool cuda_allow_tf32,
    LinkageNumericalPolicy numerical_policy) noexcept;


// Dataset and initialization semantics needed only to derive the registered
// final artifact identities. Runtime resources and paths do not enter this
// boundary; the trained model itself is hashed after the last round closes.
struct IndexArtifactRecipe final {
  std::string dataset_name;
  std::uint64_t base_source_first_row = 0U;
  std::uint64_t base_source_count = 0U;
  std::uint32_t base_source_block_rows = 0U;
  std::uint32_t base_basic_rsd_rank = 2U;
  BasicExecutorKind basic_executor = BasicExecutorKind::kCpuOnly;
  bool cuda_allow_tf32 = false;
  CandidateSearchRecipe candidate_search;
  LinkageNumericalPolicy linkage_numerical_policy =
      LinkageNumericalPolicy::kSourceTask;
};

// Product-level final-index options deliberately omit artifact hashes. The
// top-level builder derives them from the actual trained model and accepted
// recipe; only lower-level publication primitives accept explicit hashes.
struct FinalIndexBuildOptions final {
  std::filesystem::path prepared_spool;
  std::filesystem::path destination;
  bool protect_existing_outputs = false;
  bool checkpoint_enabled = false;
  CoefficientCodecRecipe coefficient_codec;
  ReconstructionNorm8Mode norm8_mode = ReconstructionNorm8Mode::kIdentityLloyd;
  std::uint32_t norm8_lloyd_iterations = 0U;
  bool use_cuda_norm8_fit = false;
  int cuda_norm_device = 0;
  bool retain_float_forest = true;
  bool retain_float_norm2 = true;
  bool publish_quantized_forest = true;
  bool publish_scaled_score8 = false;
  bool publish_float_scaled = false;
  FullTopologyPolicy topology;
  FullTopologyPipelineOptions pipeline;
  FullTopologySpoolWindowOptions spool_window;
};

struct IndexBuildStagePlan final {
  bool initialize_training = true;
  bool train = true;
  bool encode_base = true;
  bool build_linkage = true;
  // Required when encode_base=false and build_linkage=true.
  std::filesystem::path reused_base_basic_directory;
};

struct ResolvedIndexBuildPaths final {
  std::filesystem::path run_root;
  std::filesystem::path work_root;
  std::filesystem::path base_list_directory;
  std::filesystem::path index_directory;
  ArtifactHash64 base_basic_hash;
  ArtifactHash64 base_list_hash;
  ArtifactHash64 linkage_hash;
};

struct IndexBuildNamespace final {
  std::filesystem::path output_base;
  std::filesystem::path work_base;
  std::string prefix = "test";
  std::function<Status(const ResolvedIndexBuildPaths&)> on_resolved;
};

struct IndexBuildArtifactHashes final {
  ArtifactHash64 base_basic;
  ArtifactHash64 base_list;
  IndexArtifactHashes index;
};

[[nodiscard]] StatusOr<ResolvedIndexBuildPaths> ResolveIndexBuildPaths(
    const IndexBuildNamespace& space, const std::string& dataset_name,
    const IndexBuildArtifactHashes& hashes);

// Shared publication/eval identity calculation. Codec options do not alter
// the linkage directory identity.
[[nodiscard]] IndexBuildArtifactHashes ComputeIndexBuildArtifactHashesRuntime(
    const ModelShape& shape, const TrainedModelHashView& model,
    ArrayView<const std::uint8_t> bad_cluster_mask, bool base_source_is_u8,
    const BaseBasicStageOptions& base_basic,
    const IndexArtifactRecipe& artifacts,
    const FinalIndexBuildOptions& final_index);

template <int M>
[[nodiscard]] IndexBuildArtifactHashes ComputeIndexBuildArtifactHashes(
    const TrainingModel<M>& model,
    ArrayView<const std::uint8_t> bad_cluster_mask, bool base_source_is_u8,
    const BaseBasicStageOptions& base_basic,
    const IndexArtifactRecipe& artifacts,
    const FinalIndexBuildOptions& final_index);

// Complete typed product request. A fresh or reopened initialization is used
// when training.resume_checkpoint is absent; a round-checkpoint restart skips
// initialization entirely. Sub-request identities remain explicit so every
// independently restartable boundary can validate itself.
struct IndexBuildOptions {
  IndexBuildStagePlan stages;
  std::optional<SphericalRvqInitializationRequest> root_model;
  TrainRequest training;
  BaseBasicStageOptions base_basic;
  IndexArtifactRecipe artifacts;
  FinalIndexBuildOptions final_index;
  std::optional<IndexBuildNamespace> artifact_namespace;
};

template <int M>
struct IndexBuildRequest final : IndexBuildOptions {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::optional<TrainingInitializationRequest<M>> initialization;
};

// Scalar selection belongs to the source-consuming stages, not to the
// complete build. Each binding preserves the concrete source/runtime pair.
template <int M, typename T>
struct IndexInitializationInputs final {
  using Scalar = T;
  SphericalRvqInitializationBackend<T>* root_backend = nullptr;
  TrainingInitializationRuntime<M, T> runtime;
};

template <int M, typename T>
struct IndexBaseInputs final {
  using Scalar = T;
  BaseBasicBlockSourceFactory<T>* source = nullptr;
  BasicRoundRuntimeFactory<M, T>* runtime = nullptr;
};

template <int M>
struct IndexBuildRuntime final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::variant<IndexInitializationInputs<M, std::uint8_t>,
               IndexInitializationInputs<M, float>>
      initialization;
  TrainingRuntime<M> training;
  TrainingTopologyContextFactory<M>* final_topology = nullptr;
  std::variant<IndexBaseInputs<M, std::uint8_t>, IndexBaseInputs<M, float>>
      base;
  const ParallelForExecutor* parallel = nullptr;
  const ParallelForExecutor* base_list_parallel = nullptr;
  const F32GemmBackend* gemm = nullptr;
  const SpdSolverF64* coefficient_solver = nullptr;
};

struct IndexBuildStatistics final {
  double root_model_wall_seconds = 0.0;
  double initialization_wall_seconds = 0.0;
  // Complete training call: RVQ, initialization and iterations executed in this
  // invocation. With training disabled, measures trained checkpoint loading.
  double training_wall_seconds = 0.0;
  double base_basic_wall_seconds = 0.0;
  double final_linkage_publication_wall_seconds = 0.0;
  double wall_seconds = 0.0;
};

template <int M>
struct IndexBuildResult final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  using Statistics = IndexBuildStatistics;
  std::optional<ModelLayers<SphericalRvqLayerStatistics, M>> root_model;
  std::optional<TrainingInitializationResult<M>> initialization;
  TrainResult<M> training;
  BaseBasicStageResult base_basic;
  FullTopologyPipelineStatistics final_topology;
  std::optional<LinkageResourceStatistics> final_linkage_resources;
  std::filesystem::path index_directory;
  std::filesystem::path base_list_directory;
  // False when linkage borrows an existing base-list input.
  bool owns_base_list = false;
  std::optional<ResolvedIndexBuildPaths> resolved_paths;
  IndexBuildStagePlan stages;
  Statistics statistics;
};

// Executes initialization/resume, training, base encoding and final
// publication. Source scalar choices are independent stage bindings; numerical
// M remains available to the training and topology implementations.
template <int M>
[[nodiscard]] StatusOr<IndexBuildResult<M>> BuildIndex(
    const IndexBuildRequest<M>& request, IndexBuildRuntime<M>& runtime);

extern template StatusOr<IndexBuildResult<5>> BuildIndex<5>(
    const IndexBuildRequest<5>&, IndexBuildRuntime<5>&);
extern template StatusOr<IndexBuildResult<10>> BuildIndex<10>(
    const IndexBuildRequest<10>&, IndexBuildRuntime<10>&);

}  // namespace stlqpp
