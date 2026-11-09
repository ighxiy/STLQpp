#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/pipeline/root_initialization_basic_stage.h"
#include "stlqpp/training/analysis/basic_post_root_analysis.h"
#include "stlqpp/training/analysis/initialization_post_edge_analysis.h"
#include "stlqpp/training/checkpoint/initialized_training_checkpoint.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_context.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_pipeline.h"
#include "stlqpp/training/linkage/topology/full_topology_pipeline.h"
#include "stlqpp/training/trainer.h"
#include "stlqpp/training/linkage/scheduler/linkage_resource_statistics.h"
#include "stlqpp/training/update/root_only_initialization_update_sink.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace stlqpp {

enum class TrainingInitializationMode : std::uint8_t {
  kLegacyRootOnly,
  kFastInit,
};

// The registered stage recipe has no numerical specialization. Only the
// initial-model binding below carries M.
struct TrainingInitializationOptions {
  std::filesystem::path checkpoint_directory;
  std::optional<std::filesystem::path> resume_checkpoint;
  bool log_checkpoint_publication = false;
  TrainingCheckpointIdentity identity;
  std::uint32_t seed = 0U;
  TrainingInitializationMode mode =
      TrainingInitializationMode::kLegacyRootOnly;
  bool virtual_enabled = true;
  double good_cluster_fraction = 0.95;
  float bad_cluster_training_weight = 0.0F;
  std::uint32_t edge_moment_shard_count = 0U;
  RootInitializationBasicStageOptions basic;
  RootOnlyInitializationTopologyPolicy topology;
  RootOnlyInitializationPipelineOptions topology_pipeline;
  FullTopologyPolicy fast_topology;
  bool log_metrics = false;
};

template <int M>
struct TrainingInitializationRequest final : TrainingInitializationOptions {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  TrainingInitializationRequest() = default;
  explicit TrainingInitializationRequest(
      TrainingInitializationOptions options,
      const RootTrainingModel<M>* model = nullptr)
      : TrainingInitializationOptions(std::move(options)),
        initial_model(model) {}

  const RootTrainingModel<M>* initial_model = nullptr;
};

template <int M, typename Scalar = std::uint8_t>
struct TrainingInitializationRuntime final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  InitializationBasicBlockSourceFactory<Scalar>* basic_source = nullptr;
  BasicRoundRuntimeFactory<M, Scalar>* basic_runtime = nullptr;
  RootOnlyInitializationTopologyContextFactory<M>* topology = nullptr;
  TrainingTopologyContextFactory<M>* full_topology = nullptr;
  const ParallelForExecutor* parallel = nullptr;
  const ParallelForExecutor* root_accumulation_parallel = nullptr;
  const ParallelForExecutor* root_moment_parallel = nullptr;
  const ParallelForExecutor* list_parallel = nullptr;
  const SpdSolverF32* spd_solver = nullptr;
  const F32GemmBackend* metric_gemm = nullptr;
};

struct TrainingInitializationTimings final {
  double basic_wall_seconds = 0.0;
  double root_update_wall_seconds = 0.0;
  // Moment-metric finalization only; norm accumulation and RHS contraction
  // are included in the basic and root-solve phases, respectively.
  double root_mse_wall_seconds = 0.0;
  double init_linkage_wall_seconds = 0.0;
  double bad_mask_wall_seconds = 0.0;
  double c_one_update_wall_seconds = 0.0;
  double post_c_one_analysis_wall_seconds = 0.0;
  double checkpoint_wall_seconds = 0.0;
};

template <int M>
using TrainingInitializationTimingStatistics = TrainingInitializationTimings;

// Stage statistics and their ownership do not depend on the numerical model.
// Keep their construction, movement and cleanup shared across typed results.
struct TrainingInitializationSummary {
  TrainingInitializationSummary();
  ~TrainingInitializationSummary();
  TrainingInitializationSummary(TrainingInitializationSummary&&) noexcept;
  TrainingInitializationSummary& operator=(
      TrainingInitializationSummary&&) noexcept;
  TrainingInitializationSummary(const TrainingInitializationSummary&) = delete;
  TrainingInitializationSummary& operator=(
      const TrainingInitializationSummary&) = delete;

  std::filesystem::path checkpoint_directory;
  std::optional<RootInitializationBasicStageResult> basic;
  std::optional<BasicPostRootAnalysis> post_root_analysis;
  std::optional<RootOnlyInitializationPipelineStatistics> topology;
  std::optional<FullTopologyPipelineStatistics> fast_topology;
  std::optional<InitializationBadClusterResult> bad_clusters;
  std::optional<InitializationModelUpdateStatistics> update;
  std::optional<InitializationPostEdgeAnalysis> post_edge_analysis;
  std::optional<LinkageResourceStatistics> linkage_resources;
  TrainingInitializationTimings timing;
};

template <int M>
struct TrainingInitializationResult final : TrainingInitializationSummary {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  TrainingInitializationResult(
      InitializedTrainingCheckpoint<M>&& initialized,
      TrainingInitializationSummary&& summary) noexcept
      : TrainingInitializationSummary(std::move(summary)),
        checkpoint(std::move(initialized)) {}

  InitializedTrainingCheckpoint<M> checkpoint;
};

template <int M, typename Scalar>
[[nodiscard]] StatusOr<TrainingInitializationResult<M>> InitializeTraining(
    const TrainingInitializationRequest<M>& request,
    TrainingInitializationRuntime<M, Scalar>& runtime);

extern template StatusOr<TrainingInitializationResult<5>>
InitializeTraining<5>(const TrainingInitializationRequest<5>&,
                      TrainingInitializationRuntime<5, std::uint8_t>&);
extern template StatusOr<TrainingInitializationResult<10>>
InitializeTraining<10>(const TrainingInitializationRequest<10>&,
                       TrainingInitializationRuntime<10, std::uint8_t>&);
extern template StatusOr<TrainingInitializationResult<5>>
InitializeTraining<5>(const TrainingInitializationRequest<5>&,
                      TrainingInitializationRuntime<5, float>&);
extern template StatusOr<TrainingInitializationResult<10>>
InitializeTraining<10>(const TrainingInitializationRequest<10>&,
                       TrainingInitializationRuntime<10, float>&);

}  // namespace stlqpp
