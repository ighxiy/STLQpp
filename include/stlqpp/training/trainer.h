#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cpu/f32_cross_covariance.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/spd_solver_f32.h"
#include "stlqpp/runtime/cpu/square_svd_f32.h"
#include "stlqpp/training/analysis/initialization_post_edge_analysis.h"
#include "stlqpp/training/checkpoint/checkpoint_replay_loader.h"
#include "stlqpp/training/checkpoint/full_topology_checkpoint_round.h"
#include "stlqpp/training/checkpoint/initialized_training_checkpoint.h"
#include "stlqpp/training/checkpoint/replay_checkpoint_round.h"
#include "stlqpp/training/linkage/replay/period_two_topology_schedule.h"
#include "stlqpp/training/linkage/scheduler/linkage_resource_statistics.h"
#include "stlqpp/training/update/round_update_transcript.h"

namespace stlqpp {

// Holds the lifetime of a round-local filesystem workset. The concrete owner
// is intentionally opaque to the trainer; retaining the artifact keeps its
// list path valid until the round topology consumer is destroyed.
class TrainingBasicArtifactLifetime {
 public:
  virtual ~TrainingBasicArtifactLifetime() = default;
  TrainingBasicArtifactLifetime(const TrainingBasicArtifactLifetime&) = delete;
  TrainingBasicArtifactLifetime& operator=(
      const TrainingBasicArtifactLifetime&) = delete;

 protected:
  TrainingBasicArtifactLifetime() = default;
};

struct TrainingBasicRoundArtifact final {
  std::filesystem::path list_directory;
  std::uint64_t rows = 0U;
  std::shared_ptr<TrainingBasicArtifactLifetime> lifetime;
  // Real producer statistics from the encoding pass.  They travel with the
  // round artifact so the trainer can render DEV-compatible stage logs
  // without down-casting a runtime implementation or recomputing encodings.
  std::uint64_t metric_vectors = 0U;
  double initial_squared_error_sum = 0.0;
  double final_squared_error_sum = 0.0;
  double encode_wall_seconds = 0.0;
};

// DEV keeps basic encoding and linkage construction on independent random
// streams.  One typed pair is therefore the indivisible semantic authority
// for a training round; resource scheduling must never derive either seed.
struct TrainingRoundSeeds final {
  std::uint32_t basic = 0U;
  std::uint32_t topology = 0U;
};

// One complete RSD/ILS8/ICM4 training-basic pass. Implementations own data
// loading and CPU/CUDA execution, while this boundary exposes only the exact
// root moments consumed by the common round updater.
template <int M>
class TrainingBasicPass {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  virtual ~TrainingBasicPass() = default;
  TrainingBasicPass(const TrainingBasicPass&) = delete;
  TrainingBasicPass& operator=(const TrainingBasicPass&) = delete;

  [[nodiscard]] virtual StatusOr<TrainingBasicRoundArtifact> Accumulate(
      const TrainingModel<M>& model, std::uint32_t round, std::uint32_t seed,
      RoundModelUpdater<M>& updater) = 0;

  // A replay round keeps the previous forest's native CID for every stable
  // training row.  The resulting fixed-root basic artifact is consumed after
  // the C_root barrier; replay must not run an ordinary root search or
  // re-encode roots against the already-updated C_root.
  [[nodiscard]] virtual StatusOr<TrainingBasicRoundArtifact>
  AccumulateFixedRoots(const TrainingModel<M>& model,
                       const TrainingCheckpoint<M>& previous_checkpoint,
                       std::uint32_t round, std::uint32_t seed,
                       RoundModelUpdater<M>& updater) = 0;

 protected:
  TrainingBasicPass() = default;
};

// Owns every model-bound resource for one full topology round. The public
// trainer retains the scheduling and checkpoint transaction; a runtime
// factory only supplies the typed source and worker-local executors.
template <int M>
class FullTopologyRoundContext {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  virtual ~FullTopologyRoundContext() = default;
  FullTopologyRoundContext(const FullTopologyRoundContext&) = delete;
  FullTopologyRoundContext& operator=(const FullTopologyRoundContext&) = delete;

  [[nodiscard]] virtual FullTopologyClusterSource<M>& source() noexcept = 0;
  [[nodiscard]] virtual ArrayView<FullTopologyClusterExecutor<M>*>
  executors() noexcept = 0;
  [[nodiscard]] virtual std::optional<LinkageResourceStatistics>
  resource_statistics() const noexcept = 0;

 protected:
  FullTopologyRoundContext() = default;
};

// Owns the checkpoint loader and fixed-root/fixed-edge executors for one
// replay round. It cannot expose candidate search or dynamic topology.
template <int M>
class ReplayTopologyRoundContext {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  virtual ~ReplayTopologyRoundContext() = default;
  ReplayTopologyRoundContext(const ReplayTopologyRoundContext&) = delete;
  ReplayTopologyRoundContext& operator=(const ReplayTopologyRoundContext&) =
      delete;

  [[nodiscard]] virtual FixedTopologyReplayWindowLoader&
  loader() noexcept = 0;
  [[nodiscard]] virtual const FixedTopologyReplayExecutor<M>& executor()
      const noexcept = 0;

 protected:
  ReplayTopologyRoundContext() = default;
};

template <int M>
class TrainingTopologyContextFactory {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  virtual ~TrainingTopologyContextFactory() = default;
  TrainingTopologyContextFactory(const TrainingTopologyContextFactory&) =
      delete;
  TrainingTopologyContextFactory& operator=(
      const TrainingTopologyContextFactory&) = delete;

  [[nodiscard]] virtual StatusOr<std::unique_ptr<FullTopologyRoundContext<M>>>
  CreateFull(const TrainingModel<M>& topology_model,
             ArrayView<const std::uint8_t> bad_cluster_mask,
             const TrainingBasicRoundArtifact& basic, std::uint32_t round,
             const FullTopologyPolicy& policy) = 0;

  [[nodiscard]] virtual StatusOr<std::unique_ptr<ReplayTopologyRoundContext<M>>>
  CreateReplay(const TrainingCheckpoint<M>& previous_checkpoint,
               const TrainingModel<M>& topology_model,
               const TrainingBasicRoundArtifact& basic, std::uint32_t round,
               std::uint32_t rsd_rank, std::uint32_t icm_rounds,
               std::uint32_t seed) = 0;

 protected:
  TrainingTopologyContextFactory() = default;
};

template <int M>
class TrainingRotationPass {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  virtual ~TrainingRotationPass() = default;
  TrainingRotationPass(const TrainingRotationPass&) = delete;
  TrainingRotationPass& operator=(const TrainingRotationPass&) = delete;

  [[nodiscard]] virtual StatusOr<RotationCrossCovariance> Accumulate(
      const TrainingBasicRoundArtifact& basic,
      const RoundUpdateTranscript<M>& transcript,
      const TrainingModel<M>& post_edge_model) = 0;

 protected:
  TrainingRotationPass() = default;
};

struct TrainRequest final {
  std::filesystem::path checkpoint_root;
  std::optional<std::filesystem::path> initialization_checkpoint;
  // Published trained/initialized model or full checkpoint. Every invocation
  // starts with Full and rebuilds its own forest before the first Replay.
  std::optional<std::filesystem::path> resume_checkpoint;
  TrainingCheckpointIdentity identity;

  // Explicit seeds are semantic recipe state for this invocation. Their
  // count is DEV train.max_R_iters: rounds appended after the reopened state,
  // not the lifetime total round count. Seed generation is deliberately not
  // hidden in a runtime.
  std::vector<TrainingRoundSeeds> round_seeds;
  // Period-two replay consumes the previous round's compact in-memory BFS
  // forest. Enabled recovery points are written under checkpoint_root;
  // Numbered public artifacts contain only the trained model and metadata.
  bool periodic_checkpoint_enabled = false;
  std::uint32_t periodic_checkpoint_every_rounds = 0U;
  std::filesystem::path periodic_checkpoint_base_path{};
  bool update_opq_rotation = true;
  bool log_metrics = false;
  bool log_linkage_pre_c1 = false;
  float bad_cluster_training_weight = 1.0F;
  std::uint32_t edge_moment_shard_count = 0U;
  FullTopologyPolicy topology_policy;
  FullTopologyPipelineOptions topology_pipeline;
};

template <int M>
struct TrainingRuntime final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  TrainingBasicPass<M>* basic = nullptr;
  TrainingTopologyContextFactory<M>* topology = nullptr;
  TrainingRotationPass<M>* rotation = nullptr;
  // Backend-neutral R*x for linkage analysis and U*V^T for OPQ rotation.
  // Required when rotation is updated; post-C_root MSE uses basic moments.
  const F32GemmBackend* metric_gemm = nullptr;
  const F32CrossCovarianceBackend* covariance_backend = nullptr;
  const SpdSolverF32* spd_solver = nullptr;
  const SquareSvdF32* svd = nullptr;
  const ParallelForExecutor* root_accumulation_parallel = nullptr;
  const ParallelForExecutor* root_moment_parallel = nullptr;
  // Numerical covariance shards, independent of the analysis worker count.
  const ParallelForExecutor* rotation_moment_parallel = nullptr;
  const ParallelForExecutor* update_parallel = nullptr;
};

struct TrainingRoundStatistics final {
  std::uint32_t round = 0U;
  TrainingRoundSeeds seeds;
  TopologyRoundKind topology_kind = TopologyRoundKind::kFull;
  double basic_wall_seconds = 0.0;
  double root_update_wall_seconds = 0.0;
  // Final retrieval only; norm accumulation and RHS contraction are charged
  // to the basic and root-update phases.
  double root_mse_wall_seconds = 0.0;
  std::uint64_t root_metric_rows = 0U;
  double root_post_update_mse = 0.0;
  double pre_c_one_analysis_wall_seconds = 0.0;
  double edge_update_wall_seconds = 0.0;
  double post_edge_analysis_wall_seconds = 0.0;
  double topology_checkpoint_wall_seconds = 0.0;
  double wall_seconds = 0.0;
  RoundModelUpdateStatistics update;
  std::optional<FullTopologyPipelineStatistics> full;
  std::optional<FixedTopologyReplayPipelineStatistics> replay;
  std::optional<InitializationPostEdgeAnalysis> pre_c_one_analysis;
  std::optional<InitializationPostEdgeAnalysis> post_edge_analysis;
  std::optional<LinkageResourceStatistics> linkage_resources;
};

template <int M>
class TrainingFinalCheckpoint final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  explicit TrainingFinalCheckpoint(
      InitializedTrainingCheckpoint<M> checkpoint) noexcept
      : checkpoint_(std::move(checkpoint)) {}
  explicit TrainingFinalCheckpoint(TrainingCheckpoint<M> checkpoint) noexcept
      : checkpoint_(std::move(checkpoint)) {}

  TrainingFinalCheckpoint(TrainingFinalCheckpoint&&) noexcept = default;
  TrainingFinalCheckpoint& operator=(TrainingFinalCheckpoint&&) noexcept =
      default;
  TrainingFinalCheckpoint(const TrainingFinalCheckpoint&) = delete;
  TrainingFinalCheckpoint& operator=(const TrainingFinalCheckpoint&) = delete;

  [[nodiscard]] bool has_completed_round() const noexcept {
    return std::holds_alternative<TrainingCheckpoint<M>>(checkpoint_);
  }
  [[nodiscard]] std::uint32_t completed_round() const noexcept {
    return std::get<TrainingCheckpoint<M>>(checkpoint_).completed_round();
  }
  [[nodiscard]] std::uint32_t completed_round_count() const noexcept {
    return has_completed_round() ? completed_round() + 1U : 0U;
  }
  [[nodiscard]] const TrainingModel<M>& model() const noexcept {
    return has_completed_round()
               ? std::get<TrainingCheckpoint<M>>(checkpoint_).model()
               : std::get<InitializedTrainingCheckpoint<M>>(checkpoint_)
                     .model();
  }
  [[nodiscard]] ArrayView<const std::uint8_t> bad_cluster_mask()
      const noexcept {
    return has_completed_round()
               ? std::get<TrainingCheckpoint<M>>(checkpoint_).bad_cluster_mask()
               : std::get<InitializedTrainingCheckpoint<M>>(checkpoint_)
                     .bad_cluster_mask();
  }

 private:
  std::variant<InitializedTrainingCheckpoint<M>, TrainingCheckpoint<M>>
      checkpoint_;
};

template <int M>
struct TrainResult final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::filesystem::path checkpoint_directory;
  TrainingFinalCheckpoint<M> checkpoint;
  std::vector<TrainingRoundStatistics> executed_rounds;
};

// Appends the requested number of rounds after the reopened state. The
// invocation-local first round is full and later rounds alternate with replay.
// Within an invocation the model and new forest stay resident. Both public
// model artifacts and full checkpoints can seed additional training; the first
// Full round needs no forest from the preceding invocation.
template <int M>
[[nodiscard]] StatusOr<TrainResult<M>> Train(const TrainRequest& request,
                                             TrainingRuntime<M>& runtime);

extern template StatusOr<TrainResult<5>> Train<5>(const TrainRequest&,
                                                  TrainingRuntime<5>&);
extern template StatusOr<TrainResult<10>> Train<10>(const TrainRequest&,
                                                    TrainingRuntime<10>&);

}  // namespace stlqpp
