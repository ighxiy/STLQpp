#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/array_view.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/runtime/cpu/spd_solver_f32.h"
#include "stlqpp/runtime/cpu/square_svd_f32.h"
#include "stlqpp/training/basic/basic_encoding.h"
#include "stlqpp/training/update/edge_codebook_moments.h"
#include "stlqpp/training/update/edge_codebook_solver.h"
#include "stlqpp/training/update/root_codebook_moments.h"
#include "stlqpp/training/update/root_codebook_solver.h"
#include "stlqpp/training/update/rotation_update.h"
#include "stlqpp/training/update/training_model.h"

#include <cstdint>
#include <optional>

namespace stlqpp {

template <int M>
class FullTopologyCheckpointSink;
template <int M>
class ReplayRoundUpdateSink;

template <int M>
struct EdgeTrainingClusterView final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  LinkedTrainingClusterView<M> linked;
};

struct RoundModelUpdateStatistics final {
  std::uint64_t basic_rows = 0U;
  std::uint64_t linked_rows = 0U;
  std::uint64_t rotation_rows = 0U;
  float root_diagonal_bump = 0.0F;
  float edge_diagonal_bump = 0.0F;
};

// Immutable model-update semantics for ordinary full and replay rounds.
// Frozen DEV and clean Master down-weight bad-CID rows only in the C_one
// normal equations; root and rotation moments remain unweighted.
struct RoundModelUpdatePolicy final {
  ArrayView<const std::uint8_t> bad_cluster_mask;
  float bad_cluster_edge_weight = 1.0F;
};

template <int M>
struct RoundModelUpdate final {
  TrainingModel<M> model;
  RoundModelUpdateStatistics statistics;
};

// Stateful two-barrier update for one training round. Basic statistics close
// first and publish a complete root-updated model for full/replay topology.
// Final forest rows then train C_one and R before one complete next-round
// model is returned. Full and replay construction therefore cannot diverge in
// update mathematics.
template <int M>
class RoundModelUpdater final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<RoundModelUpdater<M>> Create(
      const TrainingModel<M>& previous,
      const SpdSolverF32& spd_solver,
      const SquareSvdF32& svd,
      RoundModelUpdatePolicy policy,
      const ParallelForExecutor& root_moment_parallel,
      std::uint32_t edge_moment_shard_count,
      const ParallelForExecutor* final_moment_parallel = nullptr);
  [[nodiscard]] static StatusOr<RoundModelUpdater<M>> Create(
      const TrainingModel<M>& previous,
      const SpdSolverF32& spd_solver,
      const SquareSvdF32& svd,
      RoundModelUpdatePolicy policy,
      const ParallelForExecutor& root_accumulation_parallel,
      const ParallelForExecutor& root_conversion_parallel,
      std::uint32_t edge_moment_shard_count,
      const ParallelForExecutor* final_moment_parallel = nullptr);

  RoundModelUpdater(RoundModelUpdater&&) noexcept = default;
  RoundModelUpdater& operator=(RoundModelUpdater&&) noexcept = default;
  RoundModelUpdater(const RoundModelUpdater&) = delete;
  RoundModelUpdater& operator=(const RoundModelUpdater&) = delete;

  [[nodiscard]] Status AccumulateBasic(
      ColumnMajorMatrixF32View exact_model_space,
      const BasicEncoding<M>& encoding);
  [[nodiscard]] const ModelShape& basic_model_shape() const noexcept {
    return previous_->codebooks().shape();
  }
  [[nodiscard]] Status EnablePostRootMetric();
  [[nodiscard]] std::optional<double> post_root_squared_error_sum()
      const noexcept;

  // The returned immutable model contains new C_root and the previous C_one/R
  // and is the only model topology construction may observe this round.
  [[nodiscard]] StatusOr<TrainingModel<M>> FinishRootUpdate();

  [[nodiscard]] Status AccumulateEdgeCluster(
      const EdgeTrainingClusterView<M>& cluster);

  // Fast path for a frozen forest: reconstructs only nodes that are consumed
  // as parents and accumulates the same C_one normal equations directly.
  [[nodiscard]] Status AccumulateFrozenEdgeCluster(
      const ProductCodebooks<M>& codebooks,
      const FrozenLinkedTrainingClusterView<M>& cluster,
      const CanonicalBfsForestOrder* order = nullptr);

  // Replay windows expose independent CIDs together. C_one and rotation
  // sufficient statistics retain DEV's fixed contiguous-CID shard trees.
  [[nodiscard]] Status AccumulateEdgeClusters(
      ArrayView<const EdgeTrainingClusterView<M>> clusters);
  [[nodiscard]] Status AccumulateFrozenEdgeClusters(
      const ProductCodebooks<M>& codebooks,
      ArrayView<const FrozenLinkedTrainingClusterView<M>> clusters);

  // Closes C_one before any rotation statistics may be formed. The returned
  // model contains new C_root/new C_one and the previous R.
  [[nodiscard]] StatusOr<TrainingModel<M>> FinishEdgeUpdate();

  [[nodiscard]] StatusOr<RoundModelUpdate<M>> FinishRound(
      RotationCrossCovariance covariance, const F32GemmBackend& gemm);

  // Completes C_root/C_one while retaining the previous rotation exactly.
  // This is DEV train.use_opq_rotation=false, not an identity-covariance SVD.
  [[nodiscard]] StatusOr<RoundModelUpdate<M>> FinishRoundKeepingRotation();

 private:
  friend class FullTopologyCheckpointSink<M>;
  friend class ReplayRoundUpdateSink<M>;
  [[nodiscard]] Status AccumulateFrozenEdgeClusterAfterFullSinkValidation(
      const ProductCodebooks<M>& codebooks,
      const FrozenLinkedTrainingClusterView<M>& cluster,
      const CanonicalBfsForestOrder& order);
  [[nodiscard]] Status AccumulateFrozenEdgeClusterImpl(
      const ProductCodebooks<M>& codebooks,
      const FrozenLinkedTrainingClusterView<M>& cluster,
      const CanonicalBfsForestOrder* order,
      bool full_sink_validated,
      bool replay_validated = false);
  [[nodiscard]] Status AccumulateFrozenEdgeClustersAfterReplayValidation(
      const ProductCodebooks<M>& codebooks,
      ArrayView<const FrozenLinkedTrainingClusterView<M>> clusters);
  [[nodiscard]] Status AccumulateFrozenEdgeClustersImpl(
      const ProductCodebooks<M>& codebooks,
      ArrayView<const FrozenLinkedTrainingClusterView<M>> clusters,
      bool replay_validated);

  enum class Phase : std::uint8_t {
    kBasic,
    kTopology,
    kRotation,
    kComplete,
    kFailed,
  };

  RoundModelUpdater(
      const TrainingModel<M>& previous,
      const SpdSolverF32& spd_solver,
      const SquareSvdF32& svd,
      RootCodebookMoments<M> root_moments,
      std::uint32_t edge_moment_shard_count,
      const ParallelForExecutor* final_moment_parallel,
      std::vector<std::uint8_t> bad_cluster_mask,
      float bad_cluster_edge_weight) noexcept;

  [[nodiscard]] Status Fail(Status status) noexcept;
  [[nodiscard]] Status EnsureEdgeMoments();

  const TrainingModel<M>* previous_ = nullptr;
  const SpdSolverF32* spd_solver_ = nullptr;
  const SquareSvdF32* svd_ = nullptr;
  std::optional<RootCodebookMoments<M>> root_moments_;
  std::optional<EdgeCodebookMoments<M>> edge_moments_;
  std::vector<EdgeCodebookMoments<M>> edge_moment_shards_;
  std::uint32_t edge_moment_shard_count_ = 0U;
  const ParallelForExecutor* final_moment_parallel_ = nullptr;
  std::vector<std::uint8_t> bad_cluster_mask_;
  float bad_cluster_edge_weight_ = 1.0F;
  std::optional<RootCodebookSolveResult<M>> root_result_;
  std::optional<EdgeCodebookSolveResult<M>> edge_result_;
  std::uint64_t basic_rows_ = 0U;
  Phase phase_ = Phase::kBasic;
};

using RuntimeRoundModelUpdater = RoundModelUpdater<0>;
extern template class RoundModelUpdater<0>;
extern template class RoundModelUpdater<5>;
extern template class RoundModelUpdater<10>;

}  // namespace stlqpp
