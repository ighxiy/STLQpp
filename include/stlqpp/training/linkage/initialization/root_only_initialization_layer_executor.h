#pragma once

#include <cstdint>
#include <future>
#include <optional>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_tail_evaluator.h"
#include "stlqpp/training/linkage/scheduler/linkage_evaluation_service.h"
#include "stlqpp/training/linkage/topology/causal_topology_types.h"
#include "stlqpp/training/linkage/topology/cluster_topology_state.h"

namespace stlqpp {

template <int M>
class ClusterTopologySession;

// Dedicated owner of root-only initialization scheduling. Geometry and
// candidate construction remain outside this class. This class owns only the
// inner-batch, block-local dyn_before, reverse-edge, forced-completion, commit
// and reconstruction-publication order.
template <int M>
class RootOnlyInitializationLayerExecutor final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  RootOnlyInitializationLayerExecutor(
      ClusterTopologyState<M>& state, LinkageEvaluationService& scheduler,
      LinkageClusterHandle cluster,
      RootOnlyInitializationTailEvaluator<M>& tail_evaluator,
      std::uint32_t cluster_seed, CausalTopologyBatching batching) noexcept
      : state_(&state),
        scheduler_(&scheduler),
        cluster_(cluster),
        tail_evaluator_(&tail_evaluator),
        cluster_seed_(cluster_seed),
        batching_(batching) {}

  [[nodiscard]] StatusOr<CausalTopologyWindowStatistics> Run(
      ArrayView<const CausalTopologyRow<M>> commit_rows,
      ArrayView<const std::uint32_t> physical_layer_order,
      std::uint32_t maximum_depth, std::uint64_t* generation);

 private:
  friend class ClusterTopologySession<M>;

  // Builder-owned rows and physical order derive from the same radial layer.
  [[nodiscard]] StatusOr<CausalTopologyWindowStatistics> RunValidatedLayer(
      ArrayView<const CausalTopologyRow<M>> commit_rows,
      ArrayView<const std::uint32_t> physical_layer_order,
      std::uint32_t maximum_depth, std::uint64_t* generation);
  [[nodiscard]] StatusOr<CausalTopologyWindowStatistics> RunImpl(
      ArrayView<const CausalTopologyRow<M>> commit_rows,
      ArrayView<const std::uint32_t> physical_layer_order,
      std::uint32_t maximum_depth, std::uint64_t* generation,
      bool validated_layer);
  struct Winner;
  struct EvaluationRow;
  struct DependentCandidate;
  struct PendingEvaluation final {
    std::vector<std::uint32_t> real_local_ids;
    std::vector<std::uint32_t> candidate_columns;
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> original_candidate_ranks;
    std::future<StatusOr<LinkageEvaluationResult>> result;
    std::uint32_t node_count = 0U;
    std::uint32_t pair_count = 0U;
    CausalEvaluationKind kind = CausalEvaluationKind::kInner;
  };

  [[nodiscard]] StatusOr<std::optional<PendingEvaluation>> SubmitEvaluation(
      const std::vector<EvaluationRow>& rows, std::uint64_t* generation,
      CausalTopologyWindowStatistics* statistics, std::uint32_t maximum_depth,
      CausalEvaluationKind kind);

  [[nodiscard]] Status FinishEvaluation(
      PendingEvaluation evaluation, std::vector<Winner>* winners,
      ArrayView<const std::uint32_t> commit_position_by_local,
      CausalTopologyWindowStatistics* statistics);

  [[nodiscard]] Status MergeEvaluation(
      const PendingEvaluation& evaluation, LinkageEvaluationResult result,
      std::vector<Winner>* winners,
      ArrayView<const std::uint32_t> commit_position_by_local,
      CausalTopologyWindowStatistics* statistics, bool cpu);

  [[nodiscard]] Status Evaluate(
      const std::vector<EvaluationRow>& rows, std::vector<Winner>* winners,
      ArrayView<const std::uint32_t> commit_position_by_local,
      std::uint32_t maximum_depth, std::uint64_t* generation,
      CausalTopologyWindowStatistics* statistics, CausalEvaluationKind kind);

  [[nodiscard]] Status EvaluateTinyForced(
      const EvaluationRow& row, std::vector<Winner>* winners,
      ArrayView<const std::uint32_t> commit_position_by_local,
      std::uint32_t maximum_depth, std::uint64_t generation,
      CausalTopologyWindowStatistics* statistics, CausalEvaluationKind kind);

  [[nodiscard]] Status Commit(std::uint32_t position,
                              ArrayView<const CausalTopologyRow<M>> rows,
                              std::vector<Winner>* winners,
                              std::uint32_t maximum_depth);

  [[nodiscard]] Status FlushUpdates(std::uint64_t* generation,
                                    CausalTopologyWindowStatistics* statistics);

  void AppendUpdate(ClusterTopologyReconstructionUpdate update);

  ClusterTopologyState<M>* state_ = nullptr;
  LinkageEvaluationService* scheduler_ = nullptr;
  LinkageClusterHandle cluster_;
  RootOnlyInitializationTailEvaluator<M>* tail_evaluator_ = nullptr;
  std::uint32_t cluster_seed_ = 0U;
  CausalTopologyBatching batching_;
  std::vector<std::uint32_t> update_columns_;
  std::vector<float> update_reconstructions_;
};

extern template class RootOnlyInitializationLayerExecutor<5>;
extern template class RootOnlyInitializationLayerExecutor<10>;

}  // namespace stlqpp
