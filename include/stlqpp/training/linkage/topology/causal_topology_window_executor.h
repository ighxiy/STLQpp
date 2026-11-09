#pragma once

#include <cstdint>
#include <future>
#include <optional>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/training/linkage/scheduler/linkage_evaluation_service.h"
#include "stlqpp/training/linkage/topology/causal_cpu_linkage_evaluator.h"
#include "stlqpp/training/linkage/topology/causal_topology_types.h"
#include "stlqpp/training/linkage/topology/cluster_topology_state.h"
#include "stlqpp/training/linkage/topology/topology_random_stream.h"

namespace stlqpp {

// Executes one already ordered causal sequence. Geometry providers remain
// outside this class; all parent visibility, depth and commit authority remain
// in ClusterTopologyState.
template <int M>
class CausalTopologyWindowExecutor final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  CausalTopologyWindowExecutor(
      ClusterTopologyState<M>& state, LinkageEvaluationService& scheduler,
      LinkageClusterHandle cluster,
      const CausalCpuLinkageEvaluator<M>* cpu_evaluator,
      TopologyRandomStream* random_stream,
      CausalTopologyBatching batching) noexcept
      : state_(&state),
        scheduler_(&scheduler),
        cluster_(cluster),
        cpu_evaluator_(cpu_evaluator),
        random_stream_(random_stream),
        batching_(batching),
        cpu_scratch_(cpu_evaluator == nullptr
                         ? typename CausalCpuLinkageEvaluator<M>::Scratch{}
                         : cpu_evaluator->MakeScratch()) {}

  [[nodiscard]] StatusOr<CausalTopologyWindowStatistics> RunRadialLayer(
      ArrayView<const CausalTopologyRow<M>> rows, std::uint32_t maximum_depth,
      std::uint64_t* generation);

  [[nodiscard]] StatusOr<CausalTopologyWindowStatistics> RunGraphOrder(
      ArrayView<const CausalTopologyRow<M>> rows, std::uint32_t maximum_depth,
      std::uint64_t* generation);

 private:
  struct DependentCandidate;
  struct Winner;
  struct PendingEvaluation final {
    std::vector<std::uint32_t> real_local_ids;
    std::vector<std::uint32_t> source_row_positions;
    std::vector<std::uint32_t> candidate_columns;
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> original_candidate_ranks;
    std::future<StatusOr<LinkageEvaluationResult>> result;
    std::uint32_t node_count = 0U;
    std::uint32_t pair_count = 0U;
    CausalEvaluationKind kind = CausalEvaluationKind::kInner;
  };

  [[nodiscard]] StatusOr<std::optional<PendingEvaluation>> SubmitGpuEvaluation(
      const std::vector<const CausalTopologyRow<M>*>& rows,
      const std::vector<std::vector<ClusterTopologyRankedCandidate>>&
          candidates,
      std::uint32_t maximum_depth, std::uint64_t* generation,
      CausalTopologyWindowStatistics* statistics, CausalEvaluationKind kind);

  [[nodiscard]] Status FinishGpuEvaluation(
      PendingEvaluation evaluation,
      const std::vector<const CausalTopologyRow<M>*>& rows,
      std::vector<Winner>* winners, CausalTopologyWindowStatistics* statistics);

  [[nodiscard]] Status Evaluate(
      const std::vector<const CausalTopologyRow<M>*>& rows,
      const std::vector<std::vector<ClusterTopologyRankedCandidate>>&
          candidates,
      std::vector<Winner>* winners, std::uint32_t maximum_depth,
      std::uint64_t* generation, CausalTopologyWindowStatistics* statistics,
      bool permit_cpu_tiny_tail, CausalEvaluationKind kind);

  [[nodiscard]] Status ExecuteBatch(
      const std::vector<const CausalTopologyRow<M>*>& rows,
      std::uint32_t maximum_depth, std::uint64_t* generation,
      CausalTopologyWindowStatistics* statistics);

  [[nodiscard]] Status EvaluateForcedDynamic(
      const std::vector<const CausalTopologyRow<M>*>& rows,
      std::vector<std::vector<ClusterTopologyRankedCandidate>>* pending,
      std::vector<Winner>* winners, std::uint32_t position,
      std::uint32_t maximum_depth, std::uint64_t* generation,
      CausalTopologyWindowStatistics* statistics, CausalEvaluationKind kind);

  [[nodiscard]] Status PreflushDynamic(
      const std::vector<const CausalTopologyRow<M>*>& rows,
      std::vector<std::vector<ClusterTopologyRankedCandidate>>* pending,
      std::uint32_t first_position, std::uint32_t maximum_depth,
      std::uint64_t* generation, CausalTopologyWindowStatistics* statistics,
      CausalEvaluationKind kind, std::optional<PendingEvaluation>* in_flight);

  [[nodiscard]] StatusOr<CausalTopologyWindowStatistics> RunFixedWindows(
      ArrayView<const CausalTopologyRow<M>> rows, std::uint32_t maximum_depth,
      std::uint64_t* generation, std::uint32_t window_rows,
      std::uint32_t maximum_frozen_pairs, std::uint32_t dynamic_preflush_pairs);

  [[nodiscard]] Status FlushUpdates(std::uint64_t* generation,
                                    CausalTopologyWindowStatistics* statistics);

  void AppendUpdate(ClusterTopologyReconstructionUpdate update);

  ClusterTopologyState<M>* state_ = nullptr;
  LinkageEvaluationService* scheduler_ = nullptr;
  LinkageClusterHandle cluster_;
  const CausalCpuLinkageEvaluator<M>* cpu_evaluator_ = nullptr;
  TopologyRandomStream* random_stream_ = nullptr;
  CausalTopologyBatching batching_;
  typename CausalCpuLinkageEvaluator<M>::Scratch cpu_scratch_;
  std::vector<std::uint32_t> update_columns_;
  std::vector<float> update_reconstructions_;
  std::vector<std::uint32_t> window_position_;
  std::vector<std::uint32_t> layer_position_;
  std::vector<std::uint32_t> batch_position_;
};

extern template class CausalTopologyWindowExecutor<5>;
extern template class CausalTopologyWindowExecutor<10>;

}  // namespace stlqpp
