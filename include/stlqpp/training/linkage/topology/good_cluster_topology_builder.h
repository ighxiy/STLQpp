#pragma once

#include <cstdint>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/linkage_code_tuple.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/training/linkage/candidates/cpu_exact_parent_candidates.h"
#include "stlqpp/training/linkage/scheduler/linkage_evaluation_service.h"
#include "stlqpp/training/linkage/topology/causal_cpu_linkage_evaluator.h"
#include "stlqpp/training/linkage/topology/causal_topology_types.h"
#include "stlqpp/training/linkage/topology/cluster_topology_state.h"
#include "stlqpp/training/linkage/topology/topology_random_stream.h"

namespace stlqpp {

template <int M>
class FullTopologyClusterBuilder;

template <int M>
struct GoodClusterTopologyBuildRequest final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  std::uint32_t cid = 0U;
  ColumnMajorMatrixF32View exact_real;
  bool borrow_exact_real = false;
  ColumnMajorMatrixF32View basic_real_reconstructions;
  ArrayView<const LinkageCodeTuple<M>> basic_supports;
  ArrayView<const float> basic_coefficients;
  ArrayView<const float> basic_squared_errors;
  std::optional<ValidatedBaseListBasicState> validated_basic = std::nullopt;
  ClusterTopologyInitialStateOwnership<M> basic_ownership{};
  ArrayView<const RadialLayerView> radial_layers;
  // Fresh task-local GPU permutations. Tiny CPU tails do not use this array.
  ArrayView<const std::uint8_t> icm_layer_orders;
  // Continued cluster stream shared by radial processing-order shuffles and
  // successful tiny-CPU rsd candidates.
  TopologyRandomStream* random_stream = nullptr;
  std::uint32_t depth_k = 0U;
  std::uint32_t knn_k = 0U;
  std::uint32_t maximum_depth = 0U;
};

// Synchronous cluster-level topology algorithm. Parallel outer orchestration
// runs many instances, allowing the shared scheduler to batch their one-node
// causal frontiers without changing any cluster's commit order.
template <int M>
class GoodClusterTopologyBuilder final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  GoodClusterTopologyBuilder(
      const ProductCodebooks<M>& codebooks,
      const CpuExactParentCandidateProvider& candidates,
      LinkageEvaluationService& scheduler, CausalTopologyBatching batching,
      const CausalCpuLinkageEvaluator<M>* cpu_evaluator = nullptr) noexcept
      : codebooks_(&codebooks),
        candidates_(&candidates),
        scheduler_(&scheduler),
        batching_(batching),
        cpu_evaluator_(cpu_evaluator) {}

  [[nodiscard]] StatusOr<ClusterTopologyResult<M>> Build(
      const GoodClusterTopologyBuildRequest<M>& request) const;

 private:
  friend class FullTopologyClusterBuilder<M>;

  // Only the full builder may pass RadialLayerPlan and generated ICM orders.
  [[nodiscard]] StatusOr<ClusterTopologyResult<M>> BuildProduced(
      const GoodClusterTopologyBuildRequest<M>& request) const;
  [[nodiscard]] StatusOr<ClusterTopologyResult<M>> BuildImpl(
      const GoodClusterTopologyBuildRequest<M>& request,
      bool produced_partition) const;

  const ProductCodebooks<M>* codebooks_ = nullptr;
  const CpuExactParentCandidateProvider* candidates_ = nullptr;
  LinkageEvaluationService* scheduler_ = nullptr;
  CausalTopologyBatching batching_;
  const CausalCpuLinkageEvaluator<M>* cpu_evaluator_ = nullptr;
};

extern template class GoodClusterTopologyBuilder<5>;
extern template class GoodClusterTopologyBuilder<10>;

}  // namespace stlqpp
