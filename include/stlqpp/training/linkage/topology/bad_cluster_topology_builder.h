#pragma once

#include <cstdint>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/linkage_code_tuple.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/training/linkage/candidates/cpu_exact_knn.h"
#include "stlqpp/training/linkage/scheduler/linkage_evaluation_service.h"
#include "stlqpp/training/linkage/topology/causal_cpu_linkage_evaluator.h"
#include "stlqpp/training/linkage/topology/causal_topology_types.h"
#include "stlqpp/training/linkage/topology/cluster_topology_state.h"
#include "stlqpp/training/linkage/topology/topology_random_stream.h"

namespace stlqpp {

template <int M>
class FullTopologyClusterBuilder;

template <int M>
struct BadClusterTopologyBuildRequest final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  std::uint32_t cid = 0U;
  ColumnMajorMatrixF32View exact_real;
  bool borrow_exact_real = false;
  ColumnMajorMatrixF32View basic_real_reconstructions;
  ColumnMajorMatrixF32View virtual_reconstructions;
  ArrayView<const LinkageCodeTuple<M>> basic_supports;
  ArrayView<const float> basic_coefficients;
  ArrayView<const float> basic_squared_errors;
  std::optional<ValidatedBaseListBasicState> validated_basic = std::nullopt;
  ClusterTopologyInitialStateOwnership<M> basic_ownership{};
  std::vector<float>* owned_virtual_reconstructions = nullptr;
  // Complete permutation of real cluster-local IDs. The upstream virtual-root
  // builder owns its deterministic seed and supplies this order explicitly.
  ArrayView<const std::uint32_t> processing_order;
  // One complete caller-owned permutation per real local ID.
  ArrayView<const std::uint8_t> icm_layer_orders;
  // Continued stream after processing_order was shuffled. Tiny CPU tails
  // consume it; scheduler tasks retain their independent task-local stream.
  TopologyRandomStream* random_stream = nullptr;
  std::uint32_t knn_k = 0U;
  std::uint32_t maximum_depth = 0U;
};

// Consumes geometry-only exact-KNN candidate products using their ordered-row
// contract.
template <int M>
class BadClusterTopologyBuilder final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  BadClusterTopologyBuilder(
      const ProductCodebooks<M>& codebooks, LinkageEvaluationService& scheduler,
      CausalTopologyBatching batching,
      const CausalCpuLinkageEvaluator<M>* cpu_evaluator = nullptr) noexcept
      : codebooks_(&codebooks),
        scheduler_(&scheduler),
        batching_(batching),
        cpu_evaluator_(cpu_evaluator) {}

  [[nodiscard]] StatusOr<ClusterTopologyResult<M>> Build(
      const BadClusterTopologyBuildRequest<M>& request,
      const PreparedKnnRows& real_graph,
      const PreparedKnnRows& virtual_neighbors) const;

 private:
  template <int N>
  friend class FullTopologyClusterBuilder;

  [[nodiscard]] StatusOr<ClusterTopologyResult<M>> BuildProduced(
      const BadClusterTopologyBuildRequest<M>& request,
      const PreparedKnnRows& real_graph,
      const PreparedKnnRows& virtual_neighbors) const;
  [[nodiscard]] StatusOr<ClusterTopologyResult<M>> BuildImpl(
      const BadClusterTopologyBuildRequest<M>& request,
      const PreparedKnnRows& real_graph,
      const PreparedKnnRows& virtual_neighbors,
      bool produced_rows_and_order) const;
  const ProductCodebooks<M>* codebooks_ = nullptr;
  LinkageEvaluationService* scheduler_ = nullptr;
  CausalTopologyBatching batching_;
  const CausalCpuLinkageEvaluator<M>* cpu_evaluator_ = nullptr;
};

extern template class BadClusterTopologyBuilder<5>;
extern template class BadClusterTopologyBuilder<10>;

}  // namespace stlqpp
