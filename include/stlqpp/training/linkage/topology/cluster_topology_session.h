#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_tail_evaluator.h"
#include "stlqpp/training/linkage/scheduler/linkage_evaluation_service.h"
#include "stlqpp/training/linkage/topology/causal_cpu_linkage_evaluator.h"
#include "stlqpp/training/linkage/topology/causal_topology_types.h"
#include "stlqpp/training/linkage/topology/cluster_topology_state.h"
#include "stlqpp/training/linkage/topology/topology_random_stream.h"

namespace stlqpp {

template <int M>
class RootOnlyInitializationClusterBuilder;

// Owns one scheduler registration and the generation protocol around one
// causal cluster state. Geometry/topology builders decide node and candidate
// order; this session performs the common submit, commit and R_full update
// transaction without exposing scheduler mechanics to either builder.
template <int M>
class ClusterTopologySession final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<std::unique_ptr<ClusterTopologySession<M>>>
  Create(const ProductCodebooks<M>& codebooks,
         ClusterTopologyInitialStateView<M> initial,
         LinkageEvaluationService& scheduler, CausalTopologyBatching batching,
         const CausalCpuLinkageEvaluator<M>* cpu_evaluator = nullptr,
         TopologyRandomStream* random_stream = nullptr,
         ClusterTopologyInitialStateOwnership<M> ownership = {},
         std::vector<float>* owned_virtual_reconstructions = nullptr,
         bool borrow_exact_real = false);

  [[nodiscard]] static StatusOr<std::unique_ptr<ClusterTopologySession<M>>>
  CreateRootOnlyInitialization(
      const RootCodebooks<M>& codebooks,
      ClusterTopologyInitialStateView<M> initial,
      LinkageEvaluationService& scheduler,
      RootOnlyInitializationTailEvaluator<M>& tail_evaluator,
      std::uint32_t cluster_seed, CausalTopologyBatching batching);

  ~ClusterTopologySession();
  ClusterTopologySession(const ClusterTopologySession&) = delete;
  ClusterTopologySession& operator=(const ClusterTopologySession&) = delete;

  [[nodiscard]] Status PublishInitialRoot(std::uint32_t real_local_id);

  [[nodiscard]] Status ProcessNode(
      std::uint32_t real_local_id,
      ArrayView<const LinkageParentReference> ordered_candidates,
      ArrayView<const std::uint8_t> icm_layer_order,
      std::uint32_t maximum_depth);

  [[nodiscard]] Status ProcessRootOnlyInitializationLayer(
      ArrayView<const CausalTopologyRow<M>> commit_rows,
      ArrayView<const std::uint32_t> physical_layer_order,
      std::uint32_t maximum_depth);

  [[nodiscard]] Status ProcessRadialLayerRows(
      ArrayView<const CausalTopologyRow<M>> rows, std::uint32_t maximum_depth);

  [[nodiscard]] Status ProcessGraphRows(
      ArrayView<const CausalTopologyRow<M>> rows, std::uint32_t maximum_depth);

  [[nodiscard]] const CausalTopologyWindowStatistics& window_statistics()
      const noexcept {
    return window_statistics_;
  }

  [[nodiscard]] StatusOr<ClusterTopologyResult<M>> Finish();

 private:
  friend class RootOnlyInitializationClusterBuilder<M>;

  [[nodiscard]] Status ProcessValidatedRootOnlyInitializationLayer(
      ArrayView<const CausalTopologyRow<M>> commit_rows,
      ArrayView<const std::uint32_t> physical_layer_order,
      std::uint32_t maximum_depth);
  [[nodiscard]] Status ProcessRootOnlyInitializationLayerImpl(
      ArrayView<const CausalTopologyRow<M>> commit_rows,
      ArrayView<const std::uint32_t> physical_layer_order,
      std::uint32_t maximum_depth, bool validated_layer);
  ClusterTopologySession(
      LinkageEvaluationService& scheduler, LinkageClusterHandle cluster,
      std::unique_ptr<ClusterTopologyState<M>> state,
      const CausalCpuLinkageEvaluator<M>* cpu_evaluator,
      TopologyRandomStream* random_stream,
      RootOnlyInitializationTailEvaluator<M>* initialization_tail_evaluator,
      std::uint32_t cluster_seed, CausalTopologyBatching batching) noexcept;

  void AccumulateWindowStatistics(
      const CausalTopologyWindowStatistics& statistics) noexcept;

  LinkageEvaluationService* scheduler_ = nullptr;
  LinkageClusterHandle cluster_;
  std::unique_ptr<ClusterTopologyState<M>> state_;
  const CausalCpuLinkageEvaluator<M>* cpu_evaluator_ = nullptr;
  TopologyRandomStream* random_stream_ = nullptr;
  RootOnlyInitializationTailEvaluator<M>* initialization_tail_evaluator_ =
      nullptr;
  std::uint32_t cluster_seed_ = 0U;
  CausalTopologyBatching batching_;
  std::uint64_t generation_ = 0U;
  CausalTopologyWindowStatistics window_statistics_;
  bool registered_ = false;
};

extern template class ClusterTopologySession<5>;
extern template class ClusterTopologySession<10>;

}  // namespace stlqpp
