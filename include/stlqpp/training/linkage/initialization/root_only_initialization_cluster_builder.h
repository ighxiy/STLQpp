#pragma once

#include <cstdint>
#include <optional>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/linkage_code_tuple.h"
#include "stlqpp/model/root_codebooks.h"
#include "stlqpp/training/linkage/candidates/cpu_exact_parent_candidates.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_tail_evaluator.h"
#include "stlqpp/training/linkage/scheduler/linkage_evaluation_service.h"
#include "stlqpp/training/linkage/topology/causal_topology_types.h"
#include "stlqpp/training/linkage/topology/cluster_topology_state.h"

namespace stlqpp {

struct RootOnlyInitializationTopologyPolicy final {
  std::uint32_t radial_layer_count = 0U;
  double root_fraction = 0.0;
  std::uint32_t depth_k = 0U;
  std::uint32_t knn_k = 0U;
  std::uint32_t maximum_depth = 0U;
  std::uint32_t seed = 0U;
  std::uint32_t icm_rounds = 1U;
  bool shared_cluster_seed = false;
  CausalTopologyBatching batching;
};

template <int M>
struct RootOnlyInitializationClusterRequest final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t cid = 0U;
  ColumnMajorMatrixF32View exact_real;
  ColumnMajorMatrixF32View basic_real_reconstructions;
  ArrayView<const LinkageCodeTuple<M>> basic_supports;
  ArrayView<const float> basic_coefficients;
  ArrayView<const float> basic_squared_errors;
  std::optional<ValidatedBaseListBasicState> validated_basic = std::nullopt;

  // One shared task-level ICM permutation, matching the registered
  // legacy-root-only evaluator. It is recipe input rather than hidden RNG.
  ArrayView<const std::uint8_t> icm_layer_order;
};

// Builds the pre-C_one causal forest using C_root for every accepted residual.
// There are no virtual roots and no RSD candidates in this stage.
template <int M>
class RootOnlyInitializationClusterBuilder final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  RootOnlyInitializationClusterBuilder(
      const RootCodebooks<M>& codebooks,
      const CpuExactParentCandidateProvider& candidates,
      LinkageEvaluationService& scheduler,
      RootOnlyInitializationTailEvaluator<M>& tail_evaluator) noexcept
      : codebooks_(&codebooks),
        candidates_(&candidates),
        scheduler_(&scheduler),
        tail_evaluator_(&tail_evaluator) {}

  [[nodiscard]] StatusOr<ClusterTopologyResult<M>> Build(
      const RootOnlyInitializationClusterRequest<M>& request,
      const RootOnlyInitializationTopologyPolicy& policy) const;

 private:
  const RootCodebooks<M>* codebooks_ = nullptr;
  const CpuExactParentCandidateProvider* candidates_ = nullptr;
  LinkageEvaluationService* scheduler_ = nullptr;
  RootOnlyInitializationTailEvaluator<M>* tail_evaluator_ = nullptr;
};

extern template class RootOnlyInitializationClusterBuilder<5>;
extern template class RootOnlyInitializationClusterBuilder<10>;

}  // namespace stlqpp
