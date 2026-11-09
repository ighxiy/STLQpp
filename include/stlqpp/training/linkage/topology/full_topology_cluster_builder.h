#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/linkage_code_tuple.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/training/least_squares_cost_mode.h"
#include "stlqpp/training/linkage/candidates/cpu_exact_knn.h"
#include "stlqpp/training/linkage/candidates/cluster_knn.h"
#include "stlqpp/training/linkage/candidates/cpu_exact_parent_candidates.h"
#include "stlqpp/training/linkage/scheduler/linkage_evaluation_service.h"
#include "stlqpp/training/linkage/topology/causal_cpu_linkage_evaluator.h"
#include "stlqpp/training/linkage/topology/causal_topology_types.h"
#include "stlqpp/training/linkage/topology/cluster_topology_state.h"
#include "stlqpp/training/linkage/virtual/umap_virtual_root_plan.h"
#include "stlqpp/training/linkage/virtual/virtual_root_encoder.h"

namespace stlqpp {

struct FullTopologyPolicy final {
  std::uint32_t radial_layer_count = 0U;
  double root_fraction = 0.0;
  std::uint32_t depth_k = 0U;
  std::uint32_t knn_k = 0U;
  std::uint32_t maximum_depth = 0U;
  std::uint32_t seed = 0U;
  // Maximum cumulative RSD branch rank. Rank 3 includes rank-2 and rank-3
  // single deviations; it never replaces the rank-2 candidate family.
  std::uint32_t rsd_rank = 2U;
  std::uint32_t icm_rounds = 1U;
  LeastSquaresCostMode least_squares_cost_mode =
      LeastSquaresCostMode::kNormalEquation;
  // Virtual-root re-encoding follows the basic-encoder seed of its stage:
  // train seed during ordinary rounds and base seed during final construction.
  // It is independent from the topology-processing random stream above.
  std::uint32_t virtual_root_seed = 0U;
  // Publication-only observation. It does not affect topology, candidates,
  // hashes or training; final scaled-score publication uses it to retain one
  // source norm per real row in bounded temporary state.
  bool collect_source_norm2 = false;
  CausalTopologyBatching batching;

  bool virtual_enabled = true;
  double virtual_ratio = 0.0;
  std::uint32_t minimum_virtual_roots = 0U;
  std::uint32_t maximum_virtual_roots = 0U;
  bool use_fixed_virtual_roots = false;
  std::uint32_t fixed_virtual_roots = 0U;
  ClusterKnnPolicy cluster_knn;
  UmapVirtualRootPolicy virtual_policy;
};

template <int M>
struct FullTopologyClusterBuildRequest final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t cid = 0U;
  // Required only when bad_cluster is true and virtual roots may be built.
  ColumnMajorMatrixF32View unrotated_exact_real;
  ColumnMajorMatrixF32View rotated_exact_real;
  // FullTopologyClusterWork remains alive throughout Build, so its rotated
  // rows may be borrowed by the causal state instead of copied.
  bool borrow_rotated_exact_real = false;
  ColumnMajorMatrixF32View basic_real_reconstructions;
  ArrayView<const std::uint64_t> real_sample_ids;
  ArrayView<const LinkageCodeTuple<M>> basic_supports;
  ArrayView<const float> basic_coefficients;
  ArrayView<const float> basic_squared_errors;
  std::optional<ValidatedBaseListBasicState> validated_basic = std::nullopt;
  // Optional typed ownership handoff from FullTopologyClusterWork.  It is
  // consumed only when the causal topology state is successfully created.
  ClusterTopologyInitialStateOwnership<M> basic_ownership{};
  bool bad_cluster = false;
  // Borrowed only for this Build call; never crosses the result boundary.
  ExactKnnCpuBudget* exact_knn_cpu_budget = nullptr;
};

template <int M>
struct FullTopologyClusterProduct final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  ClusterTopologyResult<M> topology;
  std::optional<EncodedVirtualRootsPublication> virtual_roots;
};

// One owned cluster crossing the source-to-worker boundary. Views handed to
// FullTopologyClusterBuilder remain valid only for the corresponding Build
// call; build-only payload must not cross the ordered publication boundary.
template <int M>
struct FullTopologyClusterWork final {
  std::uint32_t m = M;
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t cid = 0U;
  std::uint32_t dimension = 0U;
  std::vector<float> unrotated_exact_real;
  std::vector<float> rotated_exact_real;
  std::vector<float> basic_real_reconstructions;
  std::vector<std::uint64_t> real_sample_ids;
  std::vector<float> source_real_norm2;
  std::vector<LinkageCodeTuple<M>> basic_supports;
  std::vector<float> basic_coefficients;
  std::vector<float> basic_squared_errors;
  std::optional<ValidatedBaseListBasicState> validated_basic = std::nullopt;
  bool bad_cluster = false;

  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return static_cast<std::uint32_t>(real_sample_ids.size());
  }
  [[nodiscard]] FullTopologyClusterBuildRequest<M> request() noexcept {
    const std::uint32_t count = real_count();
    return {
        .cid = cid,
        .unrotated_exact_real = {unrotated_exact_real.data(), dimension, count},
        .rotated_exact_real = {rotated_exact_real.data(), dimension, count},
        .borrow_rotated_exact_real = true,
        .basic_real_reconstructions = {basic_real_reconstructions.data(),
                                       dimension, count},
        .real_sample_ids = real_sample_ids,
        .basic_supports = basic_supports,
        .basic_coefficients = basic_coefficients,
        .basic_squared_errors = basic_squared_errors,
        .validated_basic = validated_basic,
        .basic_ownership =
            {
                .real_reconstructions = &basic_real_reconstructions,
                .supports = &basic_supports,
                .coefficients = &basic_coefficients,
                .squared_errors = &basic_squared_errors,
            },
        .bad_cluster = bad_cluster,
    };
  }
};

template <int M>
class FullTopologyClusterExecutor {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  virtual ~FullTopologyClusterExecutor() = default;
  FullTopologyClusterExecutor(FullTopologyClusterExecutor&&) noexcept = default;
  FullTopologyClusterExecutor& operator=(
      FullTopologyClusterExecutor&&) noexcept = default;
  FullTopologyClusterExecutor(const FullTopologyClusterExecutor&) = delete;
  FullTopologyClusterExecutor& operator=(const FullTopologyClusterExecutor&) =
      delete;

  [[nodiscard]] virtual StatusOr<FullTopologyClusterProduct<M>> Build(
      const FullTopologyClusterBuildRequest<M>& request,
      const FullTopologyPolicy& policy) const = 0;

 protected:
  FullTopologyClusterExecutor() = default;
};

// Cluster-level full-topology composition. Good and bad geometry meet only at
// the common causal state machine; virtual-root construction is confined to
// the bad path and reuses the caller-supplied real graph.
template <int M>
class FullTopologyClusterBuilder final : public FullTopologyClusterExecutor<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  FullTopologyClusterBuilder(
      const ProductCodebooks<M>& codebooks,
      const CpuExactParentCandidateProvider& good_candidates,
      const CpuExactKnnProvider& exact_knn,
      const VirtualRootSupportEncoder<M>& root_encoder,
      LinkageEvaluationService& scheduler,
      const CausalCpuLinkageEvaluator<M>* cpu_evaluator = nullptr) noexcept
      : codebooks_(&codebooks),
        good_candidates_(&good_candidates),
        exact_knn_(&exact_knn),
        root_encoder_(&root_encoder),
        scheduler_(&scheduler),
        cpu_evaluator_(cpu_evaluator) {}

  [[nodiscard]] StatusOr<FullTopologyClusterProduct<M>> Build(
      const FullTopologyClusterBuildRequest<M>& request,
      const FullTopologyPolicy& policy) const override;

 private:
  const ProductCodebooks<M>* codebooks_ = nullptr;
  const CpuExactParentCandidateProvider* good_candidates_ = nullptr;
  const CpuExactKnnProvider* exact_knn_ = nullptr;
  const VirtualRootSupportEncoder<M>* root_encoder_ = nullptr;
  LinkageEvaluationService* scheduler_ = nullptr;
  const CausalCpuLinkageEvaluator<M>* cpu_evaluator_ = nullptr;
};

extern template class FullTopologyClusterBuilder<5>;
extern template class FullTopologyClusterBuilder<10>;

}  // namespace stlqpp
