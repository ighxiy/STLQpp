#pragma once

#include "stlqpp/model/model_shape.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/linkage_code_tuple.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/model/root_codebooks.h"
#include "stlqpp/model/topology/typed_forest.h"
#include "stlqpp/training/linkage/scheduler/linkage_evaluation.h"
#include "stlqpp/training/linkage/topology/causal_topology_types.h"

namespace stlqpp {

template <int M>
class BaseListFullTopologySource;
template <int M>
class TrainingListRootInitializationSource;

// Created only after the base-list source has decoded coefficients and
// checked the finite reconstruction error for every real row.
class ValidatedBaseListBasicState final {
 public:
  ValidatedBaseListBasicState(const ValidatedBaseListBasicState&) = default;
  ValidatedBaseListBasicState& operator=(const ValidatedBaseListBasicState&) =
      default;

 private:
  template <int M>
  friend class BaseListFullTopologySource;
  template <int M>
  friend class TrainingListRootInitializationSource;
  ValidatedBaseListBasicState() = default;
};

template <int M>
struct ClusterTopologyInitialStateView final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  std::uint32_t cid = 0U;
  ColumnMajorMatrixF32View exact_real;
  ColumnMajorMatrixF32View basic_real_reconstructions;
  ColumnMajorMatrixF32View virtual_reconstructions;
  ArrayView<const LinkageCodeTuple<M>> basic_supports;
  ArrayView<const float> basic_coefficients;
  ArrayView<const float> basic_squared_errors;
  std::optional<ValidatedBaseListBasicState> validated_basic = std::nullopt;
};

// Optional all-or-none owner transfer for the basic state represented by an
// InitialStateView.  Every pointer must back its corresponding view exactly.
// General callers leave this empty and retain the copying API.
template <int M>
struct ClusterTopologyInitialStateOwnership final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  std::vector<float>* real_reconstructions = nullptr;
  std::vector<LinkageCodeTuple<M>>* supports = nullptr;
  std::vector<float>* coefficients = nullptr;
  std::vector<float>* squared_errors = nullptr;
};

struct ClusterTopologyReconstructionUpdate final {
  std::uint32_t registered_column = 0U;
  // Borrows the committed real-reconstruction column. Cluster state owns a
  // fixed-size matrix for its full lifetime, and callers consume this view
  // synchronously into their bounded update storage.
  ArrayView<const float> reconstruction;
};

struct ClusterTopologyRankedCandidate final {
  LinkageParentReference parent;
  std::uint32_t original_rank = 0U;
};

struct ClusterTopologyEvaluationRowView final {
  std::uint32_t real_local_id = 0U;
  std::uint32_t source_position = 0U;
  ArrayView<const ClusterTopologyRankedCandidate> ranked_candidates;
  ArrayView<const std::uint8_t> icm_layer_order;
  // DEV's root-only inner batch retains rows that only have same-layer
  // candidates.  They carry no inner pairs, but still occupy a node column
  // in the batched GEMM and therefore affect the exact floating-point order.
  bool preserve_empty_candidate_row = false;
};

struct PreparedClusterTopologyEvaluation final {
  LinkageEvaluationTask task;
  std::vector<std::uint32_t> real_local_ids;
  // Position in the caller's row view for each retained evaluator node.
  std::vector<std::uint32_t> source_row_positions;
  // Flattened in the same CSR order as task.candidate_columns.
  std::vector<std::uint32_t> original_candidate_ranks;
};

template <int M>
struct ClusterTopologyEvaluationWinner final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  std::uint32_t parent_column = 0U;
  LinkageCodeTuple<M> support;
  std::array<float, (M == 0 ? kMaximumModelLayers : M)> coefficients{};
  float squared_error = 0.0F;
};

template <int M>
struct ClusterTopologyResult final {
  std::uint32_t m = M;
  static_assert(M == 0 || (M >= 2 && M <= 20));
  std::uint32_t cid = 0U;
  std::uint32_t dimension = 0U;
  std::uint32_t virtual_count = 0U;
  std::vector<LinkageNodeRole> roles;
  std::vector<std::optional<LinkageParentReference>> parents;
  std::vector<std::uint32_t> depths;
  std::vector<LinkageCodeTuple<M>> supports;
  std::vector<float> coefficients;
  std::vector<float> squared_errors;
  std::vector<float> real_reconstructions;
  std::uint64_t evaluation_tasks = 0U;
  std::uint64_t cpu_evaluation_tasks = 0U;
  std::array<std::uint64_t, kCausalEvaluationKindCount>
      evaluation_tasks_by_kind{};
  std::array<std::uint64_t, kCausalEvaluationKindCount>
      cpu_evaluation_tasks_by_kind{};
};

// Cluster-confined causal state. Candidate providers own geometry and radial
// legality; this object owns visibility, maximum depth, strict root-vs-link
// acceptance and reconstruction publication. Exactly one node may be pending.
template <int M>
class ClusterTopologyState final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<std::unique_ptr<ClusterTopologyState<M>>>
  Create(const ProductCodebooks<M>& codebooks,
         ClusterTopologyInitialStateView<M> initial,
         ClusterTopologyInitialStateOwnership<M> ownership = {},
         std::vector<float>* owned_virtual_reconstructions = nullptr,
         bool borrow_exact_real = false);

  // Distinct construction boundary for the legacy initialization forest.
  // Its accepted linked residuals are encoded and reconstructed with C_root,
  // including the potentially wide layer-zero domain. Ordinary Create()
  // remains permanently bound to C_one edge rsd.
  [[nodiscard]] static StatusOr<std::unique_ptr<ClusterTopologyState<M>>>
  CreateRootOnlyInitialization(const RootCodebooks<M>& codebooks,
                               ClusterTopologyInitialStateView<M> initial);

  ~ClusterTopologyState();
  ClusterTopologyState(const ClusterTopologyState&) = delete;
  ClusterTopologyState& operator=(const ClusterTopologyState&) = delete;

  [[nodiscard]] Status PublishInitialRoot(std::uint32_t real_local_id);

  // Builds one immutable multi-row evaluator task from candidates visible at
  // the current causal boundary. Rows whose candidates are all rejected by
  // the depth bound are omitted. This method never makes a node pending and
  // therefore cannot publish topology state.
  [[nodiscard]] StatusOr<std::optional<PreparedClusterTopologyEvaluation>>
  PrepareVisibleEvaluation(
      LinkageClusterHandle cluster, std::uint64_t required_generation,
      ArrayView<const ClusterTopologyEvaluationRowView> rows,
      std::uint32_t maximum_depth) const;

  // An empty optional means every candidate exceeded the depth bound; the
  // node has already been published as its unchanged basic root.
  [[nodiscard]] StatusOr<std::optional<LinkageEvaluationTask>> PrepareNode(
      LinkageClusterHandle cluster, std::uint64_t required_generation,
      std::uint32_t real_local_id,
      ArrayView<const LinkageParentReference> ordered_candidates,
      ArrayView<const std::uint8_t> icm_layer_order,
      std::uint32_t maximum_depth);

  // Opens the same authoritative pending/commit transaction as PrepareNode,
  // without allocating an evaluator task. The causal-window executor uses it
  // after exact candidates were evaluated in vectorized batches.
  [[nodiscard]] StatusOr<bool> PrepareNodeCommit(
      std::uint32_t real_local_id,
      ArrayView<const LinkageParentReference> ordered_candidates,
      std::uint32_t maximum_depth);

  // An empty optional means the evaluator did not strictly beat the node's
  // basic root. The root is nevertheless made causally visible.
  [[nodiscard]] StatusOr<std::optional<ClusterTopologyReconstructionUpdate>>
  CommitPending(LinkageEvaluationResult result);

  // Fixed-extent hot-path form used after batched evaluation winner merging.
  [[nodiscard]] StatusOr<std::optional<ClusterTopologyReconstructionUpdate>>
  CommitPending(const ClusterTopologyEvaluationWinner<M>& result);

  [[nodiscard]] bool visible(std::uint32_t real_local_id) const noexcept;
  [[nodiscard]] std::uint32_t depth(std::uint32_t real_local_id) const noexcept;
  [[nodiscard]] std::uint32_t real_count() const noexcept;
  [[nodiscard]] std::uint32_t virtual_count() const noexcept;
  [[nodiscard]] std::uint32_t m() const noexcept;
  [[nodiscard]] std::uint32_t dimension() const noexcept;
  [[nodiscard]] std::uint32_t cid() const noexcept;
  [[nodiscard]] LinkageNodeRole role(
      std::uint32_t real_local_id) const noexcept;
  [[nodiscard]] const std::optional<LinkageParentReference>& parent(
      std::uint32_t real_local_id) const noexcept;
  [[nodiscard]] const LinkageCodeTuple<M>& support(
      std::uint32_t real_local_id) const noexcept;
  [[nodiscard]] ArrayView<const float> coefficients(
      std::uint32_t real_local_id) const noexcept;
  [[nodiscard]] ArrayView<const float> parent_reconstruction(
      LinkageParentReference parent) const noexcept;
  [[nodiscard]] ColumnMajorMatrixF32View real_reconstructions() const noexcept;
  [[nodiscard]] ColumnMajorMatrixF32View virtual_reconstructions()
      const noexcept;

  [[nodiscard]] StatusOr<ClusterTopologyResult<M>> Finish();

 private:
  struct State;
  explicit ClusterTopologyState(std::unique_ptr<State> state) noexcept;

  [[nodiscard]] static StatusOr<std::unique_ptr<ClusterTopologyState<M>>>
  CreateWithKind(const RootCodebooks<M>& roots,
                 const ProductCodebooks<M>* product,
                 ClusterTopologyInitialStateView<M> initial,
                 LinkageEvaluationKind kind,
                 ClusterTopologyInitialStateOwnership<M> ownership,
                 std::vector<float>* owned_virtual_reconstructions,
                 bool borrow_exact_real);

  [[nodiscard]] StatusOr<bool> PreparePending(
      std::uint32_t real_local_id,
      ArrayView<const LinkageParentReference> ordered_candidates,
      std::uint32_t maximum_depth);

  std::unique_ptr<State> state_;
};

extern template class ClusterTopologyState<5>;
extern template class ClusterTopologyState<10>;

}  // namespace stlqpp
