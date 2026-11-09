#include "stlqpp/training/linkage/topology/cluster_topology_state.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"

namespace stlqpp {
namespace {

Status ValidateMatrix(ColumnMajorMatrixF32View matrix, std::uint32_t rows,
                      bool permit_empty, bool check_finite = true) {
  if (matrix.columns == 0U && permit_empty) {
    return matrix.values == nullptr &&
                   (matrix.rows == 0U || matrix.rows == rows)
               ? Status::Ok()
               : Status::InvalidArgument(
                     "empty topology matrix has an invalid extent");
  }
  if (matrix.values == nullptr || matrix.rows != rows || matrix.columns == 0U) {
    return Status::InvalidArgument("topology matrix has an invalid extent");
  }
  if (check_finite) {
    const std::size_t count =
        static_cast<std::size_t>(matrix.rows) * matrix.columns;
    for (std::size_t index = 0U; index < count; ++index) {
      if (!std::isfinite(matrix.values[index])) {
        return Status::InvalidArgument(
            "topology matrix contains non-finite data");
      }
    }
  }
  return Status::Ok();
}

}  // namespace

template <int M>
struct ClusterTopologyState<M>::State final {
  const RootCodebooks<M>* roots = nullptr;
  const ProductCodebooks<M>* product = nullptr;
  LinkageEvaluationKind evaluation_kind = LinkageEvaluationKind::kEdgeRsd;
  std::uint32_t cid = 0U;
  std::uint32_t dimension = 0U;
  std::uint32_t virtual_count = 0U;
  std::vector<float> owned_exact_real;
  ColumnMajorMatrixF32View exact_real;
  std::vector<float> virtual_reconstructions;
  std::vector<float> candidate_reconstruction;
  ClusterTopologyResult<M> result;
  std::vector<std::uint8_t> visible;
  std::optional<std::uint32_t> pending_node;
  std::vector<std::uint32_t> pending_columns;
  mutable std::vector<std::uint32_t> seen_node_epochs;
  mutable std::vector<std::uint32_t> candidate_epochs;
  mutable std::uint32_t next_epoch = 0U;

  [[nodiscard]] std::uint32_t NextEpoch() const {
    ++next_epoch;
    if (next_epoch == 0U) {
      std::fill(seen_node_epochs.begin(), seen_node_epochs.end(), 0U);
      std::fill(candidate_epochs.begin(), candidate_epochs.end(), 0U);
      ++next_epoch;
    }
    return next_epoch;
  }

  [[nodiscard]] const float* ParentReconstruction(
      LinkageParentReference parent) const noexcept {
    return parent.kind == LinkageParentKind::kVirtual
               ? virtual_reconstructions.data() +
                     static_cast<std::size_t>(parent.local_id) * dimension
               : result.real_reconstructions.data() +
                     static_cast<std::size_t>(parent.local_id) * dimension;
  }

  [[nodiscard]] std::uint32_t ParentDepth(
      LinkageParentReference parent) const noexcept {
    return parent.kind == LinkageParentKind::kVirtual
               ? 0U
               : result.depths[parent.local_id];
  }

  [[nodiscard]] std::uint32_t RegisteredColumn(
      LinkageParentReference parent) const noexcept {
    return parent.kind == LinkageParentKind::kVirtual
               ? parent.local_id
               : virtual_count + parent.local_id;
  }

  [[nodiscard]] LinkageParentReference DecodeColumn(
      std::uint32_t column) const noexcept {
    return column < virtual_count
               ? LinkageParentReference{LinkageParentKind::kVirtual, column}
               : LinkageParentReference{LinkageParentKind::kReal,
                                        column - virtual_count};
  }
};

template <int M>
ClusterTopologyState<M>::ClusterTopologyState(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
ClusterTopologyState<M>::~ClusterTopologyState() = default;

template <int M>
StatusOr<std::unique_ptr<ClusterTopologyState<M>>>
ClusterTopologyState<M>::Create(
    const ProductCodebooks<M>& codebooks,
    ClusterTopologyInitialStateView<M> initial,
    ClusterTopologyInitialStateOwnership<M> ownership,
    std::vector<float>* owned_virtual_reconstructions, bool borrow_exact_real) {
  return CreateWithKind(codebooks.root_codebooks(), &codebooks, initial,
                        LinkageEvaluationKind::kEdgeRsd, ownership,
                        owned_virtual_reconstructions, borrow_exact_real);
}

template <int M>
StatusOr<std::unique_ptr<ClusterTopologyState<M>>>
ClusterTopologyState<M>::CreateRootOnlyInitialization(
    const RootCodebooks<M>& codebooks,
    ClusterTopologyInitialStateView<M> initial) {
  return CreateWithKind(codebooks, nullptr, initial,
                        LinkageEvaluationKind::kRootOnlyInitialization, {},
                        nullptr, false);
}

template <int M>
StatusOr<std::unique_ptr<ClusterTopologyState<M>>>
ClusterTopologyState<M>::CreateWithKind(
    const RootCodebooks<M>& roots, const ProductCodebooks<M>* product,
    ClusterTopologyInitialStateView<M> initial, LinkageEvaluationKind kind,
    ClusterTopologyInitialStateOwnership<M> ownership,
    std::vector<float>* owned_virtual_reconstructions, bool borrow_exact_real) {
  const std::uint32_t dimension = roots.shape().dimension;
  const std::uint32_t width = roots.m();
  if (initial.cid >= roots.shape().nlist || initial.exact_real.columns == 0U ||
      initial.basic_real_reconstructions.columns !=
          initial.exact_real.columns) {
    return Status::InvalidArgument(
        "topology initial state has an invalid CID or real count");
  }
  const bool validated_basic = initial.validated_basic.has_value();
  Status validation =
      ValidateMatrix(initial.exact_real, dimension, false, !validated_basic);
  if (!validation.ok()) {
    return validation;
  }
  validation = ValidateMatrix(initial.basic_real_reconstructions, dimension,
                              false, !validated_basic);
  if (!validation.ok()) {
    return validation;
  }
  validation = ValidateMatrix(initial.virtual_reconstructions, dimension, true);
  if (!validation.ok()) {
    return validation;
  }
  const std::uint32_t real_count = initial.exact_real.columns;
  const std::size_t real_value_count =
      static_cast<std::size_t>(dimension) * real_count;
  const bool owns_any_basic = ownership.real_reconstructions != nullptr ||
                              ownership.supports != nullptr ||
                              ownership.coefficients != nullptr ||
                              ownership.squared_errors != nullptr;
  const bool owns_all_basic = ownership.real_reconstructions != nullptr &&
                              ownership.supports != nullptr &&
                              ownership.coefficients != nullptr &&
                              ownership.squared_errors != nullptr;
  const std::size_t virtual_value_count =
      static_cast<std::size_t>(dimension) *
      initial.virtual_reconstructions.columns;
  if (initial.basic_supports.size() != real_count ||
      initial.basic_coefficients.size() !=
          static_cast<std::size_t>(real_count) * width ||
      initial.basic_squared_errors.size() != real_count ||
      owns_any_basic != owns_all_basic ||
      (owns_all_basic &&
       (ownership.real_reconstructions->size() != real_value_count ||
        ownership.real_reconstructions->data() !=
            initial.basic_real_reconstructions.values ||
        ownership.supports->size() != real_count ||
        ownership.supports->data() != initial.basic_supports.data() ||
        ownership.coefficients->size() !=
            static_cast<std::size_t>(real_count) * width ||
        ownership.coefficients->data() != initial.basic_coefficients.data() ||
        ownership.squared_errors->size() != real_count ||
        ownership.squared_errors->data() !=
            initial.basic_squared_errors.data())) ||
      (owned_virtual_reconstructions != nullptr &&
       (owned_virtual_reconstructions->size() != virtual_value_count ||
        owned_virtual_reconstructions->data() !=
            initial.virtual_reconstructions.values))) {
    return Status::InvalidArgument(
        "topology basic state has an invalid support or coefficient extent");
  }
  if (!validated_basic) {
    for (const float coefficient : initial.basic_coefficients) {
      if (!std::isfinite(coefficient)) {
        return Status::InvalidArgument(
            "topology basic coefficient is non-finite");
      }
    }
    for (const float error : initial.basic_squared_errors) {
      if (!std::isfinite(error) || error < 0.0F) {
        return Status::InvalidArgument(
            "topology basic squared error is invalid");
      }
    }
    for (const LinkageCodeTuple<M>& support : initial.basic_supports) {
      if (support.first != initial.cid) {
        return Status::InvalidArgument(
            "topology basic root support does not match its native CID");
      }
    }
  }
#ifndef NDEBUG
  if (validated_basic) {
    assert(ValidateMatrix(initial.exact_real, dimension, false).ok());
    assert(ValidateMatrix(initial.basic_real_reconstructions, dimension, false)
               .ok());
    for (const float coefficient : initial.basic_coefficients) {
      assert(std::isfinite(coefficient));
    }
    for (const float error : initial.basic_squared_errors) {
      assert(std::isfinite(error) && error >= 0.0F);
    }
    for (const LinkageCodeTuple<M>& support : initial.basic_supports) {
      assert(support.first == initial.cid);
    }
  }
#endif

  auto state = std::make_unique<State>();
  state->roots = &roots;
  state->product = product;
  state->evaluation_kind = kind;
  state->cid = initial.cid;
  state->dimension = dimension;
  state->virtual_count = initial.virtual_reconstructions.columns;
  state->candidate_reconstruction.resize(dimension);
  if (borrow_exact_real) {
    state->exact_real = initial.exact_real;
  } else {
    state->owned_exact_real.assign(
        initial.exact_real.values,
        initial.exact_real.values + real_value_count);
    state->exact_real = {state->owned_exact_real.data(), dimension, real_count};
  }
  if (state->virtual_count != 0U) {
    if (owned_virtual_reconstructions != nullptr) {
      state->virtual_reconstructions =
          std::move(*owned_virtual_reconstructions);
    } else {
      state->virtual_reconstructions.assign(
          initial.virtual_reconstructions.values,
          initial.virtual_reconstructions.values + virtual_value_count);
    }
  }
  state->result.m = width;
  state->result.cid = initial.cid;
  state->result.dimension = dimension;
  state->result.virtual_count = state->virtual_count;
  state->result.roles.assign(real_count, LinkageNodeRole::kRoot);
  state->result.parents.resize(real_count);
  state->result.depths.assign(real_count, 0U);
  if (owns_all_basic) {
    state->result.supports = std::move(*ownership.supports);
    state->result.coefficients = std::move(*ownership.coefficients);
    state->result.squared_errors = std::move(*ownership.squared_errors);
    state->result.real_reconstructions =
        std::move(*ownership.real_reconstructions);
  } else {
    state->result.supports.assign(initial.basic_supports.begin(),
                                  initial.basic_supports.end());
    state->result.coefficients.assign(initial.basic_coefficients.begin(),
                                      initial.basic_coefficients.end());
    state->result.squared_errors.assign(initial.basic_squared_errors.begin(),
                                        initial.basic_squared_errors.end());
    state->result.real_reconstructions.assign(
        initial.basic_real_reconstructions.values,
        initial.basic_real_reconstructions.values + real_value_count);
  }
  state->visible.assign(real_count, 0U);
  state->seen_node_epochs.assign(real_count, 0U);
  state->candidate_epochs.assign(
      static_cast<std::size_t>(state->virtual_count) + real_count, 0U);
  return std::unique_ptr<ClusterTopologyState<M>>(
      new ClusterTopologyState<M>(std::move(state)));
}

template <int M>
Status ClusterTopologyState<M>::PublishInitialRoot(
    std::uint32_t real_local_id) {
  if (state_->pending_node.has_value() ||
      real_local_id >= state_->visible.size() ||
      state_->visible[real_local_id] != 0U) {
    return Status::FailedPrecondition(
        "initial topology root is duplicate, pending or outside the cluster");
  }
  state_->visible[real_local_id] = 1U;
  return Status::Ok();
}

template <int M>
StatusOr<std::optional<PreparedClusterTopologyEvaluation>>
ClusterTopologyState<M>::PrepareVisibleEvaluation(
    LinkageClusterHandle cluster, std::uint64_t required_generation,
    ArrayView<const ClusterTopologyEvaluationRowView> rows,
    std::uint32_t maximum_depth) const {
  if (cluster.value == 0U || rows.empty() || maximum_depth == 0U ||
      state_->pending_node.has_value()) {
    return Status::FailedPrecondition(
        "visible topology batch is unavailable at this causal boundary");
  }
  PreparedClusterTopologyEvaluation output;
  output.task.m = m();
  output.task.kind = state_->evaluation_kind;
  output.task.cluster = cluster;
  output.task.required_generation = required_generation;
  output.task.dimension = state_->dimension;
  output.task.row_offsets.push_back(0U);
  const std::uint32_t node_epoch = state_->NextEpoch();
  for (const ClusterTopologyEvaluationRowView& row : rows) {
    if (row.real_local_id >= state_->visible.size() ||
        state_->visible[row.real_local_id] != 0U ||
        state_->seen_node_epochs[row.real_local_id] == node_epoch) {
      return Status::FailedPrecondition(
          "visible topology batch contains an unavailable row");
    }
    if (const Status status =
            ValidateIcmLayerOrders<M>(1U, row.icm_layer_order, m());
        !status.ok()) {
      return status;
    }
    state_->seen_node_epochs[row.real_local_id] = node_epoch;
    const std::uint32_t candidate_epoch = state_->NextEpoch();
    std::uint32_t admitted = 0U;
    for (std::size_t index = 0U; index < row.ranked_candidates.size();
         ++index) {
      const ClusterTopologyRankedCandidate candidate =
          row.ranked_candidates[index];
      const LinkageParentReference parent = candidate.parent;
      const bool valid_virtual = parent.kind == LinkageParentKind::kVirtual &&
                                 parent.local_id < state_->virtual_count;
      const bool valid_real = parent.kind == LinkageParentKind::kReal &&
                              parent.local_id < state_->visible.size() &&
                              parent.local_id != row.real_local_id &&
                              state_->visible[parent.local_id] != 0U;
      if (!valid_virtual && !valid_real) {
        return Status::FailedPrecondition(
            "visible topology batch references an invisible parent");
      }
      const std::uint32_t column = state_->RegisteredColumn(parent);
      if (state_->ParentDepth(parent) + 1U <= maximum_depth) {
        if (state_->candidate_epochs[column] == candidate_epoch) {
          return Status::InvalidArgument(
              "visible topology candidate row contains a duplicate parent");
        }
        state_->candidate_epochs[column] = candidate_epoch;
        ++admitted;
        output.task.candidate_columns.push_back(column);
        output.original_candidate_ranks.push_back(candidate.original_rank);
      }
    }
    if (admitted == 0U && !row.preserve_empty_candidate_row) {
      continue;
    }
    const float* exact = state_->exact_real.column(row.real_local_id);
    output.task.node_vectors.insert(output.task.node_vectors.end(), exact,
                                    exact + state_->dimension);
    output.task.row_offsets.push_back(
        static_cast<std::uint32_t>(output.task.candidate_columns.size()));
    if (state_->evaluation_kind == LinkageEvaluationKind::kEdgeRsd) {
      output.task.icm_layer_orders.insert(output.task.icm_layer_orders.end(),
                                          row.icm_layer_order.begin(),
                                          row.icm_layer_order.end());
    } else if (output.task.icm_layer_orders.empty()) {
      output.task.icm_layer_orders.assign(row.icm_layer_order.begin(),
                                          row.icm_layer_order.end());
    } else if (!std::equal(output.task.icm_layer_orders.begin(),
                           output.task.icm_layer_orders.end(),
                           row.icm_layer_order.begin(),
                           row.icm_layer_order.end())) {
      return Status::InvalidArgument(
          "root-only evaluation rows require one shared ICM order");
    }
    output.real_local_ids.push_back(row.real_local_id);
    output.source_row_positions.push_back(row.source_position);
  }
  if (output.real_local_ids.empty()) {
    return std::optional<PreparedClusterTopologyEvaluation>{};
  }
  return std::optional<PreparedClusterTopologyEvaluation>(std::move(output));
}

template <int M>
StatusOr<bool> ClusterTopologyState<M>::PreparePending(
    std::uint32_t real_local_id,
    ArrayView<const LinkageParentReference> ordered_candidates,
    std::uint32_t maximum_depth) {
  if (maximum_depth == 0U || state_->pending_node.has_value() ||
      real_local_id >= state_->visible.size() ||
      state_->visible[real_local_id] != 0U) {
    return Status::FailedPrecondition(
        "topology node is not available for causal preparation");
  }

  state_->pending_columns.clear();
  state_->pending_columns.reserve(ordered_candidates.size());
  for (const LinkageParentReference parent : ordered_candidates) {
    const bool valid_virtual = parent.kind == LinkageParentKind::kVirtual &&
                               parent.local_id < state_->virtual_count;
    const bool valid_real = parent.kind == LinkageParentKind::kReal &&
                            parent.local_id < state_->visible.size() &&
                            parent.local_id != real_local_id &&
                            state_->visible[parent.local_id] != 0U;
    if (!valid_virtual && !valid_real) {
      return Status::FailedPrecondition(
          "topology candidate is not causally visible");
    }
    const std::uint32_t column = state_->RegisteredColumn(parent);
    if (std::find(state_->pending_columns.begin(),
                  state_->pending_columns.end(),
                  column) != state_->pending_columns.end()) {
      return Status::InvalidArgument(
          "topology candidate row contains a duplicate parent");
    }
    if (state_->ParentDepth(parent) + 1U <= maximum_depth) {
      state_->pending_columns.push_back(column);
    }
  }
  if (state_->pending_columns.empty()) {
    state_->visible[real_local_id] = 1U;
    return false;
  }
  state_->pending_node = real_local_id;
  return true;
}

template <int M>
StatusOr<std::optional<LinkageEvaluationTask>>
ClusterTopologyState<M>::PrepareNode(
    LinkageClusterHandle cluster, std::uint64_t required_generation,
    std::uint32_t real_local_id,
    ArrayView<const LinkageParentReference> ordered_candidates,
    ArrayView<const std::uint8_t> icm_layer_order,
    std::uint32_t maximum_depth) {
  if (cluster.value == 0U) {
    return Status::FailedPrecondition(
        "topology node requires a registered cluster");
  }
  if (const Status status = ValidateIcmLayerOrders<M>(1U, icm_layer_order, m());
      !status.ok()) {
    return status;
  }
  auto prepared =
      PreparePending(real_local_id, ordered_candidates, maximum_depth);
  if (!prepared.ok()) {
    return prepared.status();
  }
  if (!prepared.value()) {
    return std::optional<LinkageEvaluationTask>{};
  }
  LinkageEvaluationTask task;
  task.m = m();
  task.kind = state_->evaluation_kind;
  task.cluster = cluster;
  task.required_generation = required_generation;
  task.dimension = state_->dimension;
  const float* exact = state_->exact_real.column(real_local_id);
  task.node_vectors.assign(exact, exact + state_->dimension);
  task.candidate_columns = state_->pending_columns;
  task.row_offsets = {
      0U, static_cast<std::uint32_t>(state_->pending_columns.size())};
  task.icm_layer_orders.assign(icm_layer_order.begin(), icm_layer_order.end());
  return std::optional<LinkageEvaluationTask>(std::move(task));
}

template <int M>
StatusOr<bool> ClusterTopologyState<M>::PrepareNodeCommit(
    std::uint32_t real_local_id,
    ArrayView<const LinkageParentReference> ordered_candidates,
    std::uint32_t maximum_depth) {
  return PreparePending(real_local_id, ordered_candidates, maximum_depth);
}

template <int M>
StatusOr<std::optional<ClusterTopologyReconstructionUpdate>>
ClusterTopologyState<M>::CommitPending(LinkageEvaluationResult result) {
  if (!state_->pending_node.has_value() || result.parent_columns.size() != 1U ||
      result.supports.size() != 1U || result.supports.m() != m() ||
      result.coefficients.size() != static_cast<std::size_t>(m()) ||
      result.squared_errors.size() != 1U) {
    return Status::FailedPrecondition(
        "topology evaluator result does not match the pending node: "
        "pending=" +
        std::to_string(state_->pending_node.has_value()) +
        " parents=" + std::to_string(result.parent_columns.size()) +
        " supports=" + std::to_string(result.supports.size()) +
        " coefficients=" + std::to_string(result.coefficients.size()) +
        " errors=" + std::to_string(result.squared_errors.size()));
  }
  ClusterTopologyEvaluationWinner<M> winner;
  winner.parent_column = result.parent_columns[0];
  winner.support = result.supports.template tuple<M>(0);
  std::copy_n(result.coefficients.begin(), m(), winner.coefficients.begin());
  winner.squared_error = result.squared_errors[0];
  return CommitPending(winner);
}

template <int M>
StatusOr<std::optional<ClusterTopologyReconstructionUpdate>>
ClusterTopologyState<M>::CommitPending(
    const ClusterTopologyEvaluationWinner<M>& result) {
  if (!state_->pending_node.has_value()) {
    return Status::FailedPrecondition(
        "fixed topology evaluator result has no pending node");
  }
  const std::uint32_t node = *state_->pending_node;
  const std::uint32_t parent_column = result.parent_column;
  if (std::find(state_->pending_columns.begin(), state_->pending_columns.end(),
                parent_column) == state_->pending_columns.end()) {
    return Status::FailedPrecondition(
        "topology evaluator selected a parent outside the pending row");
  }
  for (std::uint32_t layer = 0U; layer < m(); ++layer) {
    const float coefficient = result.coefficients[layer];
    if (!std::isfinite(coefficient)) {
      return Status::FailedPrecondition(
          "topology evaluator returned a non-finite coefficient");
    }
  }
  const std::uint32_t first_limit =
      state_->evaluation_kind == LinkageEvaluationKind::kEdgeRsd
          ? 256U
          : state_->roots->shape().nlist;
  if (result.support.first >= first_limit) {
    return Status::FailedPrecondition(
        "topology evaluator returned a layer-zero code outside its domain");
  }

  state_->pending_node.reset();
  state_->pending_columns.clear();
  state_->visible[node] = 1U;
  if (!(result.squared_error < state_->result.squared_errors[node])) {
    return std::optional<ClusterTopologyReconstructionUpdate>{};
  }

  const LinkageParentReference parent = state_->DecodeColumn(parent_column);
  const float* parent_reconstruction = state_->ParentReconstruction(parent);
  float* reconstruction = state_->candidate_reconstruction.data();
  std::fill_n(reconstruction, state_->dimension, 0.0F);
  for (std::uint32_t layer = 0U; layer < m(); ++layer) {
    const float* atom = nullptr;
    if (state_->evaluation_kind ==
        LinkageEvaluationKind::kRootOnlyInitialization) {
      atom = layer == 0U ? state_->roots->root0().column(result.support.first)
                         : state_->roots->root_small(layer).column(
                               result.support.trailing[layer - 1U]);
    } else {
      assert(state_->product != nullptr);
      atom = state_->product->edge(layer).column(result.support.code(layer));
    }
    const float coefficient = result.coefficients[layer];
#pragma omp simd
    for (std::uint32_t axis = 0U; axis < state_->dimension; ++axis) {
      reconstruction[axis] += coefficient * atom[axis];
    }
  }
#pragma omp simd
  for (std::uint32_t axis = 0U; axis < state_->dimension; ++axis) {
    reconstruction[axis] = parent_reconstruction[axis] + reconstruction[axis];
  }
  const float* exact = state_->exact_real.column(node);
  float squared_error = 0.0F;
  for (std::uint32_t axis = 0U; axis < state_->dimension; ++axis) {
    const float difference = exact[axis] - reconstruction[axis];
    squared_error += difference * difference;
  }
  state_->result.roles[node] = LinkageNodeRole::kLinked;
  state_->result.parents[node] = parent;
  state_->result.depths[node] = state_->ParentDepth(parent) + 1U;
  state_->result.supports[node] = result.support;
  std::copy_n(result.coefficients.begin(), m(),
              state_->result.coefficients.begin() +
                  static_cast<std::size_t>(node) * m());
  float* committed_reconstruction =
      state_->result.real_reconstructions.data() +
      static_cast<std::size_t>(node) * state_->dimension;
  std::copy_n(reconstruction, state_->dimension, committed_reconstruction);
  state_->result.squared_errors[node] = squared_error;
  ClusterTopologyReconstructionUpdate update;
  update.registered_column = state_->virtual_count + node;
  update.reconstruction = {committed_reconstruction, state_->dimension};
  return std::optional<ClusterTopologyReconstructionUpdate>(std::move(update));
}

template <int M>
bool ClusterTopologyState<M>::visible(
    std::uint32_t real_local_id) const noexcept {
  assert(real_local_id < state_->visible.size());
  return state_->visible[real_local_id] != 0U;
}

template <int M>
std::uint32_t ClusterTopologyState<M>::depth(
    std::uint32_t real_local_id) const noexcept {
  assert(real_local_id < state_->result.depths.size());
  return state_->result.depths[real_local_id];
}

template <int M>
std::uint32_t ClusterTopologyState<M>::real_count() const noexcept {
  return static_cast<std::uint32_t>(state_->visible.size());
}

template <int M>
std::uint32_t ClusterTopologyState<M>::virtual_count() const noexcept {
  return state_->virtual_count;
}

template <int M>
std::uint32_t ClusterTopologyState<M>::m() const noexcept {
  return state_->roots->m();
}

template <int M>
std::uint32_t ClusterTopologyState<M>::dimension() const noexcept {
  return state_->dimension;
}

template <int M>
std::uint32_t ClusterTopologyState<M>::cid() const noexcept {
  return state_->cid;
}

template <int M>
LinkageNodeRole ClusterTopologyState<M>::role(
    std::uint32_t real_local_id) const noexcept {
  assert(real_local_id < state_->result.roles.size());
  return state_->result.roles[real_local_id];
}

template <int M>
const std::optional<LinkageParentReference>& ClusterTopologyState<M>::parent(
    std::uint32_t real_local_id) const noexcept {
  assert(real_local_id < state_->result.parents.size());
  return state_->result.parents[real_local_id];
}

template <int M>
const LinkageCodeTuple<M>& ClusterTopologyState<M>::support(
    std::uint32_t real_local_id) const noexcept {
  assert(real_local_id < state_->result.supports.size());
  return state_->result.supports[real_local_id];
}

template <int M>
ArrayView<const float> ClusterTopologyState<M>::coefficients(
    std::uint32_t real_local_id) const noexcept {
  assert(real_local_id < state_->result.roles.size());
  return {state_->result.coefficients.data() +
              static_cast<std::size_t>(real_local_id) * m(),
          static_cast<std::size_t>(m())};
}

template <int M>
ArrayView<const float> ClusterTopologyState<M>::parent_reconstruction(
    LinkageParentReference parent) const noexcept {
  assert(parent.kind == LinkageParentKind::kVirtual
             ? parent.local_id < state_->virtual_count
             : parent.local_id < state_->result.roles.size());
  return {state_->ParentReconstruction(parent), state_->dimension};
}

template <int M>
ColumnMajorMatrixF32View ClusterTopologyState<M>::real_reconstructions()
    const noexcept {
  return {state_->result.real_reconstructions.data(), state_->dimension,
          real_count()};
}

template <int M>
ColumnMajorMatrixF32View ClusterTopologyState<M>::virtual_reconstructions()
    const noexcept {
  return {state_->virtual_reconstructions.data(), state_->dimension,
          state_->virtual_count};
}

template <int M>
StatusOr<ClusterTopologyResult<M>> ClusterTopologyState<M>::Finish() {
  if (state_->pending_node.has_value() ||
      std::find(state_->visible.begin(), state_->visible.end(), 0U) !=
          state_->visible.end()) {
    return Status::FailedPrecondition(
        "topology cluster cannot finish before every real node is visible");
  }
  return std::move(state_->result);
}

template class ClusterTopologyState<0>;
template class ClusterTopologyState<5>;
template class ClusterTopologyState<12>;
template class ClusterTopologyState<6>;
template class ClusterTopologyState<10>;

}  // namespace stlqpp
