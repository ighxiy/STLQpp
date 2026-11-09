#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/update/edge_codebook_moments.h"

#include "stlqpp/model/topology/canonical_bfs_forest_order.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

template <int M>
Status ValidateSupportsAndCoefficients(
    std::uint32_t sample_count,
    ArrayView<const LinkageCodeTuple<M>> supports,
    ArrayView<const float> coefficients, std::uint32_t m) {
  if (supports.size() != sample_count ||
      coefficients.size() != static_cast<std::size_t>(sample_count) * m) {
    return Status::InvalidArgument(
        "edge-codebook moment streams have inconsistent row counts");
  }
  for (std::uint32_t sample = 0U; sample < sample_count; ++sample) {
    if (supports[sample].first >= 256U) {
      return Status::InvalidArgument(
          "C_one layer-zero code exceeds its byte domain");
    }
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      if (!std::isfinite(
              coefficients[static_cast<std::size_t>(sample) * m + layer])) {
        return Status::InvalidArgument(
            "edge-codebook coefficient is non-finite");
      }
    }
  }
  return Status::Ok();
}

Status ValidateTargets(ColumnMajorMatrixF32View targets,
                       std::uint32_t dimension) {
  if (targets.rows != dimension ||
      (targets.columns != 0U && targets.values == nullptr)) {
    return Status::InvalidArgument(
        "edge-codebook target matrix has an invalid shape");
  }
  const std::size_t count =
      static_cast<std::size_t>(targets.rows) * targets.columns;
  for (std::size_t index = 0U; index < count; ++index) {
    if (!std::isfinite(targets.values[index])) {
      return Status::InvalidArgument(
          "edge-codebook target matrix contains non-finite data");
    }
  }
  return Status::Ok();
}

template <int M>
void ReconstructFrozenRoot(const ProductCodebooks<M>& codebooks,
                           std::uint32_t cid,
                           const std::uint8_t* small_codes,
                           const float* coefficients,
                           float* output) noexcept {
  const std::uint32_t dimension = codebooks.shape().dimension;
  const std::uint32_t m = M == 0 ? codebooks.shape().m : M;
  const float* first = codebooks.root0().column(cid);
  for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
    output[axis] = coefficients[0U] * first[axis];
  }
  for (std::uint32_t layer = 1U;
       layer < m; ++layer) {
    const float* atom = codebooks.root_small(layer).column(
        small_codes[layer - 1U]);
    const float coefficient = coefficients[layer];
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      output[axis] += coefficient * atom[axis];
    }
  }
}

template <int M>
void ReconstructFrozenEdgeResidual(
    const ProductCodebooks<M>& codebooks,
    const LinkageCodeTuple<M>& support,
    const float* coefficients,
    float* output) noexcept {
  const std::uint32_t dimension = codebooks.shape().dimension;
  const std::uint32_t m = M == 0 ? codebooks.shape().m : M;
  const float* first = codebooks.edge(0U).column(support.first);
  for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
    output[axis] = coefficients[0U] * first[axis];
  }
  for (std::uint32_t layer = 1U;
       layer < m; ++layer) {
    const float* atom = codebooks.edge(layer).column(
        support.trailing[layer - 1U]);
    const float coefficient = coefficients[layer];
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      output[axis] += coefficient * atom[axis];
    }
  }
}

}  // namespace

template <int M>
EdgeCodebookMoments<M>::EdgeCodebookMoments(std::uint32_t dimension, std::uint32_t m)
    : m_(m), dimension_(dimension),
      packed_upper_gram_(
          static_cast<std::size_t>(atom_count()) * (atom_count() + 1U) / 2U, 0.0),
      atom_major_rhs_(static_cast<std::size_t>(atom_count()) * dimension, 0.0) {}

template <int M>
StatusOr<EdgeCodebookMoments<M>> EdgeCodebookMoments<M>::Create(
    std::uint32_t dimension, std::uint32_t m) {
  if (m < 2U || m > stlqpp::kMaximumModelLayers || (M != 0 && m != M))
    return Status::InvalidArgument("edge moment M is invalid");
  const std::uint32_t atoms = m * 256U;
  if (dimension == 0U ||
      static_cast<std::size_t>(dimension) >
          std::numeric_limits<std::size_t>::max() / atoms) {
    return Status::InvalidArgument(
        "edge-codebook moment dimension is invalid");
  }
  return EdgeCodebookMoments<M>(dimension, m);
}

template <int M>
void EdgeCodebookMoments<M>::AccumulateOne(
    const float* target,
    const LinkageCodeTuple<M>& support,
    const float* coefficients,
    double sample_weight) noexcept {
  std::array<std::uint32_t, (M == 0 ? kMaximumModelLayers : M)> atoms{};
  for (std::uint32_t layer = 0U;
       layer < m(); ++layer) {
    atoms[layer] = layer * 256U + support.code(layer);
  }
  for (std::uint32_t left = 0U;
       left < m(); ++left) {
    const std::uint32_t left_atom = atoms[left];
    const double left_coefficient = coefficients[left];
    double* rhs = atom_major_rhs_.data() +
        static_cast<std::size_t>(left_atom) * dimension_;
    for (std::uint32_t axis = 0U; axis < dimension_; ++axis) {
      rhs[axis] += sample_weight * left_coefficient * target[axis];
    }
    for (std::uint32_t right = 0U; right <= left; ++right) {
      const std::uint32_t row = std::min(left_atom, atoms[right]);
      const std::uint32_t column = std::max(left_atom, atoms[right]);
      packed_upper_gram_[PackedUpperIndex(row, column)] +=
          sample_weight * left_coefficient * coefficients[right];
    }
  }
  ++sample_count_;
}

template <int M>
Status EdgeCodebookMoments<M>::AccumulateTargets(
    ColumnMajorMatrixF32View targets,
    ArrayView<const LinkageCodeTuple<M>> supports,
    ArrayView<const float> coefficients) {
  if (const Status status = ValidateTargets(targets, dimension_);
      !status.ok()) {
    return status;
  }
  if (const Status status = ValidateSupportsAndCoefficients<M>(
          targets.columns, supports, coefficients, m());
      !status.ok()) {
    return status;
  }
  for (std::uint32_t sample = 0U; sample < targets.columns; ++sample) {
    AccumulateOne(
        targets.column(sample), supports[sample],
        coefficients.data() + static_cast<std::size_t>(sample) * m(),
        1.0);
  }
  return Status::Ok();
}

template <int M>
Status EdgeCodebookMoments<M>::AccumulateLinkedCluster(
    const LinkedTrainingClusterView<M>& cluster,
    double sample_weight) {
  if (!std::isfinite(sample_weight) || sample_weight <= 0.0) {
    return Status::InvalidArgument(
        "edge-codebook sample weight must be finite and positive");
  }
  if (cluster.topology == nullptr ||
      cluster.exact_real.rows != dimension_ ||
      cluster.exact_real.columns != cluster.topology->roles.size() ||
      cluster.topology->dimension != dimension_ || cluster.topology->m != m() ||
      cluster.topology->parents.size() != cluster.topology->roles.size() ||
      cluster.topology->depths.size() != cluster.topology->roles.size() ||
      cluster.topology->supports.size() != cluster.topology->roles.size() ||
      cluster.topology->coefficients.size() !=
          cluster.topology->roles.size() * m() ||
      cluster.real_reconstructions.rows != dimension_ ||
      cluster.real_reconstructions.columns !=
          cluster.topology->roles.size() ||
      cluster.virtual_reconstructions.rows != dimension_ ||
      cluster.virtual_reconstructions.columns !=
          cluster.topology->virtual_count ||
      (cluster.virtual_reconstructions.columns != 0U &&
       cluster.virtual_reconstructions.values == nullptr)) {
    return Status::InvalidArgument(
        "linked training cluster has inconsistent reconstruction streams");
  }
  if (const Status status = ValidateTargets(cluster.exact_real, dimension_);
      !status.ok()) {
    return status;
  }
  const std::uint32_t real_count = cluster.exact_real.columns;
  auto order = BuildCanonicalBfsForestOrder({
      .virtual_count = cluster.topology->virtual_count,
      .real_roles = cluster.topology->roles,
      .real_parents = cluster.topology->parents,
  });
  if (!order.ok()) {
    return order.status();
  }
  std::vector<float> target(dimension_);
  for (std::uint32_t bfs_real = order.value().real_root_count;
       bfs_real < real_count; ++bfs_real) {
    const std::uint32_t real = order.value().new_to_old_real[bfs_real];
    if (!cluster.topology->parents[real].has_value()) {
      return Status::FailedPrecondition(
          "linked training node has no parent");
    }
    const LinkageParentReference parent =
        *cluster.topology->parents[real];
    if (cluster.topology->supports[real].first >= 256U) {
      return Status::FailedPrecondition(
          "linked training node has an invalid C_one layer-zero code");
    }
    const float* coefficients = cluster.topology->coefficients.data() +
        static_cast<std::size_t>(real) * m();
    for (std::uint32_t layer = 0U;
         layer < m(); ++layer) {
      if (!std::isfinite(coefficients[layer])) {
        return Status::FailedPrecondition(
            "linked training node has a non-finite coefficient");
      }
    }
    const float* parent_reconstruction = nullptr;
    if (parent.kind == LinkageParentKind::kVirtual) {
      if (parent.local_id >= cluster.virtual_reconstructions.columns) {
        return Status::FailedPrecondition(
            "linked training node has an invalid virtual parent");
      }
      if (cluster.topology->depths[real] != 1U) {
        return Status::FailedPrecondition(
            "virtual-parent edge does not enter depth one");
      }
      parent_reconstruction =
          cluster.virtual_reconstructions.column(parent.local_id);
    } else {
      if (parent.local_id >= cluster.exact_real.columns ||
          cluster.topology->depths[parent.local_id] + 1U !=
              cluster.topology->depths[real]) {
        return Status::FailedPrecondition(
            "linked training node violates causal real-parent order");
      }
      parent_reconstruction =
          cluster.real_reconstructions.column(parent.local_id);
    }
    const float* exact = cluster.exact_real.column(real);
    for (std::uint32_t axis = 0U; axis < dimension_; ++axis) {
      target[axis] = exact[axis] - parent_reconstruction[axis];
    }
    AccumulateOne(
        target.data(), cluster.topology->supports[real],
        coefficients, sample_weight);
  }
  return Status::Ok();
}

template <int M>
Status EdgeCodebookMoments<M>::AccumulateFrozenLinkedCluster(
    const ProductCodebooks<M>& codebooks,
    const FrozenLinkedTrainingClusterView<M>& cluster,
    double sample_weight,
    const CanonicalBfsForestOrder* order) {
  return AccumulateFrozenLinkedClusterImpl<false, false>(
      codebooks, cluster, sample_weight, order);
}

template <int M>
Status EdgeCodebookMoments<M>::AccumulateFrozenLinkedClusterAfterValidation(
    const ProductCodebooks<M>& codebooks,
    const FrozenLinkedTrainingClusterView<M>& cluster,
    double sample_weight,
    const CanonicalBfsForestOrder& order) {
  return AccumulateFrozenLinkedClusterImpl<true, false>(
      codebooks, cluster, sample_weight, &order);
}

template <int M>
Status EdgeCodebookMoments<M>::
AccumulateFrozenLinkedClusterAfterReplayValidation(
    const ProductCodebooks<M>& codebooks,
    const FrozenLinkedTrainingClusterView<M>& cluster,
    double sample_weight) {
  return AccumulateFrozenLinkedClusterImpl<true, true>(
      codebooks, cluster, sample_weight, nullptr);
}

template <int M>
template <bool Prevalidated, bool ExactValidated>
Status EdgeCodebookMoments<M>::AccumulateFrozenLinkedClusterImpl(
    const ProductCodebooks<M>& codebooks,
    const FrozenLinkedTrainingClusterView<M>& cluster,
    double sample_weight,
    const CanonicalBfsForestOrder* order) {
  if (!std::isfinite(sample_weight) || sample_weight <= 0.0) {
    return Status::InvalidArgument(
        "edge-codebook sample weight must be finite and positive");
  }
  if (cluster.topology == nullptr) {
    return Status::InvalidArgument(
        "frozen linked training cluster has no topology");
  }
  const ClusterTopologyResult<M>& topology = *cluster.topology;
  const ModelShape& shape = codebooks.shape();
  const std::uint32_t real_count =
      static_cast<std::uint32_t>(topology.roles.size());
  const std::uint32_t virtual_count = topology.virtual_count;
  if (shape.dimension != dimension_ || shape.m != m() || topology.m != m() ||
      topology.dimension != dimension_ ||
      topology.cid >= shape.nlist ||
      cluster.exact_real.rows != dimension_ ||
      cluster.exact_real.columns != real_count) {
    return Status::InvalidArgument(
        "frozen linked training cluster has inconsistent streams");
  }
  if constexpr (Prevalidated) {
#ifndef NDEBUG
    assert(topology.parents.size() == real_count);
    assert(topology.depths.size() == real_count);
    assert(topology.supports.size() == real_count);
    assert(topology.coefficients.size() ==
           static_cast<std::size_t>(real_count) * m());
    assert(cluster.virtual_coefficients.size() ==
           static_cast<std::size_t>(virtual_count) * m());
    assert(cluster.virtual_small_codes.size() ==
           static_cast<std::size_t>(virtual_count) * (m() - 1U));
#endif
  } else if (topology.parents.size() != real_count ||
             topology.depths.size() != real_count ||
             topology.supports.size() != real_count ||
             topology.coefficients.size() !=
                 static_cast<std::size_t>(real_count) * m() ||
             cluster.virtual_coefficients.size() !=
                 static_cast<std::size_t>(virtual_count) * m() ||
             cluster.virtual_small_codes.size() !=
                 static_cast<std::size_t>(virtual_count) * (m() - 1U)) {
    return Status::InvalidArgument(
        "frozen linked training cluster has inconsistent streams");
  }
  if constexpr (ExactValidated) {
    assert(cluster.exact_real.columns == real_count);
    assert(cluster.exact_real.rows == dimension_);
    assert(real_count == 0U || cluster.exact_real.values != nullptr);
  } else {
    if (const Status status = ValidateTargets(
            cluster.exact_real, dimension_); !status.ok()) {
      return status;
    }
  }

  CanonicalBfsForestOrder local_order;
  if (order == nullptr) {
    auto built = BuildCanonicalBfsForestOrder({
        .virtual_count = virtual_count,
        .real_roles = topology.roles,
        .real_parents = topology.parents,
    });
    if (!built.ok()) return built.status();
    local_order = std::move(built).value();
    order = &local_order;
  }
  if constexpr (Prevalidated) {
#ifndef NDEBUG
    assert(order != nullptr);
    assert(order->virtual_count == virtual_count);
    assert(order->real_root_count <= real_count);
    assert(order->new_to_old_real.size() == real_count);
    assert(order->old_to_new_real.size() == real_count);
    assert(order->parent_local_1based.size() == real_count);
#endif
  } else if (order->virtual_count != virtual_count ||
             order->real_root_count > real_count ||
             order->new_to_old_real.size() != real_count ||
             order->old_to_new_real.size() != real_count ||
             order->parent_local_1based.size() != real_count) {
    return Status::InvalidArgument(
        "frozen linked cluster has an incompatible BFS order");
  }

  frozen_real_has_child_.assign(real_count, 0U);
  frozen_virtual_has_child_.assign(virtual_count, 0U);
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    if (topology.roles[real] != LinkageNodeRole::kLinked) {
      continue;
    }
    const LinkageParentReference parent = *topology.parents[real];
    if (parent.kind == LinkageParentKind::kVirtual) {
      frozen_virtual_has_child_[parent.local_id] = 1U;
    } else {
      frozen_real_has_child_[parent.local_id] = 1U;
    }
  }

  frozen_real_reconstructions_.resize(
      static_cast<std::size_t>(dimension_) * real_count);
  frozen_virtual_reconstructions_.resize(
      static_cast<std::size_t>(dimension_) * virtual_count);

  // Preserve the original reconstruction arithmetic and topological order,
  // but skip columns that cannot be consumed as a parent. Keeping this pass
  // separate from moment accumulation also preserves the old phase ordering.
  for (std::uint32_t root = 0U; root < virtual_count; ++root) {
    if (frozen_virtual_has_child_[root] == 0U) {
      continue;
    }
    ReconstructFrozenRoot<M>(
        codebooks, topology.cid,
        cluster.virtual_small_codes.data() +
            static_cast<std::size_t>(root) * (m() - 1U),
        cluster.virtual_coefficients.data() +
            static_cast<std::size_t>(root) * m(),
        frozen_virtual_reconstructions_.data() +
            static_cast<std::size_t>(root) * dimension_);
  }
  for (const std::uint32_t real : order->new_to_old_real) {
    if (frozen_real_has_child_[real] == 0U) {
      continue;
    }
    float* reconstruction = frozen_real_reconstructions_.data() +
        static_cast<std::size_t>(real) * dimension_;
    const float* coefficients = topology.coefficients.data() +
        static_cast<std::size_t>(real) * m();
    if (topology.roles[real] == LinkageNodeRole::kRoot) {
      ReconstructFrozenRoot<M>(
          codebooks, topology.cid, topology.supports[real].trailing.data(),
          coefficients, reconstruction);
      continue;
    }
    ReconstructFrozenEdgeResidual<M>(
        codebooks, topology.supports[real], coefficients, reconstruction);
    const LinkageParentReference parent = *topology.parents[real];
    const float* parent_reconstruction =
        parent.kind == LinkageParentKind::kVirtual
        ? frozen_virtual_reconstructions_.data() +
              static_cast<std::size_t>(parent.local_id) * dimension_
        : frozen_real_reconstructions_.data() +
              static_cast<std::size_t>(parent.local_id) * dimension_;
    for (std::uint32_t axis = 0U; axis < dimension_; ++axis) {
      reconstruction[axis] += parent_reconstruction[axis];
    }
  }

  frozen_target_.resize(dimension_);
  for (std::uint32_t bfs_real = order->real_root_count;
       bfs_real < real_count; ++bfs_real) {
    const std::uint32_t real = order->new_to_old_real[bfs_real];
    if constexpr (Prevalidated) {
      assert(topology.parents[real].has_value());
    } else if (!topology.parents[real].has_value()) {
      return Status::FailedPrecondition(
          "linked training node has no parent");
    }
    const LinkageParentReference parent = *topology.parents[real];
    if constexpr (Prevalidated) {
      assert(topology.supports[real].first < 256U);
    } else if (topology.supports[real].first >= 256U) {
      return Status::FailedPrecondition(
          "linked training node has an invalid C_one layer-zero code");
    }
    const float* coefficients = topology.coefficients.data() +
        static_cast<std::size_t>(real) * m();
    if constexpr (Prevalidated) {
#ifndef NDEBUG
      for (std::uint32_t layer = 0U;
           layer < m(); ++layer) {
        assert(std::isfinite(coefficients[layer]));
      }
#endif
    } else {
      for (std::uint32_t layer = 0U;
           layer < m(); ++layer) {
        if (!std::isfinite(coefficients[layer])) {
          return Status::FailedPrecondition(
              "linked training node has a non-finite coefficient");
        }
      }
    }

    const float* parent_reconstruction = nullptr;
    if (parent.kind == LinkageParentKind::kVirtual) {
      if constexpr (Prevalidated) {
        assert(parent.local_id < virtual_count);
        assert(topology.depths[real] == 1U);
      } else {
        if (parent.local_id >= virtual_count) {
          return Status::FailedPrecondition(
              "linked training node has an invalid virtual parent");
        }
        if (topology.depths[real] != 1U) {
          return Status::FailedPrecondition(
              "virtual-parent edge does not enter depth one");
        }
      }
      parent_reconstruction = frozen_virtual_reconstructions_.data() +
          static_cast<std::size_t>(parent.local_id) * dimension_;
    } else {
      if constexpr (Prevalidated) {
        assert(parent.local_id < real_count);
        assert(topology.depths[parent.local_id] + 1U ==
               topology.depths[real]);
      } else if (parent.local_id >= real_count ||
                 topology.depths[parent.local_id] + 1U !=
                     topology.depths[real]) {
        return Status::FailedPrecondition(
            "linked training node violates causal real-parent order");
      }
      parent_reconstruction = frozen_real_reconstructions_.data() +
          static_cast<std::size_t>(parent.local_id) * dimension_;
    }

    const float* exact = cluster.exact_real.column(real);
    for (std::uint32_t axis = 0U; axis < dimension_; ++axis) {
      frozen_target_[axis] = exact[axis] - parent_reconstruction[axis];
    }
    AccumulateOne(
        frozen_target_.data(), topology.supports[real], coefficients,
        sample_weight);
  }
  return Status::Ok();
}

template <int M>
Status EdgeCodebookMoments<M>::MergeFrom(
    const EdgeCodebookMoments<M>& shard) {
  if (shard.dimension_ != dimension_ ||
      shard.packed_upper_gram_.size() != packed_upper_gram_.size() ||
      shard.atom_major_rhs_.size() != atom_major_rhs_.size()) {
    return Status::InvalidArgument(
        "edge-codebook moment shard has an incompatible shape");
  }
  for (std::size_t index = 0U; index < packed_upper_gram_.size(); ++index) {
    packed_upper_gram_[index] += shard.packed_upper_gram_[index];
  }
  for (std::size_t index = 0U; index < atom_major_rhs_.size(); ++index) {
    atom_major_rhs_[index] += shard.atom_major_rhs_[index];
  }
  sample_count_ += shard.sample_count_;
  return Status::Ok();
}

template <int M>
Status EdgeCodebookMoments<M>::AccumulateRootOnlyInitializationCluster(
    const LinkedTrainingClusterView<M>& cluster,
    const RootToEdgeCodeMap& root_to_edge,
    double sample_weight) {
  return AccumulateRootOnlyInitializationClusterImpl<false>(
      cluster, root_to_edge, sample_weight);
}

template <int M>
Status EdgeCodebookMoments<M>::
AccumulateRootOnlyInitializationClusterAfterValidation(
    const LinkedTrainingClusterView<M>& cluster,
    const RootToEdgeCodeMap& root_to_edge,
    double sample_weight) {
  return AccumulateRootOnlyInitializationClusterImpl<true>(
      cluster, root_to_edge, sample_weight);
}

template <int M>
template <bool Prevalidated>
Status EdgeCodebookMoments<M>::AccumulateRootOnlyInitializationClusterImpl(
    const LinkedTrainingClusterView<M>& cluster,
    const RootToEdgeCodeMap& root_to_edge,
    double sample_weight) {
  if (!std::isfinite(sample_weight) || sample_weight <= 0.0 ||
      cluster.topology == nullptr ||
      cluster.exact_real.rows != dimension_ ||
      cluster.exact_real.columns != cluster.topology->roles.size() ||
      cluster.topology->dimension != dimension_ || cluster.topology->m != m() ||
      cluster.topology->parents.size() != cluster.topology->roles.size() ||
      cluster.topology->depths.size() != cluster.topology->roles.size() ||
      cluster.topology->supports.size() != cluster.topology->roles.size() ||
      cluster.topology->coefficients.size() !=
          cluster.topology->roles.size() * m() ||
      cluster.real_reconstructions.rows != dimension_ ||
      cluster.real_reconstructions.columns !=
          cluster.topology->roles.size() ||
      cluster.topology->virtual_count != 0U ||
      cluster.virtual_reconstructions.columns != 0U) {
    return Status::InvalidArgument(
        "root-only initialization cluster has inconsistent streams");
  }
  if (const Status status = ValidateTargets(cluster.exact_real, dimension_);
      !status.ok()) {
    return status;
  }

  std::vector<float> target(dimension_);
  const std::uint32_t real_count = cluster.exact_real.columns;
  auto order = BuildCanonicalBfsForestOrder({
      .virtual_count = 0U,
      .real_roles = cluster.topology->roles,
      .real_parents = cluster.topology->parents,
  });
  if (!order.ok()) {
    return order.status();
  }
  for (std::uint32_t bfs_real = order.value().real_root_count;
       bfs_real < real_count; ++bfs_real) {
    const std::uint32_t real = order.value().new_to_old_real[bfs_real];
    if constexpr (Prevalidated) {
      assert(cluster.topology->parents[real].has_value());
      assert(cluster.topology->parents[real]->kind ==
             LinkageParentKind::kReal);
    } else {
      if (!cluster.topology->parents[real].has_value() ||
          cluster.topology->parents[real]->kind !=
              LinkageParentKind::kReal) {
        return Status::FailedPrecondition(
            "root-only linked node requires one real parent");
      }
    }
    const std::uint32_t parent =
        cluster.topology->parents[real]->local_id;
    if constexpr (Prevalidated) {
      assert(parent < real_count);
      assert(cluster.topology->depths[parent] + 1U ==
             cluster.topology->depths[real]);
    } else {
      if (parent >= real_count ||
          cluster.topology->depths[parent] + 1U !=
              cluster.topology->depths[real]) {
        return Status::FailedPrecondition(
            "root-only linked node violates causal parent depth");
      }
    }
    const LinkageCodeTuple<M>& root_support =
        cluster.topology->supports[real];
    if constexpr (Prevalidated) {
      assert(root_support.first < root_to_edge.root_count());
    } else {
      if (root_support.first >= root_to_edge.root_count()) {
        return Status::FailedPrecondition(
            "root-only support exceeds the frozen root mapping");
      }
    }
    const float* coefficients = cluster.topology->coefficients.data() +
        static_cast<std::size_t>(real) * m();
    if constexpr (Prevalidated) {
#ifndef NDEBUG
      for (std::uint32_t layer = 0U;
           layer < m(); ++layer) {
        assert(std::isfinite(coefficients[layer]));
      }
#endif
    } else {
      for (std::uint32_t layer = 0U;
           layer < m(); ++layer) {
        if (!std::isfinite(coefficients[layer])) {
          return Status::FailedPrecondition(
              "root-only coefficient is non-finite");
        }
      }
    }
    const float* parent_reconstruction =
        cluster.real_reconstructions.column(parent);
    const float* exact = cluster.exact_real.column(real);
    for (std::uint32_t axis = 0U; axis < dimension_; ++axis) {
      target[axis] = exact[axis] - parent_reconstruction[axis];
    }
    LinkageCodeTuple<M> mapped_support = root_support;
    mapped_support.first = root_to_edge.edge_code(root_support.first);
    AccumulateOne(target.data(), mapped_support, coefficients, sample_weight);
  }
  return Status::Ok();
}

template class EdgeCodebookMoments<0>;
template class EdgeCodebookMoments<5>;
template class EdgeCodebookMoments<12>;
template class EdgeCodebookMoments<6>;
template class EdgeCodebookMoments<10>;

}  // namespace stlqpp
