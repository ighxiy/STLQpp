#include "stlqpp/training/update/frozen_forest_reconstruction.h"

#include "stlqpp/model/topology/canonical_bfs_forest_order.h"

#include <cstddef>
#include <cstdint>

namespace stlqpp {
namespace {

template <int M>
void ReconstructRoot(const ProductCodebooks<M>& codebooks,
                     std::uint32_t cid,
                     const std::uint8_t* small_codes,
                     const float* coefficients,
                     float* output) noexcept {
  const std::uint32_t dimension = codebooks.shape().dimension;
  const float* first = codebooks.root0().column(cid);
  for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
    output[axis] = coefficients[0U] * first[axis];
  }
  for (std::uint32_t layer = 1U;
       layer < codebooks.shape().m; ++layer) {
    const float* atom = codebooks.root_small(layer).column(
        small_codes[layer - 1U]);
    const float coefficient = coefficients[layer];
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      output[axis] += coefficient * atom[axis];
    }
  }
}

template <int M>
void ReconstructEdgeResidual(const ProductCodebooks<M>& codebooks,
                             const LinkageCodeTuple<M>& support,
                             const float* coefficients,
                             float* output) noexcept {
  const std::uint32_t dimension = codebooks.shape().dimension;
  const float* first = codebooks.edge(0U).column(support.first);
  for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
    output[axis] = coefficients[0U] * first[axis];
  }
  for (std::uint32_t layer = 1U;
       layer < codebooks.shape().m; ++layer) {
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
Status ReconstructImpl(
    const ProductCodebooks<M>& codebooks,
    std::uint32_t cid,
    ArrayView<const LinkageNodeRole> roles,
    ArrayView<const std::optional<LinkageParentReference>> parents,
    ArrayView<const std::uint32_t> depths,
    ArrayView<const LinkageCodeTuple<M>> supports,
    ArrayView<const float> coefficients,
    ArrayView<const std::uint8_t> virtual_small_codes,
    ArrayView<const float> virtual_coefficients,
    FrozenForestReconstruction* output) {
  if (output == nullptr) {
    return Status::InvalidArgument(
        "frozen forest reconstruction requires output storage");
  }
  const ModelShape& shape = codebooks.shape();
  const std::uint32_t virtual_count = static_cast<std::uint32_t>(
      virtual_coefficients.size() / static_cast<std::size_t>(shape.m));
  if (cid >= shape.nlist || coefficients.size() != roles.size() * shape.m ||
      parents.size() != roles.size() || depths.size() != roles.size() ||
      supports.size() != roles.size() ||
      virtual_coefficients.size() !=
          static_cast<std::size_t>(virtual_count) * shape.m ||
      virtual_small_codes.size() !=
          static_cast<std::size_t>(virtual_count) * (shape.m - 1U)) {
    return Status::InvalidArgument(
        "frozen forest reconstruction inputs are incompatible");
  }
  auto order = BuildCanonicalBfsForestOrder({
      .virtual_count = virtual_count,
      .real_roles = roles,
      .real_parents = parents,
  });
  if (!order.ok()) {
    return order.status();
  }
  output->dimension = shape.dimension;
  output->real_count = static_cast<std::uint32_t>(roles.size());
  output->virtual_count = virtual_count;
  output->real.resize(
      static_cast<std::size_t>(output->dimension) * output->real_count);
  output->virtual_roots.resize(
      static_cast<std::size_t>(output->dimension) * output->virtual_count);

  for (std::uint32_t root = 0U; root < output->virtual_count; ++root) {
    ReconstructRoot<M>(
        codebooks, cid,
        virtual_small_codes.data() +
            static_cast<std::size_t>(root) * (shape.m - 1U),
        virtual_coefficients.data() +
            static_cast<std::size_t>(root) * shape.m,
        output->virtual_roots.data() +
            static_cast<std::size_t>(root) * output->dimension);
  }
  for (const std::uint32_t real : order.value().new_to_old_real) {
    float* reconstruction = output->real.data() +
        static_cast<std::size_t>(real) * output->dimension;
    const float* row_coefficients = coefficients.data() +
        static_cast<std::size_t>(real) * shape.m;
    if (roles[real] == LinkageNodeRole::kRoot) {
      ReconstructRoot<M>(
          codebooks, cid,
          supports[real].trailing.data(), row_coefficients,
          reconstruction);
      continue;
    }
    ReconstructEdgeResidual<M>(
        codebooks, supports[real], row_coefficients, reconstruction);
    const LinkageParentReference parent = *parents[real];
    const float* parent_reconstruction =
        parent.kind == LinkageParentKind::kVirtual
        ? output->virtual_roots.data() +
              static_cast<std::size_t>(parent.local_id) * output->dimension
        : output->real.data() +
              static_cast<std::size_t>(parent.local_id) * output->dimension;
    for (std::uint32_t axis = 0U; axis < output->dimension; ++axis) {
      reconstruction[axis] += parent_reconstruction[axis];
    }
  }
  return Status::Ok();
}

template <int M>
StatusOr<FrozenForestReconstruction> ReconstructFrozenForest(
    const ProductCodebooks<M>& codebooks,
    const RoundUpdateTranscriptCluster<M>& cluster) {
  FrozenForestReconstruction output;
  if (const Status status = ReconstructFrozenForest<M>(
          codebooks, cluster, &output); !status.ok()) {
    return status;
  }
  return output;
}

template <int M>
Status ReconstructFrozenForest(
    const ProductCodebooks<M>& codebooks,
    const RoundUpdateTranscriptCluster<M>& cluster,
    FrozenForestReconstruction* output) {
  return ReconstructImpl<M>(
      codebooks, cluster.cid, cluster.roles, cluster.parents,
      cluster.depths, cluster.supports, cluster.coefficients,
      cluster.virtual_small_codes, cluster.virtual_coefficients, output);
}

template <int M>
StatusOr<FrozenForestReconstruction> ReconstructFrozenForest(
    const ProductCodebooks<M>& codebooks,
    const ClusterTopologyResult<M>& topology,
    ArrayView<const std::uint8_t> virtual_small_codes,
    ArrayView<const float> virtual_coefficients) {
  FrozenForestReconstruction output;
  if (const Status status = ReconstructFrozenForest<M>(
          codebooks, topology, virtual_small_codes, virtual_coefficients,
          &output); !status.ok()) {
    return status;
  }
  return output;
}

template <int M>
Status ReconstructFrozenForest(
    const ProductCodebooks<M>& codebooks,
    const ClusterTopologyResult<M>& topology,
    ArrayView<const std::uint8_t> virtual_small_codes,
    ArrayView<const float> virtual_coefficients,
    FrozenForestReconstruction* output) {
  return ReconstructImpl<M>(
      codebooks, topology.cid, topology.roles, topology.parents,
      topology.depths, topology.supports, topology.coefficients,
      virtual_small_codes, virtual_coefficients, output);
}

template StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest<0>(
    const ProductCodebooks<0>&,
    const RoundUpdateTranscriptCluster<0>&);
template Status ReconstructFrozenForest<0>(
    const ProductCodebooks<0>&,
    const RoundUpdateTranscriptCluster<0>&,
    FrozenForestReconstruction*);
template StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest<0>(
    const ProductCodebooks<0>&,
    const ClusterTopologyResult<0>&,
    ArrayView<const std::uint8_t>,
    ArrayView<const float>);
template Status ReconstructFrozenForest<0>(
    const ProductCodebooks<0>&,
    const ClusterTopologyResult<0>&,
    ArrayView<const std::uint8_t>,
    ArrayView<const float>,
    FrozenForestReconstruction*);
template StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest<5>(
    const ProductCodebooks<5>&,
    const RoundUpdateTranscriptCluster<5>&);
template Status ReconstructFrozenForest<5>(
    const ProductCodebooks<5>&,
    const RoundUpdateTranscriptCluster<5>&,
    FrozenForestReconstruction*);
template StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest<5>(
    const ProductCodebooks<5>&,
    const ClusterTopologyResult<5>&,
    ArrayView<const std::uint8_t>,
    ArrayView<const float>);
template Status ReconstructFrozenForest<5>(
    const ProductCodebooks<5>&,
    const ClusterTopologyResult<5>&,
    ArrayView<const std::uint8_t>,
    ArrayView<const float>,
    FrozenForestReconstruction*);
template StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest<6>(
    const ProductCodebooks<6>&,
    const RoundUpdateTranscriptCluster<6>&);
template Status ReconstructFrozenForest<6>(
    const ProductCodebooks<6>&,
    const RoundUpdateTranscriptCluster<6>&,
    FrozenForestReconstruction*);
template StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest<6>(
    const ProductCodebooks<6>&,
    const ClusterTopologyResult<6>&,
    ArrayView<const std::uint8_t>,
    ArrayView<const float>);
template Status ReconstructFrozenForest<6>(
    const ProductCodebooks<6>&,
    const ClusterTopologyResult<6>&,
    ArrayView<const std::uint8_t>,
    ArrayView<const float>,
    FrozenForestReconstruction*);
template StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest<10>(
    const ProductCodebooks<10>&,
    const RoundUpdateTranscriptCluster<10>&);
template Status ReconstructFrozenForest<10>(
    const ProductCodebooks<10>&,
    const RoundUpdateTranscriptCluster<10>&,
    FrozenForestReconstruction*);
template StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest<10>(
    const ProductCodebooks<10>&,
    const ClusterTopologyResult<10>&,
    ArrayView<const std::uint8_t>,
    ArrayView<const float>);
template Status ReconstructFrozenForest<10>(
    const ProductCodebooks<10>&,
    const ClusterTopologyResult<10>&,
    ArrayView<const std::uint8_t>,
    ArrayView<const float>,
    FrozenForestReconstruction*);
template StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest<12>(
    const ProductCodebooks<12>&,
    const RoundUpdateTranscriptCluster<12>&);
template Status ReconstructFrozenForest<12>(
    const ProductCodebooks<12>&,
    const RoundUpdateTranscriptCluster<12>&,
    FrozenForestReconstruction*);
template StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest<12>(
    const ProductCodebooks<12>&,
    const ClusterTopologyResult<12>&,
    ArrayView<const std::uint8_t>,
    ArrayView<const float>);
template Status ReconstructFrozenForest<12>(
    const ProductCodebooks<12>&,
    const ClusterTopologyResult<12>&,
    ArrayView<const std::uint8_t>,
    ArrayView<const float>,
    FrozenForestReconstruction*);

}  // namespace stlqpp
