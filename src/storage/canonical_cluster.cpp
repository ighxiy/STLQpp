#include "stlqpp/storage/canonical_cluster.h"

#include "stlqpp/model/topology/parent_louds.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace stlqpp {

template <int M>
StatusOr<TrustedClusterView<M>> TrustedClusterView<M>::Create(
    const MappedClusterView& source) {
  static_assert(M >= 2 && M <= 20);
  const std::uint64_t node_count =
      static_cast<std::uint64_t>(source.real_count) + source.virtual_count;
  if (node_count > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted("cluster node count exceeds u32 domain");
  }
  if (source.real_root_count > source.real_count) {
    return Status::DataLoss("cluster real-root count exceeds real rows");
  }

  if (source.real_count != 0U &&
      (source.real_ids == nullptr || source.parent_local_1based == nullptr ||
       source.real_codes_small == nullptr || source.edge_code0 == nullptr ||
       source.real_norm8 == nullptr || source.norm8_centers == nullptr ||
       source.norm8_center_count == 0U)) {
    return Status::DataLoss("cluster is missing a required real-row stream");
  }
  if (source.virtual_count != 0U && source.virtual_codes_small == nullptr) {
    return Status::DataLoss("cluster is missing virtual root codes");
  }
  if (node_count != 0U &&
      (source.coefficients_layer_major == nullptr ||
       source.root_scales == nullptr || source.linkage_scales == nullptr)) {
    return Status::DataLoss("cluster is missing coefficient state");
  }

  if (node_count != 0U) {
    for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(M);
         ++layer) {
      if (!std::isfinite(source.root_scales[layer]) ||
          !std::isfinite(source.linkage_scales[layer])) {
        return Status::DataLoss("cluster contains an invalid coefficient scale");
      }
    }
  }
  for (std::uint32_t center = 0U; center < source.norm8_center_count;
       ++center) {
    if (!std::isfinite(source.norm8_centers[center]) ||
        source.norm8_centers[center] < 0.0F) {
      return Status::DataLoss("cluster contains an invalid norm8 center");
    }
  }
  for (std::uint32_t pos = 0U; pos < source.real_count; ++pos) {
    if (source.norm8_center_count != 256U &&
        source.real_norm8[pos] >= source.norm8_center_count) {
      return Status::DataLoss("cluster norm8 code exceeds its center table");
    }
    const std::uint32_t parent = source.parent_local_1based[pos];
    if (pos < source.real_root_count) {
      if (parent != 0U) {
        return Status::DataLoss("real root has a parent");
      }
      continue;
    }
    const std::uint32_t local = source.virtual_count + pos;
    if (parent == 0U || parent > local) {
      return Status::DataLoss("linked row parent is not causally earlier");
    }
  }
  return TrustedClusterView<M>(source);
}

template class TrustedClusterView<5>;
template class TrustedClusterView<12>;
template class TrustedClusterView<6>;
template class TrustedClusterView<10>;

template <int M>
StatusOr<TrustedBfsClusterView<M>> TrustedBfsClusterView<M>::Create(
    const MappedBfsClusterView& source) {
  return CreateImpl(source, true);
}

template <int M>
StatusOr<TrustedBfsClusterView<M>>
TrustedBfsClusterView<M>::CreateAfterValidatedLouds(
    const MappedBfsClusterView& source) {
  return CreateImpl(source, false);
}

template <int M>
StatusOr<TrustedBfsClusterView<M>> TrustedBfsClusterView<M>::CreateImpl(
    const MappedBfsClusterView& source, bool validate_louds) {
  static_assert(M >= 2 && M <= 20);
  const std::uint64_t node_count =
      static_cast<std::uint64_t>(source.real_count) + source.virtual_count;
  if (node_count > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted("cluster node count exceeds u32 domain");
  }
  if (source.real_root_count > source.real_count) {
    return Status::DataLoss("cluster real-root count exceeds real rows");
  }
  if (source.real_count != 0U &&
      (source.real_ids == nullptr || source.real_codes_small == nullptr ||
       source.edge_code0 == nullptr || source.real_norm8 == nullptr ||
       source.norm8_centers == nullptr || source.norm8_center_count == 0U)) {
    return Status::DataLoss("cluster is missing a required real-row stream");
  }
  if (source.virtual_count != 0U && source.virtual_codes_small == nullptr) {
    return Status::DataLoss("cluster is missing virtual root codes");
  }
  if (node_count != 0U &&
      (source.coefficients_layer_major == nullptr ||
       source.root_scales == nullptr || source.linkage_scales == nullptr)) {
    return Status::DataLoss("cluster is missing coefficient state");
  }
  if (validate_louds) {
    if (const Status status = ValidateParentLouds(
            source.parent_louds_words,
            static_cast<std::uint32_t>(node_count),
            source.virtual_count + source.real_root_count);
        !status.ok()) {
      return status;
    }
  }
  if (node_count != 0U) {
    for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(M);
         ++layer) {
      if (!std::isfinite(source.root_scales[layer]) ||
          !std::isfinite(source.linkage_scales[layer])) {
        return Status::DataLoss("cluster contains an invalid coefficient scale");
      }
    }
  }
  for (std::uint32_t center = 0U; center < source.norm8_center_count;
       ++center) {
    if (!std::isfinite(source.norm8_centers[center]) ||
        source.norm8_centers[center] < 0.0F) {
      return Status::DataLoss("cluster contains an invalid norm8 center");
    }
  }
  if (source.norm8_center_count != 256U) {
    for (std::uint32_t pos = 0U; pos < source.real_count; ++pos) {
      if (source.real_norm8[pos] >= source.norm8_center_count) {
        return Status::DataLoss("cluster norm8 code exceeds its center table");
      }
    }
  }
  return TrustedBfsClusterView<M>(source);
}

template class TrustedBfsClusterView<5>;
template class TrustedBfsClusterView<12>;
template class TrustedBfsClusterView<6>;
template class TrustedBfsClusterView<10>;

}  // namespace stlqpp
