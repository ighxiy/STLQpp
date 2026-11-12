#include "stlqpp/model/model_shape.h"
#include "stlqpp/index/build/canonical_float_cluster_assembler.h"

#include "stlqpp/model/topology/canonical_bfs_forest_order.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

template <int M>
Status ValidateSource(const FinalForestClusterView<M>& source) {
  const auto m = M == 0 ? source.m : M;
  const std::size_t real_count = source.real_ids.size();
  const std::size_t virtual_count = source.forest.virtual_count;
  if (m < 2U || m > stlqpp::kMaximumModelLayers || source.m != m || real_count > std::numeric_limits<std::uint32_t>::max() ||
      real_count + virtual_count >
          std::numeric_limits<std::uint32_t>::max() ||
      source.forest.real_roles.size() != real_count ||
      source.forest.real_parents.size() != real_count ||
      source.real_supports.size() != real_count ||
      source.real_coefficients_node_major.size() != real_count * m ||
      (!source.source_real_norm2.empty() &&
       source.source_real_norm2.size() != real_count) ||
      source.virtual_codes_small.size() != virtual_count * (m - 1U) ||
      source.virtual_coefficients_node_major.size() != virtual_count * m) {
    return Status::InvalidArgument(
        "final forest cluster streams disagree with its role counts");
  }
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    const LinkageNodeRole role = source.forest.real_roles[real];
    const std::uint32_t first = source.real_supports[real].first;
    if ((role == LinkageNodeRole::kRoot && first != source.cid) ||
        (role == LinkageNodeRole::kLinked && first >= 256U)) {
      return Status::InvalidArgument(
          "final forest support does not match its root/link role");
    }
  }
  for (const float value : source.real_coefficients_node_major) {
    if (!std::isfinite(value)) {
      return Status::InvalidArgument(
          "final forest contains a non-finite real coefficient");
    }
  }
  for (const float norm2 : source.source_real_norm2) {
    if (!std::isfinite(norm2) || norm2 < 0.0F) {
      return Status::InvalidArgument(
          "final forest contains an invalid source norm");
    }
  }
  for (const float value : source.virtual_coefficients_node_major) {
    if (!std::isfinite(value)) {
      return Status::InvalidArgument(
          "final forest contains a non-finite virtual coefficient");
    }
  }
  return Status::Ok();
}

}  // namespace

template <int M>
CanonicalFloatClusterView
OwnedCanonicalFloatCluster<M>::view() const noexcept {
  return {
      .m = m_,
      .cid = cid_,
      .virtual_count = virtual_count_,
      .real_root_count = real_root_count_,
      .real_ids = real_ids_,
      .parent_local_1based = parent_local_1based_,
      .real_codes_small = real_codes_small_,
      .edge_code0 = edge_code0_,
      .virtual_codes_small = virtual_codes_small_,
      .coefficients_layer_major = coefficients_layer_major_,
      .source_real_norm2 = source_real_norm2_,
  };
}

template <int M>
StatusOr<OwnedCanonicalFloatCluster<M>>
CanonicalFloatClusterAssembler<M>::Assemble(
    const FinalForestClusterView<M>& source) {
  const auto m = M == 0 ? source.m : M;
  if (const Status status = ValidateSource(source); !status.ok()) {
    return status;
  }
  const std::uint32_t source_virtual_count = source.forest.virtual_count;
  constexpr std::uint32_t kUnavailable =
      std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint8_t> live_virtual(source_virtual_count, 0U);
  for (std::uint32_t real = 0U;
       real < static_cast<std::uint32_t>(source.real_ids.size()); ++real) {
    const auto& parent = source.forest.real_parents[real];
    if (parent.has_value() &&
        parent->kind == LinkageParentKind::kVirtual &&
        parent->local_id < source_virtual_count) {
      live_virtual[parent->local_id] = 1U;
    }
  }
  std::vector<std::uint32_t> new_to_old_virtual;
  new_to_old_virtual.reserve(source_virtual_count);
  std::vector<std::uint32_t> old_to_new_virtual(
      source_virtual_count, kUnavailable);
  for (std::uint32_t old_root = 0U; old_root < source_virtual_count;
       ++old_root) {
    if (live_virtual[old_root] == 0U) continue;
    old_to_new_virtual[old_root] =
        static_cast<std::uint32_t>(new_to_old_virtual.size());
    new_to_old_virtual.push_back(old_root);
  }
  const std::uint32_t retained_virtual_count =
      static_cast<std::uint32_t>(new_to_old_virtual.size());

  std::vector<std::optional<LinkageParentReference>> compact_parents;
  TypedForestView compact_forest = source.forest;
  if (retained_virtual_count != source_virtual_count) {
    compact_parents.assign(source.forest.real_parents.begin(),
                           source.forest.real_parents.end());
    for (auto& parent : compact_parents) {
      if (!parent.has_value() ||
          parent->kind != LinkageParentKind::kVirtual) {
        continue;
      }
      if (parent->local_id >= old_to_new_virtual.size() ||
          old_to_new_virtual[parent->local_id] == kUnavailable) {
        return Status::DataLoss(
            "final forest references an unavailable virtual root");
      }
      parent->local_id = old_to_new_virtual[parent->local_id];
    }
    compact_forest = {
        .virtual_count = retained_virtual_count,
        .real_roles = source.forest.real_roles,
        .real_parents = compact_parents,
    };
  }

  auto order_result = BuildCanonicalBfsForestOrder(compact_forest);
  if (!order_result.ok()) {
    return order_result.status();
  }
  const CanonicalBfsForestOrder& order = order_result.value();
  const std::uint32_t real_count =
      static_cast<std::uint32_t>(source.real_ids.size());
  const std::uint32_t virtual_count = retained_virtual_count;
  const std::uint32_t node_count = virtual_count + real_count;

  OwnedCanonicalFloatCluster<M> output;
  output.m_ = m;
  output.cid_ = source.cid;
  output.virtual_count_ = virtual_count;
  output.real_root_count_ = order.real_root_count;
  output.real_ids_.resize(real_count);
  output.parent_local_1based_ = order.parent_local_1based;
  output.real_codes_small_.resize(
      static_cast<std::size_t>(real_count) * (m - 1U));
  output.edge_code0_.assign(real_count, 0U);
  output.virtual_codes_small_.resize(
      static_cast<std::size_t>(virtual_count) * (m - 1U));
  output.coefficients_layer_major_.resize(
      static_cast<std::size_t>(node_count) * m);
  if (!source.source_real_norm2.empty()) {
    output.source_real_norm2_.resize(real_count);
  }

  for (std::uint32_t layer = 0U;
       layer < m; ++layer) {
    for (std::uint32_t root = 0U; root < virtual_count; ++root) {
      const std::uint32_t old_root = new_to_old_virtual[root];
      output.coefficients_layer_major_[
          static_cast<std::size_t>(layer) * node_count + root] =
          source.virtual_coefficients_node_major[
              static_cast<std::size_t>(old_root) * m + layer];
      if (layer != 0U) {
        output.virtual_codes_small_[
            static_cast<std::size_t>(root) * (m - 1U) + layer - 1U] =
            source.virtual_codes_small[
                static_cast<std::size_t>(old_root) * (m - 1U) + layer - 1U];
      }
    }
  }
  for (std::uint32_t new_real = 0U; new_real < real_count; ++new_real) {
    const std::uint32_t old_real = order.new_to_old_real[new_real];
    output.real_ids_[new_real] = source.real_ids[old_real];
    if (!source.source_real_norm2.empty()) {
      output.source_real_norm2_[new_real] =
          source.source_real_norm2[old_real];
    }
    const LinkageCodeTuple<M>& support = source.real_supports[old_real];
    for (std::uint32_t layer = 1U;
         layer < m; ++layer) {
      output.real_codes_small_[
          static_cast<std::size_t>(new_real) * (m - 1U) + layer - 1U] =
          support.trailing[layer - 1U];
    }
    if (source.forest.real_roles[old_real] == LinkageNodeRole::kLinked) {
      output.edge_code0_[new_real] =
          static_cast<std::uint8_t>(support.first);
    }
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      output.coefficients_layer_major_[
          static_cast<std::size_t>(layer) * node_count + virtual_count +
          new_real] = source.real_coefficients_node_major[
              static_cast<std::size_t>(old_real) * m + layer];
    }
  }
  return output;
}

template class OwnedCanonicalFloatCluster<0>;
template class OwnedCanonicalFloatCluster<5>;
template class OwnedCanonicalFloatCluster<12>;
template class OwnedCanonicalFloatCluster<6>;
template class OwnedCanonicalFloatCluster<10>;
template class CanonicalFloatClusterAssembler<0>;
template class CanonicalFloatClusterAssembler<5>;
template class CanonicalFloatClusterAssembler<12>;
template class CanonicalFloatClusterAssembler<6>;
template class CanonicalFloatClusterAssembler<10>;

}  // namespace stlqpp
