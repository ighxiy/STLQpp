#include "stlqpp/model/topology/canonical_bfs_forest_order.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace stlqpp {
namespace {

Status ValidateForest(TypedForestView forest) {
  if (forest.real_roles.size() != forest.real_parents.size() ||
      forest.real_roles.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      static_cast<std::uint64_t>(forest.virtual_count) +
              forest.real_roles.size() >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::InvalidArgument(
        "typed forest has inconsistent or oversized row counts");
  }
  const std::uint32_t real_count =
      static_cast<std::uint32_t>(forest.real_roles.size());
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    const LinkageNodeRole role = forest.real_roles[real];
    const auto& parent = forest.real_parents[real];
    if (role == LinkageNodeRole::kRoot) {
      if (parent.has_value()) {
        return Status::InvalidArgument(
            "typed forest root unexpectedly has a parent");
      }
      continue;
    }
    if (role != LinkageNodeRole::kLinked || !parent.has_value()) {
      return Status::InvalidArgument(
          "typed forest linked row is missing its typed parent");
    }
    const bool valid_virtual =
        parent->kind == LinkageParentKind::kVirtual &&
        parent->local_id < forest.virtual_count;
    const bool valid_real =
        parent->kind == LinkageParentKind::kReal &&
        parent->local_id < real_count && parent->local_id != real;
    if (!valid_virtual && !valid_real) {
      return Status::InvalidArgument(
          "typed forest contains an invalid parent reference");
    }
  }
  return Status::Ok();
}

std::uint32_t OldNode(std::uint32_t virtual_count,
                      LinkageParentReference parent) noexcept {
  return parent.kind == LinkageParentKind::kVirtual
      ? parent.local_id
      : virtual_count + parent.local_id;
}

}  // namespace

StatusOr<CanonicalBfsForestOrder> BuildCanonicalBfsForestOrder(
    TypedForestView forest) {
  if (const Status status = ValidateForest(forest); !status.ok()) {
    return status;
  }
  const std::uint32_t real_count =
      static_cast<std::uint32_t>(forest.real_roles.size());
  const std::uint32_t node_count = forest.virtual_count + real_count;
  std::vector<std::uint32_t> degree(node_count, 0U);
  std::uint32_t real_root_count = 0U;
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    if (forest.real_roles[real] == LinkageNodeRole::kRoot) {
      ++real_root_count;
    } else {
      ++degree[OldNode(forest.virtual_count, *forest.real_parents[real])];
    }
  }

  std::vector<std::uint32_t> offsets(
      static_cast<std::size_t>(node_count) + 1U, 0U);
  for (std::uint32_t node = 0U; node < node_count; ++node) {
    offsets[node + 1U] = offsets[node] + degree[node];
  }
  std::vector<std::uint32_t> children(offsets.back());
  std::vector<std::uint32_t> cursor(offsets.begin(), offsets.end() - 1U);
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    if (forest.real_roles[real] == LinkageNodeRole::kLinked) {
      const std::uint32_t parent =
          OldNode(forest.virtual_count, *forest.real_parents[real]);
      children[cursor[parent]++] = forest.virtual_count + real;
    }
  }

  std::vector<std::uint32_t> new_to_old_node;
  new_to_old_node.reserve(node_count);
  for (std::uint32_t root = 0U; root < forest.virtual_count; ++root) {
    new_to_old_node.push_back(root);
  }
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    if (forest.real_roles[real] == LinkageNodeRole::kRoot) {
      new_to_old_node.push_back(forest.virtual_count + real);
    }
  }
  for (std::size_t queue = 0U; queue < new_to_old_node.size(); ++queue) {
    const std::uint32_t parent = new_to_old_node[queue];
    new_to_old_node.insert(new_to_old_node.end(),
                           children.begin() + offsets[parent],
                           children.begin() + offsets[parent + 1U]);
  }
  if (new_to_old_node.size() != node_count) {
    return Status::InvalidArgument(
        "typed forest is cyclic or not reachable from its roots");
  }

  std::vector<std::uint32_t> old_to_new_node(node_count);
  for (std::uint32_t local = 0U; local < node_count; ++local) {
    old_to_new_node[new_to_old_node[local]] = local;
  }
  CanonicalBfsForestOrder result;
  result.virtual_count = forest.virtual_count;
  result.real_root_count = real_root_count;
  result.new_to_old_real.resize(real_count);
  result.old_to_new_real.resize(real_count);
  result.parent_local_1based.resize(real_count, 0U);
  for (std::uint32_t new_real = 0U; new_real < real_count; ++new_real) {
    const std::uint32_t new_node = forest.virtual_count + new_real;
    const std::uint32_t old_node = new_to_old_node[new_node];
    const std::uint32_t old_real = old_node - forest.virtual_count;
    result.new_to_old_real[new_real] = old_real;
    result.old_to_new_real[old_real] = new_real;
    if (forest.real_roles[old_real] == LinkageNodeRole::kLinked) {
      const std::uint32_t old_parent = OldNode(
          forest.virtual_count, *forest.real_parents[old_real]);
      result.parent_local_1based[new_real] =
          old_to_new_node[old_parent] + 1U;
    }
  }
  return result;
}

}  // namespace stlqpp
