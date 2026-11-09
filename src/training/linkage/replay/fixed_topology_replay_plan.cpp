#include "stlqpp/training/linkage/replay/fixed_topology_replay_plan.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace stlqpp {
namespace {

template <bool kValidatedCheckpoint>
StatusOr<FixedTopologyReplayPlan> BuildPlan(
    TypedForestView forest,
    ArrayView<const std::uint32_t> real_depths) {
  if (forest.real_roles.size() != forest.real_parents.size() ||
      forest.real_roles.size() != real_depths.size() ||
      forest.real_roles.size() >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::InvalidArgument(
        "fixed-topology replay inputs have inconsistent row counts");
  }

  const std::uint32_t real_count =
      static_cast<std::uint32_t>(forest.real_roles.size());
  std::uint32_t maximum_depth = 0U;
  FixedTopologyReplayPlan result;
  result.virtual_count = forest.virtual_count;
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    const LinkageNodeRole role = forest.real_roles[real];
    const auto& parent = forest.real_parents[real];
    if (role == LinkageNodeRole::kRoot) {
      if constexpr (kValidatedCheckpoint) {
        assert(!parent.has_value() && real_depths[real] == 0U);
      } else if (parent.has_value() || real_depths[real] != 0U) {
        return Status::InvalidArgument(
            "fixed-topology real root has parent or nonzero depth");
      }
      result.real_roots.push_back(real);
      continue;
    }
    if constexpr (kValidatedCheckpoint) {
      assert(role == LinkageNodeRole::kLinked && parent.has_value() &&
             real_depths[real] != 0U);
    } else if (role != LinkageNodeRole::kLinked || !parent.has_value() ||
               real_depths[real] == 0U) {
      return Status::InvalidArgument(
          "fixed-topology linked row has invalid role, parent or depth");
    }
    std::uint32_t parent_depth = 0U;
    if (parent->kind == LinkageParentKind::kVirtual) {
      if constexpr (kValidatedCheckpoint) {
        assert(parent->local_id < forest.virtual_count);
      } else if (parent->local_id >= forest.virtual_count) {
        return Status::InvalidArgument(
            "fixed-topology virtual parent is outside the cluster");
      }
    } else if (parent->kind == LinkageParentKind::kReal) {
      if constexpr (kValidatedCheckpoint) {
        assert(parent->local_id < real);
      } else if (parent->local_id >= real_count ||
                 parent->local_id == real) {
        return Status::InvalidArgument(
            "fixed-topology real parent is outside the frozen forest");
      }
      parent_depth = real_depths[parent->local_id];
    } else {
      if constexpr (kValidatedCheckpoint) {
        assert(false && "checkpoint decoded an invalid parent kind");
      } else {
        return Status::InvalidArgument(
            "fixed-topology parent kind is invalid");
      }
    }
    if constexpr (kValidatedCheckpoint) {
      assert(parent_depth + 1U == real_depths[real]);
    } else if (parent_depth + 1U != real_depths[real]) {
      return Status::InvalidArgument(
          "fixed-topology child depth does not follow its parent");
    }
    maximum_depth = std::max(maximum_depth, real_depths[real]);
  }

  std::vector<std::vector<std::uint32_t>> depth_rows(
      static_cast<std::size_t>(maximum_depth) + 1U);
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    if (forest.real_roles[real] == LinkageNodeRole::kLinked) {
      depth_rows[real_depths[real]].push_back(real);
    }
  }
  result.linked_by_depth.reserve(maximum_depth);
  for (std::uint32_t depth = 1U; depth <= maximum_depth; ++depth) {
    if (!depth_rows[depth].empty()) {
      result.linked_by_depth.push_back(
          {.depth = depth,
           .real_local_ids = std::move(depth_rows[depth])});
    }
  }
  return result;
}

}  // namespace

StatusOr<FixedTopologyReplayPlan> BuildFixedTopologyReplayPlan(
    TypedForestView forest,
    ArrayView<const std::uint32_t> real_depths) {
  return BuildPlan<false>(forest, real_depths);
}

StatusOr<FixedTopologyReplayPlan>
FixedTopologyReplayPlanFromCheckpoint::Build(
    TypedForestView forest,
    ArrayView<const std::uint32_t> real_depths) {
  return BuildPlan<true>(forest, real_depths);
}

}  // namespace stlqpp
