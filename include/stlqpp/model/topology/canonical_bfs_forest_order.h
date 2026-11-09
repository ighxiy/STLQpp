#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/model/topology/typed_forest.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

// Canonical persistent order. Virtual roots remain first, real roots follow,
// and linked real rows are emitted in global BFS order. Runtime AoSoA packing
// preserves this logical order and is deliberately not represented here.
struct CanonicalBfsForestOrder final {
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  std::vector<std::uint32_t> new_to_old_real;
  std::vector<std::uint32_t> old_to_new_real;
  std::vector<std::uint32_t> parent_local_1based;
};

[[nodiscard]] StatusOr<CanonicalBfsForestOrder>
BuildCanonicalBfsForestOrder(TypedForestView forest);

}  // namespace stlqpp
