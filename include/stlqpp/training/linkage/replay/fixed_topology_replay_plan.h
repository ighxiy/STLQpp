#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/topology/typed_forest.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

template <int M>
class FixedTopologyReplayExecutor;

struct ReplayDepthBatch final {
  std::uint32_t depth = 0U;
  std::vector<std::uint32_t> real_local_ids;
};

struct FixedTopologyReplayPlan final {
  std::uint32_t virtual_count = 0U;
  std::vector<std::uint32_t> real_roots;
  std::vector<ReplayDepthBatch> linked_by_depth;
};

// Validates the frozen forest and produces exact replay barriers. It does not
// encode vectors: fixed-root and fixed-edge executors consume this immutable
// plan, and no deeper batch becomes visible before the preceding batch is
// reconstructed and committed.
[[nodiscard]] StatusOr<FixedTopologyReplayPlan>
BuildFixedTopologyReplayPlan(
    TypedForestView forest,
    ArrayView<const std::uint32_t> real_depths);

// Checkpoint-only entry: the decoder has already validated the same immutable
// forest before it was copied into the replay window.
class FixedTopologyReplayPlanFromCheckpoint final {
 private:
  template <int M>
  friend class FixedTopologyReplayExecutor;

  [[nodiscard]] static StatusOr<FixedTopologyReplayPlan> Build(
      TypedForestView forest,
      ArrayView<const std::uint32_t> real_depths);
};

}  // namespace stlqpp
