#pragma once

#include "stlqpp/model/model_shape.h"
#include "stlqpp/model/topology/typed_forest.h"
#include "stlqpp/model/model_layers.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

// Numerical scheduling inputs shared by DEV initialization and ordinary
// good-cluster topology construction. They are independent recipe fields:
// changing the outer many-node batch must not change the same-layer causal
// block, or vice versa.
struct CausalTopologyBatching final {
  std::uint32_t inner_batch_rows = 0U;
  // Hard cap for inner and other frozen many-node tasks.
  std::uint32_t maximum_batch_pairs = 0U;
  // Legacy/global target used only when a stage explicitly requests fallback
  // preflush semantics. It is independent of both hard caps.
  std::uint32_t many_nodes_preflush_target_pairs = 0U;
  std::uint32_t same_layer_block_rows = 0U;
  std::uint32_t same_layer_minimum_batch_pairs = 0U;
  // Hard cap for a same-layer dynamic-window task.
  std::uint32_t same_layer_window_maximum_pairs = 0U;
  std::uint32_t same_layer_maximum_pending_nodes = 0U;
  // Positive: explicit pair trigger; zero: stage-defined registered mode;
  // negative: fall back to many_nodes_preflush_target_pairs.
  std::int32_t same_layer_preflush_target_pairs = 0;
  // Zero disables the independent pending-node trigger.
  std::uint32_t same_layer_preflush_target_nodes = 0U;
  bool tiny_forced_cpu_enabled = true;
  std::uint32_t tiny_forced_pair_limit = 0U;
  std::uint32_t tiny_forced_node_limit = 0U;
  // Ordinary single-node routing. Scheduler-backed evaluation uses DEV's
  // effective minimum of one; direct evaluation applies the configured
  // thresholds. Initialization does not use this ordinary edge policy.
  bool single_gpu_route_enabled = false;
  bool single_gpu_scheduler_backed = false;
  std::uint32_t single_gpu_min_candidates = 16U;
  std::uint32_t single_gpu_max_candidates = 0U;
  std::uint32_t same_layer_single_gpu_min_candidates = 6U;
  std::uint32_t bad_frozen_window_rows = 0U;
  std::uint32_t bad_frozen_maximum_pairs = 0U;
};

inline bool SingleNodeUsesCpu(const CausalTopologyBatching& batching,
                              std::uint32_t candidate_count,
                              bool same_layer) noexcept {
  if (!batching.single_gpu_route_enabled) return false;
  const std::uint32_t configured = same_layer &&
      batching.same_layer_single_gpu_min_candidates != 0U
      ? batching.same_layer_single_gpu_min_candidates
      : batching.single_gpu_min_candidates;
  const std::uint32_t minimum = batching.single_gpu_scheduler_backed
      ? 1U
      : same_layer ? std::max(4U, configured)
                   : std::max(1U, configured);
  return candidate_count < minimum ||
      (batching.single_gpu_max_candidates != 0U &&
       candidate_count > batching.single_gpu_max_candidates);
}

namespace causal_topology_detail {

// DEV applies the depth bound before counting frozen inner pairs toward the
// many-node batch limit. Same-layer candidates are classified later, after
// the batch boundary is fixed.
inline bool CountsTowardInnerBatchLimit(
    LinkageParentReference parent,
    std::uint32_t current_layer_position,
    std::uint32_t unavailable_position,
    std::uint32_t parent_depth,
    std::uint32_t maximum_depth) noexcept {
  if (parent.kind == LinkageParentKind::kVirtual) {
    return maximum_depth != 0U;
  }
  return current_layer_position == unavailable_position &&
      parent_depth < maximum_depth;
}

}  // namespace causal_topology_detail

template <int M>
struct CausalTopologyRow final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  std::uint32_t real_local_id = 0U;
  std::vector<LinkageParentReference> ordered_candidates;
  std::array<std::uint8_t, (M == 0 ? kMaximumModelLayers : M)> icm_layer_order{};
};

enum class CausalEvaluationKind : std::uint8_t {
  kInner = 0U,
  kSameFrozen = 1U,
  kDynamicBefore = 2U,
  kDynamicWindow = 3U,
  kGraphFrozen = 4U,
  kGraphDynamic = 5U,
  kCount = 6U,
};

inline bool UsesSameLayerSingleNodeThreshold(
    CausalEvaluationKind kind) noexcept {
  return kind == CausalEvaluationKind::kDynamicWindow;
}

inline constexpr std::size_t kCausalEvaluationKindCount =
    static_cast<std::size_t>(CausalEvaluationKind::kCount);

struct CausalTopologyWindowStatistics final {
  std::uint64_t rows = 0U;
  std::uint64_t candidate_pairs = 0U;
  std::uint64_t evaluation_tasks = 0U;
  std::uint64_t evaluation_nodes = 0U;
  std::uint64_t evaluation_pairs = 0U;
  std::uint64_t cpu_evaluation_tasks = 0U;
  std::uint64_t cpu_evaluation_nodes = 0U;
  std::uint64_t cpu_evaluation_pairs = 0U;
  std::array<std::uint64_t, kCausalEvaluationKindCount>
      evaluation_tasks_by_kind{};
  std::array<std::uint64_t, kCausalEvaluationKindCount>
      cpu_evaluation_tasks_by_kind{};
  std::uint64_t update_requests = 0U;
  std::uint64_t update_columns = 0U;
  std::uint64_t maximum_task_nodes = 0U;
  std::uint64_t maximum_task_pairs = 0U;
};

}  // namespace stlqpp
