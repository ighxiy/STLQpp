#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/model/topology/typed_forest.h"
#include "stlqpp/training/linkage/replay/fixed_edge_replay_encoder.h"
#include "stlqpp/training/linkage/topology/cluster_topology_state.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace stlqpp {

template <int M>
class FixedTopologyReplayPipeline;

// Replay transport consists of views and a runtime shape. M specialization
// starts at the numerical executor, not at the loader or gather window.
inline constexpr std::uint32_t kMaximumReplayWindowClusters = 64U;

struct FixedTopologyReplayRequest final {
  std::uint32_t m = 0U;

  std::uint32_t cid = 0U;
  // Current model coordinates R*x, consumed by root/edge replay.
  ColumnMajorMatrixF32View exact_real;
  // Corresponding source coordinates x, carried through the window for the
  // next-round Procrustes update but never consumed by replay encoders.
  ColumnMajorMatrixF32View unrotated_exact_real;
  ArrayView<const std::uint64_t> real_sample_ids;
  ArrayView<const std::uint64_t> virtual_sample_ids;
  // Real-root supports come from the replay round's fixed-root basic pass,
  // which ran before C_root was updated. Virtual-root supports remain frozen
  // from the preceding full forest, matching DEV replay semantics.
  ArrayView<const std::uint8_t> basic_small_codes;
  ArrayView<const float> basic_coefficients;
  ArrayView<const std::uint8_t> frozen_virtual_small_codes;
  ArrayView<const float> frozen_virtual_coefficients;
  ArrayView<const LinkageNodeRole> frozen_roles;
  ArrayView<const std::optional<LinkageParentReference>> frozen_parents;
  ArrayView<const std::uint32_t> frozen_depths;
  ArrayView<const std::uint8_t> edge_icm_layer_orders;
  std::uint32_t seed = 0U;
};

template <int M>
struct FixedTopologyReplayResult final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  ClusterTopologyResult<M> topology;
  // Virtual roots precede real rows in the canonical forest but are not part
  // of ClusterTopologyResult's real-row arrays.
  std::vector<std::uint8_t> virtual_small_codes;
  std::vector<float> virtual_coefficients;
  std::vector<float> virtual_reconstructions;
};

// Complete semantic replay for one frozen cluster. It gathers virtual and
// real roots, refreshes them with immutable native CIDs, scatters their new
// reconstructions, then re-encodes fixed edges behind strict depth barriers.
// No candidate search or parent selection is reachable from this executor.
template <int M>
class FixedTopologyReplayExecutor final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  static constexpr std::uint32_t kMaximumWindowClusters =
      kMaximumReplayWindowClusters;

  FixedTopologyReplayExecutor(
      const ProductCodebooks<M>& codebooks,
      const FixedEdgeReplayEncoder<M>& edge_encoder) noexcept
      : codebooks_(&codebooks),
        edge_encoder_(&edge_encoder) {}

  [[nodiscard]] StatusOr<FixedTopologyReplayResult<M>> Execute(
      const FixedTopologyReplayRequest& request) const;

  // Production batching boundary. Roots from every cluster share one fixed-
  // root call; fixed edges share one call per global depth while retaining
  // cluster-local parent spaces and barriers. The default preserves complete
  // results for direct callers; sinks that consume only codes/errors may omit
  // the real reconstruction array.
  [[nodiscard]] StatusOr<std::vector<FixedTopologyReplayResult<M>>>
  ExecuteWindow(
      ArrayView<const FixedTopologyReplayRequest> requests,
      bool retain_real_reconstructions = true) const;

 private:
  friend class FixedTopologyReplayPipeline<M>;

  [[nodiscard]] StatusOr<std::vector<FixedTopologyReplayResult<M>>>
  ExecuteWindowFromCheckpoint(
      ArrayView<const FixedTopologyReplayRequest> requests,
      bool retain_real_reconstructions) const;
  [[nodiscard]] StatusOr<std::vector<FixedTopologyReplayResult<M>>>
  ExecuteWindowImpl(
      ArrayView<const FixedTopologyReplayRequest> requests,
      bool retain_real_reconstructions,
      bool validated_checkpoint_payload) const;

  const ProductCodebooks<M>* codebooks_ = nullptr;
  const FixedEdgeReplayEncoder<M>* edge_encoder_ = nullptr;
};

extern template class FixedTopologyReplayExecutor<5>;
extern template class FixedTopologyReplayExecutor<10>;

}  // namespace stlqpp
