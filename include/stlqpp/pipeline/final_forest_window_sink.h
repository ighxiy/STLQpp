#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/index/build/final_cluster_postprocessor.h"
#include "stlqpp/training/linkage/topology/cluster_topology_state.h"

#include <cstdint>

namespace stlqpp {

// Immutable final product of either a full-topology build or a fixed-topology
// replay. Stable source IDs remain u64 until the canonical-v1 boundary.
template <int M>
struct FinalForestSpoolClusterView final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  ArrayView<const std::uint64_t> real_sample_ids;
  ArrayView<const float> source_real_norm2;
  const ClusterTopologyResult<M>* topology = nullptr;
  ArrayView<const std::uint8_t> virtual_codes_small;
  ArrayView<const float> virtual_coefficients_node_major;
};

// Typed composition boundary between final forest production and canonical
// index publication. Producers do not know whether the receiver is a spool
// writer or a complete publication transaction.
template <int M>
class FinalForestWindowSink {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  virtual ~FinalForestWindowSink() = default;

  [[nodiscard]] virtual bool RequiresOrderedCommit() const noexcept {
    return true;
  }
  [[nodiscard]] virtual bool RequiresSourceNorm2() const noexcept {
    return false;
  }

  [[nodiscard]] virtual Status AppendWindow(
      ArrayView<const FinalForestSpoolClusterView<M>> clusters) = 0;

  // Full-topology workers use these two operations to move codec/norm work
  // ahead of ordered publication. Fixed-topology replay continues to use
  // AppendWindow and therefore retains its existing window-parallel path.
  [[nodiscard]] virtual StatusOr<PreparedCanonicalCluster<M>> PrepareCluster(
      const FinalForestSpoolClusterView<M>& cluster) const = 0;
  [[nodiscard]] virtual Status AppendPreparedWindow(
      ArrayView<const PreparedCanonicalCluster<M>> clusters) = 0;
};

}  // namespace stlqpp
