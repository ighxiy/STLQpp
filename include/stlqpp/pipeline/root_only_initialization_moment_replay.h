#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/index/emit/base_list_reader.h"
#include "stlqpp/training/linkage/initialization/root_to_edge_code_map.h"
#include "stlqpp/training/update/initialization_model_updater.h"
#include "stlqpp/training/update/root_only_initialization_transcript.h"
#include "stlqpp/training/update/root_training_model.h"

#include <cstdint>

namespace stlqpp {

struct RootOnlyInitializationMomentReplayStatistics final {
  std::uint64_t clusters = 0U;
  std::uint64_t real_rows = 0U;
  std::uint64_t linked_rows = 0U;
};

// Reopens the initial native-list workset after the bad-cluster barrier,
// reconstructs the exact staged C_root forest one CID per shard at a time, and feeds
// DEV's cluster-weighted C_one sufficient statistics. It holds neither the
// training set nor its d-dimensional forest reconstructions globally. Shards
// own fixed contiguous CID ranges, preserving serial accumulation order within
// each shard and the updater's existing ordered reduction.
template <int M>
class RootOnlyInitializationMomentReplay final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<
      RootOnlyInitializationMomentReplayStatistics> Run(
      const BaseListReader<M>& lists,
      const RootTrainingModel<M>& root_model,
      const RootOnlyInitializationTranscript<M>& transcript,
      ArrayView<const std::uint8_t> bad_cluster_mask,
      double bad_cluster_weight,
      const RootToEdgeCodeMap& root_to_edge,
      InitializationModelUpdater<M>& updater);
};

extern template class RootOnlyInitializationMomentReplay<5>;
extern template class RootOnlyInitializationMomentReplay<10>;

}  // namespace stlqpp
