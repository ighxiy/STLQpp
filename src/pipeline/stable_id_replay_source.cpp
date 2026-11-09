#include "stlqpp/pipeline/stable_id_replay_source.h"

#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace stlqpp {

template <int M>
StatusOr<TrainingReplayRealCluster> StableIdReplaySource<M>::Load(
    std::uint32_t cid,
    ArrayView<const std::uint64_t> stable_ids,
    std::uint32_t round_seed) {
  if (current_basic_ == nullptr ||
      stable_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status::FailedPrecondition(
        "stable-ID replay source is unavailable or oversized");
  }
  TrainingReplayRealCluster output;
  output.cid = cid;
  output.dimension = current_basic_->store().shape().dimension;
  output.real_count = static_cast<std::uint32_t>(stable_ids.size());
  if (const Status status = current_basic_->GatherCidByStableIds(
          cid, stable_ids, &output.unrotated_exact_real,
          &output.basic_small_codes, &output.basic_coefficients);
      !status.ok()) {
    return status;
  }
  auto orders = BuildEdgeIcmLayerOrders<M>(round_seed, stable_ids, current_basic_->store().m());
  if (!orders.ok()) {
    return orders.status();
  }
  output.edge_icm_layer_orders = std::move(orders).value();
  return output;
}

template class StableIdReplaySource<0>;
template class StableIdReplaySource<5>;
template class StableIdReplaySource<12>;
template class StableIdReplaySource<6>;
template class StableIdReplaySource<10>;

}  // namespace stlqpp
