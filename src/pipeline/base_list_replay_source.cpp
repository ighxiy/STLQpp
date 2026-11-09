#include "stlqpp/pipeline/base_list_replay_source.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"

namespace stlqpp {

template <int M>
StatusOr<TrainingReplayRealCluster> BaseListReplaySource<M>::Load(
    std::uint32_t cid, ArrayView<const std::uint64_t> stable_ids,
    std::uint32_t round_seed) {
  auto loaded_result = lists_->LoadCid(cid);
  if (!loaded_result.ok()) {
    return loaded_result.status();
  }
  auto loaded = std::move(loaded_result).value();
  TrainingReplayRealCluster output;
  output.cid = cid;
  output.dimension = lists_->store().shape().dimension;
  output.real_count = static_cast<std::uint32_t>(loaded.ids().size());
  if (stable_ids.size() != loaded.ids().size()) {
    return Status::FailedPrecondition(
        "base-list replay stable-ID extent changed");
  }
  for (std::size_t row = 0U; row < stable_ids.size(); ++row) {
    if (stable_ids[row] != loaded.ids()[row]) {
      return Status::FailedPrecondition(
          "base-list replay stable-ID order changed");
    }
  }
  auto orders_result =
      BuildEdgeIcmLayerOrders<M>(round_seed, stable_ids, lists_->store().m());
  if (!orders_result.ok()) {
    return orders_result.status();
  }
  output.edge_icm_layer_orders = std::move(orders_result).value();
  output.basic_small_codes.assign(loaded.small_codes_row_major().begin(),
                                  loaded.small_codes_row_major().end());
  output.basic_coefficients.assign(loaded.coefficients_row_major().begin(),
                                   loaded.coefficients_row_major().end());
  output.unrotated_exact_real = std::move(loaded).TakeExactVectors();
  return output;
}

template class BaseListReplaySource<0>;
template class BaseListReplaySource<5>;
template class BaseListReplaySource<12>;
template class BaseListReplaySource<6>;
template class BaseListReplaySource<10>;

}  // namespace stlqpp
