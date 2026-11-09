#include "stlqpp/training/checkpoint/checkpoint_replay_sink.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

template <typename T>
bool Same(stlqpp::ArrayView<const T> frozen,
          const std::vector<T>& output) {
  return frozen.size() == output.size() &&
      std::equal(frozen.begin(), frozen.end(), output.begin());
}

template <int M>
bool FrozenFieldsMatch(
    const stlqpp::FixedTopologyReplayRequest& request,
    const stlqpp::FixedTopologyReplayResult<M>& result) {
  return Same(request.frozen_roles, result.topology.roles) &&
      Same(request.frozen_parents, result.topology.parents) &&
      Same(request.frozen_depths, result.topology.depths) &&
      Same(request.frozen_virtual_small_codes,
           result.virtual_small_codes) &&
      request.frozen_virtual_coefficients.size() ==
          result.virtual_coefficients.size() &&
      (request.frozen_virtual_coefficients.empty() ||
       std::memcmp(request.frozen_virtual_coefficients.data(),
                   result.virtual_coefficients.data(),
                   request.frozen_virtual_coefficients.size() *
                       sizeof(float)) == 0);
}

}  // namespace

namespace stlqpp {

template <int M>
Status CheckpointReplaySink<M>::Commit(
    std::uint64_t sequence,
    ArrayView<const FixedTopologyReplayRequest> requests,
    ArrayView<const FixedTopologyReplayResult<M>> results) {
  return CommitImpl<false>(sequence, requests, results);
}

template <int M>
Status CheckpointReplaySink<M>::CommitValidatedFromExecutor(
    std::uint64_t sequence,
    ArrayView<const FixedTopologyReplayRequest> requests,
    ArrayView<const FixedTopologyReplayResult<M>> results) {
  return CommitImpl<true>(sequence, requests, results);
}

template <int M>
template <bool Prevalidated>
Status CheckpointReplaySink<M>::CommitImpl(
    std::uint64_t sequence,
    ArrayView<const FixedTopologyReplayRequest> requests,
    ArrayView<const FixedTopologyReplayResult<M>> results) {
  static_cast<void>(sequence);
  if (requests.size() != results.size()) {
    return Status::InvalidArgument(
        "checkpoint replay sink received mismatched window rows");
  }
  for (std::size_t cluster = 0U; cluster < results.size(); ++cluster) {
    const FixedTopologyReplayRequest& request = requests[cluster];
    const ClusterTopologyResult<M>& topology = results[cluster].topology;
    if (request.cid != topology.cid ||
        request.virtual_sample_ids.size() != topology.virtual_count) {
      return Status::InvalidArgument(
          "checkpoint replay result changed CID or virtual-root count");
    }
    if constexpr (Prevalidated) {
      assert(FrozenFieldsMatch(request, results[cluster]));
    } else if (!FrozenFieldsMatch(request, results[cluster])) {
      return Status::FailedPrecondition(
          "checkpoint replay changed a frozen forest field");
    }
    const Status status = writer_->Append({
        .cid = request.cid,
        .dimension = topology.dimension,
        .virtual_sample_ids = request.virtual_sample_ids,
        .virtual_small_codes = results[cluster].virtual_small_codes,
        .virtual_coefficients = results[cluster].virtual_coefficients,
        .real_sample_ids = request.real_sample_ids,
        .real_roles = topology.roles,
        .real_parents = topology.parents,
        .real_depths = topology.depths,
    });
    if (!status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

template class CheckpointReplaySink<0>;
template class CheckpointReplaySink<5>;
template class CheckpointReplaySink<12>;
template class CheckpointReplaySink<6>;
template class CheckpointReplaySink<10>;

}  // namespace stlqpp
