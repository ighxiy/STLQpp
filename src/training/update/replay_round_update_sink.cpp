#include "stlqpp/training/update/replay_round_update_sink.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

template <int M>
Status ReplayRoundUpdateSink<M>::Commit(
    std::uint64_t sequence,
    ArrayView<const FixedTopologyReplayRequest> requests,
    ArrayView<const FixedTopologyReplayResult<M>> results) {
  return CommitImpl<false>(sequence, requests, results);
}

template <int M>
Status ReplayRoundUpdateSink<M>::CommitValidatedFromExecutor(
    std::uint64_t sequence,
    ArrayView<const FixedTopologyReplayRequest> requests,
    ArrayView<const FixedTopologyReplayResult<M>> results) {
  return CommitImpl<true>(sequence, requests, results);
}

template <int M>
template <bool Prevalidated>
Status ReplayRoundUpdateSink<M>::CommitImpl(
    std::uint64_t sequence,
    ArrayView<const FixedTopologyReplayRequest> requests,
    ArrayView<const FixedTopologyReplayResult<M>> results) {
  if (requests.size() != results.size()) {
    return Status::InvalidArgument(
        "replay update sink received mismatched request/result windows");
  }
  Status staged = Prevalidated
      ? staging_sink_->CommitValidatedFromExecutor(
            sequence, requests, results)
      : staging_sink_->Commit(sequence, requests, results);
  if (!staged.ok()) {
    return staged;
  }
  std::vector<FrozenLinkedTrainingClusterView<M>> clusters;
  clusters.reserve(results.size());
  for (std::size_t cluster = 0U; cluster < results.size(); ++cluster) {
    const FixedTopologyReplayResult<M>& result = results[cluster];
    const FixedTopologyReplayRequest& request = requests[cluster];
    if (const Status status = transcript_->Append(
            result.topology, request.real_sample_ids,
            result.virtual_small_codes,
            result.virtual_coefficients);
        !status.ok()) {
      return status;
    }
    clusters.push_back({
        .exact_real = request.exact_real,
        .topology = &result.topology,
        .virtual_small_codes = result.virtual_small_codes,
        .virtual_coefficients = result.virtual_coefficients,
    });
  }
  if constexpr (Prevalidated) {
    return updater_->AccumulateFrozenEdgeClustersAfterReplayValidation(
        *codebooks_, clusters);
  }
  return updater_->AccumulateFrozenEdgeClusters(*codebooks_, clusters);
}

template class ReplayRoundUpdateSink<0>;
template class ReplayRoundUpdateSink<5>;
template class ReplayRoundUpdateSink<12>;
template class ReplayRoundUpdateSink<6>;
template class ReplayRoundUpdateSink<10>;

}  // namespace stlqpp
