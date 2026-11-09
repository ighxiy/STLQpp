#include "stlqpp/training/checkpoint/full_topology_checkpoint_sink.h"

#include <cstddef>
#include <cstdint>

namespace stlqpp {

template <int M>
Status FullTopologyCheckpointSink<M>::Commit(
    std::uint64_t sequence,
    FullTopologyClusterCommitPayload payload,
    FullTopologyClusterProduct<M> product) {
  const ClusterTopologyResult<M>& topology = product.topology;
  const std::uint32_t real_count = payload.real_count();
  const std::uint32_t virtual_count = product.virtual_roots.has_value()
      ? product.virtual_roots->count()
      : 0U;
  if (sequence != payload.cid || topology.cid != payload.cid ||
      topology.dimension != payload.dimension ||
      topology.roles.size() != real_count ||
      topology.virtual_count != virtual_count ||
      payload.rotated_exact_real.size() !=
          static_cast<std::size_t>(payload.dimension) * real_count) {
    return Status::InvalidArgument(
        "full-topology result changed its ordered cluster shape");
  }

  const ArrayView<const std::uint64_t> virtual_ids =
      product.virtual_roots.has_value()
      ? product.virtual_roots->sample_ids()
      : ArrayView<const std::uint64_t>{};
  const ArrayView<const std::uint8_t> virtual_small_codes =
      product.virtual_roots.has_value()
      ? product.virtual_roots->small_codes()
      : ArrayView<const std::uint8_t>{};
  const ArrayView<const float> virtual_coefficients =
      product.virtual_roots.has_value()
      ? product.virtual_roots->coefficients()
      : ArrayView<const float>{};
  CanonicalBfsForestOrder order;
  if (const Status status = checkpoint_->Append({
          .cid = payload.cid,
          .dimension = payload.dimension,
          .virtual_sample_ids = virtual_ids,
          .virtual_small_codes = virtual_small_codes,
          .virtual_coefficients = virtual_coefficients,
          .real_sample_ids = payload.real_sample_ids,
          .real_roles = topology.roles,
          .real_parents = topology.parents,
          .real_depths = topology.depths,
      }, &order); !status.ok()) {
    return status;
  }
  // Append has validated this forest and produced its BFS order. The
  // transcript still performs the first real-support/coefficient check.
  if (const Status status = transcript_->AppendAfterForestValidation(
          topology, payload.real_sample_ids,
          product.virtual_roots.has_value()
              ? product.virtual_roots->small_codes()
              : ArrayView<const std::uint8_t>{},
          virtual_coefficients);
      !status.ok()) {
    return status;
  }
  if (real_count == 0U) {
    return Status::Ok();
  }
  return updater_->AccumulateFrozenEdgeClusterAfterFullSinkValidation(
      *codebooks_,
      {
          .exact_real = {
              payload.rotated_exact_real.data(), payload.dimension,
              real_count},
          .topology = &topology,
          .virtual_small_codes = virtual_small_codes,
          .virtual_coefficients = virtual_coefficients,
      }, order);
}

template class FullTopologyCheckpointSink<0>;
template class FullTopologyCheckpointSink<5>;
template class FullTopologyCheckpointSink<12>;
template class FullTopologyCheckpointSink<6>;
template class FullTopologyCheckpointSink<10>;

}  // namespace stlqpp
