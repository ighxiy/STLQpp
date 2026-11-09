#include "stlqpp/pipeline/full_topology_spool_sink.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace stlqpp {

template <int M>
struct FullTopologySpoolSink<M>::PreparationState final {
  std::mutex mutex;
  std::unordered_map<std::uint64_t, PreparedCanonicalCluster<M>> completed;
};

template <int M>
StatusOr<FullTopologySpoolSink<M>> FullTopologySpoolSink<M>::Create(
    FinalForestWindowSink<M>& destination,
    FullTopologySpoolWindowOptions options,
    std::uint32_t first_sequence) {
  if (options.maximum_clusters == 0U || options.target_real_rows == 0U) {
    return Status::InvalidArgument(
        "full-topology spool window bounds must be positive");
  }
  FullTopologySpoolSink<M> output(destination, options, first_sequence);
  output.preparation_ = std::make_shared<PreparationState>();
  return output;
}

template <int M>
Status FullTopologySpoolSink<M>::PrepareForCommit(
    std::uint64_t sequence,
    FullTopologyClusterCommitPayload* payload,
    FullTopologyClusterProduct<M>* product) {
  if (payload == nullptr || product == nullptr || preparation_ == nullptr) {
    return Status::InvalidArgument(
        "full-topology worker preparation requires live state");
  }
  const std::uint32_t virtual_count = product->virtual_roots.has_value()
      ? product->virtual_roots->count()
      : 0U;
  if (payload->cid != sequence || product->topology.cid != sequence ||
      product->topology.dimension != payload->dimension ||
      product->topology.roles.size() != payload->real_count() ||
      product->topology.virtual_count != virtual_count ||
      !payload->rotated_exact_real.empty()) {
    return Status::InvalidArgument(
        "full-topology worker preparation changed CID or cluster shape");
  }
  const ArrayView<const std::uint8_t> virtual_codes =
      product->virtual_roots.has_value()
      ? product->virtual_roots->small_codes()
      : ArrayView<const std::uint8_t>{};
  const ArrayView<const float> virtual_coefficients =
      product->virtual_roots.has_value()
      ? product->virtual_roots->coefficients()
      : ArrayView<const float>{};
  auto prepared = destination_->PrepareCluster({
      .real_sample_ids = payload->real_sample_ids,
      .source_real_norm2 = payload->source_real_norm2,
      .topology = &product->topology,
      .virtual_codes_small = virtual_codes,
      .virtual_coefficients_node_major = virtual_coefficients,
  });
  if (!prepared.ok()) {
    return prepared.status();
  }
  const std::uint32_t retained_virtual_count =
      prepared.value().virtual_count();
  {
    std::lock_guard<std::mutex> lock(preparation_->mutex);
    const bool inserted = preparation_->completed.emplace(
        sequence, std::move(prepared).value()).second;
    if (!inserted) {
      return Status::Internal(
          "full-topology worker prepared one CID more than once");
    }
  }
  std::vector<std::uint64_t>().swap(payload->real_sample_ids);
  std::vector<float>().swap(payload->source_real_norm2);
  ClusterTopologyResult<M> compact;
  compact.m = product->topology.m;
  compact.cid = product->topology.cid;
  compact.dimension = product->topology.dimension;
  // Preserve the post-liveness count after releasing all static forest
  // streams so the outer pipeline can report generated versus retained roots.
  compact.virtual_count = retained_virtual_count;
  product->topology = std::move(compact);
  product->virtual_roots.reset();
  return Status::Ok();
}

template <int M>
std::uint64_t FullTopologySpoolSink<M>::PreparedResidentBytes(
    std::uint64_t sequence) const noexcept {
  std::lock_guard<std::mutex> lock(preparation_->mutex);
  const auto found = preparation_->completed.find(sequence);
  return found == preparation_->completed.end()
      ? 0U : found->second.resident_bytes() + sizeof(*found);
}

template <int M>
Status FullTopologySpoolSink<M>::Commit(
    std::uint64_t sequence,
    FullTopologyClusterCommitPayload payload,
    FullTopologyClusterProduct<M> product) {
  if (failed_ || finished_) {
    return Status::FailedPrecondition(
        "full-topology spool sink is no longer accepting clusters");
  }
  if ((RequiresOrderedCommit() && sequence != next_sequence_) ||
      payload.cid != sequence || product.topology.cid != sequence ||
      product.topology.dimension != payload.dimension ||
      !payload.real_sample_ids.empty() || !payload.source_real_norm2.empty() ||
      !payload.rotated_exact_real.empty() || preparation_ == nullptr) {
    return Status::InvalidArgument(
        "full-topology spool input changed CID order or cluster shape");
  }
  PreparedCanonicalCluster<M> pending;
  {
    std::lock_guard<std::mutex> lock(preparation_->mutex);
    auto found = preparation_->completed.find(sequence);
    if (found == preparation_->completed.end()) {
      return Status::Internal(
          "full-topology ordered commit has no prepared cluster");
    }
    pending = std::move(found->second);
    preparation_->completed.erase(found);
  }
  if (pending.cid() != sequence) {
    return Status::Internal(
        "full-topology prepared cluster changed CID order");
  }
  pending_real_rows_ += pending.real_count();
  pending_.push_back(std::move(pending));
  ++next_sequence_;
  if (pending_.size() >= options_.maximum_clusters ||
      pending_real_rows_ >= options_.target_real_rows) {
    return Flush();
  }
  return Status::Ok();
}

template <int M>
Status FullTopologySpoolSink<M>::Flush() {
  if (pending_.empty()) {
    return Status::Ok();
  }
  Status status = destination_->AppendPreparedWindow(pending_);
  if (!status.ok()) {
    failed_ = true;
    return status;
  }
  pending_.clear();
  pending_real_rows_ = 0U;
  return Status::Ok();
}

template <int M>
Status FullTopologySpoolSink<M>::Finish() {
  if (failed_ || finished_) {
    return Status::FailedPrecondition(
        "full-topology spool sink cannot finish twice or after failure");
  }
  if (const Status status = Flush(); !status.ok()) {
    return status;
  }
  finished_ = true;
  return Status::Ok();
}

template class FullTopologySpoolSink<0>;
template class FullTopologySpoolSink<5>;
template class FullTopologySpoolSink<12>;
template class FullTopologySpoolSink<6>;
template class FullTopologySpoolSink<10>;

}  // namespace stlqpp
