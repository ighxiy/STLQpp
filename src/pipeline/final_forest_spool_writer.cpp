#include "stlqpp/pipeline/final_forest_spool_writer.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

template <int M>
struct PrepareWindowContext final {
  ArrayView<const FinalForestSpoolClusterView<M>> clusters;
  const FinalForestSpoolWriter<M>* writer = nullptr;
  std::vector<PreparedCanonicalCluster<M>>* prepared = nullptr;
  std::vector<Status>* statuses = nullptr;
};

template <int M>
void PrepareWindowItem(void* opaque,
                       std::uint32_t,
                       std::uint32_t local) noexcept {
  auto& context = *static_cast<PrepareWindowContext<M>*>(opaque);
  const FinalForestSpoolClusterView<M>& cluster = context.clusters[local];
  auto prepared_result = context.writer->PrepareCluster(cluster);
  if (!prepared_result.ok()) {
    (*context.statuses)[local] = prepared_result.status();
    return;
  }
  (*context.prepared)[local] = std::move(prepared_result).value();
}

}  // namespace

template <int M>
Status FinalForestSpoolWriter<M>::AppendWindow(
    ArrayView<const FinalForestSpoolClusterView<M>> clusters) {
  if (failed_) {
    return Status::FailedPrecondition(
        "final forest spool transaction already failed during append");
  }
  if (clusters.empty() ||
      clusters.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status::InvalidArgument(
        "final forest spool requires one nonempty bounded CID window");
  }
  const std::uint32_t count =
      static_cast<std::uint32_t>(clusters.size());
  if ((!spool_->direct_payloads() &&
       (next_cid_ >= shape_.nlist || count > shape_.nlist - next_cid_)) ||
      count > shape_.nlist) {
    return Status::InvalidArgument(
        "final forest window exceeds the native IVF CID domain");
  }
  std::vector<PreparedCanonicalCluster<M>> prepared(count);
  std::vector<Status> statuses(count);
  PrepareWindowContext<M> context = {
      .clusters = clusters,
      .writer = this,
      .prepared = &prepared,
      .statuses = &statuses,
  };
  parallel_->Run(count, &context, &PrepareWindowItem<M>);
  for (const Status& status : statuses) {
    if (!status.ok()) {
      return status;
    }
  }
  return AppendPreparedWindow(prepared);
}

template <int M>
StatusOr<PreparedCanonicalCluster<M>>
FinalForestSpoolWriter<M>::PrepareCluster(
    const FinalForestSpoolClusterView<M>& cluster) const {
  if (cluster.topology == nullptr ||
      cluster.topology->cid >= shape_.nlist ||
      cluster.topology->dimension != shape_.dimension ||
      cluster.real_sample_ids.size() != cluster.topology->roles.size()) {
    return Status::InvalidArgument(
        "final forest cluster disagrees with model shape");
  }
  std::vector<std::uint32_t> real_ids(cluster.real_sample_ids.size());
  for (std::size_t row = 0U; row < real_ids.size(); ++row) {
    if (cluster.real_sample_ids[row] >
        std::numeric_limits<std::uint32_t>::max()) {
      return Status::ResourceExhausted(
          "final base ID exceeds canonical v1 u32 storage");
    }
    real_ids[row] = static_cast<std::uint32_t>(
        cluster.real_sample_ids[row]);
  }
  const ClusterTopologyResult<M>& topology = *cluster.topology;
  auto assembled_result = CanonicalFloatClusterAssembler<M>::Assemble({
      .m = topology.m,
      .cid = topology.cid,
      .real_ids = real_ids,
      .forest = {
          .virtual_count = topology.virtual_count,
          .real_roles = topology.roles,
          .real_parents = topology.parents,
      },
      .real_supports = topology.supports,
      .real_coefficients_node_major = topology.coefficients,
      .source_real_norm2 = cluster.source_real_norm2,
      .virtual_codes_small = cluster.virtual_codes_small,
      .virtual_coefficients_node_major =
          cluster.virtual_coefficients_node_major,
  });
  if (!assembled_result.ok()) {
    return assembled_result.status();
  }
  auto cross_result = grams_->BuildRootCrossBlock(topology.cid, 1U);
  if (!cross_result.ok()) {
    return cross_result.status();
  }
  return postprocessor_.PrepareAssembled(
      assembled_result.value(), cross_result.value());
}

template <int M>
Status FinalForestSpoolWriter<M>::AppendPreparedWindow(
    ArrayView<const PreparedCanonicalCluster<M>> prepared) {
  if (failed_) {
    return Status::FailedPrecondition(
        "final forest spool transaction already failed during append");
  }
  if (prepared.empty() ||
      prepared.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status::InvalidArgument(
        "final forest spool requires one nonempty prepared CID window");
  }
  const std::uint32_t count =
      static_cast<std::uint32_t>(prepared.size());
  if ((!spool_->direct_payloads() &&
       (next_cid_ >= shape_.nlist || count > shape_.nlist - next_cid_)) ||
      count > shape_.nlist) {
    return Status::InvalidArgument(
        "prepared final forest window exceeds the native IVF CID domain");
  }
  for (std::uint32_t local = 0U; local < count; ++local) {
    if (!spool_->direct_payloads() &&
        prepared[local].cid() != next_cid_ + local) {
      return Status::InvalidArgument(
          "prepared final forest window changed CID order");
    }
  }
  for (const PreparedCanonicalCluster<M>& cluster : prepared) {
    if (const Status status = spool_->Append(cluster); !status.ok()) {
      failed_ = true;
      return status;
    }
  }
  next_cid_ += count;
  return Status::Ok();
}

template class FinalForestSpoolWriter<0>;
template class FinalForestSpoolWriter<5>;
template class FinalForestSpoolWriter<12>;
template class FinalForestSpoolWriter<6>;
template class FinalForestSpoolWriter<10>;

}  // namespace stlqpp
