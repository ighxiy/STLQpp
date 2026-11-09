#include "stlqpp/pipeline/root_only_initialization_moment_replay.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

template <int M>
Status ReconstructCluster(
    const RootCodebooks<M>& codebooks,
    ClusterTopologyResult<M>* topology) {
  const std::uint32_t dimension = codebooks.shape().dimension;
  const std::uint32_t rows =
      static_cast<std::uint32_t>(topology->roles.size());
  topology->m = codebooks.m();
  topology->dimension = dimension;
  topology->virtual_count = 0U;
  topology->real_reconstructions.assign(
      static_cast<std::size_t>(dimension) * rows, 0.0F);

  std::vector<std::uint32_t> causal_order(rows);
  std::iota(causal_order.begin(), causal_order.end(), 0U);
  std::stable_sort(
      causal_order.begin(), causal_order.end(),
      [&](std::uint32_t left, std::uint32_t right) {
        return topology->depths[left] < topology->depths[right];
      });

  for (const std::uint32_t row : causal_order) {
    float* reconstruction = topology->real_reconstructions.data() +
        static_cast<std::size_t>(row) * dimension;
    const LinkageCodeTuple<M>& support = topology->supports[row];
    const float* coefficients = topology->coefficients.data() +
        static_cast<std::size_t>(row) * codebooks.m();
    for (std::uint32_t layer = 0U;
         layer < codebooks.m(); ++layer) {
      const float* atom = layer == 0U
          ? codebooks.root0().column(support.first)
          : codebooks.root_small(layer).column(
                support.trailing[layer - 1U]);
      const float coefficient = coefficients[layer];
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        reconstruction[axis] += coefficient * atom[axis];
      }
    }
    if (topology->roles[row] == LinkageNodeRole::kLinked) {
      const std::uint32_t parent = topology->parents[row]->local_id;
      const float* parent_reconstruction =
          topology->real_reconstructions.data() +
          static_cast<std::size_t>(parent) * dimension;
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        reconstruction[axis] += parent_reconstruction[axis];
      }
    }
  }
  return Status::Ok();
}

}  // namespace

template <int M>
StatusOr<RootOnlyInitializationMomentReplayStatistics>
RootOnlyInitializationMomentReplay<M>::Run(
    const BaseListReader<M>& lists,
    const RootTrainingModel<M>& root_model,
    const RootOnlyInitializationTranscript<M>& transcript,
    ArrayView<const std::uint8_t> bad_cluster_mask,
    double bad_cluster_weight,
    const RootToEdgeCodeMap& root_to_edge,
    InitializationModelUpdater<M>& updater) {
  const ModelShape& shape = root_model.codebooks().shape();
  const BaseBuildShape& list_shape = lists.store().shape();
  if (transcript.m() != shape.m || lists.store().m() != shape.m ||
      transcript.dimension() != shape.dimension ||
      transcript.nlist() != shape.nlist ||
      list_shape.dimension != shape.dimension ||
      list_shape.nlist != shape.nlist ||
      bad_cluster_mask.size() != shape.nlist ||
      root_to_edge.root_count() != shape.nlist ||
      !std::isfinite(bad_cluster_weight) || bad_cluster_weight <= 0.0) {
    return Status::InvalidArgument(
        "root-only initialization moment replay inputs are incompatible");
  }

  if (updater.phase_ != InitializationModelUpdater<M>::Phase::kTopology) {
    return Status::FailedPrecondition(
        "initialization moment replay requires the topology phase");
  }
  const auto shard_count =
      static_cast<std::uint32_t>(updater.edge_moment_shards_.size() + 1U);
  std::vector<RootOnlyInitializationMomentReplayStatistics> partial(shard_count);
  std::vector<Status> errors(shard_count, Status::Ok());
  auto replay_shard = [&](std::uint32_t shard) -> Status {
    auto& statistics = partial[shard];
    // Same partition as StaticCidShard / DEV's schedule(static), including
    // remainder and empty shards. Each accumulator has exactly one writer.
    const auto short_size = shape.nlist / shard_count;
    const auto long_count = shape.nlist % shard_count;
    const auto begin = shard * short_size + std::min(shard, long_count);
    const auto end = begin + short_size + (shard < long_count ? 1U : 0U);
    for (std::uint32_t cid = begin; cid < end; ++cid) {
      auto loaded = lists.LoadCid(cid);
      if (!loaded.ok()) {
        return loaded.status();
      }
      auto staged = transcript.LoadCluster(cid);
      if (!staged.ok()) {
        return staged.status();
      }
      if (loaded.value().row_count() != staged.value().real_count()) {
        return Status::FailedPrecondition(
            "root-only initialization transcript changed native-list rows");
      }

      ClusterTopologyResult<M> topology;
      topology.cid = cid;
      topology.roles = std::move(staged.value().roles);
      topology.parents = std::move(staged.value().parents);
      topology.depths = std::move(staged.value().depths);
      topology.supports = std::move(staged.value().supports);
      topology.coefficients = std::move(staged.value().coefficients);
      if (const Status status = ReconstructCluster(
              root_model.codebooks(), &topology);
          !status.ok()) {
        return status;
      }
      const std::uint32_t linked = static_cast<std::uint32_t>(std::count(
          topology.roles.begin(), topology.roles.end(),
          LinkageNodeRole::kLinked));
      if (loaded.value().row_count() != 0U) {
        const double weight = bad_cluster_mask[cid] == 0U
            ? 1.0
            : bad_cluster_weight;
        if (const Status status = updater.AccumulateRootOnlyClusterAfterReplayValidation(
                {
                    .exact_real = loaded.value().exact_vectors(),
                    .topology = &topology,
                    .real_reconstructions = {
                        topology.real_reconstructions.data(),
                        topology.dimension,
                        static_cast<std::uint32_t>(topology.roles.size())},
                    .virtual_reconstructions = {},
                },
                root_to_edge, weight);
            !status.ok()) {
          return status;
        }
      }
      ++statistics.clusters;
      statistics.real_rows += loaded.value().row_count();
      statistics.linked_rows += linked;
    }
    return Status::Ok();
  };
  struct Work {
    decltype(replay_shard)& run;
    std::vector<Status>& errors;
  } work{replay_shard, errors};
  updater.edge_moment_parallel_->Run(
      shard_count, &work,
      [](void* opaque, std::uint32_t, std::uint32_t shard) noexcept {
        auto& state = *static_cast<Work*>(opaque);
        state.errors[shard] = state.run(shard);
      });
  RootOnlyInitializationMomentReplayStatistics statistics;
  for (std::uint32_t shard = 0U; shard < shard_count; ++shard) {
    // Only the calling thread changes updater phase, including failures.
    if (!errors[shard].ok()) return updater.Fail(errors[shard]);
    statistics.clusters += partial[shard].clusters;
    statistics.real_rows += partial[shard].real_rows;
    statistics.linked_rows += partial[shard].linked_rows;
  }
  return statistics;
}

template class RootOnlyInitializationMomentReplay<0>;
template class RootOnlyInitializationMomentReplay<5>;
template class RootOnlyInitializationMomentReplay<12>;
template class RootOnlyInitializationMomentReplay<6>;
template class RootOnlyInitializationMomentReplay<10>;

}  // namespace stlqpp
