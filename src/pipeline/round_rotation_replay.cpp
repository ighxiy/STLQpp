#include "stlqpp/pipeline/round_rotation_replay.h"

#include "stlqpp/training/update/frozen_forest_reconstruction.h"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

struct CidRange final {
  std::uint32_t begin = 0U;
  std::uint32_t end = 0U;
};

CidRange StaticShardRange(std::uint32_t shard,
                          std::uint32_t cid_count,
                          std::uint32_t shard_count) noexcept {
  const std::uint32_t short_size = cid_count / shard_count;
  const std::uint32_t long_count = cid_count % shard_count;
  const std::uint32_t begin = shard < long_count
      ? shard * (short_size + 1U)
      : long_count * (short_size + 1U) +
            (shard - long_count) * short_size;
  return {begin, begin + short_size + (shard < long_count ? 1U : 0U)};
}

}  // namespace

template <int M>
StatusOr<RotationCrossCovariance> RoundRotationReplay<M>::Accumulate(
    const TrainingBasicRoundArtifact& basic,
    const RoundUpdateTranscript<M>& transcript,
    const TrainingModel<M>& post_edge_model) {
  auto lists_result = BaseListReader<M>::Open(basic.list_directory);
  if (!lists_result.ok()) {
    return lists_result.status();
  }
  const BaseListReader<M>& lists = lists_result.value();
  const ModelShape& shape = post_edge_model.codebooks().shape();
  const BaseBuildShape& list_shape = lists.store().shape();
  if (shape.m != lists.store().m() || shape.m != transcript.m() ||
      transcript.dimension() != shape.dimension ||
      transcript.nlist() != shape.nlist ||
      list_shape.dimension != shape.dimension ||
      list_shape.nlist != shape.nlist) {
    return Status::InvalidArgument(
        "round rotation replay inputs are incompatible");
  }
  const std::uint32_t shard_count = std::min(
      parallel_->worker_count(), shape.nlist);
  std::vector<RotationCrossCovariance> shards;
  shards.reserve(shard_count);
  for (std::uint32_t shard = 0U; shard < shard_count; ++shard) {
    auto covariance = RotationCrossCovariance::Create(shape.dimension);
    if (!covariance.ok()) {
      return covariance.status();
    }
    shards.push_back(std::move(covariance).value());
  }
  std::vector<Status> statuses(shard_count, Status::Ok());
  struct Context final {
    const BaseListReader<M>* lists;
    const RoundUpdateTranscript<M>* transcript;
    const ProductCodebooks<M>* codebooks;
    const F32CrossCovarianceBackend* backend;
    ArrayView<RotationCrossCovariance> shards;
    ArrayView<Status> statuses;
    std::uint32_t cid_count;
    std::uint32_t shard_count;
  } context = {
      &lists, &transcript, &post_edge_model.codebooks(),
      covariance_backend_, shards, statuses, shape.nlist, shard_count};
  parallel_->Run(
      shard_count, &context,
      [](void* opaque, std::uint32_t, std::uint32_t shard) noexcept {
        auto& state = *static_cast<Context*>(opaque);
        const CidRange range = StaticShardRange(
            shard, state.cid_count, state.shard_count);
        LoadedBaseListCid<M> loaded;
        RoundUpdateTranscriptCluster<M> staged;
        FrozenForestReconstruction reconstruction;
        std::vector<std::pair<std::uint64_t, std::uint32_t>> list_rows;
        std::vector<float> list_order_reconstructions;
        std::vector<std::uint8_t> consumed;
        for (std::uint32_t cid = range.begin; cid < range.end; ++cid) {
          if (const Status status = state.lists->LoadCid(cid, &loaded);
              !status.ok()) {
            state.statuses[shard] = status;
            return;
          }
          if (const Status status =
                  state.transcript->LoadCluster(cid, &staged);
              !status.ok()) {
            state.statuses[shard] = status;
            return;
          }
          if (loaded.row_count() != staged.real_count()) {
            state.statuses[shard] = Status::FailedPrecondition(
                "round rotation replay changed native-list rows");
            return;
          }
          if (loaded.row_count() == 0U) {
            continue;
          }
          if (const Status status = ReconstructFrozenForest<M>(
                  *state.codebooks, staged, &reconstruction);
              !status.ok()) {
            state.statuses[shard] = status;
            return;
          }
          list_rows.clear();
          list_rows.reserve(loaded.row_count());
          for (std::uint32_t row = 0U;
               row < loaded.row_count(); ++row) {
            list_rows.emplace_back(loaded.ids()[row], row);
          }
          std::sort(list_rows.begin(), list_rows.end());
          for (std::size_t row = 1U; row < list_rows.size(); ++row) {
            if (list_rows[row - 1U].first == list_rows[row].first) {
              state.statuses[shard] = Status::DataLoss(
                  "round rotation native-list IDs are not unique");
              return;
            }
          }
          list_order_reconstructions.resize(
              static_cast<std::size_t>(state.codebooks->shape().dimension) *
              loaded.row_count());
          consumed.assign(loaded.row_count(), 0U);
          for (std::uint32_t row = 0U;
               row < staged.real_count(); ++row) {
            const std::uint64_t stable_id =
                staged.real_sample_ids[row];
            const auto found = std::lower_bound(
                list_rows.begin(), list_rows.end(),
                std::pair<std::uint64_t, std::uint32_t>{stable_id, 0U});
            if (found == list_rows.end() || found->first != stable_id ||
                consumed[found->second] != 0U) {
              state.statuses[shard] = Status::FailedPrecondition(
                  "round rotation native-list membership changed");
              return;
            }
            consumed[found->second] = 1U;
            std::memcpy(
                list_order_reconstructions.data() +
                    static_cast<std::size_t>(found->second) *
                        state.codebooks->shape().dimension,
                reconstruction.real_view().column(row),
                static_cast<std::size_t>(
                    state.codebooks->shape().dimension) * sizeof(float));
          }
          const Status status = state.shards[shard].Accumulate(
              loaded.exact_vectors(),
              {list_order_reconstructions.data(),
               state.codebooks->shape().dimension,
               loaded.row_count()},
              *state.backend);
          if (!status.ok()) {
            state.statuses[shard] = status;
            return;
          }
        }
      });
  for (const Status& status : statuses) {
    if (!status.ok()) {
      return status;
    }
  }
  for (std::uint32_t shard = 1U; shard < shard_count; ++shard) {
    if (const Status status = shards[0].MergeFrom(shards[shard]);
        !status.ok()) {
      return status;
    }
  }
  return std::move(shards[0]);
}

template class RoundRotationReplay<0>;
template class RoundRotationReplay<5>;
template class RoundRotationReplay<12>;
template class RoundRotationReplay<6>;
template class RoundRotationReplay<10>;

}  // namespace stlqpp
