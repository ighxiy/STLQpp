#include "stlqpp/training/update/round_model_updater.h"

#include "stlqpp/training/update/round_model_assembler.h"
#include "stlqpp/training/update/root_codebook_solver.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

std::uint32_t StaticCidShard(std::uint32_t cid,
                             std::uint32_t cid_count,
                             std::uint32_t shard_count) noexcept {
  const std::uint32_t short_size = cid_count / shard_count;
  const std::uint32_t long_count = cid_count % shard_count;
  const std::uint32_t long_end = long_count * (short_size + 1U);
  return cid < long_end
      ? cid / (short_size + 1U)
      : long_count + (cid - long_end) / short_size;
}

template <int M>
Status ValidateEdgeCluster(const EdgeTrainingClusterView<M>& cluster,
                           std::uint32_t dimension,
                           std::size_t cid_count) {
  if (cluster.linked.topology == nullptr) {
    return Status::FailedPrecondition("edge row has no frozen topology");
  }
  const ClusterTopologyResult<M>& topology = *cluster.linked.topology;
  if (topology.dimension != dimension || topology.cid >= cid_count) {
    return Status::InvalidArgument(
        "round edge rows do not match the frozen forest");
  }
  return Status::Ok();
}

template <int M>
Status ValidateFrozenEdgeCluster(
    const FrozenLinkedTrainingClusterView<M>& cluster,
    std::uint32_t dimension,
    std::size_t cid_count) {
  if (cluster.topology == nullptr) {
    return Status::FailedPrecondition("edge row has no frozen topology");
  }
  const ClusterTopologyResult<M>& topology = *cluster.topology;
  if (topology.dimension != dimension || topology.cid >= cid_count) {
    return Status::InvalidArgument(
        "round edge rows do not match the frozen forest");
  }
  return Status::Ok();
}

}  // namespace

template <int M>
RoundModelUpdater<M>::RoundModelUpdater(
    const TrainingModel<M>& previous,
    const SpdSolverF32& spd_solver,
    const SquareSvdF32& svd,
    RootCodebookMoments<M> root_moments,
    std::uint32_t edge_moment_shard_count,
    const ParallelForExecutor* final_moment_parallel,
    std::vector<std::uint8_t> bad_cluster_mask,
    float bad_cluster_edge_weight) noexcept
    : previous_(&previous),
      spd_solver_(&spd_solver),
      svd_(&svd),
      root_moments_(std::move(root_moments)),
      edge_moment_shard_count_(edge_moment_shard_count),
      final_moment_parallel_(final_moment_parallel),
      bad_cluster_mask_(std::move(bad_cluster_mask)),
      bad_cluster_edge_weight_(bad_cluster_edge_weight) {}

template <int M>
StatusOr<RoundModelUpdater<M>> RoundModelUpdater<M>::Create(
    const TrainingModel<M>& previous,
    const SpdSolverF32& spd_solver,
    const SquareSvdF32& svd,
    RoundModelUpdatePolicy policy,
    const ParallelForExecutor& root_moment_parallel,
    std::uint32_t edge_moment_shard_count,
    const ParallelForExecutor* final_moment_parallel) {
  return Create(
      previous, spd_solver, svd, policy, root_moment_parallel,
      root_moment_parallel,
      edge_moment_shard_count, final_moment_parallel);
}

template <int M>
StatusOr<RoundModelUpdater<M>> RoundModelUpdater<M>::Create(
    const TrainingModel<M>& previous,
    const SpdSolverF32& spd_solver,
    const SquareSvdF32& svd,
    RoundModelUpdatePolicy policy,
    const ParallelForExecutor& root_accumulation_parallel,
    const ParallelForExecutor& root_conversion_parallel,
    std::uint32_t edge_moment_shard_count,
    const ParallelForExecutor* final_moment_parallel) {
  const ModelShape& shape = previous.codebooks().shape();
  if (policy.bad_cluster_mask.size() != shape.nlist ||
      edge_moment_shard_count == 0U ||
      edge_moment_shard_count > root_conversion_parallel.worker_count() ||
      !std::isfinite(policy.bad_cluster_edge_weight) ||
      policy.bad_cluster_edge_weight <= 0.0F) {
    return Status::InvalidArgument(
        "round model update has an invalid bad-cluster policy");
  }
  for (const std::uint8_t value : policy.bad_cluster_mask) {
    if (value > 1U) {
      return Status::InvalidArgument(
          "round model update bad-cluster mask is not binary");
    }
  }
  auto root = RootCodebookMoments<M>::Create(
      shape.dimension, shape.nlist, root_accumulation_parallel,
      root_conversion_parallel, shape.m);
  if (!root.ok()) {
    return root.status();
  }
  return RoundModelUpdater<M>(
      previous, spd_solver, svd, std::move(root).value(),
      edge_moment_shard_count,
      final_moment_parallel,
      std::vector<std::uint8_t>(policy.bad_cluster_mask.begin(),
                                policy.bad_cluster_mask.end()),
      policy.bad_cluster_edge_weight);
}

template <int M>
Status RoundModelUpdater<M>::Fail(Status status) noexcept {
  phase_ = Phase::kFailed;
  return status;
}

template <int M>
Status RoundModelUpdater<M>::EnsureEdgeMoments() {
  if (edge_moments_.has_value()) {
    return Status::Ok();
  }
  auto primary = EdgeCodebookMoments<M>::Create(
      previous_->codebooks().shape().dimension, previous_->codebooks().shape().m);
  if (!primary.ok()) {
    return primary.status();
  }
  std::vector<EdgeCodebookMoments<M>> shards;
  shards.reserve(edge_moment_shard_count_ - 1U);
  for (std::uint32_t shard = 1U;
       shard < edge_moment_shard_count_; ++shard) {
    auto moments = EdgeCodebookMoments<M>::Create(
        previous_->codebooks().shape().dimension, previous_->codebooks().shape().m);
    if (!moments.ok()) {
      return moments.status();
    }
    shards.push_back(std::move(moments).value());
  }
  edge_moments_.emplace(std::move(primary).value());
  edge_moment_shards_ = std::move(shards);
  return Status::Ok();
}

template <int M>
Status RoundModelUpdater<M>::AccumulateBasic(
    ColumnMajorMatrixF32View exact_model_space,
    const BasicEncoding<M>& encoding) {
  if (phase_ != Phase::kBasic) {
    return Status::FailedPrecondition(
        "basic rows arrived outside the root-update phase");
  }
  if (!root_moments_.has_value()) {
    return Fail(Status::Internal("round root moments are unavailable"));
  }
  const Status status = root_moments_->Accumulate(
      exact_model_space, encoding);
  return status.ok() ? status : Fail(status);
}

template <int M>
Status RoundModelUpdater<M>::EnablePostRootMetric() {
  if (phase_ != Phase::kBasic || !root_moments_.has_value()) {
    return Status::FailedPrecondition(
        "post-root metric may only be enabled before basic accumulation");
  }
  return root_moments_->EnablePostRootMetric();
}

template <int M>
std::optional<double> RoundModelUpdater<M>::post_root_squared_error_sum()
    const noexcept {
  return root_result_.has_value()
      ? root_result_->post_root_squared_error_sum()
      : std::nullopt;
}

template <int M>
StatusOr<TrainingModel<M>> RoundModelUpdater<M>::FinishRootUpdate() {
  if (phase_ != Phase::kBasic) {
    return Status::FailedPrecondition(
        "root update may finish exactly once after basic accumulation");
  }
  RootCodebookSolver<M> solver(*spd_solver_);
  if (!root_moments_.has_value()) {
    phase_ = Phase::kFailed;
    return Status::Internal("round root moments are unavailable");
  }
  auto root = solver.SolveValidatedPreviousRoot(
      *root_moments_, previous_->codebooks().root0());
  if (!root.ok()) {
    phase_ = Phase::kFailed;
    return root.status();
  }
  SolvedRoundModelView<M> solved;
  solved.shape = previous_->codebooks().shape();
  if constexpr (M == 0) {
    solved.root_small.resize(solved.shape.m - 1U);
    solved.edge.resize(solved.shape.m);
  }
  solved.rotation = previous_->rotation();
  solved.root0 = root.value().root0();
  for (std::uint32_t layer = 1U;
       layer < previous_->codebooks().root_codebooks().m(); ++layer) {
    solved.root_small[layer - 1U] = root.value().small_layer(layer);
  }
  for (std::uint32_t layer = 0U;
       layer < previous_->codebooks().root_codebooks().m(); ++layer) {
    solved.edge[layer] = previous_->codebooks().edge(layer);
  }
  auto topology_model = RoundModelAssembler<M>::Assemble(solved);
  if (!topology_model.ok()) {
    phase_ = Phase::kFailed;
    return topology_model.status();
  }
  root_result_.emplace(std::move(root).value());
  basic_rows_ = root_moments_->sample_count();
  root_moments_.reset();
  phase_ = Phase::kTopology;
  return std::move(topology_model).value();
}

template <int M>
Status RoundModelUpdater<M>::AccumulateEdgeCluster(
    const EdgeTrainingClusterView<M>& cluster) {
  if (phase_ != Phase::kTopology) {
    return Status::FailedPrecondition(
        "edge rows arrived outside the topology-update phase");
  }
  if (const Status status = EnsureEdgeMoments(); !status.ok()) {
    return Fail(status);
  }
  if (const Status status = ValidateEdgeCluster<M>(
          cluster, edge_moments_->dimension(), bad_cluster_mask_.size());
      !status.ok()) {
    return Fail(status);
  }
  const std::uint32_t cid = cluster.linked.topology->cid;
  const double weight = bad_cluster_mask_[cid] != 0U
      ? static_cast<double>(bad_cluster_edge_weight_)
      : 1.0;
  const std::uint32_t shard_count =
      edge_moment_shard_count_;
  const std::uint32_t shard = StaticCidShard(
      cid, static_cast<std::uint32_t>(bad_cluster_mask_.size()),
      shard_count);
  EdgeCodebookMoments<M>& moments = shard == 0U
      ? *edge_moments_
      : edge_moment_shards_[shard - 1U];
  const Status status = moments.AccumulateLinkedCluster(
      cluster.linked, weight);
  return status.ok() ? status : Fail(status);
}

template <int M>
Status RoundModelUpdater<M>::AccumulateFrozenEdgeCluster(
    const ProductCodebooks<M>& codebooks,
    const FrozenLinkedTrainingClusterView<M>& cluster,
    const CanonicalBfsForestOrder* order) {
  return AccumulateFrozenEdgeClusterImpl(
      codebooks, cluster, order, false);
}

template <int M>
Status RoundModelUpdater<M>::
AccumulateFrozenEdgeClusterAfterFullSinkValidation(
    const ProductCodebooks<M>& codebooks,
    const FrozenLinkedTrainingClusterView<M>& cluster,
    const CanonicalBfsForestOrder& order) {
  return AccumulateFrozenEdgeClusterImpl(
      codebooks, cluster, &order, true);
}

template <int M>
Status RoundModelUpdater<M>::AccumulateFrozenEdgeClusterImpl(
    const ProductCodebooks<M>& codebooks,
    const FrozenLinkedTrainingClusterView<M>& cluster,
    const CanonicalBfsForestOrder* order,
    bool full_sink_validated,
    bool replay_validated) {
  if (phase_ != Phase::kTopology) {
    return Status::FailedPrecondition(
        "edge rows arrived outside the topology-update phase");
  }
  if (const Status status = EnsureEdgeMoments(); !status.ok()) {
    return Fail(status);
  }
  if (const Status status = ValidateFrozenEdgeCluster<M>(
          cluster, edge_moments_->dimension(), bad_cluster_mask_.size());
      !status.ok()) {
    return Fail(status);
  }
  const std::uint32_t cid = cluster.topology->cid;
  const double weight = bad_cluster_mask_[cid] != 0U
      ? static_cast<double>(bad_cluster_edge_weight_)
      : 1.0;
  const std::uint32_t shard = StaticCidShard(
      cid, static_cast<std::uint32_t>(bad_cluster_mask_.size()),
      edge_moment_shard_count_);
  EdgeCodebookMoments<M>& moments = shard == 0U
      ? *edge_moments_
      : edge_moment_shards_[shard - 1U];
  const Status status = full_sink_validated
      ? moments.AccumulateFrozenLinkedClusterAfterValidation(
            codebooks, cluster, weight, *order)
      : replay_validated
          ? moments.AccumulateFrozenLinkedClusterAfterReplayValidation(
                codebooks, cluster, weight)
          : moments.AccumulateFrozenLinkedCluster(
                codebooks, cluster, weight, order);
  return status.ok() ? status : Fail(status);
}

template <int M>
Status RoundModelUpdater<M>::AccumulateEdgeClusters(
    ArrayView<const EdgeTrainingClusterView<M>> clusters) {
  if (phase_ != Phase::kTopology) {
    return Status::FailedPrecondition(
        "edge rows arrived outside the topology-update phase");
  }
  if (clusters.empty()) {
    return Status::Ok();
  }
  if (const Status status = EnsureEdgeMoments(); !status.ok()) {
    return Fail(status);
  }
  if (final_moment_parallel_ == nullptr) {
    for (const EdgeTrainingClusterView<M>& cluster : clusters) {
      if (const Status status = AccumulateEdgeCluster(cluster);
          !status.ok()) {
        return status;
      }
    }
    return Status::Ok();
  }
  for (const EdgeTrainingClusterView<M>& cluster : clusters) {
    if (const Status status = ValidateEdgeCluster<M>(
            cluster, edge_moments_->dimension(), bad_cluster_mask_.size());
        !status.ok()) {
      return Fail(status);
    }
  }
  const std::uint32_t shard_count =
      edge_moment_shard_count_;
  std::vector<Status> shard_status(shard_count, Status::Ok());
  struct Context final {
    RoundModelUpdater<M>* updater;
    ArrayView<const EdgeTrainingClusterView<M>> clusters;
    ArrayView<Status> status;
    std::uint32_t shard_count;
  } context = {this, clusters, shard_status, shard_count};
  final_moment_parallel_->Run(
      shard_count, &context,
      [](void* opaque, std::uint32_t, std::uint32_t shard) noexcept {
        auto& state = *static_cast<Context*>(opaque);
        EdgeCodebookMoments<M>& moments = shard == 0U
            ? *state.updater->edge_moments_
            : state.updater->edge_moment_shards_[shard - 1U];
        const std::uint32_t cid_count = static_cast<std::uint32_t>(
            state.updater->bad_cluster_mask_.size());
        for (const EdgeTrainingClusterView<M>& cluster : state.clusters) {
          const std::uint32_t cid = cluster.linked.topology->cid;
          if (StaticCidShard(cid, cid_count, state.shard_count) != shard) {
            continue;
          }
          const double weight = state.updater->bad_cluster_mask_[cid] != 0U
              ? static_cast<double>(
                    state.updater->bad_cluster_edge_weight_)
              : 1.0;
          const Status status = moments.AccumulateLinkedCluster(
              cluster.linked, weight);
          if (!status.ok()) {
            state.status[shard] = status;
            return;
          }
        }
      });
  for (const Status& status : shard_status) {
    if (!status.ok()) {
      return Fail(status);
    }
  }
  return Status::Ok();
}

template <int M>
Status RoundModelUpdater<M>::AccumulateFrozenEdgeClusters(
    const ProductCodebooks<M>& codebooks,
    ArrayView<const FrozenLinkedTrainingClusterView<M>> clusters) {
  return AccumulateFrozenEdgeClustersImpl(
      codebooks, clusters, false);
}

template <int M>
Status RoundModelUpdater<M>::
AccumulateFrozenEdgeClustersAfterReplayValidation(
    const ProductCodebooks<M>& codebooks,
    ArrayView<const FrozenLinkedTrainingClusterView<M>> clusters) {
  return AccumulateFrozenEdgeClustersImpl(
      codebooks, clusters, true);
}

template <int M>
Status RoundModelUpdater<M>::AccumulateFrozenEdgeClustersImpl(
    const ProductCodebooks<M>& codebooks,
    ArrayView<const FrozenLinkedTrainingClusterView<M>> clusters,
    bool replay_validated) {
  if (phase_ != Phase::kTopology) {
    return Status::FailedPrecondition(
        "edge rows arrived outside the topology-update phase");
  }
  if (clusters.empty()) {
    return Status::Ok();
  }
  if (const Status status = EnsureEdgeMoments(); !status.ok()) {
    return Fail(status);
  }
  if (final_moment_parallel_ == nullptr) {
    for (const FrozenLinkedTrainingClusterView<M>& cluster : clusters) {
      if (const Status status = replay_validated
              ? AccumulateFrozenEdgeClusterImpl(
                    codebooks, cluster, nullptr, false, true)
              : AccumulateFrozenEdgeCluster(codebooks, cluster);
          !status.ok()) {
        return status;
      }
    }
    return Status::Ok();
  }
  for (const FrozenLinkedTrainingClusterView<M>& cluster : clusters) {
    if (const Status status = ValidateFrozenEdgeCluster<M>(
            cluster, edge_moments_->dimension(), bad_cluster_mask_.size());
        !status.ok()) {
      return Fail(status);
    }
  }
  const std::uint32_t shard_count = edge_moment_shard_count_;
  std::vector<Status> shard_status(shard_count, Status::Ok());
  struct Context final {
    RoundModelUpdater<M>* updater;
    const ProductCodebooks<M>* codebooks;
    ArrayView<const FrozenLinkedTrainingClusterView<M>> clusters;
    ArrayView<Status> status;
    std::uint32_t shard_count;
    bool replay_validated;
  } context = {this, &codebooks, clusters, shard_status, shard_count,
               replay_validated};
  const auto accumulate_shard =
      [](void* opaque, std::uint32_t, std::uint32_t shard) noexcept {
        auto& state = *static_cast<Context*>(opaque);
        EdgeCodebookMoments<M>& moments = shard == 0U
            ? *state.updater->edge_moments_
            : state.updater->edge_moment_shards_[shard - 1U];
        const std::uint32_t cid_count = static_cast<std::uint32_t>(
            state.updater->bad_cluster_mask_.size());
        for (const FrozenLinkedTrainingClusterView<M>& cluster :
             state.clusters) {
          const std::uint32_t cid = cluster.topology->cid;
          if (StaticCidShard(cid, cid_count, state.shard_count) != shard) {
            continue;
          }
          const double weight = state.updater->bad_cluster_mask_[cid] != 0U
              ? static_cast<double>(
                    state.updater->bad_cluster_edge_weight_)
              : 1.0;
          const Status status = state.replay_validated
              ? moments.AccumulateFrozenLinkedClusterAfterReplayValidation(
                    *state.codebooks, cluster, weight)
              : moments.AccumulateFrozenLinkedCluster(
                    *state.codebooks, cluster, weight);
          if (!status.ok()) {
            state.status[shard] = status;
            return;
          }
        }
      };
  // A replay window contains at most 64 nearby CIDs, while a large IVF's
  // static reduction shard spans thousands. Usually only one shard has any
  // work: launching an OpenMP team then only makes the other workers wait.
  // Keep the exact CID-to-shard mapping and within-shard accumulation order.
  std::vector<std::uint32_t> active_shards;
  active_shards.reserve(std::min<std::size_t>(clusters.size(), shard_count));
  for (const auto& cluster : clusters) {
    const auto shard = StaticCidShard(
        cluster.topology->cid,
        static_cast<std::uint32_t>(bad_cluster_mask_.size()), shard_count);
    if (std::find(active_shards.begin(), active_shards.end(), shard) ==
        active_shards.end()) active_shards.push_back(shard);
  }
  if (active_shards.size() == 1U) {
    accumulate_shard(&context, 0U, active_shards.front());
  } else {
    struct ActiveWork {
      Context& context;
      const std::vector<std::uint32_t>& shards;
      ParallelForTask accumulate;
    } work{context, active_shards, accumulate_shard};
    final_moment_parallel_->Run(
        static_cast<std::uint32_t>(active_shards.size()), &work,
        [](void* opaque, std::uint32_t worker, std::uint32_t item) noexcept {
          auto& state = *static_cast<ActiveWork*>(opaque);
          state.accumulate(&state.context, worker, state.shards[item]);
        });
  }
  for (const Status& status : shard_status) {
    if (!status.ok()) {
      return Fail(status);
    }
  }
  return Status::Ok();
}

template <int M>
StatusOr<TrainingModel<M>> RoundModelUpdater<M>::FinishEdgeUpdate() {
  if (phase_ != Phase::kTopology || !root_result_.has_value()) {
    return Status::FailedPrecondition(
        "C_one update may finish exactly once after frozen forest rows");
  }
  if (const Status status = EnsureEdgeMoments(); !status.ok()) {
    return Fail(status);
  }
  for (const EdgeCodebookMoments<M>& shard : edge_moment_shards_) {
    if (const Status status = edge_moments_->MergeFrom(shard);
        !status.ok()) {
      return Fail(status);
    }
  }
  EdgeCodebookSolver<M> solver(*spd_solver_);
  auto edge = solver.Solve(*edge_moments_);
  if (!edge.ok()) {
    phase_ = Phase::kFailed;
    return edge.status();
  }
  SolvedRoundModelView<M> solved;
  solved.shape = previous_->codebooks().shape();
  if constexpr (M == 0) {
    solved.root_small.resize(solved.shape.m - 1U);
    solved.edge.resize(solved.shape.m);
  }
  solved.rotation = previous_->rotation();
  solved.root0 = root_result_->root0();
  for (std::uint32_t layer = 1U;
       layer < previous_->codebooks().root_codebooks().m(); ++layer) {
    solved.root_small[layer - 1U] = root_result_->small_layer(layer);
  }
  for (std::uint32_t layer = 0U;
       layer < previous_->codebooks().root_codebooks().m(); ++layer) {
    solved.edge[layer] = edge.value().layer(layer);
  }
  auto model = RoundModelAssembler<M>::Assemble(solved);
  if (!model.ok()) {
    phase_ = Phase::kFailed;
    return model.status();
  }
  edge_result_.emplace(std::move(edge).value());
  phase_ = Phase::kRotation;
  return std::move(model).value();
}

template <int M>
StatusOr<RoundModelUpdate<M>> RoundModelUpdater<M>::FinishRound(
    RotationCrossCovariance covariance, const F32GemmBackend& gemm) {
  if (phase_ != Phase::kRotation || !root_result_.has_value() ||
      !edge_result_.has_value()) {
    return Status::FailedPrecondition(
        "round update may finish exactly once after post-C_one covariance");
  }
  RotationUpdater rotation_updater(*svd_, gemm);
  auto rotation = rotation_updater.Solve(covariance);
  if (!rotation.ok()) {
    phase_ = Phase::kFailed;
    return rotation.status();
  }
  auto model = RoundModelAssembler<M>::Assemble(
      previous_->codebooks().shape(), *root_result_, *edge_result_,
      rotation.value());
  if (!model.ok()) {
    phase_ = Phase::kFailed;
    return model.status();
  }
  RoundModelUpdateStatistics statistics = {
      .basic_rows = basic_rows_,
      .linked_rows = edge_moments_->sample_count(),
      .rotation_rows = covariance.sample_count(),
      .root_diagonal_bump = root_result_->diagonal_bump(),
      .edge_diagonal_bump = edge_result_->diagonal_bump(),
  };
  phase_ = Phase::kComplete;
  return RoundModelUpdate<M>{
      .model = std::move(model).value(),
      .statistics = statistics,
  };
}

template <int M>
StatusOr<RoundModelUpdate<M>>
RoundModelUpdater<M>::FinishRoundKeepingRotation() {
  if (phase_ != Phase::kRotation || !root_result_.has_value() ||
      !edge_result_.has_value()) {
    return Status::FailedPrecondition(
        "round update may finish exactly once after the C_one update");
  }
  SolvedRoundModelView<M> solved;
  solved.shape = previous_->codebooks().shape();
  if constexpr (M == 0) {
    solved.root_small.resize(solved.shape.m - 1U);
    solved.edge.resize(solved.shape.m);
  }
  solved.rotation = previous_->rotation();
  solved.root0 = root_result_->root0();
  for (std::uint32_t layer = 1U;
       layer < previous_->codebooks().root_codebooks().m(); ++layer) {
    solved.root_small[layer - 1U] = root_result_->small_layer(layer);
  }
  for (std::uint32_t layer = 0U;
       layer < previous_->codebooks().root_codebooks().m(); ++layer) {
    solved.edge[layer] = edge_result_->layer(layer);
  }
  auto model = RoundModelAssembler<M>::Assemble(solved);
  if (!model.ok()) {
    phase_ = Phase::kFailed;
    return model.status();
  }
  RoundModelUpdateStatistics statistics = {
      .basic_rows = basic_rows_,
      .linked_rows = edge_moments_->sample_count(),
      .rotation_rows = 0U,
      .root_diagonal_bump = root_result_->diagonal_bump(),
      .edge_diagonal_bump = edge_result_->diagonal_bump(),
  };
  phase_ = Phase::kComplete;
  return RoundModelUpdate<M>{
      .model = std::move(model).value(),
      .statistics = statistics,
  };
}

template class RoundModelUpdater<0>;
template class RoundModelUpdater<5>;
template class RoundModelUpdater<12>;
template class RoundModelUpdater<6>;
template class RoundModelUpdater<10>;

}  // namespace stlqpp
