#include "stlqpp/training/update/initialization_model_updater.h"

#include "stlqpp/training/update/edge_codebook_solver.h"

#include <cstdint>
#include <utility>

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

}  // namespace

template <int M>
StatusOr<InitializationModelUpdater<M>>
InitializationModelUpdater<M>::Create(
    const RootTrainingModel<M>& initial,
    const SpdSolverF32& solver,
    const ParallelForExecutor& root_moment_parallel,
    std::uint32_t edge_moment_shard_count) {
  return Create(
      initial, solver, root_moment_parallel, root_moment_parallel,
      edge_moment_shard_count);
}

template <int M>
StatusOr<InitializationModelUpdater<M>>
InitializationModelUpdater<M>::Create(
    const RootTrainingModel<M>& initial,
    const SpdSolverF32& solver,
    const ParallelForExecutor& root_accumulation_parallel,
    const ParallelForExecutor& root_conversion_parallel,
    std::uint32_t edge_moment_shard_count) {
  const ModelShape& shape = initial.codebooks().shape();
  if (edge_moment_shard_count == 0U ||
      edge_moment_shard_count > root_conversion_parallel.worker_count()) {
    return Status::InvalidArgument(
        "initialization edge moment shard count is invalid");
  }
  auto root = RootCodebookMoments<M>::Create(
      shape.dimension, shape.nlist, root_accumulation_parallel,
      root_conversion_parallel, shape.m);
  if (!root.ok()) {
    return root.status();
  }
  auto edge = EdgeCodebookMoments<M>::Create(shape.dimension, shape.m);
  if (!edge.ok()) {
    return edge.status();
  }
  std::vector<EdgeCodebookMoments<M>> edge_moment_shards;
  edge_moment_shards.reserve(edge_moment_shard_count - 1U);
  for (std::uint32_t shard = 1U;
       shard < edge_moment_shard_count; ++shard) {
    auto moments = EdgeCodebookMoments<M>::Create(shape.dimension, shape.m);
    if (!moments.ok()) {
      return moments.status();
    }
    edge_moment_shards.push_back(std::move(moments).value());
  }
  return InitializationModelUpdater<M>(
      initial, solver, root_conversion_parallel,
      std::move(root).value(), std::move(edge).value(),
      std::move(edge_moment_shards));
}

template <int M>
Status InitializationModelUpdater<M>::Fail(Status status) noexcept {
  phase_ = Phase::kFailed;
  return status;
}

template <int M>
Status InitializationModelUpdater<M>::AccumulateBasic(
    ColumnMajorMatrixF32View exact_model_space,
    const BasicEncoding<M>& encoding) {
  if (phase_ != Phase::kBasic) {
    return Status::FailedPrecondition(
        "initialization basic rows arrived outside the root phase");
  }
  if (!root_moments_.has_value()) {
    return Fail(Status::FailedPrecondition(
        "initialization root moments were already released"));
  }
  const Status status = root_moments_->Accumulate(
      exact_model_space, encoding);
  return status.ok() ? status : Fail(status);
}

template <int M>
Status InitializationModelUpdater<M>::EnablePostRootMetric() {
  if (phase_ != Phase::kBasic || !root_moments_.has_value()) {
    return Status::FailedPrecondition(
        "post-root metric may only be enabled before basic accumulation");
  }
  return root_moments_->EnablePostRootMetric();
}

template <int M>
std::optional<double>
InitializationModelUpdater<M>::post_root_squared_error_sum() const noexcept {
  return root_result_.has_value()
      ? root_result_->post_root_squared_error_sum()
      : std::nullopt;
}

template <int M>
StatusOr<RootTrainingModel<M>>
InitializationModelUpdater<M>::FinishRootUpdate() {
  if (phase_ != Phase::kBasic) {
    return Status::FailedPrecondition(
        "initialization root update may finish exactly once");
  }
  RootCodebookSolver<M> root_solver(*solver_);
  auto solved = root_solver.SolveValidatedPreviousRoot(
      *root_moments_, initial_->codebooks().root0());
  if (!solved.ok()) {
    phase_ = Phase::kFailed;
    return solved.status();
  }
  basic_rows_ = root_moments_->sample_count();
  root_moments_.reset();

  RootCodebookView<M> view;
  view.shape = initial_->codebooks().shape();
  if constexpr (M == 0) view.root_small.resize(view.shape.m - 1U);
  view.root0 = solved.value().root0();
  for (std::uint32_t layer = 1U;
       layer < initial_->codebooks().m(); ++layer) {
    view.root_small[layer - 1U] = solved.value().small_layer(layer);
  }
  auto model = RootTrainingModel<M>::Create(initial_->rotation(), view);
  if (!model.ok()) {
    phase_ = Phase::kFailed;
    return model.status();
  }
  root_result_.emplace(std::move(solved).value());
  phase_ = Phase::kTopology;
  return std::move(model).value();
}

template <int M>
Status InitializationModelUpdater<M>::AccumulateRootOnlyCluster(
    const LinkedTrainingClusterView<M>& cluster,
    const RootToEdgeCodeMap& root_to_edge,
    double sample_weight) {
  return AccumulateRootOnlyClusterImpl<false>(
      cluster, root_to_edge, sample_weight);
}

template <int M>
Status InitializationModelUpdater<M>::AccumulateFrozenEdgeCluster(
    const ProductCodebooks<M>& codebooks,
    const FrozenLinkedTrainingClusterView<M>& cluster) {
  if (phase_ != Phase::kTopology || cluster.topology == nullptr) {
    return Status::FailedPrecondition(
        "fast initialization forest arrived outside the topology phase");
  }
  const std::uint32_t shard_count =
      static_cast<std::uint32_t>(edge_moment_shards_.size() + 1U);
  const std::uint32_t shard = StaticCidShard(
      cluster.topology->cid, initial_->codebooks().shape().nlist,
      shard_count);
  EdgeCodebookMoments<M>& moments = shard == 0U
      ? edge_moments_ : edge_moment_shards_[shard - 1U];
  const Status status = moments.AccumulateFrozenLinkedCluster(
      codebooks, cluster, 1.0);
  return status.ok() ? status : Fail(status);
}

template <int M>
Status InitializationModelUpdater<M>::
AccumulateRootOnlyClusterAfterReplayValidation(
    const LinkedTrainingClusterView<M>& cluster,
    const RootToEdgeCodeMap& root_to_edge,
    double sample_weight) {
  return AccumulateRootOnlyClusterImpl<true>(
      cluster, root_to_edge, sample_weight);
}

template <int M>
template <bool Prevalidated>
Status InitializationModelUpdater<M>::AccumulateRootOnlyClusterImpl(
    const LinkedTrainingClusterView<M>& cluster,
    const RootToEdgeCodeMap& root_to_edge,
    double sample_weight) {
  if (phase_ != Phase::kTopology) {
    return Status::FailedPrecondition(
        "initialization forest rows arrived outside the topology phase");
  }
  const std::uint32_t shard_count =
      static_cast<std::uint32_t>(edge_moment_shards_.size() + 1U);
  const std::uint32_t shard = StaticCidShard(
      cluster.topology->cid, initial_->codebooks().shape().nlist,
      shard_count);
  EdgeCodebookMoments<M>& moments = shard == 0U
      ? edge_moments_
      : edge_moment_shards_[shard - 1U];
  const Status status = Prevalidated
      ? moments.AccumulateRootOnlyInitializationClusterAfterValidation(
            cluster, root_to_edge, sample_weight)
      : moments.AccumulateRootOnlyInitializationCluster(
            cluster, root_to_edge, sample_weight);
  // Parallel replay owns one accumulator per shard. Its caller joins all
  // workers before changing the shared phase on failure.
  if constexpr (Prevalidated) return status;
  return status.ok() ? status : Fail(status);
}

template <int M>
StatusOr<InitializationModelUpdate<M>>
InitializationModelUpdater<M>::Finish() {
  if (phase_ != Phase::kTopology || !root_result_.has_value()) {
    return Status::FailedPrecondition(
        "initialization may finish exactly once after its forest");
  }
  for (const EdgeCodebookMoments<M>& shard : edge_moment_shards_) {
    if (const Status status = edge_moments_.MergeFrom(shard);
        !status.ok()) {
      return Fail(status);
    }
  }
  EdgeCodebookSolver<M> edge_solver(*solver_);
  auto edge = edge_solver.Solve(edge_moments_);
  if (!edge.ok()) {
    phase_ = Phase::kFailed;
    return edge.status();
  }

  ProductCodebookView<M> codebooks;
  codebooks.shape = initial_->codebooks().shape();
  if constexpr (M == 0) {
    codebooks.root_small.resize(codebooks.shape.m - 1U);
    codebooks.edge.resize(codebooks.shape.m);
  }
  codebooks.root0 = root_result_->root0();
  for (std::uint32_t layer = 1U;
       layer < initial_->codebooks().m(); ++layer) {
    codebooks.root_small[layer - 1U] = root_result_->small_layer(layer);
  }
  for (std::uint32_t layer = 0U;
       layer < initial_->codebooks().m(); ++layer) {
    codebooks.edge[layer] = edge.value().layer(layer);
  }
  auto model = TrainingModel<M>::Create(initial_->rotation(), codebooks);
  if (!model.ok()) {
    phase_ = Phase::kFailed;
    return model.status();
  }

  InitializationModelUpdateStatistics statistics = {
      .basic_rows = basic_rows_,
      .linked_rows = edge_moments_.sample_count(),
      .root_diagonal_bump = root_result_->diagonal_bump(),
      .edge_diagonal_bump = edge.value().diagonal_bump(),
  };
  phase_ = Phase::kComplete;
  return InitializationModelUpdate<M>{
      .model = std::move(model).value(),
      .statistics = statistics,
  };
}

template class InitializationModelUpdater<0>;
template class InitializationModelUpdater<5>;
template class InitializationModelUpdater<12>;
template class InitializationModelUpdater<6>;
template class InitializationModelUpdater<10>;

}  // namespace stlqpp
