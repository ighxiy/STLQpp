#include "stlqpp/training/linkage/topology/cluster_topology_session.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "stlqpp/training/linkage/initialization/root_only_initialization_layer_executor.h"
#include "stlqpp/training/linkage/topology/causal_topology_window_executor.h"

namespace stlqpp {

template <int M>
ClusterTopologySession<M>::ClusterTopologySession(
    LinkageEvaluationService& scheduler, LinkageClusterHandle cluster,
    std::unique_ptr<ClusterTopologyState<M>> state,
    const CausalCpuLinkageEvaluator<M>* cpu_evaluator,
    TopologyRandomStream* random_stream,
    RootOnlyInitializationTailEvaluator<M>* initialization_tail_evaluator,
    std::uint32_t cluster_seed, CausalTopologyBatching batching) noexcept
    : scheduler_(&scheduler),
      cluster_(cluster),
      state_(std::move(state)),
      cpu_evaluator_(cpu_evaluator),
      random_stream_(random_stream),
      initialization_tail_evaluator_(initialization_tail_evaluator),
      cluster_seed_(cluster_seed),
      batching_(batching),
      registered_(true) {}

template <int M>
ClusterTopologySession<M>::~ClusterTopologySession() {
  if (registered_) {
    (void)scheduler_->UnregisterCluster(cluster_);
  }
}

template <int M>
StatusOr<std::unique_ptr<ClusterTopologySession<M>>>
ClusterTopologySession<M>::Create(
    const ProductCodebooks<M>& codebooks,
    ClusterTopologyInitialStateView<M> initial,
    LinkageEvaluationService& scheduler, CausalTopologyBatching batching,
    const CausalCpuLinkageEvaluator<M>* cpu_evaluator,
    TopologyRandomStream* random_stream,
    ClusterTopologyInitialStateOwnership<M> ownership,
    std::vector<float>* owned_virtual_reconstructions, bool borrow_exact_real) {
  auto state = ClusterTopologyState<M>::Create(codebooks, initial, ownership,
                                               owned_virtual_reconstructions,
                                               borrow_exact_real);
  if (!state.ok()) {
    return state.status();
  }
  auto registered = scheduler.RegisterCluster({
      .affinity_key = state.value()->cid(),
      .first = state.value()->virtual_reconstructions(),
      .second = state.value()->real_reconstructions(),
  });
  if (!registered.ok()) {
    return registered.status();
  }
  return std::unique_ptr<ClusterTopologySession<M>>(
      new ClusterTopologySession<M>(scheduler, registered.value(),
                                    std::move(state).value(), cpu_evaluator,
                                    random_stream, nullptr, 0U, batching));
}

template <int M>
StatusOr<std::unique_ptr<ClusterTopologySession<M>>>
ClusterTopologySession<M>::CreateRootOnlyInitialization(
    const RootCodebooks<M>& codebooks,
    ClusterTopologyInitialStateView<M> initial,
    LinkageEvaluationService& scheduler,
    RootOnlyInitializationTailEvaluator<M>& tail_evaluator,
    std::uint32_t cluster_seed, CausalTopologyBatching batching) {
  auto state =
      ClusterTopologyState<M>::CreateRootOnlyInitialization(codebooks, initial);
  if (!state.ok()) {
    return state.status();
  }
  auto registered = scheduler.RegisterCluster({
      .affinity_key = state.value()->cid(),
      .first = state.value()->virtual_reconstructions(),
      .second = state.value()->real_reconstructions(),
  });
  if (!registered.ok()) {
    return registered.status();
  }
  return std::unique_ptr<ClusterTopologySession<M>>(
      new ClusterTopologySession<M>(scheduler, registered.value(),
                                    std::move(state).value(), nullptr, nullptr,
                                    &tail_evaluator, cluster_seed, batching));
}

template <int M>
Status ClusterTopologySession<M>::PublishInitialRoot(
    std::uint32_t real_local_id) {
  return state_->PublishInitialRoot(real_local_id);
}

template <int M>
Status ClusterTopologySession<M>::ProcessNode(
    std::uint32_t real_local_id,
    ArrayView<const LinkageParentReference> ordered_candidates,
    ArrayView<const std::uint8_t> icm_layer_order,
    std::uint32_t maximum_depth) {
  auto task =
      state_->PrepareNode(cluster_, generation_, real_local_id,
                          ordered_candidates, icm_layer_order, maximum_depth);
  if (!task.ok()) {
    return task.status();
  }
  if (!task.value().has_value()) {
    return Status::Ok();
  }
  auto submitted = scheduler_->Submit(std::move(*task.value()));
  if (!submitted.ok()) {
    return submitted.status();
  }
  auto evaluated = submitted.value().get();
  if (!evaluated.ok()) {
    return evaluated.status();
  }
  auto committed = state_->CommitPending(std::move(evaluated).value());
  if (!committed.ok()) {
    return committed.status();
  }
  if (!committed.value().has_value()) {
    return Status::Ok();
  }

  ClusterTopologyReconstructionUpdate& update = *committed.value();
  const std::uint32_t column = update.registered_column;
  auto ticket = scheduler_->SubmitUpdate(
      cluster_, {
                    .columns = &column,
                    .reconstructions = {update.reconstruction.data(),
                                        static_cast<std::uint32_t>(
                                            update.reconstruction.size()),
                                        1U},
                });
  if (!ticket.ok()) {
    return ticket.status();
  }
  generation_ = ticket.value().generation;
  return Status::Ok();
}

template <int M>
void ClusterTopologySession<M>::AccumulateWindowStatistics(
    const CausalTopologyWindowStatistics& value) noexcept {
  window_statistics_.rows += value.rows;
  window_statistics_.candidate_pairs += value.candidate_pairs;
  window_statistics_.evaluation_tasks += value.evaluation_tasks;
  window_statistics_.evaluation_nodes += value.evaluation_nodes;
  window_statistics_.evaluation_pairs += value.evaluation_pairs;
  window_statistics_.cpu_evaluation_tasks += value.cpu_evaluation_tasks;
  window_statistics_.cpu_evaluation_nodes += value.cpu_evaluation_nodes;
  window_statistics_.cpu_evaluation_pairs += value.cpu_evaluation_pairs;
  for (std::size_t kind = 0U; kind < kCausalEvaluationKindCount; ++kind) {
    window_statistics_.evaluation_tasks_by_kind[kind] +=
        value.evaluation_tasks_by_kind[kind];
    window_statistics_.cpu_evaluation_tasks_by_kind[kind] +=
        value.cpu_evaluation_tasks_by_kind[kind];
  }
  window_statistics_.update_requests += value.update_requests;
  window_statistics_.update_columns += value.update_columns;
  window_statistics_.maximum_task_nodes =
      std::max(window_statistics_.maximum_task_nodes, value.maximum_task_nodes);
  window_statistics_.maximum_task_pairs =
      std::max(window_statistics_.maximum_task_pairs, value.maximum_task_pairs);
}

template <int M>
Status ClusterTopologySession<M>::ProcessRootOnlyInitializationLayer(
    ArrayView<const CausalTopologyRow<M>> commit_rows,
    ArrayView<const std::uint32_t> physical_layer_order,
    std::uint32_t maximum_depth) {
  return ProcessRootOnlyInitializationLayerImpl(
      commit_rows, physical_layer_order, maximum_depth, false);
}

template <int M>
Status ClusterTopologySession<M>::ProcessValidatedRootOnlyInitializationLayer(
    ArrayView<const CausalTopologyRow<M>> commit_rows,
    ArrayView<const std::uint32_t> physical_layer_order,
    std::uint32_t maximum_depth) {
  return ProcessRootOnlyInitializationLayerImpl(
      commit_rows, physical_layer_order, maximum_depth, true);
}

template <int M>
Status ClusterTopologySession<M>::ProcessRootOnlyInitializationLayerImpl(
    ArrayView<const CausalTopologyRow<M>> commit_rows,
    ArrayView<const std::uint32_t> physical_layer_order,
    std::uint32_t maximum_depth, bool validated_layer) {
  RootOnlyInitializationLayerExecutor<M> executor(
      *state_, *scheduler_, cluster_, *initialization_tail_evaluator_,
      cluster_seed_, batching_);
  auto statistics =
      validated_layer
          ? executor.RunValidatedLayer(commit_rows, physical_layer_order,
                                       maximum_depth, &generation_)
          : executor.Run(commit_rows, physical_layer_order, maximum_depth,
                         &generation_);
  if (!statistics.ok()) {
    return statistics.status();
  }
  AccumulateWindowStatistics(statistics.value());
  return Status::Ok();
}

template <int M>
Status ClusterTopologySession<M>::ProcessRadialLayerRows(
    ArrayView<const CausalTopologyRow<M>> rows, std::uint32_t maximum_depth) {
  CausalTopologyWindowExecutor<M> executor(*state_, *scheduler_, cluster_,
                                           cpu_evaluator_, random_stream_,
                                           batching_);
  auto statistics = executor.RunRadialLayer(rows, maximum_depth, &generation_);
  if (!statistics.ok()) {
    return statistics.status();
  }
  AccumulateWindowStatistics(statistics.value());
  return Status::Ok();
}

template <int M>
Status ClusterTopologySession<M>::ProcessGraphRows(
    ArrayView<const CausalTopologyRow<M>> rows, std::uint32_t maximum_depth) {
  CausalTopologyWindowExecutor<M> executor(*state_, *scheduler_, cluster_,
                                           cpu_evaluator_, random_stream_,
                                           batching_);
  auto statistics = executor.RunGraphOrder(rows, maximum_depth, &generation_);
  if (!statistics.ok()) {
    return statistics.status();
  }
  AccumulateWindowStatistics(statistics.value());
  return Status::Ok();
}

template <int M>
StatusOr<ClusterTopologyResult<M>> ClusterTopologySession<M>::Finish() {
  auto result = state_->Finish();
  if (!result.ok()) {
    return result.status();
  }
  result.value().evaluation_tasks = window_statistics_.evaluation_tasks;
  result.value().cpu_evaluation_tasks = window_statistics_.cpu_evaluation_tasks;
  result.value().evaluation_tasks_by_kind =
      window_statistics_.evaluation_tasks_by_kind;
  result.value().cpu_evaluation_tasks_by_kind =
      window_statistics_.cpu_evaluation_tasks_by_kind;
  const Status unregistered = scheduler_->UnregisterCluster(cluster_);
  if (!unregistered.ok()) {
    return unregistered;
  }
  registered_ = false;
  return std::move(result).value();
}

template class ClusterTopologySession<0>;
template class ClusterTopologySession<5>;
template class ClusterTopologySession<12>;
template class ClusterTopologySession<6>;
template class ClusterTopologySession<10>;

}  // namespace stlqpp
