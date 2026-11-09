#include "stlqpp/training/linkage/topology/causal_topology_window_executor.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace stlqpp {

template <int M>
struct CausalTopologyWindowExecutor<M>::DependentCandidate final {
  LinkageParentReference parent;
  std::uint32_t child_position = 0U;
  std::uint32_t original_rank = 0U;
};

template <int M>
struct CausalTopologyWindowExecutor<M>::Winner final {
  bool valid = false;
  std::uint32_t original_rank = 0U;
  ClusterTopologyEvaluationWinner<M> result;
};

template <int M>
Status CausalTopologyWindowExecutor<M>::FlushUpdates(
    std::uint64_t* generation, CausalTopologyWindowStatistics* statistics) {
  if (update_columns_.empty()) {
    return Status::Ok();
  }
  auto submitted = scheduler_->SubmitUpdate(
      cluster_, {
                    .columns = update_columns_.data(),
                    .reconstructions =
                        {
                            update_reconstructions_.data(),
                            state_->dimension(),
                            static_cast<std::uint32_t>(update_columns_.size()),
                        },
                });
  if (!submitted.ok()) {
    return submitted.status();
  }
  *generation = submitted.value().generation;
  ++statistics->update_requests;
  statistics->update_columns += update_columns_.size();
  update_columns_.clear();
  update_reconstructions_.clear();
  return Status::Ok();
}

template <int M>
void CausalTopologyWindowExecutor<M>::AppendUpdate(
    ClusterTopologyReconstructionUpdate update) {
  update_columns_.push_back(update.registered_column);
  update_reconstructions_.insert(update_reconstructions_.end(),
                                 update.reconstruction.begin(),
                                 update.reconstruction.end());
}

template <int M>
StatusOr<
    std::optional<typename CausalTopologyWindowExecutor<M>::PendingEvaluation>>
CausalTopologyWindowExecutor<M>::SubmitGpuEvaluation(
    const std::vector<const CausalTopologyRow<M>*>& rows,
    const std::vector<std::vector<ClusterTopologyRankedCandidate>>& candidates,
    std::uint32_t maximum_depth, std::uint64_t* generation,
    CausalTopologyWindowStatistics* statistics, CausalEvaluationKind kind) {
  std::vector<ClusterTopologyEvaluationRowView> views;
  views.reserve(rows.size());
  for (std::size_t row = 0U; row < rows.size(); ++row) {
    if (!candidates[row].empty()) {
      views.push_back({
          .real_local_id = rows[row]->real_local_id,
          .source_position = static_cast<std::uint32_t>(row),
          .ranked_candidates = candidates[row],
          .icm_layer_order = {rows[row]->icm_layer_order.data(), state_->m()},
      });
    }
  }
  if (views.empty()) {
    return std::optional<PendingEvaluation>{};
  }
  if (const Status status = FlushUpdates(generation, statistics);
      !status.ok()) {
    return status;
  }
  auto prepared = state_->PrepareVisibleEvaluation(cluster_, *generation, views,
                                                   maximum_depth);
  if (!prepared.ok()) {
    return prepared.status();
  }
  if (!prepared.value().has_value()) {
    return std::optional<PendingEvaluation>{};
  }
  PreparedClusterTopologyEvaluation task = std::move(*prepared.value());
  PendingEvaluation pending;
  pending.real_local_ids = std::move(task.real_local_ids);
  pending.source_row_positions = std::move(task.source_row_positions);
  pending.candidate_columns = task.task.candidate_columns;
  pending.row_offsets = task.task.row_offsets;
  pending.original_candidate_ranks = std::move(task.original_candidate_ranks);
  pending.node_count = task.task.node_count();
  pending.pair_count = task.task.pair_count();
  pending.kind = kind;
  auto submitted = scheduler_->Submit(std::move(task.task));
  if (!submitted.ok()) {
    return submitted.status();
  }
  pending.result = std::move(submitted).value();
  return std::optional<PendingEvaluation>(std::move(pending));
}

template <int M>
Status CausalTopologyWindowExecutor<M>::FinishGpuEvaluation(
    PendingEvaluation evaluation,
    const std::vector<const CausalTopologyRow<M>*>& rows,
    std::vector<Winner>* winners, CausalTopologyWindowStatistics* statistics) {
  static_cast<void>(rows);  // Only needed by the debug mapping assertion.
  auto evaluated = evaluation.result.get();
  if (!evaluated.ok()) {
    return evaluated.status();
  }
  LinkageEvaluationResult result = std::move(evaluated).value();
  if (result.parent_columns.size() != evaluation.real_local_ids.size() ||
      result.supports.size() != evaluation.real_local_ids.size() ||
      result.supports.m() != state_->m() ||
      result.squared_errors.size() != evaluation.real_local_ids.size() ||
      result.coefficients.size() !=
          evaluation.real_local_ids.size() * state_->m()) {
    return Status::Internal(
        "causal-window evaluator returned an invalid result extent");
  }
  for (std::size_t index = 0U; index < evaluation.real_local_ids.size();
       ++index) {
    const std::uint32_t position = evaluation.source_row_positions[index];
    assert(position < rows.size());
    assert(rows[position]->real_local_id == evaluation.real_local_ids[index]);
    const std::uint32_t parent_column = result.parent_columns[index];
    std::optional<std::uint32_t> original_rank;
    for (std::uint32_t candidate = evaluation.row_offsets[index];
         candidate < evaluation.row_offsets[index + 1U]; ++candidate) {
      if (evaluation.candidate_columns[candidate] == parent_column) {
        original_rank = evaluation.original_candidate_ranks[candidate];
        break;
      }
    }
    if (!original_rank.has_value()) {
      return Status::Internal(
          "causal-window evaluator selected an unregistered candidate");
    }
    Winner& current = (*winners)[position];
    const bool graph_order =
        evaluation.kind == CausalEvaluationKind::kGraphFrozen ||
        evaluation.kind == CausalEvaluationKind::kGraphDynamic;
    if (!current.valid ||
        result.squared_errors[index] < current.result.squared_error ||
        (graph_order &&
         result.squared_errors[index] == current.result.squared_error &&
         *original_rank < current.original_rank)) {
      current.valid = true;
      current.original_rank = *original_rank;
      current.result.parent_column = parent_column;
      current.result.support = result.supports.template tuple<M>(index);
      std::copy_n(result.coefficients.begin() + index * state_->m(),
                  state_->m(), current.result.coefficients.begin());
      current.result.squared_error = result.squared_errors[index];
    }
  }
  ++statistics->evaluation_tasks;
  const std::size_t kind_index = static_cast<std::size_t>(evaluation.kind);
  ++statistics->evaluation_tasks_by_kind[kind_index];
  statistics->evaluation_nodes += evaluation.node_count;
  statistics->evaluation_pairs += evaluation.pair_count;
  statistics->maximum_task_nodes = std::max<std::uint64_t>(
      statistics->maximum_task_nodes, evaluation.node_count);
  statistics->maximum_task_pairs = std::max<std::uint64_t>(
      statistics->maximum_task_pairs, evaluation.pair_count);
  return Status::Ok();
}

template <int M>
Status CausalTopologyWindowExecutor<M>::Evaluate(
    const std::vector<const CausalTopologyRow<M>*>& rows,
    const std::vector<std::vector<ClusterTopologyRankedCandidate>>& candidates,
    std::vector<Winner>* winners, std::uint32_t maximum_depth,
    std::uint64_t* generation, CausalTopologyWindowStatistics* statistics,
    bool permit_cpu_tiny_tail, CausalEvaluationKind kind) {
  std::vector<ClusterTopologyEvaluationRowView> views;
  views.reserve(rows.size());
  for (std::size_t row = 0U; row < rows.size(); ++row) {
    if (candidates[row].empty()) {
      continue;
    }
    views.push_back({
        .real_local_id = rows[row]->real_local_id,
        .source_position = static_cast<std::uint32_t>(row),
        .ranked_candidates = candidates[row],
        .icm_layer_order = {rows[row]->icm_layer_order.data(), state_->m()},
    });
  }
  if (views.empty()) {
    return Status::Ok();
  }
  std::uint32_t visible_pairs = 0U;
  for (const auto& view : views) {
    visible_pairs += static_cast<std::uint32_t>(view.ranked_candidates.size());
  }
  const bool tiny_tail_cpu =
      permit_cpu_tiny_tail && batching_.tiny_forced_cpu_enabled &&
      visible_pairs <= batching_.tiny_forced_pair_limit &&
      (batching_.tiny_forced_node_limit == 0U ||
       views.size() <= batching_.tiny_forced_node_limit);
  auto prepared = state_->PrepareVisibleEvaluation(cluster_, *generation, views,
                                                   maximum_depth);
  if (!prepared.ok()) {
    return prepared.status();
  }
  if (!prepared.value().has_value()) {
    return Status::Ok();
  }
  PreparedClusterTopologyEvaluation task = std::move(*prepared.value());
  const std::uint32_t task_nodes = task.task.node_count();
  const std::uint32_t task_pairs = task.task.pair_count();
  const bool resource_cpu =
      cpu_evaluator_ != nullptr && task_nodes == 1U &&
      SingleNodeUsesCpu(batching_, task_pairs,
                        UsesSameLayerSingleNodeThreshold(kind));
  const bool tiny_batch_cpu =
      !resource_cpu && tiny_tail_cpu && cpu_evaluator_ != nullptr &&
      random_stream_ != nullptr &&
      (state_->virtual_count() == 0U ||
       std::all_of(task.task.candidate_columns.begin(),
                   task.task.candidate_columns.end(),
                   [this](std::uint32_t column) {
                     return column >= state_->virtual_count();
                   }));
  const bool use_cpu = resource_cpu || tiny_batch_cpu;
  if (!use_cpu) {
    if (const Status status = FlushUpdates(generation, statistics);
        !status.ok()) {
      return status;
    }
    task.task.required_generation = *generation;
  }
  StatusOr<LinkageEvaluationResult> evaluated =
      use_cpu ? (resource_cpu
                     ? cpu_evaluator_->Evaluate(*state_, task.task, &cpu_scratch_)
                     : cpu_evaluator_->EvaluateTinyTail(*state_, task.task,
                                                       *random_stream_, &cpu_scratch_))
              : [&]() -> StatusOr<LinkageEvaluationResult> {
    auto submitted = scheduler_->Submit(std::move(task.task));
    if (!submitted.ok()) {
      return submitted.status();
    }
    return submitted.value().get();
  }();
  if (!evaluated.ok()) {
    return evaluated.status();
  }
  LinkageEvaluationResult result = std::move(evaluated).value();
  if (result.parent_columns.size() != task.real_local_ids.size() ||
      result.supports.size() != task.real_local_ids.size() ||
      result.supports.m() != state_->m() ||
      result.squared_errors.size() != task.real_local_ids.size() ||
      result.coefficients.size() != task.real_local_ids.size() * state_->m()) {
    return Status::Internal(
        "causal-window evaluator returned an invalid result extent");
  }
  for (std::size_t index = 0U; index < task.real_local_ids.size(); ++index) {
    const std::uint32_t position = task.source_row_positions[index];
    assert(position < rows.size());
    assert(rows[position]->real_local_id == task.real_local_ids[index]);
    const std::uint32_t parent_column = result.parent_columns[index];
    std::optional<std::uint32_t> original_rank;
    for (const ClusterTopologyRankedCandidate& candidate :
         candidates[position]) {
      const std::uint32_t column =
          candidate.parent.kind == LinkageParentKind::kVirtual
              ? candidate.parent.local_id
              : state_->virtual_count() + candidate.parent.local_id;
      if (column == parent_column) {
        original_rank = candidate.original_rank;
        break;
      }
    }
    if (!original_rank.has_value()) {
      return Status::Internal(
          "causal-window evaluator selected an unregistered candidate");
    }
    Winner& current = (*winners)[position];
    // DEV merges ordinary GOOD evaluator outputs in source order and keeps
    // the first winner on an exact FP32 tie. original_rank remains only the
    // mapping from an evaluator result to the row's ordered candidate. BAD
    // graph order is different: one original candidate row is split by
    // causal visibility, so an exact tie must recover its original order.
    const bool graph_order = kind == CausalEvaluationKind::kGraphFrozen ||
                             kind == CausalEvaluationKind::kGraphDynamic;
    if (!current.valid ||
        result.squared_errors[index] < current.result.squared_error ||
        (graph_order &&
         result.squared_errors[index] == current.result.squared_error &&
         *original_rank < current.original_rank)) {
      current.valid = true;
      current.original_rank = *original_rank;
      current.result.parent_column = parent_column;
      current.result.support = result.supports.template tuple<M>(index);
      std::copy_n(result.coefficients.begin() + index * state_->m(),
                  state_->m(), current.result.coefficients.begin());
      current.result.squared_error = result.squared_errors[index];
    }
  }
  ++statistics->evaluation_tasks;
  const std::size_t kind_index = static_cast<std::size_t>(kind);
  ++statistics->evaluation_tasks_by_kind[kind_index];
  statistics->evaluation_nodes += task_nodes;
  statistics->evaluation_pairs += task_pairs;
  if (use_cpu) {
    ++statistics->cpu_evaluation_tasks;
    ++statistics->cpu_evaluation_tasks_by_kind[kind_index];
    statistics->cpu_evaluation_nodes += task_nodes;
    statistics->cpu_evaluation_pairs += task_pairs;
  }
  statistics->maximum_task_nodes =
      std::max<std::uint64_t>(statistics->maximum_task_nodes, task_nodes);
  statistics->maximum_task_pairs =
      std::max<std::uint64_t>(statistics->maximum_task_pairs, task_pairs);
  return Status::Ok();
}

template <int M>
Status CausalTopologyWindowExecutor<M>::EvaluateForcedDynamic(
    const std::vector<const CausalTopologyRow<M>*>& rows,
    std::vector<std::vector<ClusterTopologyRankedCandidate>>* pending,
    std::vector<Winner>* winners, std::uint32_t position,
    std::uint32_t maximum_depth, std::uint64_t* generation,
    CausalTopologyWindowStatistics* statistics, CausalEvaluationKind kind) {
  if (batching_.same_layer_minimum_batch_pairs == 0U) {
    std::vector<const CausalTopologyRow<M>*> current_row{rows[position]};
    std::vector<std::vector<ClusterTopologyRankedCandidate>> current_candidates;
    current_candidates.push_back(std::move((*pending)[position]));
    std::vector<Winner> current_winner;
    current_winner.push_back(std::move((*winners)[position]));
    const Status status =
        Evaluate(current_row, current_candidates, &current_winner,
                 maximum_depth, generation, statistics, false, kind);
    (*winners)[position] = std::move(current_winner[0]);
    return status;
  }
  std::uint32_t pending_pairs = 0U;
  std::uint32_t pending_nodes = 0U;
  for (std::size_t child = position; child < pending->size(); ++child) {
    pending_pairs += static_cast<std::uint32_t>((*pending)[child].size());
    pending_nodes += static_cast<std::uint32_t>(!(*pending)[child].empty());
  }
  const bool routed_cpu = SingleNodeUsesCpu(
      batching_, static_cast<std::uint32_t>((*pending)[position].size()),
      UsesSameLayerSingleNodeThreshold(kind));
  if (cpu_evaluator_ != nullptr &&
      (routed_cpu || (batching_.tiny_forced_cpu_enabled &&
                      pending_pairs <= batching_.tiny_forced_pair_limit &&
                      (batching_.tiny_forced_node_limit == 0U ||
                       pending_nodes <= batching_.tiny_forced_node_limit)))) {
    std::vector<const CausalTopologyRow<M>*> current_row{rows[position]};
    std::vector<std::vector<ClusterTopologyRankedCandidate>> current_candidates;
    current_candidates.push_back(std::move((*pending)[position]));
    std::vector<Winner> current_winner;
    current_winner.push_back(std::move((*winners)[position]));
    const Status status =
        Evaluate(current_row, current_candidates, &current_winner,
                 maximum_depth, generation, statistics, true, kind);
    (*winners)[position] = std::move(current_winner[0]);
    return status;
  }
  const Status status = Evaluate(rows, *pending, winners, maximum_depth,
                                 generation, statistics, false, kind);
  if (status.ok()) {
    for (auto& candidates : *pending) {
      candidates.clear();
    }
  }
  return status;
}

template <int M>
Status CausalTopologyWindowExecutor<M>::PreflushDynamic(
    const std::vector<const CausalTopologyRow<M>*>& rows,
    std::vector<std::vector<ClusterTopologyRankedCandidate>>* pending,
    std::uint32_t first_position, std::uint32_t maximum_depth,
    std::uint64_t* generation, CausalTopologyWindowStatistics* statistics,
    CausalEvaluationKind kind, std::optional<PendingEvaluation>* in_flight) {
  if (scheduler_->ProducerTaskPolicy() != LinkageProducerTaskPolicy::kStable ||
      !scheduler_->SupportsAsynchronousPreflush() ||
      batching_.same_layer_minimum_batch_pairs == 0U) {
    return Status::Ok();
  }
  if (in_flight == nullptr || in_flight->has_value()) {
    return Status::Ok();
  }
  const std::int32_t configured_pairs =
      batching_.same_layer_preflush_target_pairs;
  const std::uint32_t target_pairs =
      configured_pairs < 0
          ? std::max(batching_.same_layer_minimum_batch_pairs,
                     batching_.many_nodes_preflush_target_pairs)
          : static_cast<std::uint32_t>(configured_pairs);
  const std::uint32_t target_nodes = batching_.same_layer_preflush_target_nodes;
  if (target_pairs == 0U && target_nodes == 0U) {
    return Status::Ok();
  }
  std::uint32_t available_pairs = 0U;
  std::uint32_t available_nodes = 0U;
  for (std::size_t child = first_position; child < pending->size(); ++child) {
    const std::uint32_t pairs =
        static_cast<std::uint32_t>((*pending)[child].size());
    available_pairs += pairs;
    available_nodes += pairs == 0U ? 0U : 1U;
    if ((target_pairs != 0U && available_pairs >= target_pairs) ||
        (target_nodes != 0U && available_nodes >= target_nodes)) {
      break;
    }
  }
  if ((target_pairs == 0U || available_pairs < target_pairs) &&
      (target_nodes == 0U || available_nodes < target_nodes)) {
    return Status::Ok();
  }

  std::vector<std::vector<ClusterTopologyRankedCandidate>> selected(
      pending->size());
  std::uint32_t selected_pairs = 0U;
  std::uint32_t selected_nodes = 0U;
  for (std::size_t child = first_position; child < pending->size(); ++child) {
    if ((*pending)[child].empty()) {
      continue;
    }
    const std::uint32_t remaining_capacity =
        batching_.same_layer_window_maximum_pairs - selected_pairs;
    const std::uint32_t take = std::min<std::uint32_t>(
        static_cast<std::uint32_t>((*pending)[child].size()),
        remaining_capacity);
    selected[child].assign((*pending)[child].begin(),
                           (*pending)[child].begin() + take);
    selected_pairs += take;
    ++selected_nodes;
    if (selected_pairs == batching_.same_layer_window_maximum_pairs ||
        selected_nodes == batching_.same_layer_maximum_pending_nodes) {
      break;
    }
  }
  if (selected_pairs < batching_.same_layer_minimum_batch_pairs) {
    return Status::Ok();
  }
  auto submitted = SubmitGpuEvaluation(rows, selected, maximum_depth,
                                       generation, statistics, kind);
  if (!submitted.ok()) {
    return submitted.status();
  }
  if (submitted.value().has_value()) {
    *in_flight = std::move(*submitted.value());
  }
  for (std::size_t child = first_position; child < pending->size(); ++child) {
    const std::size_t consumed = selected[child].size();
    if (consumed != 0U) {
      (*pending)[child].erase((*pending)[child].begin(),
                              (*pending)[child].begin() + consumed);
    }
  }
  return Status::Ok();
}

template <int M>
Status CausalTopologyWindowExecutor<M>::ExecuteBatch(
    const std::vector<const CausalTopologyRow<M>*>& rows,
    std::uint32_t maximum_depth, std::uint64_t* generation,
    CausalTopologyWindowStatistics* statistics) {
  const std::uint32_t unavailable = std::numeric_limits<std::uint32_t>::max();
  for (std::uint32_t position = 0U; position < rows.size(); ++position) {
    batch_position_[rows[position]->real_local_id] = position;
  }
  struct BatchPositionReset final {
    std::vector<std::uint32_t>* positions = nullptr;
    const std::vector<const CausalTopologyRow<M>*>* rows = nullptr;
    std::uint32_t unavailable = 0U;

    ~BatchPositionReset() {
      for (const CausalTopologyRow<M>* row : *rows) {
        (*positions)[row->real_local_id] = unavailable;
      }
    }
  } reset{&batch_position_, &rows, unavailable};

  std::vector<std::vector<ClusterTopologyRankedCandidate>> inner(rows.size());
  std::vector<std::vector<ClusterTopologyRankedCandidate>> same_frozen(
      rows.size());
  for (std::uint32_t child = 0U; child < rows.size(); ++child) {
    const CausalTopologyRow<M>& row = *rows[child];
    for (std::uint32_t rank = 0U; rank < row.ordered_candidates.size();
         ++rank) {
      const LinkageParentReference parent = row.ordered_candidates[rank];
      if (parent.kind == LinkageParentKind::kVirtual) {
        inner[child].push_back({parent, rank});
        continue;
      }
      const std::uint32_t parent_batch = batch_position_[parent.local_id];
      if (parent_batch != unavailable) {
        if (parent_batch >= child) {
          return Status::FailedPrecondition(
              "causal topology batch references a future parent");
        }
        continue;
      }
      if (layer_position_[parent.local_id] == unavailable) {
        inner[child].push_back({parent, rank});
      } else {
        same_frozen[child].push_back({parent, rank});
      }
    }
  }

  std::vector<Winner> winners(rows.size());
  if (const Status status =
          Evaluate(rows, inner, &winners, maximum_depth, generation, statistics,
                   false, CausalEvaluationKind::kInner);
      !status.ok()) {
    return status;
  }
  if (const Status status =
          Evaluate(rows, same_frozen, &winners, maximum_depth, generation,
                   statistics, false, CausalEvaluationKind::kSameFrozen);
      !status.ok()) {
    return status;
  }

  for (std::uint32_t window_begin = 0U; window_begin < rows.size();
       window_begin += batching_.same_layer_block_rows) {
    const std::uint32_t window_size = std::min<std::uint32_t>(
        batching_.same_layer_block_rows,
        static_cast<std::uint32_t>(rows.size()) - window_begin);
    std::vector<const CausalTopologyRow<M>*> window(
        rows.begin() + window_begin, rows.begin() + window_begin + window_size);
    std::vector<std::vector<ClusterTopologyRankedCandidate>> dynamic_before(
        window_size);
    std::vector<std::vector<ClusterTopologyRankedCandidate>> pending(
        window_size);
    std::vector<std::vector<DependentCandidate>> outgoing(window_size);
    for (std::uint32_t child = 0U; child < window_size; ++child) {
      const CausalTopologyRow<M>& row = *window[child];
      const std::uint32_t batch_child = window_begin + child;
      for (std::uint32_t rank = 0U; rank < row.ordered_candidates.size();
           ++rank) {
        const LinkageParentReference parent = row.ordered_candidates[rank];
        if (parent.kind != LinkageParentKind::kReal) {
          continue;
        }
        const std::uint32_t parent_batch = batch_position_[parent.local_id];
        if (parent_batch == unavailable) {
          continue;
        }
        if (parent_batch < window_begin) {
          dynamic_before[child].push_back({parent, rank});
          continue;
        }
        if (parent_batch >= batch_child) {
          return Status::FailedPrecondition(
              "causal topology window references a future parent");
        }
        const std::uint32_t parent_position = parent_batch - window_begin;
        outgoing[parent_position].push_back({parent, child, rank});
      }
    }

    std::vector<Winner> window_winners(window_size);
    std::optional<PendingEvaluation> dynamic_in_flight;
    for (std::uint32_t position = 0U; position < window_size; ++position) {
      window_winners[position] = std::move(winners[window_begin + position]);
    }
    if (const Status status = Evaluate(
            window, dynamic_before, &window_winners, maximum_depth, generation,
            statistics, false, CausalEvaluationKind::kDynamicBefore);
        !status.ok()) {
      return status;
    }
    for (std::uint32_t position = 0U; position < window_size; ++position) {
      if (dynamic_in_flight.has_value()) {
        if (const Status status =
                FinishGpuEvaluation(std::move(*dynamic_in_flight), window,
                                    &window_winners, statistics);
            !status.ok()) {
          return status;
        }
        dynamic_in_flight.reset();
      }
      if (!pending[position].empty()) {
        if (const Status status = EvaluateForcedDynamic(
                window, &pending, &window_winners, position, maximum_depth,
                generation, statistics, CausalEvaluationKind::kDynamicWindow);
            !status.ok()) {
          return status;
        }
      }
      const CausalTopologyRow<M>& row = *window[position];
      const LinkageParentReference* winner_parent =
          window_winners[position].valid
              ? &row.ordered_candidates[window_winners[position].original_rank]
              : nullptr;
      auto prepared = state_->PrepareNodeCommit(
          row.real_local_id,
          {winner_parent, window_winners[position].valid ? 1U : 0U},
          maximum_depth);
      if (!prepared.ok()) {
        return prepared.status();
      }
      if (prepared.value()) {
        if (!window_winners[position].valid) {
          return Status::Internal(
              "causal topology node has admitted candidates without a winner");
        }
        auto committed =
            state_->CommitPending(std::move(window_winners[position].result));
        if (!committed.ok()) {
          return committed.status();
        }
        if (committed.value().has_value()) {
          AppendUpdate(std::move(*committed.value()));
        }
      }

      // A completed root is a legal parent using its existing basic reconstruction.
      const std::uint32_t parent_depth = state_->depth(row.real_local_id);
      if (parent_depth < maximum_depth) {
        for (const DependentCandidate& dependency : outgoing[position]) {
          pending[dependency.child_position].push_back(
              {dependency.parent, dependency.original_rank});
        }
      }
      if (const Status status = PreflushDynamic(
              window, &pending, position + 1U, maximum_depth, generation,
              statistics, CausalEvaluationKind::kDynamicWindow,
              &dynamic_in_flight);
          !status.ok()) {
        return status;
      }
    }
    if (dynamic_in_flight.has_value()) {
      if (const Status status =
              FinishGpuEvaluation(std::move(*dynamic_in_flight), window,
                                  &window_winners, statistics);
          !status.ok()) {
        return status;
      }
    }
  }
  return Status::Ok();
}

template <int M>
StatusOr<CausalTopologyWindowStatistics>
CausalTopologyWindowExecutor<M>::RunFixedWindows(
    ArrayView<const CausalTopologyRow<M>> rows, std::uint32_t maximum_depth,
    std::uint64_t* generation, std::uint32_t window_rows,
    std::uint32_t maximum_frozen_pairs, std::uint32_t dynamic_preflush_pairs) {
  if (rows.empty() || maximum_depth == 0U || generation == nullptr ||
      cluster_.value == 0U || window_rows == 0U) {
    return Status::InvalidArgument(
        "causal topology window requires rows, depth and generation");
  }
  const std::uint32_t unavailable = std::numeric_limits<std::uint32_t>::max();
  CausalTopologyWindowStatistics statistics;
  statistics.rows = rows.size();
  layer_position_.assign(state_->real_count(), unavailable);
  window_position_.assign(state_->real_count(), unavailable);
  for (std::uint32_t position = 0U; position < rows.size(); ++position) {
    const CausalTopologyRow<M>& row = rows[position];
    statistics.candidate_pairs += row.ordered_candidates.size();
    if (row.real_local_id >= layer_position_.size() ||
        layer_position_[row.real_local_id] != unavailable) {
      return Status::FailedPrecondition(
          "causal topology sequence contains a duplicate or invalid node");
    }
    layer_position_[row.real_local_id] = position;
  }

  std::uint32_t window_begin = 0U;
  while (window_begin < rows.size()) {
    std::uint32_t window_end = window_begin;
    std::uint32_t frozen_pairs = 0U;
    while (window_end < rows.size() &&
           window_end - window_begin < window_rows) {
      const CausalTopologyRow<M>& row = rows[window_end];
      std::uint32_t row_frozen_pairs = 0U;
      for (const LinkageParentReference parent : row.ordered_candidates) {
        if (parent.kind == LinkageParentKind::kVirtual ||
            layer_position_[parent.local_id] < window_begin ||
            layer_position_[parent.local_id] == unavailable) {
          ++row_frozen_pairs;
        }
      }
      if (window_end != window_begin && maximum_frozen_pairs != 0U &&
          frozen_pairs + row_frozen_pairs > maximum_frozen_pairs) {
        break;
      }
      frozen_pairs += row_frozen_pairs;
      ++window_end;
    }

    const std::uint32_t window_size = window_end - window_begin;
    std::vector<const CausalTopologyRow<M>*> window;
    window.reserve(window_size);
    for (std::uint32_t position = 0U; position < window_size; ++position) {
      const CausalTopologyRow<M>* row = &rows[window_begin + position];
      window.push_back(row);
      window_position_[row->real_local_id] = position;
    }

    std::vector<std::vector<ClusterTopologyRankedCandidate>> frozen(
        window_size);
    std::vector<std::vector<ClusterTopologyRankedCandidate>> pending(
        window_size);
    std::vector<std::vector<DependentCandidate>> outgoing(window_size);
    for (std::uint32_t child = 0U; child < window_size; ++child) {
      const CausalTopologyRow<M>& row = *window[child];
      for (std::uint32_t rank = 0U; rank < row.ordered_candidates.size();
           ++rank) {
        const LinkageParentReference parent = row.ordered_candidates[rank];
        if (parent.kind == LinkageParentKind::kVirtual ||
            state_->visible(parent.local_id)) {
          frozen[child].push_back({parent, rank});
          continue;
        }
        const std::uint32_t parent_position = window_position_[parent.local_id];
        if (parent_position >= child) {
          return Status::FailedPrecondition(
              "causal topology window references a future parent");
        }
        outgoing[parent_position].push_back({parent, child, rank});
      }
    }

    std::vector<Winner> winners(window_size);
    if (const Status status =
            Evaluate(window, frozen, &winners, maximum_depth, generation,
                     &statistics, false, CausalEvaluationKind::kGraphFrozen);
        !status.ok()) {
      return status;
    }
    for (std::uint32_t position = 0U; position < window_size; ++position) {
      if (!pending[position].empty()) {
        if (const Status status = EvaluateForcedDynamic(
                window, &pending, &winners, position, maximum_depth, generation,
                &statistics, CausalEvaluationKind::kGraphDynamic);
            !status.ok()) {
          return status;
        }
      }
      const CausalTopologyRow<M>& row = *window[position];
      const LinkageParentReference* winner_parent =
          winners[position].valid
              ? &row.ordered_candidates[winners[position].original_rank]
              : nullptr;
      auto prepared = state_->PrepareNodeCommit(
          row.real_local_id, {winner_parent, winners[position].valid ? 1U : 0U},
          maximum_depth);
      if (!prepared.ok()) {
        return prepared.status();
      }
      if (prepared.value()) {
        if (!winners[position].valid) {
          return Status::Internal(
              "causal topology node has admitted candidates without a winner");
        }
        auto committed =
            state_->CommitPending(std::move(winners[position].result));
        if (!committed.ok()) {
          return committed.status();
        }
        if (committed.value().has_value()) {
          AppendUpdate(std::move(*committed.value()));
        }
      }
      const std::uint32_t parent_depth = state_->depth(row.real_local_id);
      for (const DependentCandidate& dependency : outgoing[position]) {
        if (parent_depth + 1U <= maximum_depth) {
          pending[dependency.child_position].push_back(
              {dependency.parent, dependency.original_rank});
        }
      }
      if (dynamic_preflush_pairs != 0U) {
        std::uint32_t pending_pairs = 0U;
        for (std::uint32_t child = position + 1U; child < window_size;
             ++child) {
          pending_pairs += static_cast<std::uint32_t>(pending[child].size());
        }
        if (pending_pairs >= dynamic_preflush_pairs) {
          if (const Status status = Evaluate(
                  window, pending, &winners, maximum_depth, generation,
                  &statistics, false, CausalEvaluationKind::kGraphDynamic);
              !status.ok()) {
            return status;
          }
          for (auto& candidate_row : pending) {
            candidate_row.clear();
          }
        }
      }
    }
    for (const CausalTopologyRow<M>* row : window) {
      window_position_[row->real_local_id] = unavailable;
    }
    window_begin = window_end;
  }
  if (const Status status = FlushUpdates(generation, &statistics);
      !status.ok()) {
    return status;
  }
  return statistics;
}

template <int M>
StatusOr<CausalTopologyWindowStatistics>
CausalTopologyWindowExecutor<M>::RunGraphOrder(
    ArrayView<const CausalTopologyRow<M>> rows, std::uint32_t maximum_depth,
    std::uint64_t* generation) {
  return RunFixedWindows(rows, maximum_depth, generation,
                         batching_.bad_frozen_window_rows,
                         batching_.bad_frozen_maximum_pairs,
                         batching_.same_layer_minimum_batch_pairs);
}

template <int M>
StatusOr<CausalTopologyWindowStatistics>
CausalTopologyWindowExecutor<M>::RunRadialLayer(
    ArrayView<const CausalTopologyRow<M>> rows, std::uint32_t maximum_depth,
    std::uint64_t* generation) {
  if (rows.empty() || maximum_depth == 0U || generation == nullptr ||
      cluster_.value == 0U || batching_.inner_batch_rows == 0U ||
      batching_.maximum_batch_pairs == 0U ||
      batching_.same_layer_block_rows == 0U ||
      batching_.same_layer_window_maximum_pairs == 0U ||
      batching_.same_layer_maximum_pending_nodes == 0U) {
    return Status::InvalidArgument(
        "causal topology window requires rows, depth and generation");
  }
  const std::uint32_t unavailable = std::numeric_limits<std::uint32_t>::max();
  CausalTopologyWindowStatistics statistics;
  statistics.rows = rows.size();
  layer_position_.assign(state_->real_count(), unavailable);
  batch_position_.assign(state_->real_count(), unavailable);
  for (std::uint32_t position = 0U; position < rows.size(); ++position) {
    const CausalTopologyRow<M>& row = rows[position];
    statistics.candidate_pairs += row.ordered_candidates.size();
    if (row.real_local_id >= layer_position_.size() ||
        layer_position_[row.real_local_id] != unavailable) {
      return Status::FailedPrecondition(
          "causal topology layer contains a duplicate or invalid node");
    }
    layer_position_[row.real_local_id] = position;
  }

  std::vector<const CausalTopologyRow<M>*> batch;
  batch.reserve(batching_.inner_batch_rows);
  std::uint32_t inner_pairs = 0U;
  for (const CausalTopologyRow<M>& row : rows) {
    if (row.ordered_candidates.empty()) {
      auto prepared = state_->PrepareNodeCommit(
          row.real_local_id, row.ordered_candidates, maximum_depth);
      if (!prepared.ok()) {
        return prepared.status();
      }
      continue;
    }
    batch.push_back(&row);
    for (const LinkageParentReference parent : row.ordered_candidates) {
      const std::uint32_t layer_position =
          parent.kind == LinkageParentKind::kVirtual
              ? unavailable
              : layer_position_[parent.local_id];
      const std::uint32_t parent_depth =
          parent.kind == LinkageParentKind::kVirtual
              ? 0U
              : state_->depth(parent.local_id);
      if (causal_topology_detail::CountsTowardInnerBatchLimit(
              parent, layer_position, unavailable, parent_depth,
              maximum_depth)) {
        ++inner_pairs;
      }
    }
    if (batch.size() >= batching_.inner_batch_rows ||
        inner_pairs >= batching_.maximum_batch_pairs) {
      if (const Status status =
              ExecuteBatch(batch, maximum_depth, generation, &statistics);
          !status.ok()) {
        return status;
      }
      batch.clear();
      inner_pairs = 0U;
    }
  }
  if (!batch.empty()) {
    if (const Status status =
            ExecuteBatch(batch, maximum_depth, generation, &statistics);
        !status.ok()) {
      return status;
    }
  }
  if (const Status status = FlushUpdates(generation, &statistics);
      !status.ok()) {
    return status;
  }
  return statistics;
}

template class CausalTopologyWindowExecutor<0>;
template class CausalTopologyWindowExecutor<5>;
template class CausalTopologyWindowExecutor<12>;
template class CausalTopologyWindowExecutor<6>;
template class CausalTopologyWindowExecutor<10>;

}  // namespace stlqpp
