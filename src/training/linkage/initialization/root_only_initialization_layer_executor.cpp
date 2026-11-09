#include "stlqpp/training/linkage/initialization/root_only_initialization_layer_executor.h"

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
struct RootOnlyInitializationLayerExecutor<M>::Winner final {
  bool valid = false;
  std::uint32_t original_rank = 0U;
  ClusterTopologyEvaluationWinner<M> result;
};

template <int M>
struct RootOnlyInitializationLayerExecutor<M>::EvaluationRow final {
  const CausalTopologyRow<M>* row = nullptr;
  std::vector<ClusterTopologyRankedCandidate> candidates;
};

template <int M>
struct RootOnlyInitializationLayerExecutor<M>::DependentCandidate final {
  std::uint32_t child_offset = 0U;
  ClusterTopologyRankedCandidate candidate;
};

template <int M>
Status RootOnlyInitializationLayerExecutor<M>::FlushUpdates(
    std::uint64_t* generation, CausalTopologyWindowStatistics* statistics) {
  if (update_columns_.empty()) {
    return Status::Ok();
  }
  auto submitted = scheduler_->SubmitUpdate(
      cluster_, {
                    .columns = update_columns_.data(),
                    .reconstructions = {update_reconstructions_.data(),
                                        state_->dimension(),
                                        static_cast<std::uint32_t>(
                                            update_columns_.size())},
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
void RootOnlyInitializationLayerExecutor<M>::AppendUpdate(
    ClusterTopologyReconstructionUpdate update) {
  update_columns_.push_back(update.registered_column);
  update_reconstructions_.insert(update_reconstructions_.end(),
                                 update.reconstruction.begin(),
                                 update.reconstruction.end());
}

template <int M>
StatusOr<std::optional<
    typename RootOnlyInitializationLayerExecutor<M>::PendingEvaluation>>
RootOnlyInitializationLayerExecutor<M>::SubmitEvaluation(
    const std::vector<EvaluationRow>& rows, std::uint64_t* generation,
    CausalTopologyWindowStatistics* statistics, std::uint32_t maximum_depth,
    CausalEvaluationKind kind) {
  std::vector<ClusterTopologyEvaluationRowView> views;
  views.reserve(rows.size());
  bool have_candidate = false;
  for (std::size_t index = 0U; index < rows.size(); ++index) {
    const bool preserve_empty = kind == CausalEvaluationKind::kInner;
    if (rows[index].candidates.empty() && !preserve_empty) {
      continue;
    }
    have_candidate = have_candidate || !rows[index].candidates.empty();
    views.push_back({
        .real_local_id = rows[index].row->real_local_id,
        .source_position = static_cast<std::uint32_t>(index),
        .ranked_candidates = rows[index].candidates,
        .icm_layer_order = {rows[index].row->icm_layer_order.data(),
                            state_->m()},
        .preserve_empty_candidate_row = preserve_empty,
    });
  }
  if (views.empty() || !have_candidate) {
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
Status RootOnlyInitializationLayerExecutor<M>::MergeEvaluation(
    const PendingEvaluation& evaluation, LinkageEvaluationResult result,
    std::vector<Winner>* winners,
    ArrayView<const std::uint32_t> commit_position_by_local,
    CausalTopologyWindowStatistics* statistics, bool cpu) {
  if (result.parent_columns.size() != evaluation.real_local_ids.size() ||
      result.supports.size() != evaluation.real_local_ids.size() ||
      result.supports.m() != state_->m() ||
      result.coefficients.size() !=
          evaluation.real_local_ids.size() * state_->m() ||
      result.squared_errors.size() != evaluation.real_local_ids.size()) {
    return Status::Internal(
        "root-only initialization evaluator returned an invalid extent");
  }

  for (std::size_t output = 0U; output < evaluation.real_local_ids.size();
       ++output) {
    if (evaluation.row_offsets[output] == evaluation.row_offsets[output + 1U]) {
      continue;
    }
    const std::uint32_t node = evaluation.real_local_ids[output];
    const std::uint32_t position = commit_position_by_local[node];
    const std::uint32_t parent_column = result.parent_columns[output];
    std::optional<std::uint32_t> selected_rank;
    for (std::uint32_t candidate = evaluation.row_offsets[output];
         candidate < evaluation.row_offsets[output + 1U]; ++candidate) {
      if (evaluation.candidate_columns[candidate] == parent_column) {
        selected_rank = evaluation.original_candidate_ranks[candidate];
        break;
      }
    }
    if (!selected_rank.has_value()) {
      return Status::Internal(
          "root-only initialization evaluator selected an unknown parent");
    }
    Winner& current = (*winners)[position];
    if (!current.valid ||
        result.squared_errors[output] < current.result.squared_error ||
        (result.squared_errors[output] == current.result.squared_error &&
         *selected_rank < current.original_rank)) {
      current.valid = true;
      current.original_rank = *selected_rank;
      current.result.parent_column = parent_column;
      current.result.support = result.supports.template tuple<M>(output);
      std::copy_n(result.coefficients.begin() + output * state_->m(),
                  state_->m(), current.result.coefficients.begin());
      current.result.squared_error = result.squared_errors[output];
    }
  }

  ++statistics->evaluation_tasks;
  const std::size_t kind = static_cast<std::size_t>(evaluation.kind);
  ++statistics->evaluation_tasks_by_kind[kind];
  statistics->evaluation_nodes += evaluation.node_count;
  statistics->evaluation_pairs += evaluation.pair_count;
  if (cpu) {
    ++statistics->cpu_evaluation_tasks;
    ++statistics->cpu_evaluation_tasks_by_kind[kind];
    statistics->cpu_evaluation_nodes += evaluation.node_count;
    statistics->cpu_evaluation_pairs += evaluation.pair_count;
  }
  statistics->maximum_task_nodes = std::max<std::uint64_t>(
      statistics->maximum_task_nodes, evaluation.node_count);
  statistics->maximum_task_pairs = std::max<std::uint64_t>(
      statistics->maximum_task_pairs, evaluation.pair_count);
  return Status::Ok();
}

template <int M>
Status RootOnlyInitializationLayerExecutor<M>::FinishEvaluation(
    PendingEvaluation evaluation, std::vector<Winner>* winners,
    ArrayView<const std::uint32_t> commit_position_by_local,
    CausalTopologyWindowStatistics* statistics) {
  auto evaluated = evaluation.result.get();
  if (!evaluated.ok()) {
    return evaluated.status();
  }
  return MergeEvaluation(evaluation, std::move(evaluated).value(), winners,
                         commit_position_by_local, statistics, false);
}

template <int M>
Status RootOnlyInitializationLayerExecutor<M>::Evaluate(
    const std::vector<EvaluationRow>& rows, std::vector<Winner>* winners,
    ArrayView<const std::uint32_t> commit_position_by_local,
    std::uint32_t maximum_depth, std::uint64_t* generation,
    CausalTopologyWindowStatistics* statistics, CausalEvaluationKind kind) {
  auto pending =
      SubmitEvaluation(rows, generation, statistics, maximum_depth, kind);
  if (!pending.ok()) {
    return pending.status();
  }
  if (!pending.value().has_value()) {
    return Status::Ok();
  }
  return FinishEvaluation(std::move(*pending.value()), winners,
                          commit_position_by_local, statistics);
}

template <int M>
Status RootOnlyInitializationLayerExecutor<M>::EvaluateTinyForced(
    const EvaluationRow& row, std::vector<Winner>* winners,
    ArrayView<const std::uint32_t> commit_position_by_local,
    std::uint32_t maximum_depth, std::uint64_t generation,
    CausalTopologyWindowStatistics* statistics, CausalEvaluationKind kind) {
  const ClusterTopologyEvaluationRowView view{
      .real_local_id = row.row->real_local_id,
      .source_position = 0U,
      .ranked_candidates = row.candidates,
      .icm_layer_order = {row.row->icm_layer_order.data(), state_->m()},
  };
  auto prepared = state_->PrepareVisibleEvaluation(cluster_, generation,
                                                   {&view, 1U}, maximum_depth);
  if (!prepared.ok()) {
    return prepared.status();
  }
  if (!prepared.value().has_value()) {
    return Status::Ok();
  }
  PreparedClusterTopologyEvaluation task = std::move(*prepared.value());
  PendingEvaluation metadata;
  metadata.real_local_ids = task.real_local_ids;
  metadata.candidate_columns = task.task.candidate_columns;
  metadata.row_offsets = task.task.row_offsets;
  metadata.original_candidate_ranks = task.original_candidate_ranks;
  metadata.node_count = task.task.node_count();
  metadata.pair_count = task.task.pair_count();
  metadata.kind = kind;
  auto evaluated = tail_evaluator_->Evaluate(*state_, task.task, cluster_seed_,
                                             row.row->real_local_id);
  if (!evaluated.ok()) {
    return evaluated.status();
  }
  return MergeEvaluation(metadata, std::move(evaluated).value(), winners,
                         commit_position_by_local, statistics, true);
}

template <int M>
Status RootOnlyInitializationLayerExecutor<M>::Commit(
    std::uint32_t position, ArrayView<const CausalTopologyRow<M>> rows,
    std::vector<Winner>* winners, std::uint32_t maximum_depth) {
  const CausalTopologyRow<M>& row = rows[position];
  Winner& winner = (*winners)[position];
  const LinkageParentReference* parent =
      winner.valid ? &row.ordered_candidates[winner.original_rank] : nullptr;
  auto prepared = state_->PrepareNodeCommit(
      row.real_local_id, {parent, winner.valid ? 1U : 0U}, maximum_depth);
  if (!prepared.ok()) {
    return prepared.status();
  }
  if (!prepared.value()) {
    return Status::Ok();
  }
  if (!winner.valid) {
    return Status::Internal(
        "root-only initialization admitted candidates without a winner");
  }
  auto committed = state_->CommitPending(std::move(winner.result));
  if (!committed.ok()) {
    return committed.status();
  }
  if (committed.value().has_value()) {
    AppendUpdate(std::move(*committed.value()));
  }
  return Status::Ok();
}

template <int M>
StatusOr<CausalTopologyWindowStatistics>
RootOnlyInitializationLayerExecutor<M>::Run(
    ArrayView<const CausalTopologyRow<M>> commit_rows,
    ArrayView<const std::uint32_t> physical_layer_order,
    std::uint32_t maximum_depth, std::uint64_t* generation) {
  return RunImpl(commit_rows, physical_layer_order, maximum_depth, generation,
                 false);
}

template <int M>
StatusOr<CausalTopologyWindowStatistics>
RootOnlyInitializationLayerExecutor<M>::RunValidatedLayer(
    ArrayView<const CausalTopologyRow<M>> commit_rows,
    ArrayView<const std::uint32_t> physical_layer_order,
    std::uint32_t maximum_depth, std::uint64_t* generation) {
  return RunImpl(commit_rows, physical_layer_order, maximum_depth, generation,
                 true);
}

template <int M>
StatusOr<CausalTopologyWindowStatistics>
RootOnlyInitializationLayerExecutor<M>::RunImpl(
    ArrayView<const CausalTopologyRow<M>> commit_rows,
    ArrayView<const std::uint32_t> physical_layer_order,
    std::uint32_t maximum_depth, std::uint64_t* generation,
    bool validated_layer) {
  if (commit_rows.empty() ||
      commit_rows.size() != physical_layer_order.size() ||
      maximum_depth == 0U || generation == nullptr || cluster_.value == 0U ||
      batching_.inner_batch_rows == 0U || batching_.maximum_batch_pairs == 0U ||
      batching_.same_layer_block_rows == 0U ||
      batching_.same_layer_window_maximum_pairs == 0U ||
      batching_.same_layer_maximum_pending_nodes == 0U) {
    return Status::InvalidArgument(
        "root-only initialization layer has an invalid extent");
  }
  const std::uint32_t unavailable = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> commit_position(state_->real_count(), unavailable);
  CausalTopologyWindowStatistics statistics;
  statistics.rows = commit_rows.size();
  for (std::uint32_t position = 0U; position < commit_rows.size(); ++position) {
    const CausalTopologyRow<M>& row = commit_rows[position];
    if (!validated_layer) {
      if (row.real_local_id >= commit_position.size() ||
          commit_position[row.real_local_id] != unavailable) {
        return Status::InvalidArgument(
            "root-only initialization commit order is not a layer permutation");
      }
    } else {
      assert(row.real_local_id < commit_position.size());
      assert(commit_position[row.real_local_id] == unavailable);
    }
    commit_position[row.real_local_id] = position;
    statistics.candidate_pairs += row.ordered_candidates.size();
  }
  if (!validated_layer) {
    for (const std::uint32_t local : physical_layer_order) {
      if (local >= commit_position.size() ||
          commit_position[local] == unavailable) {
        return Status::InvalidArgument(
            "root-only initialization physical order is not the same layer");
      }
    }
  }

  std::vector<Winner> winners(commit_rows.size());
  std::vector<EvaluationRow> inner_batch;
  inner_batch.reserve(batching_.inner_batch_rows);
  std::uint32_t inner_pairs = 0U;
  std::optional<PendingEvaluation> pending_inner;
  const auto flush_inner = [&](bool asynchronous) -> Status {
    if (inner_batch.empty()) {
      return Status::Ok();
    }
    Status status = Status::Ok();
    if (asynchronous) {
      auto submitted =
          SubmitEvaluation(inner_batch, generation, &statistics, maximum_depth,
                           CausalEvaluationKind::kInner);
      if (!submitted.ok()) {
        status = submitted.status();
      } else {
        pending_inner = std::move(submitted).value();
      }
    } else {
      status = Evaluate(inner_batch, &winners, commit_position, maximum_depth,
                        generation, &statistics, CausalEvaluationKind::kInner);
    }
    inner_batch.clear();
    inner_pairs = 0U;
    return status;
  };
  for (const std::uint32_t local : physical_layer_order) {
    const std::uint32_t position = commit_position[local];
    const CausalTopologyRow<M>& row = commit_rows[position];
    EvaluationRow selected;
    selected.row = &row;
    for (std::uint32_t rank = 0U; rank < row.ordered_candidates.size();
         ++rank) {
      const LinkageParentReference parent = row.ordered_candidates[rank];
      if (parent.kind == LinkageParentKind::kReal &&
          commit_position[parent.local_id] == unavailable &&
          state_->depth(parent.local_id) + 1U <= maximum_depth) {
        selected.candidates.push_back({parent, rank});
      }
    }
    const std::uint32_t new_pairs =
        static_cast<std::uint32_t>(selected.candidates.size());
    if (!inner_batch.empty() &&
        (inner_batch.size() >= batching_.inner_batch_rows ||
         inner_pairs + new_pairs > batching_.maximum_batch_pairs)) {
      if (const Status status = flush_inner(false); !status.ok()) {
        return status;
      }
    }
    inner_pairs += new_pairs;
    inner_batch.push_back(std::move(selected));
  }
  if (const Status status = flush_inner(true); !status.ok()) {
    return status;
  }

  for (std::uint32_t block_begin = 0U; block_begin < commit_rows.size();
       block_begin += batching_.same_layer_block_rows) {
    const std::uint32_t block_end =
        std::min<std::uint32_t>(static_cast<std::uint32_t>(commit_rows.size()),
                                block_begin + batching_.same_layer_block_rows);
    const std::uint32_t block_size = block_end - block_begin;

    std::vector<EvaluationRow> dynamic_before;
    dynamic_before.reserve(block_size);
    if (block_begin != 0U) {
      for (std::uint32_t position = block_begin; position < block_end;
           ++position) {
        EvaluationRow selected;
        selected.row = &commit_rows[position];
        for (std::uint32_t rank = 0U;
             rank < selected.row->ordered_candidates.size(); ++rank) {
          const LinkageParentReference parent =
              selected.row->ordered_candidates[rank];
          if (parent.kind == LinkageParentKind::kReal &&
              commit_position[parent.local_id] < block_begin) {
            selected.candidates.push_back({parent, rank});
          }
        }
        if (!selected.candidates.empty()) {
          dynamic_before.push_back(std::move(selected));
        }
      }
      if (const Status status = Evaluate(
              dynamic_before, &winners, commit_position, maximum_depth,
              generation, &statistics, CausalEvaluationKind::kDynamicBefore);
          !status.ok()) {
        return status;
      }
    }

    std::vector<std::vector<DependentCandidate>> outgoing(block_size);
    for (std::uint32_t position = block_begin; position < block_end;
         ++position) {
      const CausalTopologyRow<M>& row = commit_rows[position];
      for (std::uint32_t rank = 0U; rank < row.ordered_candidates.size();
           ++rank) {
        const LinkageParentReference parent = row.ordered_candidates[rank];
        if (parent.kind != LinkageParentKind::kReal) {
          continue;
        }
        const std::uint32_t parent_position = commit_position[parent.local_id];
        if (parent_position >= block_begin && parent_position < position) {
          outgoing[parent_position - block_begin].push_back({
              .child_offset = position - block_begin,
              .candidate = {parent, rank},
          });
        }
      }
    }

    std::vector<std::vector<ClusterTopologyRankedCandidate>> pending(
        block_size);
    std::vector<std::uint32_t> consumed(block_size, 0U);
    std::optional<PendingEvaluation> pending_dynamic;
    const auto preflush_dynamic = [&](std::uint32_t first_offset) -> Status {
      if (batching_.same_layer_minimum_batch_pairs == 0U) {
        return Status::Ok();
      }
      const bool source_async = scheduler_->ProducerTaskPolicy() ==
                                    LinkageProducerTaskPolicy::kStable &&
                                scheduler_->SupportsAsynchronousPreflush();
      if (source_async && pending_dynamic.has_value()) {
        return Status::Ok();
      }
      const std::int32_t configured_pairs =
          batching_.same_layer_preflush_target_pairs;
      const std::uint32_t target_pairs =
          configured_pairs < 0
              ? std::max(batching_.same_layer_minimum_batch_pairs,
                         batching_.many_nodes_preflush_target_pairs)
          : configured_pairs == 0
              ? (source_async
                     ? 0U
                     : std::max(batching_.same_layer_minimum_batch_pairs,
                                batching_.inner_batch_rows))
              : static_cast<std::uint32_t>(configured_pairs);
      const std::uint32_t target_nodes =
          batching_.same_layer_preflush_target_nodes;
      std::uint32_t available_pairs = 0U;
      std::uint32_t available_nodes = 0U;
      for (std::uint32_t child = first_offset; child < block_size; ++child) {
        const std::uint32_t pairs =
            static_cast<std::uint32_t>(pending[child].size()) - consumed[child];
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

      std::vector<EvaluationRow> selected_rows;
      std::vector<std::uint32_t> selected_offsets;
      std::vector<std::uint32_t> selected_counts;
      selected_rows.reserve(batching_.same_layer_maximum_pending_nodes);
      std::uint32_t selected_pairs = 0U;
      for (std::uint32_t child = first_offset; child < block_size; ++child) {
        const std::uint32_t done = consumed[child];
        if (done == pending[child].size()) {
          continue;
        }
        const std::uint32_t take = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(pending[child].size()) - done,
            batching_.same_layer_window_maximum_pairs - selected_pairs);
        EvaluationRow selected;
        selected.row = &commit_rows[block_begin + child];
        selected.candidates.insert(selected.candidates.end(),
                                   pending[child].begin() + done,
                                   pending[child].begin() + done + take);
        selected_rows.push_back(std::move(selected));
        selected_offsets.push_back(child);
        selected_counts.push_back(take);
        selected_pairs += take;
        if (selected_pairs == batching_.same_layer_window_maximum_pairs ||
            selected_rows.size() ==
                batching_.same_layer_maximum_pending_nodes) {
          break;
        }
      }
      if (selected_pairs < batching_.same_layer_minimum_batch_pairs) {
        return Status::Ok();
      }
      if (source_async) {
        auto submitted = SubmitEvaluation(selected_rows, generation,
                                          &statistics, maximum_depth,
                                          CausalEvaluationKind::kDynamicWindow);
        if (!submitted.ok()) {
          return submitted.status();
        }
        pending_dynamic = std::move(submitted).value();
      } else {
        if (const Status status = Evaluate(
                selected_rows, &winners, commit_position, maximum_depth,
                generation, &statistics, CausalEvaluationKind::kDynamicWindow);
            !status.ok()) {
          return status;
        }
      }
      for (std::size_t selected = 0U; selected < selected_offsets.size();
           ++selected) {
        consumed[selected_offsets[selected]] += selected_counts[selected];
      }
      return Status::Ok();
    };
    if (pending_inner.has_value()) {
      if (const Status status =
              FinishEvaluation(std::move(*pending_inner), &winners,
                               commit_position, &statistics);
          !status.ok()) {
        return status;
      }
      pending_inner.reset();
    }

    for (std::uint32_t offset = 0U; offset < block_size; ++offset) {
      if (pending_dynamic.has_value()) {
        if (const Status status =
                FinishEvaluation(std::move(*pending_dynamic), &winners,
                                 commit_position, &statistics);
            !status.ok()) {
          return status;
        }
        pending_dynamic.reset();
      }
      while (consumed[offset] < pending[offset].size()) {
        std::vector<EvaluationRow> forced;
        std::vector<std::uint32_t> selected_offsets;
        std::vector<std::uint32_t> selected_counts;
        std::uint32_t total_pairs = 0U;
        const std::uint32_t forced_end =
            batching_.same_layer_minimum_batch_pairs == 0U ? offset + 1U
                                                           : block_size;
        for (std::uint32_t child = offset;
             child < forced_end &&
             total_pairs < batching_.same_layer_window_maximum_pairs;
             ++child) {
          auto& candidates = pending[child];
          const std::uint32_t done = consumed[child];
          if (done >= candidates.size()) {
            continue;
          }
          const std::uint32_t take = std::min<std::uint32_t>(
              static_cast<std::uint32_t>(candidates.size()) - done,
              batching_.same_layer_window_maximum_pairs - total_pairs);
          EvaluationRow selected;
          selected.row = &commit_rows[block_begin + child];
          selected.candidates.insert(selected.candidates.end(),
                                     candidates.begin() + done,
                                     candidates.begin() + done + take);
          forced.push_back(std::move(selected));
          selected_offsets.push_back(child);
          selected_counts.push_back(take);
          total_pairs += take;
        }
        if (batching_.tiny_forced_cpu_enabled &&
            total_pairs <= batching_.tiny_forced_pair_limit &&
            (batching_.tiny_forced_node_limit == 0U ||
             forced.size() <= batching_.tiny_forced_node_limit)) {
          EvaluationRow current;
          current.row = &commit_rows[block_begin + offset];
          current.candidates.insert(current.candidates.end(),
                                    pending[offset].begin() + consumed[offset],
                                    pending[offset].end());
          if (const Status status =
                  EvaluateTinyForced(current, &winners, commit_position,
                                     maximum_depth, *generation, &statistics,
                                     CausalEvaluationKind::kDynamicWindow);
              !status.ok()) {
            return status;
          }
          consumed[offset] = static_cast<std::uint32_t>(pending[offset].size());
        } else {
          if (const Status status = Evaluate(
                  forced, &winners, commit_position, maximum_depth, generation,
                  &statistics, CausalEvaluationKind::kDynamicWindow);
              !status.ok()) {
            return status;
          }
          for (std::size_t selected = 0U; selected < selected_offsets.size();
               ++selected) {
            consumed[selected_offsets[selected]] += selected_counts[selected];
          }
        }
      }

      const std::uint32_t position = block_begin + offset;
      if (const Status status =
              Commit(position, commit_rows, &winners, maximum_depth);
          !status.ok()) {
        return status;
      }
      if (state_->depth(commit_rows[position].real_local_id) + 1U <=
          maximum_depth) {
        for (const DependentCandidate& dependent : outgoing[offset]) {
          pending[dependent.child_offset].push_back(dependent.candidate);
        }
      }
      if (const Status status = preflush_dynamic(offset + 1U); !status.ok()) {
        return status;
      }
    }
    if (pending_dynamic.has_value()) {
      if (const Status status =
              FinishEvaluation(std::move(*pending_dynamic), &winners,
                               commit_position, &statistics);
          !status.ok()) {
        return status;
      }
    }
    if (const Status status = FlushUpdates(generation, &statistics);
        !status.ok()) {
      return status;
    }
  }
  if (pending_inner.has_value()) {
    if (const Status status = FinishEvaluation(
            std::move(*pending_inner), &winners, commit_position, &statistics);
        !status.ok()) {
      return status;
    }
  }
  return statistics;
}

template class RootOnlyInitializationLayerExecutor<0>;
template class RootOnlyInitializationLayerExecutor<5>;
template class RootOnlyInitializationLayerExecutor<12>;
template class RootOnlyInitializationLayerExecutor<6>;
template class RootOnlyInitializationLayerExecutor<10>;

}  // namespace stlqpp
