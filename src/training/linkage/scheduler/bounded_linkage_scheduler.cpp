#include "stlqpp/training/linkage/scheduler/bounded_linkage_scheduler.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stlqpp/runtime/concurrency/byte_admission_gate.h"
#include "stlqpp/runtime/cpu/openmp_affinity.h"
#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"

namespace stlqpp {
namespace {

struct RegisterCommand final {
  LinkageBackendRegistration registration;
  std::promise<StatusOr<LinkageBackendCluster>> completion;
};

struct SchedulerClusterState final {
  mutable std::mutex admission_mutex;
  std::size_t worker = 0U;
  LinkageBackendCluster backend = 0U;
  std::uint32_t dimension = 0U;
  std::uint32_t columns = 0U;
  std::uint64_t reserved_bytes = 0U;
  std::atomic<std::uint64_t> submitted_generation{0U};
  std::atomic<std::uint64_t> enqueued_generation{0U};
  std::atomic<std::uint64_t> queued_tasks{0U};
  std::atomic<bool> retiring{false};
  std::atomic<bool> poisoned{false};
  std::atomic<bool> live{true};
  // Protected by admission_mutex. This orders update/evaluation admission for
  // one CID without imposing an order across independent CIDs.
  std::uint64_t next_causal_sequence = 1U;
};

struct UpdateCommand final {
  LinkageClusterHandle public_cluster;
  std::shared_ptr<SchedulerClusterState> state;
  LinkageBackendUpdate update;
  std::uint64_t causal_sequence = 0U;
  std::promise<Status> completion;
};

struct UnregisterCommand final {
  std::shared_ptr<SchedulerClusterState> state;
  std::promise<Status> completion;
};

struct ResourceStatsCommand final {
  std::promise<LinkageBackendResourceStats> completion;
};

struct EvaluationCommand final {
  LinkageClusterHandle public_cluster;
  std::shared_ptr<SchedulerClusterState> state;
  LinkageBackendEvaluation evaluation;
  std::uint64_t causal_sequence = 0U;
  std::promise<StatusOr<LinkageEvaluationResult>> completion;
};

Status ValidateOptions(const LinkageSchedulerOptions& options,
                       std::size_t backend_count) {
  if (backend_count == 0U || options.queue_capacity == 0U ||
      options.maximum_live_registration_bytes == 0U ||
      options.target_batch_tasks == 0U || options.target_batch_nodes == 0U ||
      options.target_batch_pairs == 0U || options.maximum_batch_tasks == 0U ||
      options.maximum_batch_nodes == 0U || options.maximum_batch_pairs == 0U ||
      options.maximum_update_requests == 0U ||
      options.maximum_update_columns == 0U ||
      options.target_batch_tasks > options.maximum_batch_tasks ||
      options.target_batch_nodes > options.maximum_batch_nodes ||
      options.target_batch_pairs > options.maximum_batch_pairs ||
      options.gather_window.count() < 0) {
    return Status::InvalidArgument(
        "linkage scheduler requires nonzero bounded resources");
  }
  return Status::Ok();
}

struct RegistrationLayout final {
  std::uint32_t dimension = 0U;
  std::uint32_t columns = 0U;
  std::size_t values = 0U;
  std::uint64_t bytes = 0U;
};

StatusOr<RegistrationLayout> ValidateRegistration(
    LinkageClusterRegistrationView input) {
  const bool have_first = input.first.columns != 0U;
  const bool have_second = input.second.columns != 0U;
  if ((!have_first && !have_second) ||
      (have_first &&
       (input.first.values == nullptr || input.first.rows == 0U)) ||
      (have_second &&
       (input.second.values == nullptr || input.second.rows == 0U))) {
    return Status::InvalidArgument(
        "cluster registration requires at least one non-empty matrix");
  }
  const std::uint32_t dimension =
      have_first ? input.first.rows : input.second.rows;
  if ((have_first && input.first.rows != dimension) ||
      (have_second && input.second.rows != dimension)) {
    return Status::InvalidArgument(
        "cluster registration parts must share one dimension");
  }
  const std::uint64_t columns64 =
      static_cast<std::uint64_t>(input.first.columns) + input.second.columns;
  if (columns64 > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "cluster registration column count exceeds uint32");
  }
  if (columns64 > std::numeric_limits<std::size_t>::max() / dimension ||
      columns64 > std::numeric_limits<std::uint64_t>::max() / dimension /
                      sizeof(float)) {
    return Status::ResourceExhausted(
        "cluster registration byte extent overflows");
  }
  const auto values = static_cast<std::size_t>(dimension) * columns64;
  return RegistrationLayout{
      .dimension = dimension,
      .columns = static_cast<std::uint32_t>(columns64),
      .values = static_cast<std::size_t>(values),
      .bytes = static_cast<std::uint64_t>(values) * sizeof(float),
  };
}

LinkageBackendRegistration CopyRegistration(
    LinkageClusterRegistrationView input, const RegistrationLayout& layout) {
  LinkageBackendRegistration output;
  output.dimension = layout.dimension;
  output.column_count = layout.columns;
  output.reconstructions.reserve(layout.values);
  const auto dimension = layout.dimension;
  const bool have_first = input.first.columns != 0U;
  const bool have_second = input.second.columns != 0U;
  if (have_first) {
    const std::size_t count =
        static_cast<std::size_t>(dimension) * input.first.columns;
    output.reconstructions.insert(output.reconstructions.end(),
                                  input.first.values,
                                  input.first.values + count);
  }
  if (have_second) {
    const std::size_t count =
        static_cast<std::size_t>(dimension) * input.second.columns;
    output.reconstructions.insert(output.reconstructions.end(),
                                  input.second.values,
                                  input.second.values + count);
  }
  return output;
}

// The credit covers the copy as well as the backend registration. Transfer it
// to cluster ownership only after registration has completed successfully.
class RegistrationCredit final {
 public:
  RegistrationCredit(ByteAdmissionGate& gate, std::uint64_t bytes) noexcept
      : gate_(&gate), bytes_(bytes) {}
  ~RegistrationCredit() {
    if (gate_ != nullptr) gate_->Release(bytes_);
  }
  RegistrationCredit(const RegistrationCredit&) = delete;
  RegistrationCredit& operator=(const RegistrationCredit&) = delete;
  void Transfer() noexcept { gate_ = nullptr; }

 private:
  ByteAdmissionGate* gate_;
  std::uint64_t bytes_;
};

Status ValidateTaskShape(const LinkageEvaluationTask& task) {
  const std::uint32_t nodes = task.node_count();
  const std::uint32_t pairs = task.pair_count();
  if (task.cluster.value == 0U || task.dimension == 0U || nodes == 0U ||
      pairs == 0U || task.row_offsets.front() != 0U ||
      task.row_offsets.back() != pairs ||
      task.node_vectors.size() !=
          static_cast<std::size_t>(task.dimension) * nodes) {
    return Status::InvalidArgument(
        "linkage evaluation task has an invalid matrix or CSR shape");
  }
  for (std::size_t row = 1U; row < task.row_offsets.size(); ++row) {
    const bool empty_root_only_row =
        task.kind == LinkageEvaluationKind::kRootOnlyInitialization &&
        task.row_offsets[row] == task.row_offsets[row - 1U];
    if (task.row_offsets[row] < task.row_offsets[row - 1U] ||
        (task.row_offsets[row] == task.row_offsets[row - 1U] &&
         !empty_root_only_row)) {
      return Status::InvalidArgument(
          "every linkage evaluation row requires at least one candidate");
    }
  }
  const std::uint32_t order_count =
      task.kind == LinkageEvaluationKind::kEdgeRsd ? nodes : 1U;
  return ValidateIcmLayerOrders<0>(order_count, task.icm_layer_orders, task.m);
}

bool Compatible(const EvaluationCommand& left,
                const EvaluationCommand& right) noexcept {
  const LinkageEvaluationTask& left_task = left.evaluation.task;
  const LinkageEvaluationTask& right_task = right.evaluation.task;
  return left_task.kind == LinkageEvaluationKind::kEdgeRsd &&
         right_task.kind == LinkageEvaluationKind::kEdgeRsd &&
         left_task.m == right_task.m &&
         left_task.dimension == right_task.dimension;
}

}  // namespace

struct BoundedLinkageScheduler::Impl final {
  struct Worker final {
    Worker(Impl* owner_in, std::unique_ptr<LinkageWorkerBackend> backend_in,
           const LinkageSchedulerOptions& options)
        : owner(owner_in),
          backend(std::move(backend_in)),
          capacity(options.queue_capacity),
          enqueue_evaluation(options.stable ? &EnqueueStableEvaluation
                                            : &EnqueueThroughputEvaluation) {
      update_batch.reserve(options.maximum_update_requests);
      backend_updates.reserve(options.maximum_update_requests);
      deferred_update_states.reserve(options.queue_capacity);
      failed_update_states.reserve(options.maximum_update_requests);
      failed_evaluations.reserve(options.queue_capacity);
      evaluation_batch.reserve(options.maximum_batch_tasks);
      backend_evaluations.reserve(options.maximum_batch_tasks);
      evaluation_clusters.reserve(options.maximum_batch_tasks);
      pending_evaluation_sequences.reserve(options.queue_capacity);
    }

    [[nodiscard]] Status Start() {
      std::promise<Status> startup;
      auto ready = startup.get_future();
      thread = std::thread([this, startup = std::move(startup)]() mutable {
        const Status status = RestoreNativeThreadAffinity();
        startup.set_value(status);
        if (status.ok()) {
          if (owner->options.stable) {
            owner->WorkerLoop<true>(this);
          } else {
            owner->WorkerLoop<false>(this);
          }
        }
      });
      return ready.get();
    }

    template <typename Command>
    bool Push(std::deque<Command>* target, Command command) {
      std::unique_lock<std::mutex> lock(mutex);
      not_full.wait(lock,
                    [this] { return closed || queued_commands < capacity; });
      if (closed) {
        return false;
      }
      target->push_back(std::move(command));
      ++queued_commands;
      not_empty.notify_one();
      return true;
    }

    template <bool Stable>
    bool PushEvaluation(EvaluationCommand command) {
      std::unique_lock<std::mutex> lock(mutex);
      not_full.wait(lock,
                    [this] { return closed || queued_commands < capacity; });
      if (closed || command.state->poisoned.load(std::memory_order_acquire)) {
        return false;
      }
      evaluations.push_back(std::move(command));
      if constexpr (Stable) {
        auto& sequences =
            pending_evaluation_sequences[evaluations.back().state.get()];
        assert(sequences.empty() ||
               sequences.back() < evaluations.back().causal_sequence);
        sequences.push_back(evaluations.back().causal_sequence);
      }
      ++queued_commands;
      not_empty.notify_one();
      return true;
    }

    [[nodiscard]] bool PushConfiguredEvaluation(EvaluationCommand command) {
      return enqueue_evaluation(this, std::move(command));
    }

    [[nodiscard]] bool IsEarliestEvaluationUnsafe(
        const EvaluationCommand& command) const noexcept {
      const auto found = pending_evaluation_sequences.find(command.state.get());
      return found != pending_evaluation_sequences.end() &&
             !found->second.empty() &&
             found->second.front() == command.causal_sequence;
    }

    [[nodiscard]] bool HasEarlierEvaluationUnsafe(
        SchedulerClusterState* state,
        std::uint64_t causal_sequence) const noexcept {
      const auto found = pending_evaluation_sequences.find(state);
      return found != pending_evaluation_sequences.end() &&
             !found->second.empty() && found->second.front() < causal_sequence;
    }

    void RetireEvaluationUnsafe(const EvaluationCommand& command) noexcept {
      auto found = pending_evaluation_sequences.find(command.state.get());
      assert(found != pending_evaluation_sequences.end());
      assert(!found->second.empty());
      assert(found->second.front() == command.causal_sequence);
      found->second.pop_front();
      if (found->second.empty()) {
        pending_evaluation_sequences.erase(found);
      }
    }

    void ConsumeUnsafe(std::size_t count = 1U) noexcept {
      assert(count <= queued_commands);
      queued_commands -= count;
      not_full.notify_all();
    }

    [[nodiscard]] bool HasWorkUnsafe() const noexcept {
      return queued_commands != 0U;
    }

    [[nodiscard]] bool WaitForWork() {
      std::unique_lock<std::mutex> lock(mutex);
      not_empty.wait(lock, [this] { return closed || HasWorkUnsafe(); });
      return HasWorkUnsafe();
    }

    void Close() noexcept {
      std::lock_guard<std::mutex> lock(mutex);
      closed = true;
      not_empty.notify_all();
      not_full.notify_all();
    }

    Impl* owner = nullptr;
    std::unique_ptr<LinkageWorkerBackend> backend;
    const std::size_t capacity;
    using EvaluationEnqueue = bool (*)(Worker*, EvaluationCommand);
    static bool EnqueueStableEvaluation(Worker* worker,
                                        EvaluationCommand command) {
      return worker->template PushEvaluation<true>(std::move(command));
    }
    static bool EnqueueThroughputEvaluation(Worker* worker,
                                            EvaluationCommand command) {
      return worker->template PushEvaluation<false>(std::move(command));
    }
    const EvaluationEnqueue enqueue_evaluation;
    std::mutex mutex;
    std::condition_variable not_empty;
    std::condition_variable not_full;
    std::deque<RegisterCommand> registrations;
    std::deque<UpdateCommand> updates;
    std::deque<EvaluationCommand> evaluations;
    std::deque<UnregisterCommand> unregisters;
    std::deque<ResourceStatsCommand> resource_queries;
    std::vector<UpdateCommand> update_batch;
    std::vector<LinkageBackendUpdate> backend_updates;
    std::unordered_set<SchedulerClusterState*> deferred_update_states;
    std::unordered_set<SchedulerClusterState*> failed_update_states;
    std::vector<EvaluationCommand> failed_evaluations;
    std::vector<EvaluationCommand> evaluation_batch;
    std::vector<LinkageBackendEvaluation> backend_evaluations;
    std::unordered_set<std::uint64_t> evaluation_clusters;
    // Stable mode admits one monotonically ordered sequence per CID.
    // The front answers both update-predecessor and evaluation-order queries
    // in O(1), replacing a worker-wide nested scan under the queue mutex.
    std::unordered_map<SchedulerClusterState*, std::deque<std::uint64_t>>
        pending_evaluation_sequences;
    std::size_t queued_commands = 0U;
    bool closed = false;
    std::thread thread;
  };

  explicit Impl(LinkageSchedulerOptions options_in)
      : options(options_in),
        registration_admission(options_in.maximum_live_registration_bytes) {}

  ~Impl() {
    for (const auto& worker : workers) {
      worker->Close();
    }
    for (const auto& worker : workers) {
      if (worker->thread.joinable()) {
        worker->thread.join();
      }
    }
  }

  template <bool Stable>
  void WorkerLoop(Worker* worker) {
    while (worker->WaitForWork()) {
      ProcessOneRegistration(worker);
      ProcessUpdates<Stable>(worker);
      if (ProcessEvaluations<Stable>(worker)) {
        continue;
      }
      if (ProcessOneUnregistration(worker)) {
        continue;
      }
      ProcessOneResourceQuery(worker);
    }
  }

  bool ProcessOneRegistration(Worker* worker) {
    RegisterCommand command;
    {
      std::lock_guard<std::mutex> lock(worker->mutex);
      if (worker->registrations.empty()) {
        return false;
      }
      command = std::move(worker->registrations.front());
      worker->registrations.pop_front();
      worker->ConsumeUnsafe();
    }
    command.completion.set_value(
        worker->backend->Register(std::move(command.registration)));
    return true;
  }

  template <bool Stable>
  bool ProcessUpdates(Worker* worker) {
    std::vector<UpdateCommand>& commands = worker->update_batch;
    assert(commands.empty());
    std::uint64_t columns = 0U;
    {
      std::lock_guard<std::mutex> lock(worker->mutex);
      if (worker->updates.empty()) {
        return false;
      }
      if constexpr (!Stable) {
        const std::uint32_t dimension =
            worker->updates.front().update.dimension;
        while (!worker->updates.empty() &&
               worker->updates.front().update.dimension == dimension &&
               commands.size() < options.maximum_update_requests) {
          const std::uint64_t next_columns =
              worker->updates.front().update.columns.size();
          if (!commands.empty() &&
              columns + next_columns > options.maximum_update_columns) {
            break;
          }
          columns += next_columns;
          commands.push_back(std::move(worker->updates.front()));
          worker->updates.pop_front();
          worker->ConsumeUnsafe();
        }
      } else {
        std::unordered_set<SchedulerClusterState*>& deferred_states =
            worker->deferred_update_states;
        assert(deferred_states.empty());
        std::uint32_t dimension = 0U;
        for (auto it = worker->updates.begin();
             it != worker->updates.end() &&
             commands.size() < options.maximum_update_requests;) {
          if (deferred_states.contains(it->state.get())) {
            ++it;
            continue;
          }
          const bool earlier_evaluation = worker->HasEarlierEvaluationUnsafe(
              it->state.get(), it->causal_sequence);
          if (earlier_evaluation ||
              (!commands.empty() && it->update.dimension != dimension)) {
            deferred_states.insert(it->state.get());
            ++it;
            continue;
          }
          const std::uint64_t next_columns = it->update.columns.size();
          if (!commands.empty() &&
              columns + next_columns > options.maximum_update_columns) {
            deferred_states.insert(it->state.get());
            ++it;
            continue;
          }
          if (commands.empty()) {
            dimension = it->update.dimension;
          }
          columns += next_columns;
          commands.push_back(std::move(*it));
          it = worker->updates.erase(it);
          worker->ConsumeUnsafe();
        }
        deferred_states.clear();
      }
      if (commands.empty()) {
        return false;
      }
    }

    std::vector<LinkageBackendUpdate>& updates = worker->backend_updates;
    assert(updates.empty());
    for (UpdateCommand& command : commands) {
      updates.push_back(std::move(command.update));
    }
    const Status status = worker->backend->ApplyUpdates(updates);
    if (!status.ok()) {
      std::unordered_set<SchedulerClusterState*>& failed_states =
          worker->failed_update_states;
      assert(failed_states.empty());
      for (const UpdateCommand& command : commands) {
        command.state->poisoned.store(true, std::memory_order_release);
        failed_states.insert(command.state.get());
      }
      std::vector<EvaluationCommand>& failed_evaluations =
          worker->failed_evaluations;
      assert(failed_evaluations.empty());
      {
        std::lock_guard<std::mutex> lock(worker->mutex);
        for (auto it = worker->evaluations.begin();
             it != worker->evaluations.end();) {
          if (!failed_states.contains(it->state.get())) {
            ++it;
            continue;
          }
          if constexpr (Stable) {
            worker->RetireEvaluationUnsafe(*it);
          }
          failed_evaluations.push_back(std::move(*it));
          it = worker->evaluations.erase(it);
          worker->ConsumeUnsafe();
        }
      }
      for (EvaluationCommand& evaluation : failed_evaluations) {
        evaluation.state->queued_tasks.fetch_sub(1U, std::memory_order_release);
        evaluation.completion.set_value(
            StatusOr<LinkageEvaluationResult>(status));
      }
      failed_evaluations.clear();
      failed_states.clear();
    } else {
      for (const UpdateCommand& command : commands) {
        const std::uint64_t generation = command.update.generation;
        std::uint64_t observed =
            command.state->enqueued_generation.load(std::memory_order_relaxed);
        while (observed < generation &&
               !command.state->enqueued_generation.compare_exchange_weak(
                   observed, generation, std::memory_order_release,
                   std::memory_order_relaxed)) {
        }
      }
    }
    for (UpdateCommand& command : commands) {
      command.completion.set_value(status);
    }
    updates.clear();
    commands.clear();
    std::lock_guard<std::mutex> stats_lock(stats_mutex);
    stats.update_batches += 1U;
    return true;
  }

  [[nodiscard]] static bool EvaluationReady(
      const EvaluationCommand& command) noexcept {
    return command.state->live.load(std::memory_order_acquire) &&
           command.state->enqueued_generation.load(std::memory_order_acquire) >=
               command.evaluation.task.required_generation;
  }

  template <bool Stable>
  bool ProcessEvaluations(Worker* worker) {
    std::vector<EvaluationCommand>& commands = worker->evaluation_batch;
    assert(commands.empty());
    std::uint64_t nodes = 0U;
    std::uint64_t pairs = 0U;
    std::unique_lock<std::mutex> lock(worker->mutex);
    const auto append_ready = [&] {
      for (auto it = worker->evaluations.begin();
           it != worker->evaluations.end();) {
        const bool stable_out_of_order =
            Stable && !worker->IsEarliestEvaluationUnsafe(*it);
        if (!EvaluationReady(*it) || stable_out_of_order ||
            (!commands.empty() && !Compatible(commands.front(), *it))) {
          ++it;
          continue;
        }
        const std::uint64_t next_nodes =
            nodes + it->evaluation.task.node_count();
        const std::uint64_t next_pairs =
            pairs + it->evaluation.task.pair_count();
        if (!commands.empty() &&
            (commands.size() >= options.maximum_batch_tasks ||
             next_nodes > options.maximum_batch_nodes ||
             next_pairs > options.maximum_batch_pairs)) {
          ++it;
          continue;
        }
        nodes = next_nodes;
        pairs = next_pairs;
        if constexpr (Stable) {
          worker->RetireEvaluationUnsafe(*it);
        }
        commands.push_back(std::move(*it));
        it = worker->evaluations.erase(it);
        worker->ConsumeUnsafe();
        if (commands.size() >= options.target_batch_tasks ||
            nodes >= options.target_batch_nodes ||
            pairs >= options.target_batch_pairs) {
          break;
        }
      }
    };
    append_ready();
    if (commands.empty()) {
      return false;
    }
    // Frozen DEV starts the bounded gather interval after its first complete
    // ready-queue scan. Starting it before lock acquisition/scan makes queue
    // traversal consume the 250-us budget. In throughput mode that also
    // changes merged TF32 widths; in stable mode it changes only the
    // transport membership.
    const auto deadline =
        std::chrono::steady_clock::now() + options.gather_window;
    while (commands.size() < options.target_batch_tasks &&
           nodes < options.target_batch_nodes &&
           pairs < options.target_batch_pairs &&
           std::chrono::steady_clock::now() < deadline) {
      // Updates waiting on this worker cannot make more evaluations ready
      // while we collect. Finish the selected batch, then service the updates
      // in the next worker iteration; do not reorder them ahead of this batch.
      if (!worker->updates.empty()) {
        break;
      }
      worker->not_empty.wait_until(lock, deadline);
      append_ready();
    }
    lock.unlock();

    std::unordered_set<std::uint64_t>& public_clusters =
        worker->evaluation_clusters;
    std::vector<LinkageBackendEvaluation>& evaluations =
        worker->backend_evaluations;
    assert(public_clusters.empty());
    assert(evaluations.empty());
    for (EvaluationCommand& command : commands) {
      public_clusters.insert(command.public_cluster.value);
      evaluations.push_back(std::move(command.evaluation));
    }
    auto results = [&]() -> StatusOr<std::vector<LinkageEvaluationResult>> {
      if constexpr (!Stable) {
        return worker->backend->Evaluate(evaluations);
      } else {
        return worker->backend->EvaluateStable(evaluations);
      }
    }();
    // DEV completes all batch-owned scheduler bookkeeping before waking any
    // producer future.  Waking one CID early lets it enqueue its next update
    // while this transport batch is still being retired and changes later
    // merge order.
    for (EvaluationCommand& command : commands) {
      command.state->queued_tasks.fetch_sub(1U, std::memory_order_release);
    }
    {
      std::lock_guard<std::mutex> stats_lock(stats_mutex);
      ++stats.transport_batches;
      stats.cross_cluster_transport_batches +=
          public_clusters.size() > 1U ? 1U : 0U;
      stats.numerical_batches += Stable ? commands.size() : 1U;
      stats.cross_cluster_numerical_batches +=
          !Stable && public_clusters.size() > 1U ? 1U : 0U;
      stats.merged_numerical_source_tasks += Stable ? 0U : commands.size();
      stats.maximum_source_tasks_per_transport_batch = std::max<std::uint64_t>(
          stats.maximum_source_tasks_per_transport_batch, commands.size());
      stats.maximum_source_tasks_per_numerical_batch = std::max<std::uint64_t>(
          stats.maximum_source_tasks_per_numerical_batch,
          Stable ? 1U : commands.size());
    }

    if (!results.ok()) {
      for (EvaluationCommand& command : commands) {
        command.completion.set_value(
            StatusOr<LinkageEvaluationResult>(results.status()));
      }
    } else if (results.value().size() != commands.size()) {
      const Status mismatch =
          Status::Internal("linkage backend returned a different task count");
      for (EvaluationCommand& command : commands) {
        command.completion.set_value(
            StatusOr<LinkageEvaluationResult>(mismatch));
      }
    } else {
      auto& values = results.value();
      for (std::size_t index = 0U; index < commands.size(); ++index) {
        commands[index].completion.set_value(
            StatusOr<LinkageEvaluationResult>(std::move(values[index])));
      }
    }
    evaluations.clear();
    public_clusters.clear();
    commands.clear();
    return true;
  }

  bool ProcessOneUnregistration(Worker* worker) {
    UnregisterCommand command;
    {
      std::lock_guard<std::mutex> lock(worker->mutex);
      if (worker->unregisters.empty()) {
        return false;
      }
      const std::shared_ptr<SchedulerClusterState>& state =
          worker->unregisters.front().state;
      if (state->queued_tasks.load(std::memory_order_acquire) != 0U ||
          (!state->poisoned.load(std::memory_order_acquire) &&
           state->enqueued_generation.load(std::memory_order_acquire) <
               state->submitted_generation.load(std::memory_order_acquire))) {
        return false;
      }
      command = std::move(worker->unregisters.front());
      worker->unregisters.pop_front();
      worker->ConsumeUnsafe();
    }
    const Status status = worker->backend->Unregister(command.state->backend);
    if (status.ok()) {
      command.state->live.store(false, std::memory_order_release);
    }
    command.completion.set_value(status);
    return true;
  }

  bool ProcessOneResourceQuery(Worker* worker) {
    ResourceStatsCommand command;
    {
      std::lock_guard<std::mutex> lock(worker->mutex);
      if (worker->resource_queries.empty()) {
        return false;
      }
      command = std::move(worker->resource_queries.front());
      worker->resource_queries.pop_front();
      worker->ConsumeUnsafe();
    }
    command.completion.set_value(worker->backend->ResourceStats());
    return true;
  }

  LinkageSchedulerOptions options;
  ByteAdmissionGate registration_admission;
  std::vector<std::unique_ptr<Worker>> workers;

  mutable std::mutex clusters_mutex;
  std::uint64_t next_handle = 1U;
  std::unordered_map<std::uint64_t, std::shared_ptr<SchedulerClusterState>>
      clusters;

  mutable std::mutex stats_mutex;
  LinkageSchedulerStats stats;
};

BoundedLinkageScheduler::BoundedLinkageScheduler(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

BoundedLinkageScheduler::~BoundedLinkageScheduler() = default;

StatusOr<std::unique_ptr<BoundedLinkageScheduler>>
BoundedLinkageScheduler::Create(
    std::vector<std::unique_ptr<LinkageWorkerBackend>> backends,
    LinkageSchedulerOptions options) {
  if (const Status status = ValidateOptions(options, backends.size());
      !status.ok()) {
    return status;
  }
  for (const auto& backend : backends) {
    if (backend == nullptr) {
      return Status::InvalidArgument(
          "linkage scheduler backend must not be null");
    }
  }
  auto impl = std::make_unique<Impl>(options);
  impl->workers.reserve(backends.size());
  for (auto& backend : backends) {
    impl->workers.push_back(std::make_unique<typename Impl::Worker>(
        impl.get(), std::move(backend), options));
  }
  // Every worker, queue and backend is fully constructed before any thread can
  // observe the owning Impl.
  for (const auto& worker : impl->workers) {
    if (const Status status = worker->Start(); !status.ok()) {
      return status;
    }
  }
  return std::unique_ptr<BoundedLinkageScheduler>(
      new BoundedLinkageScheduler(std::move(impl)));
}

StatusOr<LinkageClusterHandle> BoundedLinkageScheduler::RegisterCluster(
    LinkageClusterRegistrationView registration) {
  auto layout = ValidateRegistration(registration);
  if (!layout.ok()) return layout.status();
  const std::uint64_t reserved_bytes = layout.value().bytes;
  if (const Status status =
          impl_->registration_admission.Acquire(reserved_bytes);
      !status.ok()) {
    return status;
  }
  RegistrationCredit credit(impl_->registration_admission, reserved_bytes);
  LinkageClusterHandle handle;
  const auto abandon = [&] {
    if (handle.value == 0U) return;
    std::lock_guard<std::mutex> lock(impl_->clusters_mutex);
    impl_->clusters.erase(handle.value);
  };
  try {
    auto copied = CopyRegistration(registration, layout.value());
    auto state = std::make_shared<SchedulerClusterState>();
    state->live.store(false, std::memory_order_relaxed);
    state->dimension = layout.value().dimension;
    state->columns = layout.value().columns;
    state->reserved_bytes = reserved_bytes;
    state->worker = static_cast<std::size_t>(registration.affinity_key %
                                            impl_->workers.size());
    RegisterCommand command;
    command.registration = std::move(copied);
    auto completion = command.completion.get_future();
    // Finish frontend allocations before the backend takes ownership. The
    // caller cannot use this handle until registration returns successfully.
    {
      std::lock_guard<std::mutex> lock(impl_->clusters_mutex);
      handle.value = impl_->next_handle++;
      impl_->clusters.emplace(handle.value, state);
    }
    if (!impl_->workers[state->worker]->Push(
            &impl_->workers[state->worker]->registrations, std::move(command))) {
      abandon();
      return Status::FailedPrecondition("linkage scheduler is closed");
    }
    auto registered = completion.get();
    if (!registered.ok()) {
      abandon();
      return registered.status();
    }
    state->backend = registered.value();
    state->live.store(true, std::memory_order_release);
    credit.Transfer();
    {
      std::lock_guard<std::mutex> lock(impl_->stats_mutex);
      impl_->stats.registration_columns += layout.value().columns;
    }
    return handle;
  } catch (const std::bad_alloc&) {
    abandon();
    return Status::ResourceExhausted("linkage registration exhausted host memory");
  }

}

StatusOr<LinkageUpdateTicket> BoundedLinkageScheduler::SubmitUpdate(
    LinkageClusterHandle cluster, LinkageReconstructionUpdateView update) {
  if (cluster.value == 0U || update.columns == nullptr ||
      update.reconstructions.values == nullptr ||
      update.reconstructions.columns == 0U) {
    return Status::InvalidArgument(
        "linkage update requires columns and reconstruction values");
  }
  if (update.reconstructions.columns > impl_->options.maximum_update_columns) {
    return Status::ResourceExhausted(
        "one linkage update exceeds the bounded column admission limit");
  }
  std::shared_ptr<SchedulerClusterState> state;
  {
    std::lock_guard<std::mutex> lock(impl_->clusters_mutex);
    const auto found = impl_->clusters.find(cluster.value);
    if (found == impl_->clusters.end()) {
      return Status::FailedPrecondition(
          "linkage update references an unavailable cluster");
    }
    state = found->second;
  }
  std::unique_lock<std::mutex> admission(state->admission_mutex);
  if (!state->live.load(std::memory_order_acquire) ||
      state->retiring.load(std::memory_order_acquire) ||
      state->poisoned.load(std::memory_order_acquire) ||
      update.reconstructions.rows != state->dimension) {
    return Status::FailedPrecondition(
        "linkage update references an unavailable cluster");
  }
  LinkageBackendUpdate backend_update;
  backend_update.cluster = state->backend;
  backend_update.generation =
      state->submitted_generation.fetch_add(1U, std::memory_order_acq_rel) + 1U;
  const std::uint64_t generation = backend_update.generation;
  backend_update.dimension = state->dimension;
  backend_update.columns.assign(
      update.columns, update.columns + update.reconstructions.columns);
#ifndef NDEBUG
  for (std::uint32_t column : backend_update.columns) {
    assert(column < state->columns);
  }
#endif
  const std::size_t value_count = static_cast<std::size_t>(state->dimension) *
                                  update.reconstructions.columns;
  backend_update.reconstructions.assign(
      update.reconstructions.values,
      update.reconstructions.values + value_count);

  UpdateCommand command;
  command.public_cluster = cluster;
  command.state = state;
  command.update = std::move(backend_update);
  command.causal_sequence = state->next_causal_sequence++;
  std::shared_future<Status> completion =
      command.completion.get_future().share();
  const std::size_t worker_index = state->worker;
  if (!impl_->workers[worker_index]->Push(
          &impl_->workers[worker_index]->updates, std::move(command))) {
    state->submitted_generation.fetch_sub(1U, std::memory_order_acq_rel);
    return Status::FailedPrecondition("linkage scheduler is closed");
  }
  admission.unlock();
  {
    std::lock_guard<std::mutex> stats_lock(impl_->stats_mutex);
    impl_->stats.update_requests += 1U;
    impl_->stats.update_columns += update.reconstructions.columns;
  }
  return LinkageUpdateTicket{
      .generation = generation,
      .completion = std::move(completion),
  };
}

StatusOr<std::future<StatusOr<LinkageEvaluationResult>>>
BoundedLinkageScheduler::Submit(LinkageEvaluationTask task) {
  if (const Status status = ValidateTaskShape(task); !status.ok()) {
    return status;
  }
  if (task.node_count() > impl_->options.maximum_batch_nodes ||
      task.pair_count() > impl_->options.maximum_batch_pairs) {
    return Status::ResourceExhausted(
        "one linkage evaluation exceeds the bounded batch admission limit");
  }
  std::shared_ptr<SchedulerClusterState> state;
  {
    std::lock_guard<std::mutex> lock(impl_->clusters_mutex);
    const auto found = impl_->clusters.find(task.cluster.value);
    if (found == impl_->clusters.end()) {
      return Status::FailedPrecondition(
          "linkage task references a stale or unavailable cluster generation");
    }
    state = found->second;
  }
  std::unique_lock<std::mutex> admission(state->admission_mutex);
  if (!state->live.load(std::memory_order_acquire) ||
      state->retiring.load(std::memory_order_acquire) ||
      state->poisoned.load(std::memory_order_acquire) ||
      task.dimension != state->dimension ||
      task.required_generation !=
          state->submitted_generation.load(std::memory_order_acquire)) {
    return Status::FailedPrecondition(
        "linkage task references a stale or unavailable cluster generation");
  }
#ifndef NDEBUG
  for (std::uint32_t column : task.candidate_columns) {
    assert(column < state->columns);
  }
#endif
  EvaluationCommand command;
  command.public_cluster = task.cluster;
  command.state = state;
  command.evaluation.cluster = state->backend;
  command.evaluation.task = std::move(task);
  command.causal_sequence = state->next_causal_sequence++;
  const std::uint32_t nodes = command.evaluation.task.node_count();
  const std::uint32_t pairs = command.evaluation.task.pair_count();
  auto completion = command.completion.get_future();
  state->queued_tasks.fetch_add(1U, std::memory_order_acq_rel);
  const std::size_t worker_index = state->worker;
  const bool queued = impl_->workers[worker_index]->PushConfiguredEvaluation(
      std::move(command));
  if (!queued) {
    state->queued_tasks.fetch_sub(1U, std::memory_order_acq_rel);
    return Status::FailedPrecondition(
        "linkage scheduler is closed or the cluster is unavailable");
  }
  admission.unlock();
  {
    std::lock_guard<std::mutex> stats_lock(impl_->stats_mutex);
    impl_->stats.source_tasks_submitted += 1U;
    impl_->stats.nodes += nodes;
    impl_->stats.pairs += pairs;
  }
  return completion;
}

Status BoundedLinkageScheduler::UnregisterCluster(
    LinkageClusterHandle cluster) {
  std::shared_ptr<SchedulerClusterState> state;
  {
    std::lock_guard<std::mutex> lock(impl_->clusters_mutex);
    const auto found = impl_->clusters.find(cluster.value);
    if (found == impl_->clusters.end()) {
      return Status::FailedPrecondition(
          "linkage unregister references an unavailable cluster");
    }
    state = found->second;
  }
  std::unique_lock<std::mutex> admission(state->admission_mutex);
  if (!state->live.load(std::memory_order_acquire) ||
      state->retiring.load(std::memory_order_acquire)) {
    return Status::FailedPrecondition(
        "linkage unregister references an unavailable cluster");
  }
  state->retiring.store(true, std::memory_order_release);
  const std::size_t worker_index = state->worker;
  UnregisterCommand command;
  command.state = state;
  auto completion = command.completion.get_future();
  if (!impl_->workers[worker_index]->Push(
          &impl_->workers[worker_index]->unregisters, std::move(command))) {
    state->retiring.store(false, std::memory_order_release);
    return Status::FailedPrecondition("linkage scheduler is closed");
  }
  admission.unlock();
  const Status status = completion.get();
  if (status.ok()) {
    const std::uint64_t reserved_bytes = state->reserved_bytes;
    {
      std::lock_guard<std::mutex> lock(impl_->clusters_mutex);
      const auto current = impl_->clusters.find(cluster.value);
      if (current != impl_->clusters.end() && current->second == state) {
        impl_->clusters.erase(current);
      }
    }
    impl_->registration_admission.Release(reserved_bytes);
  } else {
    std::lock_guard<std::mutex> lock(state->admission_mutex);
    state->retiring.store(false, std::memory_order_release);
  }
  return status;
}

LinkageSchedulerStats BoundedLinkageScheduler::Stats() const {
  LinkageSchedulerStats output;
  {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    output = impl_->stats;
  }
  const ByteAdmissionStats admission = impl_->registration_admission.Stats();
  output.registration_admission_capacity_bytes = admission.capacity_bytes;
  output.registration_admission_peak_bytes = admission.peak_live_bytes;
  for (const auto& worker : impl_->workers) {
    ResourceStatsCommand command;
    auto completion = command.completion.get_future();
    const bool queued =
        worker->Push(&worker->resource_queries, std::move(command));
    assert(queued);
    if (!queued) {
      continue;
    }
    const LinkageBackendResourceStats resources = completion.get();
    output.resources.live_registration_bytes +=
        resources.live_registration_bytes;
    output.resources.idle_cache_buffers += resources.idle_cache_buffers;
    output.resources.idle_cache_bytes += resources.idle_cache_bytes;
    output.resources.idle_cache_peak_bytes += resources.idle_cache_peak_bytes;
    output.resources.idle_cache_evictions += resources.idle_cache_evictions;
    output.resources.allocation_retries += resources.allocation_retries;
  }
  return output;
}

LinkageProducerTaskPolicy BoundedLinkageScheduler::ProducerTaskPolicy()
    const noexcept {
  return impl_ != nullptr && impl_->options.stable
             ? LinkageProducerTaskPolicy::kStable
             : LinkageProducerTaskPolicy::kEvaluatorSelected;
}

}  // namespace stlqpp
