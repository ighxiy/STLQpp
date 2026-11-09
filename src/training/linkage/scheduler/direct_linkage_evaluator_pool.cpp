#include "stlqpp/training/linkage/scheduler/direct_linkage_evaluator_pool.h"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"

namespace stlqpp {
namespace {

struct DirectRegistrationLayout final {
  std::uint32_t dimension = 0U;
  std::uint32_t columns = 0U;
  std::size_t values = 0U;
  std::uint64_t bytes = 0U;
};

StatusOr<DirectRegistrationLayout> ValidateDirectRegistration(
    LinkageClusterRegistrationView input) {
  const bool have_first = input.first.columns != 0U;
  const bool have_second = input.second.columns != 0U;
  if ((!have_first && !have_second) ||
      (have_first &&
       (input.first.values == nullptr || input.first.rows == 0U)) ||
      (have_second &&
       (input.second.values == nullptr || input.second.rows == 0U))) {
    return Status::InvalidArgument(
        "direct linkage registration requires a non-empty matrix");
  }
  const std::uint32_t dimension =
      have_first ? input.first.rows : input.second.rows;
  if ((have_first && input.first.rows != dimension) ||
      (have_second && input.second.rows != dimension)) {
    return Status::InvalidArgument(
        "direct linkage registration parts have different dimensions");
  }
  const std::uint64_t columns =
      static_cast<std::uint64_t>(input.first.columns) + input.second.columns;
  if (columns > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "direct linkage registration exceeds uint32 columns");
  }
  if (columns > std::numeric_limits<std::size_t>::max() / dimension ||
      columns > std::numeric_limits<std::uint64_t>::max() / dimension /
                    sizeof(float)) {
    return Status::ResourceExhausted(
        "direct linkage registration byte extent overflows");
  }
  const std::size_t values =
      static_cast<std::size_t>(dimension) * static_cast<std::size_t>(columns);
  return DirectRegistrationLayout{
      .dimension = dimension,
      .columns = static_cast<std::uint32_t>(columns),
      .values = values,
      .bytes = static_cast<std::uint64_t>(values) * sizeof(float),
  };
}

LinkageBackendRegistration CopyDirectRegistration(
    LinkageClusterRegistrationView input, DirectRegistrationLayout layout) {
  const bool have_first = input.first.columns != 0U;
  const bool have_second = input.second.columns != 0U;
  LinkageBackendRegistration output;
  output.dimension = layout.dimension;
  output.column_count = layout.columns;
  output.reconstructions.reserve(layout.values);
  const std::uint32_t dimension = layout.dimension;
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

Status ValidateDirectTask(const LinkageEvaluationTask& task) {
  const std::uint32_t nodes = task.node_count();
  const std::uint32_t pairs = task.pair_count();
  if (task.cluster.value == 0U || task.dimension == 0U || nodes == 0U ||
      pairs == 0U || task.row_offsets.front() != 0U ||
      task.row_offsets.back() != pairs ||
      task.node_vectors.size() !=
          static_cast<std::size_t>(task.dimension) * nodes) {
    return Status::InvalidArgument(
        "direct linkage task has an invalid matrix or CSR shape");
  }
  for (std::size_t row = 1U; row < task.row_offsets.size(); ++row) {
    const bool empty_root_only_row =
        task.kind == LinkageEvaluationKind::kRootOnlyInitialization &&
        task.row_offsets[row] == task.row_offsets[row - 1U];
    if (task.row_offsets[row] < task.row_offsets[row - 1U] ||
        (task.row_offsets[row] == task.row_offsets[row - 1U] &&
         !empty_root_only_row)) {
      return Status::InvalidArgument(
          "direct linkage task has an empty rsd row");
    }
  }
  const std::uint32_t order_count =
      task.kind == LinkageEvaluationKind::kEdgeRsd ? nodes : 1U;
  return ValidateIcmLayerOrders<0>(order_count, task.icm_layer_orders, task.m);
}

template <typename T>
std::future<T> ReadyFuture(T value) {
  std::promise<T> promise;
  auto future = promise.get_future();
  promise.set_value(std::move(value));
  return future;
}

std::shared_future<Status> ReadySharedStatus(Status value) {
  std::promise<Status> promise;
  auto future = promise.get_future().share();
  promise.set_value(std::move(value));
  return future;
}

}  // namespace

struct DirectLinkageEvaluatorPool::Impl final {
  struct Cluster final {
    std::size_t backend_index = 0U;
    LinkageBackendCluster backend_cluster = 0U;
    std::uint32_t dimension = 0U;
    std::uint32_t columns = 0U;
    std::uint64_t generation = 0U;
    std::uint64_t registration_bytes = 0U;
    std::mutex operation_mutex;
    std::condition_variable operation_cv;
    bool evaluation_in_flight = false;
    bool poisoned = false;
    bool lane_retired = false;
    Status poison_status;
  };

  Impl(std::vector<std::unique_ptr<LinkageWorkerBackend>> input_backends,
       DirectLinkageEvaluatorPoolOptions input_options,
       std::shared_ptr<ExclusiveIndexPool> input_shared_lanes)
      : backends(std::move(input_backends)),
        options(input_options),
        lanes(std::move(input_shared_lanes)) {
    stats.registration_admission_capacity_bytes =
        options.maximum_live_registration_bytes;
  }

  std::vector<std::unique_ptr<LinkageWorkerBackend>> backends;
  DirectLinkageEvaluatorPoolOptions options;
  std::shared_ptr<ExclusiveIndexPool> lanes;
  mutable std::mutex mutex;
  std::condition_variable available_cv;
  bool closed = false;
  std::uint64_t next_handle = 1U;
  std::uint64_t live_registration_bytes = 0U;
  std::unordered_map<std::uint64_t, std::shared_ptr<Cluster>> clusters;
  LinkageEvaluationServiceStats stats;
};

DirectLinkageEvaluatorPool::DirectLinkageEvaluatorPool(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

DirectLinkageEvaluatorPool::~DirectLinkageEvaluatorPool() {
  if (impl_ == nullptr) {
    return;
  }
  std::vector<std::shared_ptr<typename Impl::Cluster>> live;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->closed = true;
    live.reserve(impl_->clusters.size());
    for (const auto& item : impl_->clusters) {
      live.push_back(item.second);
    }
    impl_->clusters.clear();
  }
  impl_->available_cv.notify_all();
  for (const auto& item : live) {
    std::unique_lock<std::mutex> operation(item->operation_mutex);
    item->operation_cv.wait(operation,
                            [&] { return !item->evaluation_in_flight; });
    (void)impl_->backends[item->backend_index]->Unregister(
        item->backend_cluster);
    if (!item->lane_retired) {
      impl_->lanes->Release(item->backend_index);
    }
  }
}

StatusOr<std::unique_ptr<DirectLinkageEvaluatorPool>>
DirectLinkageEvaluatorPool::Create(
    std::vector<std::unique_ptr<LinkageWorkerBackend>> backends,
    DirectLinkageEvaluatorPoolOptions options,
    std::shared_ptr<ExclusiveIndexPool> shared_lanes) {
  if (backends.empty() || options.maximum_live_registration_bytes == 0U ||
      options.maximum_task_nodes == 0U || options.maximum_task_pairs == 0U ||
      options.maximum_update_columns == 0U) {
    return Status::InvalidArgument(
        "direct linkage pool requires nonzero bounded resources");
  }
  for (const auto& backend : backends) {
    if (backend == nullptr) {
      return Status::InvalidArgument(
          "direct linkage pool backend must not be null");
    }
    if (options.asynchronous_evaluation &&
        !backend->SupportsAsynchronousEvaluation()) {
      return Status::InvalidArgument(
          "direct asynchronous evaluation requires asynchronous backends");
    }
  }
  if (shared_lanes == nullptr) {
    auto lanes = ExclusiveIndexPool::Create(backends.size());
    if (!lanes.ok()) {
      return lanes.status();
    }
    shared_lanes = std::move(lanes).value();
  } else if (shared_lanes->size() != backends.size()) {
    return Status::InvalidArgument(
        "direct linkage backend count does not match shared lane count");
  }
  return std::unique_ptr<DirectLinkageEvaluatorPool>(
      new DirectLinkageEvaluatorPool(std::make_unique<Impl>(
          std::move(backends), options, std::move(shared_lanes))));
}

StatusOr<LinkageClusterHandle> DirectLinkageEvaluatorPool::RegisterCluster(
    LinkageClusterRegistrationView registration) {
  auto layout = ValidateDirectRegistration(registration);
  if (!layout.ok()) return layout.status();
  const std::uint64_t bytes = layout.value().bytes;
  if (bytes > impl_->options.maximum_live_registration_bytes) {
    return Status::ResourceExhausted(
        "one direct linkage registration exceeds phase capacity");
  }
  {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->available_cv.wait(lock, [&] {
      return impl_->closed ||
             bytes <= impl_->options.maximum_live_registration_bytes -
                          impl_->live_registration_bytes;
    });
    if (impl_->closed) {
      return Status::FailedPrecondition("direct linkage pool is closed");
    }
    impl_->live_registration_bytes += bytes;
    impl_->stats.registration_admission_peak_bytes =
        std::max(impl_->stats.registration_admission_peak_bytes,
                 impl_->live_registration_bytes);
  }
  bool reservation_active = true;
  const auto release_reservation = [&]() {
    if (!reservation_active) return;
    {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      impl_->live_registration_bytes -= bytes;
    }
    reservation_active = false;
    impl_->available_cv.notify_all();
  };
  // Allocate and copy only after admission. While waiting above, the caller's
  // immutable source remains the sole owner and no unaccounted duplicate is
  // live in this pool.
  LinkageBackendRegistration copied;
  try {
    copied = CopyDirectRegistration(registration, layout.value());
  } catch (const std::bad_alloc&) {
    release_reservation();
    return Status::ResourceExhausted(
        "direct linkage registration copy exhausted host memory");
  }
  auto acquired = impl_->lanes->Acquire();
  if (!acquired.ok()) {
    release_reservation();
    return acquired.status();
  }
  const std::size_t backend_index = acquired.value();
  auto backend_cluster =
      impl_->backends[backend_index]->Register(std::move(copied));
  if (!backend_cluster.ok()) {
    impl_->lanes->Release(backend_index);
    release_reservation();
    return backend_cluster.status();
  }
  auto state = std::make_shared<typename Impl::Cluster>();
  state->backend_index = backend_index;
  state->backend_cluster = backend_cluster.value();
  state->dimension = layout.value().dimension;
  state->columns = layout.value().columns;
  state->registration_bytes = bytes;
  LinkageClusterHandle handle;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    handle.value = impl_->next_handle++;
    impl_->clusters.emplace(handle.value, state);
    impl_->stats.registration_columns += state->columns;
  }
  reservation_active = false;
  return handle;
}

StatusOr<LinkageUpdateTicket> DirectLinkageEvaluatorPool::SubmitUpdate(
    LinkageClusterHandle cluster, LinkageReconstructionUpdateView update) {
  if (cluster.value == 0U || update.columns == nullptr ||
      update.reconstructions.values == nullptr ||
      update.reconstructions.columns == 0U ||
      update.reconstructions.columns > impl_->options.maximum_update_columns) {
    return Status::InvalidArgument(
        "direct linkage update has invalid or excessive columns");
  }
  std::shared_ptr<typename Impl::Cluster> state;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->clusters.find(cluster.value);
    if (found == impl_->clusters.end()) {
      return Status::FailedPrecondition(
          "direct linkage update references an unavailable cluster");
    }
    state = found->second;
  }
  std::unique_lock<std::mutex> operation(state->operation_mutex);
  state->operation_cv.wait(operation,
                           [&] { return !state->evaluation_in_flight; });
  if (state->poisoned) {
    return Status::FailedPrecondition(
        "direct linkage lane is unavailable after asynchronous failure: " +
        state->poison_status.message());
  }
  if (update.reconstructions.rows != state->dimension) {
    return Status::InvalidArgument(
        "direct linkage update dimension does not match registration");
  }
  LinkageBackendUpdate backend_update;
  backend_update.cluster = state->backend_cluster;
  backend_update.generation = ++state->generation;
  backend_update.dimension = state->dimension;
  backend_update.columns.assign(
      update.columns, update.columns + update.reconstructions.columns);
  for (const std::uint32_t column : backend_update.columns) {
    if (column >= state->columns) {
      --state->generation;
      return Status::InvalidArgument(
          "direct linkage update column is outside registration");
    }
  }
  const std::size_t values = static_cast<std::size_t>(state->dimension) *
                             update.reconstructions.columns;
  backend_update.reconstructions.assign(update.reconstructions.values,
                                        update.reconstructions.values + values);
  const Status status = impl_->backends[state->backend_index]->ApplyUpdates(
      {&backend_update, 1U});
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->stats.update_requests += 1U;
    impl_->stats.update_batches += 1U;
    impl_->stats.update_columns += update.reconstructions.columns;
  }
  return LinkageUpdateTicket{
      .generation = state->generation,
      .completion = ReadySharedStatus(status),
  };
}

StatusOr<std::future<StatusOr<LinkageEvaluationResult>>>
DirectLinkageEvaluatorPool::Submit(LinkageEvaluationTask task) {
  if (const Status status = ValidateDirectTask(task); !status.ok()) {
    return status;
  }
  if (task.node_count() > impl_->options.maximum_task_nodes ||
      task.pair_count() > impl_->options.maximum_task_pairs) {
    return Status::ResourceExhausted(
        "one direct linkage task exceeds the evaluator workspace");
  }
  std::shared_ptr<typename Impl::Cluster> state;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->clusters.find(task.cluster.value);
    if (found == impl_->clusters.end()) {
      return Status::FailedPrecondition(
          "direct linkage task references an unavailable cluster");
    }
    state = found->second;
  }
  const std::uint32_t nodes = task.node_count();
  const std::uint32_t pairs = task.pair_count();
  std::unique_lock<std::mutex> operation(state->operation_mutex);
  state->operation_cv.wait(operation,
                           [&] { return !state->evaluation_in_flight; });
  if (state->poisoned) {
    return Status::FailedPrecondition(
        "direct linkage lane is unavailable after asynchronous failure: " +
        state->poison_status.message());
  }
  if (task.dimension != state->dimension ||
      task.required_generation != state->generation) {
    return Status::FailedPrecondition(
        "direct linkage task references stale reconstruction state");
  }
  for (const std::uint32_t column : task.candidate_columns) {
    if (column >= state->columns) {
      return Status::InvalidArgument(
          "direct linkage candidate is outside registration");
    }
  }
  LinkageBackendEvaluation evaluation;
  evaluation.cluster = state->backend_cluster;
  evaluation.task = std::move(task);
  auto unpack = [](StatusOr<std::vector<LinkageEvaluationResult>> evaluated)
      -> StatusOr<LinkageEvaluationResult> {
    if (!evaluated.ok()) {
      return evaluated.status();
    }
    if (evaluated.value().size() != 1U) {
      return Status::Internal(
          "direct linkage backend returned a different task count");
    }
    return std::move(evaluated.value().front());
  };
  std::future<StatusOr<LinkageEvaluationResult>> future;
  if (impl_->options.asynchronous_evaluation) {
    auto promise =
        std::make_shared<std::promise<StatusOr<LinkageEvaluationResult>>>();
    future = promise->get_future();
    state->evaluation_in_flight = true;
    operation.unlock();
    const Status begin = impl_->backends[state->backend_index]->BeginEvaluate(
        {&evaluation, 1U},
        [state, promise, unpack, lanes = impl_->lanes](
            StatusOr<std::vector<LinkageEvaluationResult>> evaluated) mutable {
          StatusOr<LinkageEvaluationResult> result =
              unpack(std::move(evaluated));
          {
            std::lock_guard<std::mutex> lock(state->operation_mutex);
            if (!state->evaluation_in_flight) return;
            if (!result.ok() &&
                result.status().code() == StatusCode::kInternal) {
              state->poisoned = true;
              state->lane_retired = true;
              state->poison_status = result.status();
              lanes->Retire(state->backend_index);
            }
            promise->set_value(std::move(result));
            state->evaluation_in_flight = false;
          }
          state->operation_cv.notify_all();
        });
    if (!begin.ok()) {
      {
        std::lock_guard<std::mutex> lock(state->operation_mutex);
        state->evaluation_in_flight = false;
      }
      state->operation_cv.notify_all();
      return begin;
    }
  } else {
    future = ReadyFuture(unpack(
        impl_->backends[state->backend_index]->EvaluatePrevalidatedDirect(
            {&evaluation, 1U})));
  }
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->stats.source_tasks_submitted += 1U;
    impl_->stats.transport_batches += 1U;
    impl_->stats.numerical_batches += 1U;
    impl_->stats.maximum_source_tasks_per_transport_batch = 1U;
    impl_->stats.maximum_source_tasks_per_numerical_batch = 1U;
    impl_->stats.nodes += nodes;
    impl_->stats.pairs += pairs;
  }
  return future;
}

Status DirectLinkageEvaluatorPool::UnregisterCluster(
    LinkageClusterHandle cluster) {
  std::shared_ptr<typename Impl::Cluster> state;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->clusters.find(cluster.value);
    if (found == impl_->clusters.end()) {
      return Status::FailedPrecondition(
          "direct linkage unregister references an unavailable cluster");
    }
    state = found->second;
    impl_->clusters.erase(found);
  }
  std::unique_lock<std::mutex> operation(state->operation_mutex);
  state->operation_cv.wait(operation,
                           [&] { return !state->evaluation_in_flight; });
  const Status status =
      impl_->backends[state->backend_index]->Unregister(state->backend_cluster);
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->live_registration_bytes -= state->registration_bytes;
  }
  if (!state->lane_retired) {
    impl_->lanes->Release(state->backend_index);
  }
  impl_->available_cv.notify_all();
  return status;
}

LinkageEvaluationServiceStats DirectLinkageEvaluatorPool::Stats() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  LinkageEvaluationServiceStats output = impl_->stats;
  output.resources.live_registration_bytes = impl_->live_registration_bytes;
  for (const auto& backend : impl_->backends) {
    const LinkageBackendResourceStats value = backend->ResourceStats();
    output.resources.idle_cache_buffers += value.idle_cache_buffers;
    output.resources.idle_cache_bytes += value.idle_cache_bytes;
    output.resources.idle_cache_peak_bytes += value.idle_cache_peak_bytes;
    output.resources.idle_cache_evictions += value.idle_cache_evictions;
    output.resources.allocation_retries += value.allocation_retries;
  }
  return output;
}

bool DirectLinkageEvaluatorPool::SupportsAsynchronousPreflush() const noexcept {
  return impl_ != nullptr && impl_->options.asynchronous_evaluation;
}

}  // namespace stlqpp
