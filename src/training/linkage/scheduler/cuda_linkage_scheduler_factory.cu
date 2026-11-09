#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "stlqpp/runtime/cuda/status.h"
#include "stlqpp/training/linkage/rsd/linkage_rsd_device_plan.h"
#include "stlqpp/training/linkage/rsd/ranked_single_deviation.h"
#include "stlqpp/training/linkage/scheduler/cuda_linkage_scheduler_factory.h"
#include "stlqpp/training/linkage/scheduler/cuda_linkage_worker_backend.h"

namespace stlqpp {

LinkageResourceStatistics ObserveCudaLinkageResources(
    const CudaLinkageSchedulerResourcePlan& plan,
    const LinkageSchedulerStats& scheduler) noexcept {
  return {
      .scheduler_backend_count = plan.scheduler_backend_count,
      .direct_backend_count = plan.direct_backend_count,
      .device_visible_bytes = plan.visible_total_bytes,
      .device_free_before_construction_bytes =
          plan.free_before_construction_bytes,
      .fixed_workspace_resident_bytes =
          plan.shared_model_resident_bytes + plan.worker_resident_bytes,
      .device_free_after_workspace_bytes =
          plan.free_after_workspace_reservation_bytes,
      .idle_rfull_reserve_bytes = plan.idle_rfull_reserve_bytes,
      .deferred_workspace_reserve_bytes = plan.deferred_workspace_reserve_bytes,
      .runtime_safety_reserve_bytes = plan.runtime_safety_reserve_bytes,
      .live_registration_capacity_bytes = plan.live_registration_capacity_bytes,
      .live_registration_peak_bytes =
          scheduler.registration_admission_peak_bytes,
      .worker_idle_cache_peak_sum_bytes =
          scheduler.resources.idle_cache_peak_bytes,
      .idle_cache_evictions = scheduler.resources.idle_cache_evictions,
      .allocation_retries = scheduler.resources.allocation_retries,
      .evaluation_source_tasks = scheduler.source_tasks_submitted,
      .evaluation_transport_batches = scheduler.transport_batches,
      .evaluation_cross_cluster_transport_batches =
          scheduler.cross_cluster_transport_batches,
      .evaluation_numerical_batches = scheduler.numerical_batches,
      .evaluation_cross_cluster_numerical_batches =
          scheduler.cross_cluster_numerical_batches,
      .merged_numerical_source_tasks = scheduler.merged_numerical_source_tasks,
      .maximum_source_tasks_per_transport_batch =
          scheduler.maximum_source_tasks_per_transport_batch,
      .maximum_source_tasks_per_numerical_batch =
          scheduler.maximum_source_tasks_per_numerical_batch,
      .evaluation_nodes = scheduler.nodes,
      .evaluation_pairs = scheduler.pairs,
      .update_requests = scheduler.update_requests,
      .update_batches = scheduler.update_batches,
      .update_columns = scheduler.update_columns,
  };
}

namespace {

struct DeviceMemorySnapshot final {
  std::uint64_t free_bytes = 0U;
  std::uint64_t total_bytes = 0U;
};

StatusOr<DeviceMemorySnapshot> ReadDeviceMemory(int device) {
  if (const cudaError_t result = cudaSetDevice(device); result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice linkage factory", result);
  }
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  if (const cudaError_t result = cudaMemGetInfo(&free_bytes, &total_bytes);
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemGetInfo linkage factory", result);
  }
  return DeviceMemorySnapshot{
      .free_bytes = static_cast<std::uint64_t>(free_bytes),
      .total_bytes = static_cast<std::uint64_t>(total_bytes),
  };
}

std::uint64_t ResidentDelta(std::uint64_t before,
                            std::uint64_t after) noexcept {
  return before >= after ? before - after : 0U;
}

Status ValidateOptions(const CudaLinkageSchedulerFactoryOptions& options) {
  const std::size_t execution_workers = options.scheduler_enabled
                                            ? options.worker_count
                                            : options.direct_worker_count;
  if (execution_workers == 0U || options.queue_capacity == 0U ||
      !IsSupportedRsdRank(options.rsd_rank) ||
      options.direct_maximum_task_pairs == 0U ||
      options.direct_maximum_task_pairs > 32768U ||
      options.direct_admitted_maximum_task_pairs <
          options.direct_maximum_task_pairs ||
      options.direct_admitted_maximum_task_pairs > 32768U ||
      options.worker_rfull_cache_limits.maximum_buffers == 0U ||
      options.worker_rfull_cache_limits.maximum_bytes == 0U ||
      options.minimum_safety_reserve_bytes == 0U) {
    return Status::InvalidArgument(
        "CUDA linkage factory received zero resources");
  }
  if (execution_workers > std::numeric_limits<std::uint64_t>::max() /
                              options.worker_rfull_cache_limits.maximum_bytes) {
    return Status::ResourceExhausted(
        "CUDA linkage idle-cache reservation overflows uint64");
  }
  if (options.direct_deferred_workspace_bytes_per_worker != 0U &&
      options.direct_worker_count >
          std::numeric_limits<std::uint64_t>::max() /
              options.direct_deferred_workspace_bytes_per_worker) {
    return Status::ResourceExhausted(
        "CUDA linkage deferred-workspace reservation overflows uint64");
  }
  return Status::Ok();
}

std::uint64_t RsdDeferredWorkspaceBytes(
    std::uint32_t actual_m, std::uint32_t dimension,
    std::uint32_t initially_reserved_pairs, std::uint32_t admitted_pairs,
    std::uint32_t internal_chunk_max_pairs) noexcept {
  if (initially_reserved_pairs >= admitted_pairs) {
    return 0U;
  }
  const std::uint64_t additional_pair_maps =
      admitted_pairs - initially_reserved_pairs;
  const std::uint32_t effective_chunk = internal_chunk_max_pairs == 0U
                                            ? admitted_pairs
                                            : internal_chunk_max_pairs;
  const std::uint64_t additional_work_pairs =
      std::min(admitted_pairs, effective_chunk) -
      std::min(initially_reserved_pairs, effective_chunk);
  // Complete-task candidate/pair-node maps still grow, but the residual,
  // projection and RSD output buffers stop growing at the internal cap.
  const std::uint64_t bytes_per_pair =
      12U + 4ULL * dimension + 1030ULL * actual_m;
  return additional_pair_maps * 2U * sizeof(std::uint32_t) +
         additional_work_pairs * bytes_per_pair;
}

void ApplyNumericalWorkspaceBudget(
    std::uint32_t actual_m, std::uint32_t dimension,
    CudaLinkageSchedulerFactoryOptions* options) noexcept {
  if (options->numerical_workspace_budget_mebibytes == 0U) {
    return;
  }
  // Conservative pair-local RSD scratch estimate. This is a planning hint,
  // not a total-device-memory cap: model and complete-task maps are separate.
  const std::uint64_t bytes_per_pair =
      12U + 4ULL * dimension + 1030ULL * actual_m;
  const std::uint64_t budget =
      static_cast<std::uint64_t>(
          options->numerical_workspace_budget_mebibytes) *
      1024U * 1024U;
  const std::uint64_t budget_pairs =
      std::max<std::uint64_t>(1U, budget / bytes_per_pair);
  if (budget_pairs >= 32768U) {
    return;
  }
  const auto limit = static_cast<std::uint32_t>(budget_pairs);
  options->internal_chunk_max_pairs =
      options->internal_chunk_max_pairs == 0U
          ? limit
          : std::min(options->internal_chunk_max_pairs, limit);
}

std::uint64_t RootOnlyDeferredWorkspaceBytes(
    std::uint32_t initially_reserved_pairs,
    std::uint32_t admitted_pairs) noexcept {
  if (initially_reserved_pairs >= admitted_pairs) {
    return 0U;
  }
  // Root-only residual/solver storage is bounded by its device-plan chunk.
  // Only the two device-side pair maps span a complete producer task.
  return static_cast<std::uint64_t>(admitted_pairs - initially_reserved_pairs) *
         2U * sizeof(std::uint32_t);
}

CudaLinkageSchedulerResourcePlan MakeResourcePlan(
    const CudaLinkageSchedulerFactoryOptions& options, std::size_t worker_count,
    bool scheduled, const DeviceMemorySnapshot& before,
    const DeviceMemorySnapshot& after_shared,
    const DeviceMemorySnapshot& after_workers,
    std::uint64_t shared_model_logical_bytes, std::uint64_t idle_reserve,
    std::uint64_t deferred_workspace_reserve, std::uint64_t safety_reserve,
    std::uint64_t live_capacity) noexcept {
  return {
      .device = options.device,
      .worker_count = worker_count,
      .scheduler_backend_count = scheduled ? worker_count : 0U,
      .direct_backend_count = scheduled ? 0U : worker_count,
      .visible_total_bytes = after_workers.total_bytes,
      .free_before_construction_bytes = before.free_bytes,
      .shared_model_logical_bytes = shared_model_logical_bytes,
      .shared_model_resident_bytes =
          ResidentDelta(before.free_bytes, after_shared.free_bytes),
      .worker_resident_bytes =
          ResidentDelta(after_shared.free_bytes, after_workers.free_bytes),
      .free_after_workspace_reservation_bytes = after_workers.free_bytes,
      .idle_rfull_reserve_bytes = idle_reserve,
      .deferred_workspace_reserve_bytes = deferred_workspace_reserve,
      .runtime_safety_reserve_bytes = safety_reserve,
      .live_registration_capacity_bytes = live_capacity,
  };
}

LinkageSchedulerOptions MakePoolOptions(
    const CudaLinkageSchedulerFactoryOptions& options) {
  LinkageSchedulerOptions pool_options;
  if (options.scheduler_enabled) {
    pool_options.queue_capacity = options.queue_capacity;
    pool_options.stable = options.stable;
  } else {
    pool_options.maximum_batch_pairs = options.direct_maximum_task_pairs;
  }
  if (!options.scheduler_enabled || !options.transport_batching_enabled) {
    pool_options.target_batch_tasks = 1U;
    pool_options.maximum_batch_tasks = 1U;
    pool_options.gather_window = std::chrono::microseconds(0);
  }
  return pool_options;
}

StatusOr<CudaLinkageSchedulerFactoryResult> BuildSchedulerPool(
    std::uint32_t dimension, const DeviceMemorySnapshot& before,
    const DeviceMemorySnapshot& after_shared,
    std::uint64_t shared_model_logical_bytes,
    const CudaLinkageSchedulerFactoryOptions& options,
    LinkageSchedulerOptions scheduler_options,
    std::vector<std::unique_ptr<LinkageWorkerBackend>> backends) {
  auto after_workers_result = ReadDeviceMemory(options.device);
  if (!after_workers_result.ok()) {
    return after_workers_result.status();
  }
  const DeviceMemorySnapshot after_workers = after_workers_result.value();
  const std::uint64_t idle_reserve =
      static_cast<std::uint64_t>(options.worker_count) *
      options.worker_rfull_cache_limits.maximum_bytes;
  const std::uint64_t safety_reserve = std::max(
      options.minimum_safety_reserve_bytes, after_workers.total_bytes / 16U);
  if (idle_reserve > after_workers.free_bytes ||
      safety_reserve > after_workers.free_bytes - idle_reserve) {
    return Status::ResourceExhausted(
        "CUDA linkage workspaces leave no memory for live registrations");
  }
  const std::uint64_t automatic_live_capacity =
      after_workers.free_bytes - idle_reserve - safety_reserve;
  const std::uint64_t live_capacity =
      options.maximum_live_registration_bytes == 0U
          ? automatic_live_capacity
          : std::min(automatic_live_capacity,
                     options.maximum_live_registration_bytes);
  const std::uint64_t minimum_registration =
      static_cast<std::uint64_t>(dimension) * sizeof(float);
  if (live_capacity < minimum_registration) {
    return Status::ResourceExhausted(
        "CUDA linkage live-registration capacity is smaller than one column");
  }
  scheduler_options.maximum_live_registration_bytes = live_capacity;

  auto scheduler =
      BoundedLinkageScheduler::Create(std::move(backends), scheduler_options);
  if (!scheduler.ok()) {
    return scheduler.status();
  }
  return CudaLinkageSchedulerFactoryResult{
      .evaluator = std::move(scheduler).value(),
      .shared_direct_lanes = {},
      .resources = MakeResourcePlan(options, options.worker_count, true, before,
                                    after_shared, after_workers,
                                    shared_model_logical_bytes, idle_reserve,
                                    0U, safety_reserve, live_capacity),
  };
}

StatusOr<CudaLinkageSchedulerFactoryResult> BuildDirectPool(
    std::uint32_t dimension, const DeviceMemorySnapshot& before,
    const DeviceMemorySnapshot& after_shared,
    std::uint64_t shared_model_logical_bytes,
    const CudaLinkageSchedulerFactoryOptions& options,
    const LinkageSchedulerOptions& workspace_options,
    std::vector<std::unique_ptr<LinkageWorkerBackend>> backends) {
  auto after_workers_result = ReadDeviceMemory(options.device);
  if (!after_workers_result.ok()) {
    return after_workers_result.status();
  }
  const DeviceMemorySnapshot after_workers = after_workers_result.value();
  const std::uint64_t idle_reserve =
      static_cast<std::uint64_t>(options.direct_worker_count) *
      options.worker_rfull_cache_limits.maximum_bytes;
  const std::uint64_t deferred_workspace_reserve =
      static_cast<std::uint64_t>(options.direct_worker_count) *
      options.direct_deferred_workspace_bytes_per_worker;
  const std::uint64_t safety_reserve = std::max(
      options.minimum_safety_reserve_bytes, after_workers.total_bytes / 16U);
  if (idle_reserve > after_workers.free_bytes ||
      deferred_workspace_reserve > after_workers.free_bytes - idle_reserve ||
      safety_reserve > after_workers.free_bytes - idle_reserve -
                           deferred_workspace_reserve) {
    return Status::ResourceExhausted(
        "CUDA direct linkage workspaces leave no registration memory");
  }
  const std::uint64_t automatic_live_capacity =
      after_workers.free_bytes - idle_reserve - deferred_workspace_reserve -
      safety_reserve;
  const std::uint64_t live_capacity =
      options.maximum_live_registration_bytes == 0U
          ? automatic_live_capacity
          : std::min(automatic_live_capacity,
                     options.maximum_live_registration_bytes);
  if (live_capacity < static_cast<std::uint64_t>(dimension) * sizeof(float)) {
    return Status::ResourceExhausted(
        "CUDA direct linkage registration capacity is too small");
  }
  auto lanes = ExclusiveIndexPool::Create(options.direct_worker_count);
  if (!lanes.ok()) {
    return lanes.status();
  }
  auto shared_lanes = std::move(lanes).value();
  auto evaluator = DirectLinkageEvaluatorPool::Create(
      std::move(backends),
      {
          .maximum_live_registration_bytes = live_capacity,
          .maximum_task_nodes = workspace_options.maximum_batch_nodes,
          // DEV flushes only after appending a complete producer row.  That
          // row may cross the normal pair target, and both direct and
          // scheduled routes must admit the same unmerged task.  Keep the
          // smaller initial reservation above; the selected direct lane's
          // geometrically growing buffers cover a target-crossing task.
          .maximum_task_pairs = options.direct_admitted_maximum_task_pairs,
          .maximum_update_columns = workspace_options.maximum_update_columns,
          .asynchronous_evaluation =
              options.direct_async_pinned_mebibytes != 0U,
      },
      shared_lanes);
  if (!evaluator.ok()) {
    return evaluator.status();
  }
  return CudaLinkageSchedulerFactoryResult{
      .evaluator = std::move(evaluator).value(),
      .shared_direct_lanes = std::move(shared_lanes),
      .resources = MakeResourcePlan(
          options, options.direct_worker_count, false, before, after_shared,
          after_workers, shared_model_logical_bytes, idle_reserve,
          deferred_workspace_reserve, safety_reserve, live_capacity),
  };
}

// Only worker creation depends on the typed numerical model. Ownership moves
// into the common pool construction after every workspace has been reserved.
template <typename CreateWorker>
StatusOr<CudaLinkageSchedulerFactoryResult> BuildPool(
    std::uint32_t dimension, const DeviceMemorySnapshot& before,
    const DeviceMemorySnapshot& after_shared,
    std::uint64_t shared_model_logical_bytes,
    const CudaLinkageSchedulerFactoryOptions& options,
    CreateWorker&& create_worker) {
  const LinkageSchedulerOptions pool_options = MakePoolOptions(options);
  const CudaLinkageWorkerWorkspaceLimits workspace_limits{
      .maximum_batch_tasks = pool_options.maximum_batch_tasks,
      .maximum_batch_nodes = pool_options.maximum_batch_nodes,
      .maximum_batch_pairs = pool_options.maximum_batch_pairs,
      .maximum_update_columns = pool_options.maximum_update_columns,
      .internal_chunk_max_pairs = options.internal_chunk_max_pairs,
  };
  const std::size_t worker_count = options.scheduler_enabled
                                       ? options.worker_count
                                       : options.direct_worker_count;
  std::vector<std::unique_ptr<LinkageWorkerBackend>> backends;
  backends.reserve(worker_count);
  for (std::size_t worker = 0U; worker < worker_count; ++worker) {
    auto backend = create_worker(worker, workspace_limits);
    if (!backend.ok()) {
      return backend.status();
    }
    backends.push_back(std::move(backend).value());
  }
  return options.scheduler_enabled
             ? BuildSchedulerPool(dimension, before, after_shared,
                                  shared_model_logical_bytes, options,
                                  pool_options, std::move(backends))
             : BuildDirectPool(dimension, before, after_shared,
                               shared_model_logical_bytes, options,
                               pool_options, std::move(backends));
}

}  // namespace

template <int M>
StatusOr<CudaLinkageSchedulerFactoryResult> CreateCudaLinkageScheduler(
    const LinkageCodebookAlgebra<M>& edge_algebra,
    CudaLinkageSchedulerFactoryOptions options) {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  ApplyNumericalWorkspaceBudget(edge_algebra.shape().m,
                                edge_algebra.shape().dimension, &options);
  options.direct_deferred_workspace_bytes_per_worker =
      RsdDeferredWorkspaceBytes(edge_algebra.shape().m,
                                edge_algebra.shape().dimension,
                                options.direct_maximum_task_pairs,
                                options.direct_admitted_maximum_task_pairs,
                                options.internal_chunk_max_pairs);
  if (const Status status = ValidateOptions(options); !status.ok()) {
    return status;
  }
  auto before = ReadDeviceMemory(options.device);
  if (!before.ok()) {
    return before.status();
  }
  auto shared_model =
      CudaLinkageRsdDeviceModel<M>::Create(edge_algebra, options.device);
  if (!shared_model.ok()) {
    return shared_model.status();
  }
  auto after_shared = ReadDeviceMemory(options.device);
  if (!after_shared.ok()) {
    return after_shared.status();
  }
  const std::uint64_t shared_model_logical_bytes =
      shared_model.value()->logical_bytes();
  const auto create_worker = [&](std::size_t,
                                 CudaLinkageWorkerWorkspaceLimits limits)
      -> StatusOr<std::unique_ptr<LinkageWorkerBackend>> {
    auto backend = CudaLinkageWorkerBackend<M>::CreateRsd(
        shared_model.value(), options.device, options.allow_tf32,
        options.worker_rfull_cache_limits, options.rsd_rank,
        options.least_squares_cost_mode, options.icm_rounds);
    if (!backend.ok()) {
      return backend.status();
    }
    if (const Status status = backend.value()->ReserveWorkspace(limits);
        !status.ok()) {
      return status;
    }
    return std::unique_ptr<LinkageWorkerBackend>(std::move(backend).value());
  };
  return BuildPool(edge_algebra.shape().dimension, before.value(),
                   after_shared.value(), shared_model_logical_bytes, options,
                   create_worker);
}

template <int M>
StatusOr<CudaLinkageSchedulerFactoryResult> CreateCudaDirectLinkageEvaluator(
    const LinkageCodebookAlgebra<M>& edge_algebra,
    const std::vector<std::shared_ptr<CudaContext>>& contexts,
    CudaLinkageSchedulerFactoryOptions options) {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  options.scheduler_enabled = false;
  options.direct_worker_count = contexts.size();
  ApplyNumericalWorkspaceBudget(edge_algebra.shape().m,
                                edge_algebra.shape().dimension, &options);
  options.direct_deferred_workspace_bytes_per_worker =
      RsdDeferredWorkspaceBytes(edge_algebra.shape().m,
                                edge_algebra.shape().dimension,
                                options.direct_maximum_task_pairs,
                                options.direct_admitted_maximum_task_pairs,
                                options.internal_chunk_max_pairs);
  if (const Status status = ValidateOptions(options); !status.ok()) {
    return status;
  }
  for (const auto& context : contexts) {
    if (context == nullptr || context->device() != options.device ||
        context->allow_tf32() != options.allow_tf32) {
      return Status::InvalidArgument(
          "direct linkage context does not match factory device policy");
    }
  }
  auto before = ReadDeviceMemory(options.device);
  if (!before.ok()) {
    return before.status();
  }
  auto shared_model =
      CudaLinkageRsdDeviceModel<M>::Create(edge_algebra, options.device);
  if (!shared_model.ok()) {
    return shared_model.status();
  }
  auto after_shared = ReadDeviceMemory(options.device);
  if (!after_shared.ok()) {
    return after_shared.status();
  }
  const std::uint64_t shared_model_logical_bytes =
      shared_model.value()->logical_bytes();
  const auto create_worker = [&](std::size_t worker,
                                 CudaLinkageWorkerWorkspaceLimits limits)
      -> StatusOr<std::unique_ptr<LinkageWorkerBackend>> {
    auto backend = CudaLinkageWorkerBackend<M>::CreateRsdOnContext(
        shared_model.value(), contexts[worker],
        options.worker_rfull_cache_limits, options.rsd_rank,
        options.least_squares_cost_mode, options.icm_rounds);
    if (!backend.ok()) {
      return backend.status();
    }
    if (const Status status = backend.value()->ReserveWorkspace(limits);
        !status.ok()) {
      return status;
    }
    return std::unique_ptr<LinkageWorkerBackend>(std::move(backend).value());
  };
  return BuildPool(edge_algebra.shape().dimension, before.value(),
                   after_shared.value(), shared_model_logical_bytes, options,
                   create_worker);
}

template <int M>
StatusOr<CudaLinkageSchedulerFactoryResult> CreateCudaRootOnlyLinkageScheduler(
    const RootCodebooks<M>& codebooks, const F32GemmBackend& gemm,
    CudaLinkageSchedulerFactoryOptions options,
    CudaRootOnlyLinkageModelOptions model_options) {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  options.direct_deferred_workspace_bytes_per_worker =
      RootOnlyDeferredWorkspaceBytes(
          options.direct_maximum_task_pairs,
          options.direct_admitted_maximum_task_pairs);
  if (const Status status = ValidateOptions(options); !status.ok()) {
    return status;
  }
  auto before = ReadDeviceMemory(options.device);
  if (!before.ok()) {
    return before.status();
  }
  auto root_model = CudaRootOnlyLinkageModel<M>::Create(
      codebooks, gemm, options.device, options.allow_tf32, model_options);
  if (!root_model.ok()) {
    return root_model.status();
  }
  auto after_root = ReadDeviceMemory(options.device);
  if (!after_root.ok()) {
    return after_root.status();
  }
  const auto shared_model = std::move(root_model).value();
  const auto create_worker = [&](std::size_t,
                                 CudaLinkageWorkerWorkspaceLimits limits)
      -> StatusOr<std::unique_ptr<LinkageWorkerBackend>> {
    auto backend = CudaRootOnlyLinkageWorkerBackend<M>::CreateRootOnly(
        shared_model, options.device, options.allow_tf32,
        options.worker_rfull_cache_limits, options.icm_rounds);
    if (!backend.ok()) {
      return backend.status();
    }
    if (const Status status = backend.value()->ReserveWorkspace(limits);
        !status.ok()) {
      return status;
    }
    return std::unique_ptr<LinkageWorkerBackend>(std::move(backend).value());
  };
  return BuildPool(codebooks.shape().dimension, before.value(),
                   after_root.value(), shared_model->device_bytes(), options,
                   create_worker);
}

template StatusOr<CudaLinkageSchedulerFactoryResult> CreateCudaLinkageScheduler(
    const LinkageCodebookAlgebra<0>&, CudaLinkageSchedulerFactoryOptions);
template StatusOr<CudaLinkageSchedulerFactoryResult> CreateCudaLinkageScheduler(
    const LinkageCodebookAlgebra<5>&, CudaLinkageSchedulerFactoryOptions);
template StatusOr<CudaLinkageSchedulerFactoryResult> CreateCudaLinkageScheduler(
    const LinkageCodebookAlgebra<12>&, CudaLinkageSchedulerFactoryOptions);
template StatusOr<CudaLinkageSchedulerFactoryResult> CreateCudaLinkageScheduler(
    const LinkageCodebookAlgebra<6>&, CudaLinkageSchedulerFactoryOptions);
template StatusOr<CudaLinkageSchedulerFactoryResult> CreateCudaLinkageScheduler(
    const LinkageCodebookAlgebra<10>&, CudaLinkageSchedulerFactoryOptions);
template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaDirectLinkageEvaluator(
    const LinkageCodebookAlgebra<0>&,
    const std::vector<std::shared_ptr<CudaContext>>&,
    CudaLinkageSchedulerFactoryOptions);
template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaDirectLinkageEvaluator(
    const LinkageCodebookAlgebra<5>&,
    const std::vector<std::shared_ptr<CudaContext>>&,
    CudaLinkageSchedulerFactoryOptions);
template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaDirectLinkageEvaluator(
    const LinkageCodebookAlgebra<12>&,
    const std::vector<std::shared_ptr<CudaContext>>&,
    CudaLinkageSchedulerFactoryOptions);
template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaDirectLinkageEvaluator(
    const LinkageCodebookAlgebra<6>&,
    const std::vector<std::shared_ptr<CudaContext>>&,
    CudaLinkageSchedulerFactoryOptions);
template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaDirectLinkageEvaluator(
    const LinkageCodebookAlgebra<10>&,
    const std::vector<std::shared_ptr<CudaContext>>&,
    CudaLinkageSchedulerFactoryOptions);
template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaRootOnlyLinkageScheduler(const RootCodebooks<0>&,
                                   const F32GemmBackend&,
                                   CudaLinkageSchedulerFactoryOptions,
                                   CudaRootOnlyLinkageModelOptions);
template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaRootOnlyLinkageScheduler(const RootCodebooks<5>&,
                                   const F32GemmBackend&,
                                   CudaLinkageSchedulerFactoryOptions,
                                   CudaRootOnlyLinkageModelOptions);
template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaRootOnlyLinkageScheduler(const RootCodebooks<12>&,
                                   const F32GemmBackend&,
                                   CudaLinkageSchedulerFactoryOptions,
                                   CudaRootOnlyLinkageModelOptions);
template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaRootOnlyLinkageScheduler(const RootCodebooks<6>&,
                                   const F32GemmBackend&,
                                   CudaLinkageSchedulerFactoryOptions,
                                   CudaRootOnlyLinkageModelOptions);
template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaRootOnlyLinkageScheduler(const RootCodebooks<10>&,
                                   const F32GemmBackend&,
                                   CudaLinkageSchedulerFactoryOptions,
                                   CudaRootOnlyLinkageModelOptions);

}  // namespace stlqpp
