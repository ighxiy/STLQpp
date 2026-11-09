#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "stlqpp/core/status_or.h"
#include "stlqpp/training/basic/large_root_algebra.h"
#include "stlqpp/training/least_squares_cost_mode.h"
#include "stlqpp/training/linkage/root_only/root_only_linkage_device_plan.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/scheduler/bounded_best_fit_cache.h"
#include "stlqpp/training/linkage/scheduler/bounded_linkage_scheduler.h"
#include "stlqpp/training/linkage/scheduler/direct_linkage_evaluator_pool.h"
#include "stlqpp/training/linkage/scheduler/linkage_resource_statistics.h"

namespace stlqpp {

// CUDA construction policy. `allow_tf32=true` is part of the registered DEV
// production numerical semantics; the application fixes it rather than
// exposing a per-run model variant. Evaluation/update batch shapes remain the
// registered production constants owned by BoundedLinkageScheduler.
struct CudaLinkageSchedulerFactoryOptions final {
  int device = 0;
  bool allow_tf32 = true;
  // This controls scheduler construction. It is independent from merge.
  // Disabled mode constructs a direct evaluator pool and zero scheduler
  // threads/backends.
  bool scheduler_enabled = true;
  // Allows several ready tasks to share one transport batch. Stable mode
  // still preserves every producer task's numerical call boundary.
  bool transport_batching_enabled = true;
  // Selects the complete deterministic scheduler/backend path at worker
  // construction. It is not sampled inside the numerical hot loop.
  bool stable = false;
  // RSD rank is cumulative: rank 3 retains the complete rank-2 candidate set
  // and appends one rank-3 deviation at every native step.
  std::uint32_t rsd_rank = 2U;
  std::uint32_t icm_rounds = 1U;
  // Zero preserves the complete numerical batch. A positive value bounds
  // evaluator-local residual/projection/output workspace, independently of
  // the producer's complete-task admission limit.
  std::uint32_t internal_chunk_max_pairs = 0U;
  // Per-worker numerical workspace planning hint. The complete task maps,
  // shared model, registrations and caches are accounted separately.
  std::uint32_t numerical_workspace_budget_mebibytes = 0U;
  LeastSquaresCostMode least_squares_cost_mode =
      LeastSquaresCostMode::kNormalEquation;
  std::size_t worker_count = 4U;
  std::size_t direct_worker_count = 32U;
  // The producer's normal flush target and the direct lane's initial
  // workspace reservation. A complete final row may cross this target; the
  // direct service grows that lane on demand.
  std::uint32_t direct_maximum_task_pairs = 8192U;
  // Source-derived upper bound for one complete producer task. This may be
  // larger than the normal flush target above, but never larger than the
  // scheduler's 32,768-pair hard boundary. Memory ownership reserves the
  // difference for every concurrently active direct lane.
  std::uint32_t direct_admitted_maximum_task_pairs = 32768U;
  // DEV async_eval_enabled is a direct-only capability. A zero budget keeps
  // direct evaluation synchronous; scheduler construction ignores it.
  std::uint32_t direct_async_pinned_mebibytes = 0U;
  // Device bytes not materialized by the initial direct workspace but
  // reachable at the admitted complete-row task bound.
  std::uint64_t direct_deferred_workspace_bytes_per_worker = 0U;
  std::size_t queue_capacity = 256U;
  BestFitCacheLimits worker_rfull_cache_limits;
  std::uint64_t minimum_safety_reserve_bytes = 512ULL * 1024ULL * 1024ULL;
  // Optional experiment/deployment ceiling. Zero retains automatic sizing
  // from the memory remaining after every resident workspace is constructed.
  std::uint64_t maximum_live_registration_bytes = 0U;
};

struct CudaLinkageSchedulerResourcePlan final {
  int device = 0;
  std::size_t worker_count = 0U;
  std::size_t scheduler_backend_count = 0U;
  std::size_t direct_backend_count = 0U;
  std::uint64_t visible_total_bytes = 0U;
  std::uint64_t free_before_construction_bytes = 0U;
  std::uint64_t shared_model_logical_bytes = 0U;
  std::uint64_t shared_model_resident_bytes = 0U;
  std::uint64_t worker_resident_bytes = 0U;
  std::uint64_t free_after_workspace_reservation_bytes = 0U;
  std::uint64_t idle_rfull_reserve_bytes = 0U;
  std::uint64_t deferred_workspace_reserve_bytes = 0U;
  std::uint64_t runtime_safety_reserve_bytes = 0U;
  std::uint64_t live_registration_capacity_bytes = 0U;
};

[[nodiscard]] LinkageResourceStatistics ObserveCudaLinkageResources(
    const CudaLinkageSchedulerResourcePlan& plan,
    const LinkageSchedulerStats& scheduler) noexcept;

struct CudaLinkageSchedulerFactoryResult final {
  std::unique_ptr<LinkageEvaluationService> evaluator;
  // Present only for a direct pool. Ordinary virtual-root encoding uses the
  // same lane leases so scheduler-off owns exactly direct_worker_count CUDA
  // contexts rather than two independent pools.
  std::shared_ptr<ExclusiveIndexPool> shared_direct_lanes;
  CudaLinkageSchedulerResourcePlan resources;
};

// Constructs the complete production CUDA pool before starting any scheduler
// thread. Every worker workspace is materialized at its admitted high-water
// extent, after which the remaining device memory is partitioned into bounded
// idle R_full, runtime safety and live-registration ownership.
template <int M>
[[nodiscard]] StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaLinkageScheduler(const LinkageCodebookAlgebra<M>& edge_algebra,
                           CudaLinkageSchedulerFactoryOptions options = {});

template <int M>
[[nodiscard]] StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaDirectLinkageEvaluator(
    const LinkageCodebookAlgebra<M>& edge_algebra,
    const std::vector<std::shared_ptr<CudaContext>>& contexts,
    CudaLinkageSchedulerFactoryOptions options = {});

// Initialization owns only C_root. This factory constructs workers that
// contain the root-only device plan and cannot execute rsd tasks.
template <int M>
[[nodiscard]] StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaRootOnlyLinkageScheduler(
    const RootCodebooks<M>& codebooks, const F32GemmBackend& gemm,
    CudaLinkageSchedulerFactoryOptions options = {},
    CudaRootOnlyLinkageModelOptions model_options = {});

extern template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaLinkageScheduler(const LinkageCodebookAlgebra<5>&,
                           CudaLinkageSchedulerFactoryOptions);
extern template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaLinkageScheduler(const LinkageCodebookAlgebra<10>&,
                           CudaLinkageSchedulerFactoryOptions);
extern template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaDirectLinkageEvaluator(
    const LinkageCodebookAlgebra<5>&,
    const std::vector<std::shared_ptr<CudaContext>>&,
    CudaLinkageSchedulerFactoryOptions);
extern template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaDirectLinkageEvaluator(
    const LinkageCodebookAlgebra<10>&,
    const std::vector<std::shared_ptr<CudaContext>>&,
    CudaLinkageSchedulerFactoryOptions);
extern template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaRootOnlyLinkageScheduler(const RootCodebooks<5>&,
                                   const F32GemmBackend&,
                                   CudaLinkageSchedulerFactoryOptions,
                                   CudaRootOnlyLinkageModelOptions);
extern template StatusOr<CudaLinkageSchedulerFactoryResult>
CreateCudaRootOnlyLinkageScheduler(const RootCodebooks<10>&,
                                   const F32GemmBackend&,
                                   CudaLinkageSchedulerFactoryOptions,
                                   CudaRootOnlyLinkageModelOptions);

}  // namespace stlqpp
