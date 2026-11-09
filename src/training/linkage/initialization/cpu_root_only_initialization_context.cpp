#include "stlqpp/training/linkage/initialization/cpu_root_only_initialization_context.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "stlqpp/training/basic/large_root_algebra.h"
#include "stlqpp/training/linkage/candidates/cpu_exact_parent_candidates.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_cluster_builder.h"
#include "stlqpp/training/linkage/root_only/root_only_initialization_cpu_tail.h"
#include "stlqpp/training/linkage/scheduler/cpu_linkage_worker_backend.h"
#include "stlqpp/training/linkage/scheduler/direct_linkage_evaluator_pool.h"

namespace stlqpp {
namespace {

template <int M>
class CpuRootOnlyInitializationTopologyContext final
    : public RootOnlyInitializationTopologyContext<M> {
 public:
  static StatusOr<std::unique_ptr<RootOnlyInitializationTopologyContext<M>>>
  Create(const RootCodebooks<M>& source_codebooks, const F32GemmBackend& gemm,
         const SpdSolverF32& solver, const ParallelForExecutor& parallel,
         const CpuRootOnlyInitializationContextOptions& options) {
    if (options.outer_worker_count == 0U || options.direct_lane_count == 0U ||
        options.maximum_registration_and_workspace_bytes == 0U ||
        options.maximum_task_nodes == 0U || options.maximum_task_pairs == 0U ||
        options.maximum_update_columns == 0U) {
      return Status::InvalidArgument(
          "CPU root-only initialization requires bounded workers");
    }
    auto codebooks = RootCodebooks<M>::Create(source_codebooks.view());
    if (!codebooks.ok()) return codebooks.status();
    auto algebra = BasicCodebookAlgebra<M>::Create(codebooks.value(), gemm, &parallel);
    if (!algebra.ok()) return algebra.status();
    auto context = std::unique_ptr<CpuRootOnlyInitializationTopologyContext<M>>(
        new CpuRootOnlyInitializationTopologyContext<M>(
            std::move(codebooks).value(), std::move(algebra).value(),
            gemm, solver, options.direct_lane_count));
    if (const Status status = context->Initialize(parallel, options);
        !status.ok()) {
      return status;
    }
    return std::unique_ptr<RootOnlyInitializationTopologyContext<M>>(
        std::move(context));
  }

  [[nodiscard]] ArrayView<RootOnlyInitializationClusterExecutor<M>*>
  executors() noexcept override {
    return executor_views_;
  }

  [[nodiscard]] std::optional<LinkageResourceStatistics> resource_statistics()
      const noexcept override {
    const LinkageEvaluationServiceStats stats = evaluator_->Stats();
    LinkageResourceStatistics output;
    output.direct_backend_count = direct_lane_count_;
    output.live_registration_capacity_bytes =
        stats.registration_admission_capacity_bytes;
    output.live_registration_peak_bytes =
        stats.registration_admission_peak_bytes;
    output.deferred_workspace_reserve_bytes = workspace_reserve_bytes_;
    output.evaluation_source_tasks = stats.source_tasks_submitted;
    output.evaluation_transport_batches = stats.transport_batches;
    output.evaluation_cross_cluster_transport_batches =
        stats.cross_cluster_transport_batches;
    output.evaluation_numerical_batches = stats.numerical_batches;
    output.evaluation_cross_cluster_numerical_batches =
        stats.cross_cluster_numerical_batches;
    output.merged_numerical_source_tasks = stats.merged_numerical_source_tasks;
    output.maximum_source_tasks_per_transport_batch =
        stats.maximum_source_tasks_per_transport_batch;
    output.maximum_source_tasks_per_numerical_batch =
        stats.maximum_source_tasks_per_numerical_batch;
    output.evaluation_nodes = stats.nodes;
    output.evaluation_pairs = stats.pairs;
    output.update_requests = stats.update_requests;
    output.update_batches = stats.update_batches;
    output.update_columns = stats.update_columns;
    return output;
  }

 private:
  CpuRootOnlyInitializationTopologyContext(
      RootCodebooks<M> codebooks, BasicCodebookAlgebra<M> algebra,
      const F32GemmBackend& gemm, const SpdSolverF32& solver,
      std::size_t direct_lane_count) noexcept
      : codebooks_(std::move(codebooks)),
        algebra_(std::move(algebra)),
        candidates_(gemm),
        gemm_(&gemm),
        solver_(&solver),
        direct_lane_count_(direct_lane_count) {}

  Status Initialize(const ParallelForExecutor& parallel,
                    const CpuRootOnlyInitializationContextOptions& options) {
    // The tail model borrows the codebooks object, not just its buffers.
    // Bind only after the owner has reached its final address in the context.
    auto tail_model = RootOnlyInitializationCpuTailModel<M>::Create(codebooks_,
                                                                   *gemm_);
    if (!tail_model.ok()) return tail_model.status();
    tail_model_ = std::move(tail_model).value();
    std::vector<std::unique_ptr<LinkageWorkerBackend>> backends;
    const std::uint64_t workspace_per_lane =
        CpuRootLinkageMatrixWorkspaceBytes<M>(codebooks_.shape().dimension,
                                              options.maximum_task_pairs,
                                              codebooks_.m());
    if (options.direct_lane_count >
        std::numeric_limits<std::uint64_t>::max() / workspace_per_lane) {
      return Status::ResourceExhausted(
          "CPU root-only matrix workspace reservation overflows");
    }
    workspace_reserve_bytes_ = workspace_per_lane * options.direct_lane_count;
    if (workspace_reserve_bytes_ >=
        options.maximum_registration_and_workspace_bytes) {
      return Status::ResourceExhausted(
          "CPU root-only matrix workspace exhausts the phase budget");
    }
    const std::uint64_t registration_capacity =
        options.maximum_registration_and_workspace_bytes -
        workspace_reserve_bytes_;
    backends.reserve(options.direct_lane_count);
    for (std::size_t lane = 0U; lane < options.direct_lane_count; ++lane) {
      backends.push_back(std::make_unique<CpuRootOnlyLinkageWorkerBackend<M>>(
          algebra_, *gemm_, parallel, options.icm_rounds));
    }
    auto evaluator = DirectLinkageEvaluatorPool::Create(
        std::move(backends),
        {
            .maximum_live_registration_bytes = registration_capacity,
            .maximum_task_nodes = options.maximum_task_nodes,
            .maximum_task_pairs = options.maximum_task_pairs,
            .maximum_update_columns = options.maximum_update_columns,
            .asynchronous_evaluation = false,
        });
    if (!evaluator.ok()) return evaluator.status();
    evaluator_ = std::move(evaluator).value();

    const std::size_t worker_count = options.outer_worker_count;
    tail_evaluators_.reserve(worker_count);
    builders_.reserve(worker_count);
    executors_.reserve(worker_count);
    executor_views_.reserve(worker_count);
    for (std::size_t worker = 0U; worker < worker_count; ++worker) {
      tail_evaluators_.push_back(
          std::make_unique<RootOnlyInitializationCpuTailEvaluator<M>>(
              *tail_model_, *gemm_, *solver_, options.icm_rounds));
      builders_.push_back(
          std::make_unique<RootOnlyInitializationClusterBuilder<M>>(
              codebooks_, candidates_, *evaluator_, *tail_evaluators_.back()));
      executors_.push_back(
          std::make_unique<RootOnlyInitializationBuilderExecutor<M>>(
              *builders_.back()));
      executor_views_.push_back(executors_.back().get());
    }
    return Status::Ok();
  }

  RootCodebooks<M> codebooks_;
  // The evaluator borrows this algebra; keep it before the evaluator owner.
  BasicCodebookAlgebra<M> algebra_;
  CpuExactParentCandidateProvider candidates_;
  const F32GemmBackend* gemm_ = nullptr;
  const SpdSolverF32* solver_ = nullptr;
  std::unique_ptr<RootOnlyInitializationCpuTailModel<M>> tail_model_;
  std::unique_ptr<LinkageEvaluationService> evaluator_;
  std::size_t direct_lane_count_ = 0U;
  std::uint64_t workspace_reserve_bytes_ = 0U;
  std::vector<std::unique_ptr<RootOnlyInitializationCpuTailEvaluator<M>>>
      tail_evaluators_;
  std::vector<std::unique_ptr<RootOnlyInitializationClusterBuilder<M>>>
      builders_;
  std::vector<std::unique_ptr<RootOnlyInitializationBuilderExecutor<M>>>
      executors_;
  std::vector<RootOnlyInitializationClusterExecutor<M>*> executor_views_;
};

}  // namespace

template <int M>
StatusOr<std::unique_ptr<RootOnlyInitializationTopologyContext<M>>>
CpuRootOnlyInitializationTopologyContextFactory<M>::Create(
    const RootCodebooks<M>& codebooks) {
  return CpuRootOnlyInitializationTopologyContext<M>::Create(
      codebooks, *gemm_, *solver_, direct_lane_parallel_, options_);
}

template class CpuRootOnlyInitializationTopologyContextFactory<0>;
template class CpuRootOnlyInitializationTopologyContextFactory<5>;
template class CpuRootOnlyInitializationTopologyContextFactory<12>;
template class CpuRootOnlyInitializationTopologyContextFactory<6>;
template class CpuRootOnlyInitializationTopologyContextFactory<10>;

}  // namespace stlqpp
