#include "stlqpp/training/linkage/initialization/cuda_root_only_initialization_context.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "stlqpp/runtime/cpu/blas_thread_policy.h"
#include "stlqpp/training/linkage/candidates/cpu_exact_parent_candidates.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_cluster_builder.h"
#include "stlqpp/training/linkage/root_only/root_only_initialization_cpu_tail.h"

namespace stlqpp {
namespace {

template <int M>
class CudaRootOnlyInitializationTopologyContext final
    : public RootOnlyInitializationTopologyContext<M> {
 public:
  static StatusOr<std::unique_ptr<RootOnlyInitializationTopologyContext<M>>>
  Create(const RootCodebooks<M>& source_codebooks, const F32GemmBackend& gemm,
         const SpdSolverF32& solver,
         CudaRootOnlyInitializationContextOptions options) {
    if (options.outer_worker_count == 0U) {
      return Status::InvalidArgument(
          "root-only initialization requires an outer worker");
    }
    // Root-only retains its native greedy/ICM semantics in either execution
    // mode. Seed-incompatible tasks remain separate in the scheduler.
    options.linkage_scheduler.icm_rounds = options.icm_rounds;
    auto codebooks = RootCodebooks<M>::Create(source_codebooks.view());
    if (!codebooks.ok()) {
      return codebooks.status();
    }
    auto scheduled = [&]() -> StatusOr<CudaLinkageSchedulerFactoryResult> {
      ScopedBlasThreadCount blas_threads(options.precompute_thread_count);
      return CreateCudaRootOnlyLinkageScheduler<M>(
          codebooks.value(), gemm, options.linkage_scheduler, options.model);
    }();
    if (!scheduled.ok()) {
      return scheduled.status();
    }
    auto context =
        std::unique_ptr<CudaRootOnlyInitializationTopologyContext<M>>(
            new CudaRootOnlyInitializationTopologyContext<M>(
                std::move(codebooks).value(), gemm, solver,
                std::move(scheduled).value()));
    auto tail_model = [&]()
        -> StatusOr<std::unique_ptr<RootOnlyInitializationCpuTailModel<M>>> {
      ScopedBlasThreadCount blas_threads(options.precompute_thread_count);
      return RootOnlyInitializationCpuTailModel<M>::Create(context->codebooks_,
                                                           gemm);
    }();
    if (!tail_model.ok()) {
      return tail_model.status();
    }
    context->tail_model_ = std::move(tail_model).value();
    context->BuildExecutors(options.outer_worker_count, options.icm_rounds);
    return std::unique_ptr<RootOnlyInitializationTopologyContext<M>>(
        std::move(context));
  }

  [[nodiscard]] ArrayView<RootOnlyInitializationClusterExecutor<M>*>
  executors() noexcept override {
    return executor_views_;
  }

  [[nodiscard]] std::optional<LinkageResourceStatistics> resource_statistics()
      const noexcept override {
    return ObserveCudaLinkageResources(resources_, evaluator_->Stats());
  }

 private:
  CudaRootOnlyInitializationTopologyContext(
      RootCodebooks<M> codebooks, const F32GemmBackend& gemm,
      const SpdSolverF32& solver,
      CudaLinkageSchedulerFactoryResult scheduled) noexcept
      : codebooks_(std::move(codebooks)),
        candidates_(gemm),
        gemm_(&gemm),
        solver_(&solver),
        evaluator_(std::move(scheduled.evaluator)),
        resources_(scheduled.resources) {}

  void BuildExecutors(std::size_t worker_count, std::uint32_t icm_rounds) {
    builders_.reserve(worker_count);
    tail_evaluators_.reserve(worker_count);
    executors_.reserve(worker_count);
    executor_views_.reserve(worker_count);
    for (std::size_t worker = 0U; worker < worker_count; ++worker) {
      tail_evaluators_.push_back(
          std::make_unique<RootOnlyInitializationCpuTailEvaluator<M>>(
              *tail_model_, *gemm_, *solver_, icm_rounds));
      builders_.push_back(
          std::make_unique<RootOnlyInitializationClusterBuilder<M>>(
              codebooks_, candidates_, *evaluator_, *tail_evaluators_.back()));
      executors_.push_back(
          std::make_unique<RootOnlyInitializationBuilderExecutor<M>>(
              *builders_.back()));
      executor_views_.push_back(executors_.back().get());
    }
  }

  RootCodebooks<M> codebooks_;
  CpuExactParentCandidateProvider candidates_;
  const F32GemmBackend* gemm_ = nullptr;
  const SpdSolverF32* solver_ = nullptr;
  std::unique_ptr<RootOnlyInitializationCpuTailModel<M>> tail_model_;
  std::unique_ptr<LinkageEvaluationService> evaluator_;
  CudaLinkageSchedulerResourcePlan resources_;
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
CudaRootOnlyInitializationTopologyContextFactory<M>::Create(
    const RootCodebooks<M>& codebooks) {
  return CudaRootOnlyInitializationTopologyContext<M>::Create(
      codebooks, *gemm_, *solver_, options_);
}

template class CudaRootOnlyInitializationTopologyContextFactory<0>;
template class CudaRootOnlyInitializationTopologyContextFactory<5>;
template class CudaRootOnlyInitializationTopologyContextFactory<12>;
template class CudaRootOnlyInitializationTopologyContextFactory<6>;
template class CudaRootOnlyInitializationTopologyContextFactory<10>;

}  // namespace stlqpp
