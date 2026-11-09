#include "stlqpp/application/cuda_index_build_runtime.h"

#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "index_build_sources.h"
#include "stlqpp/application/rsd_basic_round_runtime.h"
#include "stlqpp/pipeline/round_rotation_replay.h"
#include "stlqpp/platform/process_heap.h"
#include "stlqpp/runtime/cpu/blas_thread_policy.h"
#include "stlqpp/runtime/cpu/cblas_f32_cross_covariance.h"
#include "stlqpp/runtime/cpu/cblas_f32_gemm.h"
#include "stlqpp/runtime/cpu/lapack_spd_solver_f32.h"
#include "stlqpp/runtime/cpu/lapack_spd_solver_f64.h"
#include "stlqpp/runtime/cpu/lapack_square_svd_f32.h"
#include "stlqpp/runtime/cpu/openmp_parallel_for.h"
#include "stlqpp/runtime/cuda/context.h"

namespace stlqpp {
namespace {

CudaRootOnlyInitializationContextOptions InitializationTopologyOptions(
    CudaRootOnlyInitializationContextOptions options,
    std::uint32_t cpu_worker_count) {
  options.precompute_thread_count = cpu_worker_count;
  return options;
}

}  // namespace

template <int M>
struct CudaIndexBuildRuntime<M>::State final {
  State(CudaIndexBuildRuntimeOptions options,
        OpenMpParallelFor parallel_executor,
        OpenMpParallelFor base_list_parallel_executor,
        OpenMpParallelFor root_accumulation_parallel_executor,
        OpenMpParallelFor root_moment_parallel_executor,
        OpenMpParallelFor rotation_moment_parallel_executor,
        std::optional<CudaContext> cuda_context)
      : parallel(std::move(parallel_executor)),
        base_list_parallel(std::move(base_list_parallel_executor)),
        root_accumulation_parallel(
            std::move(root_accumulation_parallel_executor)),
        root_moment_parallel(std::move(root_moment_parallel_executor)),
        rotation_moment_parallel(std::move(rotation_moment_parallel_executor)),
        rotation_replay(covariance, rotation_moment_parallel),
        spd_solver(options.cpu_worker_count),
        svd(parallel.worker_count()),
        cuda(std::move(cuda_context)) {
    const auto& training_spec = options.create_root_initialization ||
                                        options.create_training_initialization
                                    ? options.initialization_source
                                    : options.training_source;
    std::visit(
        [&](const auto& spec) {
          using Scalar = std::conditional_t<
              std::is_same_v<std::decay_t<decltype(spec)>, BvecsU8SourceSpec>,
              std::uint8_t, float>;
          training_sources = MakeTrainingSources<Scalar>(options);
        },
        training_spec);
    std::visit(
        [&](const auto& spec) {
          using Scalar = std::conditional_t<
              std::is_same_v<std::decay_t<decltype(spec)>, BvecsU8SourceSpec>,
              std::uint8_t, float>;
          base_sources = MakeBaseSources<Scalar>(options);
        },
        options.base_source);

    if (options.create_fast_initialization) {
      fast_initialization_topology =
          std::make_unique<CudaTrainingTopologyContextFactory<M>>(
              gemm, spd_solver, options.fast_initialization_topology);
    }

    if (options.create_training_initialization &&
        !options.create_fast_initialization) {
      initialization_topology =
          std::make_unique<CudaRootOnlyInitializationTopologyContextFactory<M>>(
              gemm, spd_solver,
              InitializationTopologyOptions(
                  std::move(options.initialization_topology),
                  options.cpu_worker_count));
    }
    if (options.create_training) {
      training_topology =
          std::make_unique<CudaTrainingTopologyContextFactory<M>>(
              gemm, spd_solver, options.training_topology);
    }

    if (options.create_final_linkage) {
      final_topology = std::make_unique<CudaTrainingTopologyContextFactory<M>>(
          gemm, spd_solver, options.final_topology);
    }
  }

  template <typename Scalar>
  detail::IndexTrainingSources<M, Scalar> MakeTrainingSources(
      CudaIndexBuildRuntimeOptions& options) {
    detail::IndexTrainingSources<M, Scalar> sources;
    if (options.create_training_initialization) {
      sources.initialization_source = std::make_unique<
          typename IndexSourceTraits<Scalar>::InitializationFactory>(
          std::get<typename IndexSourceTraits<Scalar>::Spec>(
              std::move(options.initialization_source)),
          parallel);
    }

    if (options.create_training) {
      sources.training_source =
          std::make_unique<typename IndexSourceTraits<Scalar>::TrainingFactory>(
              std::get<typename IndexSourceTraits<Scalar>::Spec>(
                  std::move(options.training_source)),
              parallel);
    }

    if (options.create_training_initialization) {
      sources.initialization_basic_runtime =
          std::make_unique<CudaRsdBasicRoundRuntimeFactory<M, Scalar>>(
              *cuda, options.training_basic_rsd_rank,
              options.training_topology.least_squares_cost_mode);
    }

    if (options.create_training) {
      sources.training_basic_runtime =
          std::make_unique<CudaRsdBasicRoundRuntimeFactory<M, Scalar>>(
              *cuda, options.training_basic_rsd_rank,
              options.training_topology.least_squares_cost_mode);

    }

    if (options.create_root_initialization) {
      sources.root_backend =
          std::make_unique<CudaSphericalRvqInitializationBackend<Scalar>>(
              *cuda, parallel, std::move(options.root_initialization));
    }

    if (options.create_training) {
      sources.training_basic =
          std::make_unique<ScalarTrainingBasicPass<M, Scalar>>(
              *sources.training_source, *sources.training_basic_runtime,
              base_list_parallel, std::move(options.training_basic));
    }
    return sources;
  }

  template <typename Scalar>
  detail::IndexBaseSources<M, Scalar> MakeBaseSources(
      CudaIndexBuildRuntimeOptions& options) {
    detail::IndexBaseSources<M, Scalar> sources;

    if (options.create_base_encode) {
      sources.base_source =
          std::make_unique<typename IndexSourceTraits<Scalar>::BaseFactory>(
              std::get<typename IndexSourceTraits<Scalar>::Spec>(
                  std::move(options.base_source)),
              parallel);
    }

    if (options.create_base_encode) {
      sources.base_basic_runtime =
          std::make_unique<CudaRsdBasicRoundRuntimeFactory<M, Scalar>>(
              *cuda, options.base_basic_rsd_rank,
              options.final_topology.least_squares_cost_mode);

    }
    return sources;
  }

  OpenMpParallelFor parallel;
  OpenMpParallelFor base_list_parallel;
  OpenMpParallelFor root_accumulation_parallel;
  OpenMpParallelFor root_moment_parallel;
  OpenMpParallelFor rotation_moment_parallel;
  CblasF32Gemm gemm;
  CblasF32CrossCovariance covariance;
  RoundRotationReplay<M> rotation_replay;
  LapackSpdSolverF32 spd_solver;
  LapackSpdSolverF64 coefficient_solver;
  LapackSquareSvdF32 svd;
  std::optional<CudaContext> cuda;
  detail::IndexTrainingSourceStorage<M> training_sources;
  detail::IndexBaseSourceStorage<M> base_sources;
  std::unique_ptr<CudaRootOnlyInitializationTopologyContextFactory<M>>
      initialization_topology;
  std::unique_ptr<CudaTrainingTopologyContextFactory<M>>
      fast_initialization_topology;
  std::unique_ptr<CudaTrainingTopologyContextFactory<M>> training_topology;
  std::unique_ptr<CudaTrainingTopologyContextFactory<M>> final_topology;
};

template <int M>
StatusOr<std::unique_ptr<CudaIndexBuildRuntime<M>>>
CudaIndexBuildRuntime<M>::Create(CudaIndexBuildRuntimeOptions options) {
  if ((options.create_root_initialization ||
       options.create_training_initialization) &&
      options.create_training &&
      options.initialization_source.index() !=
          options.training_source.index()) {
    return Status::InvalidArgument(
        "initialization and training source scalar types must match");
  }
  if (const Status status = ConfigureOfflineBuildProcessHeap(); !status.ok()) {
    return status;
  }
  auto parallel = OpenMpParallelFor::Create(options.cpu_worker_count);
  if (!parallel.ok()) {
    return parallel.status();
  }
  auto base_list_parallel =
      OpenMpParallelFor::Create(options.base_list_worker_count);
  if (!base_list_parallel.ok()) {
    return base_list_parallel.status();
  }
  auto root_accumulation_parallel =
      OpenMpParallelFor::Create(options.root_moment_partition_count);
  if (!root_accumulation_parallel.ok()) {
    return root_accumulation_parallel.status();
  }
  auto root_moment_parallel =
      OpenMpParallelFor::Create(options.cpu_worker_count);
  if (!root_moment_parallel.ok()) {
    return root_moment_parallel.status();
  }
  auto rotation_moment_parallel =
      OpenMpParallelFor::Create(options.rotation_moment_partition_count);
  if (!rotation_moment_parallel.ok()) {
    return rotation_moment_parallel.status();
  }
  if (const Status status = ConfigureBlasThreadCount(1U); !status.ok()) {
    return status;
  }
  const bool need_cuda =
      options.create_root_initialization ||
      options.create_training_initialization || options.create_training ||
      options.create_base_encode || options.create_final_linkage;
  std::optional<CudaContext> cuda;
  if (need_cuda) {
    auto created =
        CudaContext::Create(options.cuda_device, options.cuda_allow_tf32);
    if (!created.ok()) return created.status();
    cuda.emplace(std::move(created).value());
  }
  auto state = std::make_unique<State>(
      std::move(options), std::move(parallel).value(),
      std::move(base_list_parallel).value(),
      std::move(root_accumulation_parallel).value(),
      std::move(root_moment_parallel).value(),
      std::move(rotation_moment_parallel).value(), std::move(cuda));
  return std::unique_ptr<CudaIndexBuildRuntime<M>>(
      new CudaIndexBuildRuntime<M>(std::move(state)));
}

template <int M>
CudaIndexBuildRuntime<M>::~CudaIndexBuildRuntime() = default;

template <int M>
StatusOr<IndexBuildResult<M>> CudaIndexBuildRuntime<M>::Build(
    const IndexBuildRequest<M>& request) {
  IndexBuildRuntime<M> runtime;
  std::visit(
      [&](auto& sources) {
        using Scalar = typename std::decay_t<decltype(sources)>::Scalar;
        runtime.initialization = IndexInitializationInputs<M, Scalar>{
            .root_backend = sources.root_backend.get(),
            .runtime =
                {
                    .basic_source = sources.initialization_source.get(),
                    .basic_runtime = sources.initialization_basic_runtime.get(),
                    .topology = state_->initialization_topology.get(),
                    .full_topology = state_->fast_initialization_topology.get(),
                    .parallel = &state_->parallel,
                    .root_accumulation_parallel =
                        &state_->root_accumulation_parallel,
                    .root_moment_parallel = &state_->root_moment_parallel,
                    .list_parallel = &state_->base_list_parallel,
                    .spd_solver = &state_->spd_solver,
                    .metric_gemm = &state_->gemm,
                },
        };
      },
      state_->training_sources);
  runtime.training = {
      .basic = std::visit(
          [](auto& sources) -> TrainingBasicPass<M>* {
            return sources.training_basic.get();
          },
          state_->training_sources),
      .topology = state_->training_topology.get(),
      .rotation = &state_->rotation_replay,
      .metric_gemm = &state_->gemm,
      .covariance_backend = &state_->covariance,
      .spd_solver = &state_->spd_solver,
      .svd = &state_->svd,
      .root_accumulation_parallel = &state_->root_accumulation_parallel,
      .root_moment_parallel = &state_->root_moment_parallel,
      .rotation_moment_parallel = &state_->rotation_moment_parallel,
      .update_parallel = &state_->parallel,
  };
  runtime.final_topology = state_->final_topology.get();
  std::visit(
      [&](auto& sources) {
        using Scalar = typename std::decay_t<decltype(sources)>::Scalar;
        runtime.base = IndexBaseInputs<M, Scalar>{
            sources.base_source.get(), sources.base_basic_runtime.get()};
      },
      state_->base_sources);
  runtime.parallel = &state_->parallel;
  runtime.base_list_parallel = &state_->base_list_parallel;
  runtime.gemm = &state_->gemm;
  runtime.coefficient_solver = &state_->coefficient_solver;
  return BuildIndex<M>(request, runtime);
}

template class CudaIndexBuildRuntime<0>;
template class CudaIndexBuildRuntime<5>;
template class CudaIndexBuildRuntime<6>;
template class CudaIndexBuildRuntime<10>;
template class CudaIndexBuildRuntime<12>;

}  // namespace stlqpp
