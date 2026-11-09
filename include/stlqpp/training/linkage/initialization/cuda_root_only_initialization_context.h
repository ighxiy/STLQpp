#pragma once

#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/spd_solver_f32.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_context.h"
#include "stlqpp/training/linkage/scheduler/cuda_linkage_scheduler_factory.h"

#include <cstddef>
#include <cstdint>

namespace stlqpp {

struct CudaRootOnlyInitializationContextOptions final {
  std::uint32_t icm_rounds = 1U;
  std::size_t outer_worker_count = 32U;
  std::uint32_t precompute_thread_count = 1U;
  CudaRootOnlyLinkageModelOptions model;
  CudaLinkageSchedulerFactoryOptions linkage_scheduler;
};

template <int M>
class CudaRootOnlyInitializationTopologyContextFactory final
    : public RootOnlyInitializationTopologyContextFactory<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  CudaRootOnlyInitializationTopologyContextFactory(
      const F32GemmBackend& gemm,
      const SpdSolverF32& solver,
      CudaRootOnlyInitializationContextOptions options) noexcept
      : gemm_(&gemm), solver_(&solver), options_(options) {}

  [[nodiscard]] StatusOr<std::unique_ptr<
      RootOnlyInitializationTopologyContext<M>>> Create(
      const RootCodebooks<M>& codebooks) override;

 private:
  const F32GemmBackend* gemm_ = nullptr;
  const SpdSolverF32* solver_ = nullptr;
  CudaRootOnlyInitializationContextOptions options_;
};

extern template class CudaRootOnlyInitializationTopologyContextFactory<5>;
extern template class CudaRootOnlyInitializationTopologyContextFactory<10>;

}  // namespace stlqpp
