#pragma once

#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/runtime/cpu/spd_solver_f32.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace stlqpp {

struct CpuRootOnlyInitializationContextOptions final {
  std::uint32_t icm_rounds = 1U;
  std::size_t outer_worker_count = 1U;
  std::size_t direct_lane_count = 1U;
  // Combined direct-evaluation phase budget; see the ordinary CPU topology
  // option with the same name.
  std::uint64_t maximum_registration_and_workspace_bytes =
      64ULL * 1024ULL * 1024ULL * 1024ULL;
  std::uint32_t maximum_task_nodes = 1024U;
  std::uint32_t maximum_task_pairs = 32768U;
  std::uint32_t maximum_update_columns = 4096U;
};

template <int M>
class CpuRootOnlyInitializationTopologyContextFactory final
    : public RootOnlyInitializationTopologyContextFactory<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  CpuRootOnlyInitializationTopologyContextFactory(
      const F32GemmBackend& gemm,
      const SpdSolverF32& solver,
      CpuRootOnlyInitializationContextOptions options) noexcept
      : gemm_(&gemm), solver_(&solver), options_(options) {}

  [[nodiscard]] StatusOr<std::unique_ptr<
      RootOnlyInitializationTopologyContext<M>>> Create(
      const RootCodebooks<M>& codebooks) override;

 private:
  const F32GemmBackend* gemm_ = nullptr;
  const SpdSolverF32* solver_ = nullptr;
  // Root-initialization executors own the outer cluster parallelism. Keep one
  // direct lane serial so outer_worker_count lanes do not each create a full
  // OpenMP team.
  InlineParallelFor direct_lane_parallel_;
  CpuRootOnlyInitializationContextOptions options_;
};

extern template class CpuRootOnlyInitializationTopologyContextFactory<5>;
extern template class CpuRootOnlyInitializationTopologyContextFactory<10>;

}  // namespace stlqpp
