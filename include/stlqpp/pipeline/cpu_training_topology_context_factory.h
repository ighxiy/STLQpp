#pragma once

#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/runtime/cpu/spd_solver_f32.h"
#include "stlqpp/training/basic/basic_schedule.h"
#include "stlqpp/training/trainer.h"
#include "stlqpp/training/least_squares_cost_mode.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace stlqpp {

struct CpuTrainingTopologyContextOptions final {
  std::size_t full_topology_worker_count = 1U;
  std::size_t direct_lane_count = 1U;
  // Combined direct-evaluation phase budget. Per-lane residual/atom-dot
  // matrix capacity is reserved first; registration admission receives the
  // remainder. This is not a whole-process RSS limit.
  std::uint64_t maximum_registration_and_workspace_bytes =
      64ULL * 1024ULL * 1024ULL * 1024ULL;
  std::uint32_t maximum_task_nodes = 1024U;
  std::uint32_t maximum_task_pairs = 32768U;
  std::uint32_t maximum_update_columns = 4096U;
  BasicSchedule virtual_root_schedule =
      BasicSchedule::TrainingVirtualRoots();
  LeastSquaresCostMode least_squares_cost_mode =
      LeastSquaresCostMode::kNormalEquation;
};

// CPU-only production composition of the existing candidate, topology, RSD
// and replay components.  It owns no CUDA-compatible placeholder resources.
template <int M>
class CpuTrainingTopologyContextFactory final
    : public TrainingTopologyContextFactory<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  CpuTrainingTopologyContextFactory(
      const F32GemmBackend& gemm,
      const F32GemmBackend& replay_gemm,
      const SpdSolverF32& solver,
      const ParallelForExecutor& parallel,
      CpuTrainingTopologyContextOptions options) noexcept
      : gemm_(&gemm), replay_gemm_(&replay_gemm), solver_(&solver),
        parallel_(&parallel),
        options_(options) {}

  [[nodiscard]] StatusOr<std::unique_ptr<FullTopologyRoundContext<M>>>
  CreateFull(
      const TrainingModel<M>& topology_model,
      ArrayView<const std::uint8_t> bad_cluster_mask,
      const TrainingBasicRoundArtifact& basic,
      std::uint32_t round,
      const FullTopologyPolicy& policy) override;

  [[nodiscard]] StatusOr<std::unique_ptr<ReplayTopologyRoundContext<M>>>
  CreateReplay(
      const TrainingCheckpoint<M>& previous_checkpoint,
      const TrainingModel<M>& topology_model,
      const TrainingBasicRoundArtifact& basic,
      std::uint32_t round,
      std::uint32_t rsd_rank,
      std::uint32_t icm_rounds,
      std::uint32_t seed) override;

 private:
  // Full runs parallel clusters; Replay runs serial windows/depths and can
  // use the ordinary basic GEMM thread budget. Both backends are runtime-owned.
  const F32GemmBackend* gemm_ = nullptr;
  const F32GemmBackend* replay_gemm_ = nullptr;
  const SpdSolverF32* solver_ = nullptr;
  const ParallelForExecutor* parallel_ = nullptr;
  // Full-topology workers already provide the cluster-level concurrency.
  // A direct CPU lane must stay serial inside one source task; otherwise N
  // outer workers each open an N-thread OpenMP team and oversubscribe the
  // process quadratically. Replay has no such outer worker fan-out and keeps
  // using parallel_.
  InlineParallelFor direct_lane_parallel_;
  CpuTrainingTopologyContextOptions options_;
};

extern template class CpuTrainingTopologyContextFactory<5>;
extern template class CpuTrainingTopologyContextFactory<10>;

}  // namespace stlqpp
