#pragma once

#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/spd_solver_f32.h"
#include "stlqpp/training/basic/basic_schedule.h"
#include "stlqpp/training/linkage/scheduler/cuda_linkage_scheduler_factory.h"
#include "stlqpp/training/trainer.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace stlqpp {

// Operational resource policy for production topology rounds. The outer
// ordinary direct lanes encode virtual roots and, in scheduler-off mode, also
// execute RSD work under the same lease table. Scheduler-on adds only
// its explicit cross-cluster workers.
struct CudaTrainingTopologyContextOptions final {
  std::size_t full_topology_worker_count = 4U;
  std::size_t ordinary_direct_lane_count = 32U;
  BasicSchedule virtual_root_schedule =
      BasicSchedule::TrainingVirtualRoots();
  LeastSquaresCostMode least_squares_cost_mode =
      LeastSquaresCostMode::kNormalEquation;
  CudaLinkageSchedulerFactoryOptions linkage_scheduler;
};

// Opens the same-round list artifact and binds the immutable round model to
// the production full/replay CUDA machinery. Candidate search remains exact
// on CPU, including the bad-cluster real graph.
template <int M>
class CudaTrainingTopologyContextFactory final
    : public TrainingTopologyContextFactory<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  CudaTrainingTopologyContextFactory(
      const F32GemmBackend& gemm,
      const SpdSolverF32& solver,
      CudaTrainingTopologyContextOptions options) noexcept
      : gemm_(&gemm), solver_(&solver), options_(options) {}

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
  const F32GemmBackend* gemm_ = nullptr;
  const SpdSolverF32* solver_ = nullptr;
  CudaTrainingTopologyContextOptions options_;
};

extern template class CudaTrainingTopologyContextFactory<5>;
extern template class CudaTrainingTopologyContextFactory<10>;

}  // namespace stlqpp
