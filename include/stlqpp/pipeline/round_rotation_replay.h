#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/index/emit/base_list_reader.h"
#include "stlqpp/runtime/cpu/f32_cross_covariance.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/update/rotation_update.h"
#include "stlqpp/training/update/round_update_transcript.h"
#include "stlqpp/training/update/training_model.h"
#include "stlqpp/training/trainer.h"

namespace stlqpp {

// DEV's post-C_one OPQ pass. Each worker owns one contiguous CID shard,
// reconstructs the frozen forest with the newly solved C_one, and issues one
// covariance GEMM per non-empty CID. Shards merge in ascending ID order.
template <int M>
class RoundRotationReplay final : public TrainingRotationPass<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  RoundRotationReplay(
      const F32CrossCovarianceBackend& covariance_backend,
      const ParallelForExecutor& parallel) noexcept
      : covariance_backend_(&covariance_backend), parallel_(&parallel) {}

  [[nodiscard]] StatusOr<RotationCrossCovariance> Accumulate(
      const TrainingBasicRoundArtifact& basic,
      const RoundUpdateTranscript<M>& transcript,
      const TrainingModel<M>& post_edge_model) override;

 private:
  const F32CrossCovarianceBackend* covariance_backend_ = nullptr;
  const ParallelForExecutor* parallel_ = nullptr;
};

extern template class RoundRotationReplay<5>;
extern template class RoundRotationReplay<10>;

}  // namespace stlqpp
