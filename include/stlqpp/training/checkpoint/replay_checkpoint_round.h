#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/training/checkpoint/training_checkpoint.h"
#include "stlqpp/training/linkage/replay/fixed_topology_replay_pipeline.h"
#include "stlqpp/training/update/round_model_updater.h"
#include "stlqpp/training/update/round_update_transcript.h"

#include <filesystem>

namespace stlqpp {

template <int M>
struct ReplayCheckpointRoundResult final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  RoundUpdateTranscript<M> transcript;
  FixedTopologyReplayPipelineStatistics replay;
};

// Executes the topology half of one replay round, accumulates its forest into
// the common model updater, and closes the immutable update transcript. The
// caller must already have crossed the root-update barrier and owns the
// explicit pre-C_one analysis and FinishEdgeUpdate barrier.
template <int M>
class ReplayCheckpointRound final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<ReplayCheckpointRoundResult<M>> Run(
      FixedTopologyReplayWindowLoader& loader,
      const FixedTopologyReplayExecutor<M>& executor,
      const TrainingModel<M>& topology_model,
      const std::filesystem::path& transcript_destination,
      RoundModelUpdater<M>& updater,
      TrainingCheckpointWriter<M>& checkpoint,
      bool collect_quality_statistics);
};

extern template class ReplayCheckpointRound<5>;
extern template class ReplayCheckpointRound<10>;

}  // namespace stlqpp
