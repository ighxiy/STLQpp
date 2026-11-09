#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/training/checkpoint/training_checkpoint.h"
#include "stlqpp/training/linkage/topology/full_topology_pipeline.h"
#include "stlqpp/training/update/round_model_updater.h"
#include "stlqpp/training/update/round_update_transcript.h"

#include <filesystem>

namespace stlqpp {

template <int M>
struct FullTopologyCheckpointRoundResult final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  RoundUpdateTranscript<M> transcript;
  FullTopologyPipelineStatistics topology;
};

// Executes the topology half of one full round, accumulates its forest into
// the common model updater, and closes the immutable update transcript. The
// root-update barrier must already be complete. The caller owns the explicit
// pre-C_one analysis and FinishEdgeUpdate barrier.
template <int M>
class FullTopologyCheckpointRound final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<FullTopologyCheckpointRoundResult<M>> Run(
      FullTopologyClusterSource<M>& source,
      ArrayView<FullTopologyClusterExecutor<M>*> executors,
      const FullTopologyPolicy& policy,
      FullTopologyPipelineOptions options,
      const TrainingModel<M>& topology_model,
      const std::filesystem::path& transcript_destination,
      RoundModelUpdater<M>& updater,
      TrainingCheckpointWriter<M>& checkpoint);
};

extern template class FullTopologyCheckpointRound<5>;
extern template class FullTopologyCheckpointRound<10>;

}  // namespace stlqpp
