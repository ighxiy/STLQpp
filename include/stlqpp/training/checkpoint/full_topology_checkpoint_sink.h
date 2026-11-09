#pragma once

#include "stlqpp/training/checkpoint/training_checkpoint.h"
#include "stlqpp/training/linkage/topology/full_topology_pipeline.h"
#include "stlqpp/training/update/round_model_updater.h"
#include "stlqpp/training/update/round_update_transcript.h"

namespace stlqpp {

// Ordered full-topology adapter shared by compact in-memory forest capture
// and the common C_one/rotation updater. Checkpoint publication follows a
// successful round update at configured boundaries.
template <int M>
class FullTopologyCheckpointSink final
    : public FullTopologyClusterSink<M> {
 public:
  FullTopologyCheckpointSink(
      RoundModelUpdater<M>& updater,
      TrainingCheckpointWriter<M>& checkpoint,
      RoundUpdateTranscriptWriter<M>& transcript,
      const ProductCodebooks<M>& codebooks) noexcept
      : updater_(&updater), checkpoint_(&checkpoint),
        transcript_(&transcript), codebooks_(&codebooks) {}

  [[nodiscard]] bool RequiresRotatedExactReal()
      const noexcept override {
    return true;
  }
  [[nodiscard]] Status Commit(
      std::uint64_t sequence,
      FullTopologyClusterCommitPayload payload,
      FullTopologyClusterProduct<M> product) override;

 private:
  RoundModelUpdater<M>* updater_ = nullptr;
  TrainingCheckpointWriter<M>* checkpoint_ = nullptr;
  RoundUpdateTranscriptWriter<M>* transcript_ = nullptr;
  const ProductCodebooks<M>* codebooks_ = nullptr;
};

extern template class FullTopologyCheckpointSink<5>;
extern template class FullTopologyCheckpointSink<10>;

}  // namespace stlqpp
