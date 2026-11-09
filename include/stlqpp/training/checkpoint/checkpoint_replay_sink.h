#pragma once

#include "stlqpp/training/checkpoint/training_checkpoint.h"
#include "stlqpp/training/linkage/replay/fixed_topology_replay_pipeline.h"

namespace stlqpp {

// Verifies replay kept its frozen topology and advances the in-memory forest
// snapshot in source CID order without copying its immutable BFS bytes.
template <int M>
class CheckpointReplaySink final
    : public FixedTopologyReplayWindowSink<M> {
 public:
  explicit CheckpointReplaySink(
      TrainingCheckpointWriter<M>& writer) noexcept
      : writer_(&writer) {}

  [[nodiscard]] bool RequiresRealReconstructions() const noexcept override {
    return false;
  }

  [[nodiscard]] Status Commit(
      std::uint64_t sequence,
      ArrayView<const FixedTopologyReplayRequest> requests,
      ArrayView<const FixedTopologyReplayResult<M>> results) override;

 private:
  [[nodiscard]] Status CommitValidatedFromExecutor(
      std::uint64_t sequence,
      ArrayView<const FixedTopologyReplayRequest> requests,
      ArrayView<const FixedTopologyReplayResult<M>> results) override;
  template <bool Prevalidated>
  [[nodiscard]] Status CommitImpl(
      std::uint64_t sequence,
      ArrayView<const FixedTopologyReplayRequest> requests,
      ArrayView<const FixedTopologyReplayResult<M>> results);
  TrainingCheckpointWriter<M>* writer_ = nullptr;
};

extern template class CheckpointReplaySink<5>;
extern template class CheckpointReplaySink<10>;

}  // namespace stlqpp
