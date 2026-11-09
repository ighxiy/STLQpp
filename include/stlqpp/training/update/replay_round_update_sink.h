#pragma once

#include "stlqpp/training/linkage/replay/fixed_topology_replay_pipeline.h"
#include "stlqpp/training/update/round_model_updater.h"
#include "stlqpp/training/update/round_update_transcript.h"

namespace stlqpp {

// Replay pipeline adapter for the common round-update path. The downstream
// writer publishes into round-local staging first; only then are its completed
// forest rows reduced into C_one/R moments. Any later failure discards the
// whole staging round, never a previously published checkpoint.
template <int M>
class ReplayRoundUpdateSink final
    : public FixedTopologyReplayWindowSink<M> {
 public:
  ReplayRoundUpdateSink(
      RoundModelUpdater<M>& updater,
      FixedTopologyReplayWindowSink<M>& staging_sink,
      RoundUpdateTranscriptWriter<M>& transcript,
      const ProductCodebooks<M>& codebooks) noexcept
      : updater_(&updater), staging_sink_(&staging_sink),
        transcript_(&transcript), codebooks_(&codebooks) {}

  [[nodiscard]] bool RequiresRealReconstructions() const noexcept override {
    return staging_sink_->RequiresRealReconstructions();
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
  RoundModelUpdater<M>* updater_ = nullptr;
  FixedTopologyReplayWindowSink<M>* staging_sink_ = nullptr;
  RoundUpdateTranscriptWriter<M>* transcript_ = nullptr;
  const ProductCodebooks<M>* codebooks_ = nullptr;
};

extern template class ReplayRoundUpdateSink<5>;
extern template class ReplayRoundUpdateSink<10>;

}  // namespace stlqpp
