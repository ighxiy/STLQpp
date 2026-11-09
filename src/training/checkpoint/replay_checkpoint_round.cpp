#include "stlqpp/training/checkpoint/replay_checkpoint_round.h"

#include "stlqpp/training/checkpoint/checkpoint_replay_sink.h"
#include "stlqpp/training/update/replay_round_update_sink.h"

#include <utility>

namespace stlqpp {

template <int M>
StatusOr<ReplayCheckpointRoundResult<M>> ReplayCheckpointRound<M>::Run(
    FixedTopologyReplayWindowLoader& loader,
    const FixedTopologyReplayExecutor<M>& executor,
    const TrainingModel<M>& topology_model,
    const std::filesystem::path& transcript_destination,
    RoundModelUpdater<M>& updater,
    TrainingCheckpointWriter<M>& checkpoint,
    bool collect_quality_statistics) {
  const ModelShape& shape = topology_model.codebooks().shape();
  auto transcript_writer = RoundUpdateTranscriptWriter<M>::Create(
      transcript_destination, shape.dimension, shape.nlist, shape.m);
  if (!transcript_writer.ok()) {
    return transcript_writer.status();
  }
  CheckpointReplaySink<M> checkpoint_sink(checkpoint);
  ReplayRoundUpdateSink<M> update_sink(
      updater, checkpoint_sink, transcript_writer.value(),
      topology_model.codebooks());
  auto replay = FixedTopologyReplayPipeline<M>::Run(
      loader, executor, update_sink, collect_quality_statistics);
  if (!replay.ok()) {
    return replay.status();
  }
  auto transcript = transcript_writer.value().Finish();
  if (!transcript.ok()) {
    return transcript.status();
  }
  return ReplayCheckpointRoundResult<M>{
      .transcript = std::move(transcript).value(),
      .replay = replay.value(),
  };
}

template class ReplayCheckpointRound<0>;
template class ReplayCheckpointRound<5>;
template class ReplayCheckpointRound<12>;
template class ReplayCheckpointRound<6>;
template class ReplayCheckpointRound<10>;

}  // namespace stlqpp
