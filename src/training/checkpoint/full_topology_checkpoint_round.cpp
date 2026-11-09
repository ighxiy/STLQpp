#include "stlqpp/training/checkpoint/full_topology_checkpoint_round.h"

#include "stlqpp/training/checkpoint/full_topology_checkpoint_sink.h"

#include <utility>

namespace stlqpp {

template <int M>
StatusOr<FullTopologyCheckpointRoundResult<M>>
FullTopologyCheckpointRound<M>::Run(
    FullTopologyClusterSource<M>& source,
    ArrayView<FullTopologyClusterExecutor<M>*> executors,
    const FullTopologyPolicy& policy,
    FullTopologyPipelineOptions options,
    const TrainingModel<M>& topology_model,
    const std::filesystem::path& transcript_destination,
    RoundModelUpdater<M>& updater,
    TrainingCheckpointWriter<M>& checkpoint) {
  const ModelShape& shape = topology_model.codebooks().shape();
  auto transcript_writer = RoundUpdateTranscriptWriter<M>::Create(
      transcript_destination, shape.dimension, shape.nlist, shape.m);
  if (!transcript_writer.ok()) {
    return transcript_writer.status();
  }
  FullTopologyCheckpointSink<M> sink(
      updater, checkpoint, transcript_writer.value(),
      topology_model.codebooks());
  auto topology = FullTopologyPipeline<M>::Run(
      source, sink, executors, policy, options);
  if (!topology.ok()) {
    return topology.status();
  }
  auto transcript = transcript_writer.value().Finish();
  if (!transcript.ok()) {
    return transcript.status();
  }
  return FullTopologyCheckpointRoundResult<M>{
      .transcript = std::move(transcript).value(),
      .topology = std::move(topology).value(),
  };
}

template class FullTopologyCheckpointRound<0>;
template class FullTopologyCheckpointRound<5>;
template class FullTopologyCheckpointRound<12>;
template class FullTopologyCheckpointRound<6>;
template class FullTopologyCheckpointRound<10>;

}  // namespace stlqpp
