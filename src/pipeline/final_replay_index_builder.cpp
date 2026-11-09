#include "stlqpp/pipeline/final_replay_index_builder.h"

#include <memory>
#include <utility>

#include "stlqpp/pipeline/base_list_replay_source.h"
#include "stlqpp/pipeline/final_replay_spool_sink.h"
#include "stlqpp/training/checkpoint/checkpoint_replay_loader.h"
#include "stlqpp/training/linkage/replay/fixed_topology_replay_executor.h"

namespace stlqpp {

template <int M>
StatusOr<FixedTopologyReplayPipelineStatistics> BuildFinalReplayIndex(
    const FinalReplayIndexBuildRequest& request,
    const TrainingCheckpoint<M>& checkpoint,
    const BaseListReader<M>& base_lists, const SelectedAtomGramTables<M>& grams,
    const F32GemmBackend& gemm, const SpdSolverF64& coefficient_solver,
    const FixedEdgeReplayEncoder<M>& edge_encoder,
    const ParallelForExecutor& parallel) {
  const TrainingModel<M>& model = checkpoint.model();
  auto session_result = FinalIndexBuildSession<M>::Begin(
      request.index, model, grams, coefficient_solver, parallel);
  if (!session_result.ok()) {
    return session_result.status();
  }
  std::unique_ptr<FinalIndexBuildSession<M>> session =
      std::move(session_result).value();

  BaseListReplaySource<M> source(base_lists);
  auto loader_result = CheckpointReplayWindowLoader<M>::Create(
      checkpoint, model, source, gemm, request.round_seed);
  if (!loader_result.ok()) {
    return loader_result.status();
  }
  auto loader = std::move(loader_result).value();
  const FixedTopologyReplayExecutor<M> executor(model.codebooks(),
                                                edge_encoder);
  FinalReplaySpoolSink<M> sink(*session);
  auto statistics =
      FixedTopologyReplayPipeline<M>::Run(loader, executor, sink, true);
  if (!statistics.ok()) {
    return statistics.status();
  }
  if (const Status status = session->Publish(parallel); !status.ok()) {
    return status;
  }
  return statistics;
}

template StatusOr<FixedTopologyReplayPipelineStatistics>
BuildFinalReplayIndex<5>(const FinalReplayIndexBuildRequest&,
                         const TrainingCheckpoint<5>&, const BaseListReader<5>&,
                         const SelectedAtomGramTables<5>&,
                         const F32GemmBackend&, const SpdSolverF64&,
                         const FixedEdgeReplayEncoder<5>&,
                         const ParallelForExecutor&);

template StatusOr<FixedTopologyReplayPipelineStatistics>
BuildFinalReplayIndex<12>(const FinalReplayIndexBuildRequest&,
                          const TrainingCheckpoint<12>&,
                          const BaseListReader<12>&,
                          const SelectedAtomGramTables<12>&,
                          const F32GemmBackend&, const SpdSolverF64&,
                          const FixedEdgeReplayEncoder<12>&,
                          const ParallelForExecutor&);

template StatusOr<FixedTopologyReplayPipelineStatistics>
BuildFinalReplayIndex<6>(const FinalReplayIndexBuildRequest&,
                         const TrainingCheckpoint<6>&, const BaseListReader<6>&,
                         const SelectedAtomGramTables<6>&,
                         const F32GemmBackend&, const SpdSolverF64&,
                         const FixedEdgeReplayEncoder<6>&,
                         const ParallelForExecutor&);
template StatusOr<FixedTopologyReplayPipelineStatistics>
BuildFinalReplayIndex<10>(const FinalReplayIndexBuildRequest&,
                          const TrainingCheckpoint<10>&,
                          const BaseListReader<10>&,
                          const SelectedAtomGramTables<10>&,
                          const F32GemmBackend&, const SpdSolverF64&,
                          const FixedEdgeReplayEncoder<10>&,
                          const ParallelForExecutor&);

template StatusOr<FixedTopologyReplayPipelineStatistics>
BuildFinalReplayIndex<0>(const FinalReplayIndexBuildRequest&,
                         const TrainingCheckpoint<0>&, const BaseListReader<0>&,
                         const SelectedAtomGramTables<0>&,
                         const F32GemmBackend&, const SpdSolverF64&,
                         const FixedEdgeReplayEncoder<0>&,
                         const ParallelForExecutor&);

}  // namespace stlqpp
