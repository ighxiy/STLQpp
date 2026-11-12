#include "stlqpp/pipeline/final_full_topology_index_builder.h"

#include "stlqpp/pipeline/base_list_full_topology_source.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace stlqpp {

template <int M>
StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource(
    const FinalFullTopologyIndexBuildRequest& request,
    const TrainingModel<M>& model,
    FullTopologyClusterSource<M>& source,
    ArrayView<FullTopologyClusterExecutor<M>*> executors,
    const SelectedAtomGramTables<M>& grams,
    const SpdSolverF64& coefficient_solver,
    const ParallelForExecutor& parallel,
    FinalIndexPublicationStatistics* publication_statistics) {
  auto session_result = FinalIndexBuildSession<M>::Begin(
      request.index, model, grams, coefficient_solver, parallel);
  if (!session_result.ok()) {
    return session_result.status();
  }
  std::unique_ptr<FinalIndexBuildSession<M>> session =
      std::move(session_result).value();
  const std::uint32_t first_cid = session->next_cid();
  if (const Status status = source.SkipTo(first_cid); !status.ok()) {
    return status;
  }
  auto sink_result = FullTopologySpoolSink<M>::Create(
      *session, request.spool_window, first_cid);
  if (!sink_result.ok()) {
    return sink_result.status();
  }
  auto sink = std::move(sink_result).value();
  FullTopologyPipelineOptions pipeline = request.pipeline;
  pipeline.first_sequence = first_cid;
  pipeline.committed_real_rows = session->committed_real_rows();
  pipeline.completed_cids = session->completed_cids();
  pipeline.collect_quality_statistics = true;
  auto statistics = FullTopologyPipeline<M>::Run(
      source, sink, executors, request.topology, pipeline);
  if (!statistics.ok()) {
    return statistics.status();
  }
  if (const Status status = sink.Finish(); !status.ok()) {
    return status;
  }
  if (const Status status = session->Publish(
          parallel, publication_statistics);
      !status.ok()) {
    return status;
  }
  return statistics;
}

template <int M>
StatusOr<FullTopologyPipelineStatistics> BuildFinalFullTopologyIndex(
    const FinalFullTopologyIndexBuildRequest& request,
    const TrainingModel<M>& model,
    const BaseListReader<M>& base_lists,
    ArrayView<const std::uint8_t> bad_cluster_mask,
    ArrayView<FullTopologyClusterExecutor<M>*> executors,
    const SelectedAtomGramTables<M>& grams,
    const F32GemmBackend& gemm,
    const SpdSolverF64& coefficient_solver,
    const ParallelForExecutor& parallel,
    FinalIndexPublicationStatistics* publication_statistics) {
  auto source = BaseListFullTopologySource<M>::Create(
      base_lists, model, gemm, bad_cluster_mask,
      request.index.publish_scaled_score8 ||
          request.index.publish_float_scaled);
  if (!source.ok()) {
    return source.status();
  }
  return BuildFinalFullTopologyIndexFromSource<M>(
      request, model, source.value(), executors, grams,
      coefficient_solver, parallel, publication_statistics);
}

template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<0>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<0>&,
    const BaseListReader<0>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<0>*>,
    const SelectedAtomGramTables<0>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<5>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<5>&,
    const BaseListReader<5>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<5>*>,
    const SelectedAtomGramTables<5>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<12>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<12>&,
    const BaseListReader<12>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<12>*>,
    const SelectedAtomGramTables<12>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<6>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<6>&,
    const BaseListReader<6>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<6>*>,
    const SelectedAtomGramTables<6>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<10>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<10>&,
    const BaseListReader<10>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<10>*>,
    const SelectedAtomGramTables<10>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<0>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<0>&,
    FullTopologyClusterSource<0>&,
    ArrayView<FullTopologyClusterExecutor<0>*>,
    const SelectedAtomGramTables<0>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<5>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<5>&,
    FullTopologyClusterSource<5>&,
    ArrayView<FullTopologyClusterExecutor<5>*>,
    const SelectedAtomGramTables<5>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<12>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<12>&,
    FullTopologyClusterSource<12>&,
    ArrayView<FullTopologyClusterExecutor<12>*>,
    const SelectedAtomGramTables<12>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<6>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<6>&,
    FullTopologyClusterSource<6>&,
    ArrayView<FullTopologyClusterExecutor<6>*>,
    const SelectedAtomGramTables<6>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<10>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<10>&,
    FullTopologyClusterSource<10>&,
    ArrayView<FullTopologyClusterExecutor<10>*>,
    const SelectedAtomGramTables<10>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);

}  // namespace stlqpp
