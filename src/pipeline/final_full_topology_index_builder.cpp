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
BuildFinalFullTopologyIndex<2>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<2>&,
    const BaseListReader<2>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<2>*>,
    const SelectedAtomGramTables<2>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<3>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<3>&,
    const BaseListReader<3>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<3>*>,
    const SelectedAtomGramTables<3>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<4>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<4>&,
    const BaseListReader<4>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<4>*>,
    const SelectedAtomGramTables<4>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<7>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<7>&,
    const BaseListReader<7>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<7>*>,
    const SelectedAtomGramTables<7>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<8>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<8>&,
    const BaseListReader<8>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<8>*>,
    const SelectedAtomGramTables<8>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<9>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<9>&,
    const BaseListReader<9>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<9>*>,
    const SelectedAtomGramTables<9>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<11>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<11>&,
    const BaseListReader<11>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<11>*>,
    const SelectedAtomGramTables<11>&,
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
BuildFinalFullTopologyIndex<13>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<13>&,
    const BaseListReader<13>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<13>*>,
    const SelectedAtomGramTables<13>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<14>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<14>&,
    const BaseListReader<14>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<14>*>,
    const SelectedAtomGramTables<14>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<15>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<15>&,
    const BaseListReader<15>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<15>*>,
    const SelectedAtomGramTables<15>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<16>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<16>&,
    const BaseListReader<16>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<16>*>,
    const SelectedAtomGramTables<16>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<17>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<17>&,
    const BaseListReader<17>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<17>*>,
    const SelectedAtomGramTables<17>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<18>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<18>&,
    const BaseListReader<18>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<18>*>,
    const SelectedAtomGramTables<18>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<19>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<19>&,
    const BaseListReader<19>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<19>*>,
    const SelectedAtomGramTables<19>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex<20>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<20>&,
    const BaseListReader<20>&,
    ArrayView<const std::uint8_t>,
    ArrayView<FullTopologyClusterExecutor<20>*>,
    const SelectedAtomGramTables<20>&,
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
BuildFinalFullTopologyIndexFromSource<2>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<2>&,
    FullTopologyClusterSource<2>&,
    ArrayView<FullTopologyClusterExecutor<2>*>,
    const SelectedAtomGramTables<2>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<3>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<3>&,
    FullTopologyClusterSource<3>&,
    ArrayView<FullTopologyClusterExecutor<3>*>,
    const SelectedAtomGramTables<3>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<4>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<4>&,
    FullTopologyClusterSource<4>&,
    ArrayView<FullTopologyClusterExecutor<4>*>,
    const SelectedAtomGramTables<4>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<7>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<7>&,
    FullTopologyClusterSource<7>&,
    ArrayView<FullTopologyClusterExecutor<7>*>,
    const SelectedAtomGramTables<7>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<8>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<8>&,
    FullTopologyClusterSource<8>&,
    ArrayView<FullTopologyClusterExecutor<8>*>,
    const SelectedAtomGramTables<8>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<9>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<9>&,
    FullTopologyClusterSource<9>&,
    ArrayView<FullTopologyClusterExecutor<9>*>,
    const SelectedAtomGramTables<9>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<11>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<11>&,
    FullTopologyClusterSource<11>&,
    ArrayView<FullTopologyClusterExecutor<11>*>,
    const SelectedAtomGramTables<11>&,
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
BuildFinalFullTopologyIndexFromSource<13>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<13>&,
    FullTopologyClusterSource<13>&,
    ArrayView<FullTopologyClusterExecutor<13>*>,
    const SelectedAtomGramTables<13>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<14>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<14>&,
    FullTopologyClusterSource<14>&,
    ArrayView<FullTopologyClusterExecutor<14>*>,
    const SelectedAtomGramTables<14>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<15>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<15>&,
    FullTopologyClusterSource<15>&,
    ArrayView<FullTopologyClusterExecutor<15>*>,
    const SelectedAtomGramTables<15>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<16>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<16>&,
    FullTopologyClusterSource<16>&,
    ArrayView<FullTopologyClusterExecutor<16>*>,
    const SelectedAtomGramTables<16>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<17>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<17>&,
    FullTopologyClusterSource<17>&,
    ArrayView<FullTopologyClusterExecutor<17>*>,
    const SelectedAtomGramTables<17>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<18>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<18>&,
    FullTopologyClusterSource<18>&,
    ArrayView<FullTopologyClusterExecutor<18>*>,
    const SelectedAtomGramTables<18>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<19>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<19>&,
    FullTopologyClusterSource<19>&,
    ArrayView<FullTopologyClusterExecutor<19>*>,
    const SelectedAtomGramTables<19>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<20>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<20>&,
    FullTopologyClusterSource<20>&,
    ArrayView<FullTopologyClusterExecutor<20>*>,
    const SelectedAtomGramTables<20>&,
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
