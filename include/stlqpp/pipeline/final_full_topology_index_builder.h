#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/index/codec/selected_atom_gram.h"
#include "stlqpp/index/emit/base_list_reader.h"
#include "stlqpp/pipeline/final_index_build_session.h"
#include "stlqpp/pipeline/full_topology_spool_sink.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/linkage/topology/full_topology_pipeline.h"
#include "stlqpp/training/update/training_model.h"

namespace stlqpp {

struct FinalFullTopologyIndexBuildRequest final {
  FinalIndexBuildRequest index;
  FullTopologyPolicy topology;
  FullTopologyPipelineOptions pipeline;
  FullTopologySpoolWindowOptions spool_window;
};

// Common publication boundary for an already bound full-topology source.
// Production training and final base construction can therefore share one
// model-bound context without rebuilding candidate or CUDA resources.
template <int M>
[[nodiscard]] StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource(
    const FinalFullTopologyIndexBuildRequest& request,
    const TrainingModel<M>& model,
    FullTopologyClusterSource<M>& source,
    ArrayView<FullTopologyClusterExecutor<M>*> executors,
    const SelectedAtomGramTables<M>& grams,
    const SpdSolverF64& coefficient_solver,
    const ParallelForExecutor& parallel,
    FinalIndexPublicationStatistics* publication_statistics = nullptr);

// Complete bounded final full-topology publication entry point. The caller
// owns one executor per topology worker;
// all cluster products converge on the same canonical spool and norm8 path as
// fixed-topology replay.
template <int M>
[[nodiscard]] StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndex(
    const FinalFullTopologyIndexBuildRequest& request,
    const TrainingModel<M>& model,
    const BaseListReader<M>& base_lists,
    ArrayView<const std::uint8_t> bad_cluster_mask,
    ArrayView<FullTopologyClusterExecutor<M>*> executors,
    const SelectedAtomGramTables<M>& grams,
    const F32GemmBackend& gemm,
    const SpdSolverF64& coefficient_solver,
    const ParallelForExecutor& parallel,
    FinalIndexPublicationStatistics* publication_statistics = nullptr);

extern template StatusOr<FullTopologyPipelineStatistics>
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
extern template StatusOr<FullTopologyPipelineStatistics>
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
extern template StatusOr<FullTopologyPipelineStatistics>
BuildFinalFullTopologyIndexFromSource<5>(
    const FinalFullTopologyIndexBuildRequest&,
    const TrainingModel<5>&,
    FullTopologyClusterSource<5>&,
    ArrayView<FullTopologyClusterExecutor<5>*>,
    const SelectedAtomGramTables<5>&,
    const SpdSolverF64&,
    const ParallelForExecutor&,
    FinalIndexPublicationStatistics*);
extern template StatusOr<FullTopologyPipelineStatistics>
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
