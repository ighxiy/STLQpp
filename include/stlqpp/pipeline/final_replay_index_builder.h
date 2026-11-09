#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/index/codec/selected_atom_gram.h"
#include "stlqpp/index/emit/base_list_reader.h"
#include "stlqpp/pipeline/final_index_build_session.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/checkpoint/training_checkpoint.h"
#include "stlqpp/training/linkage/replay/fixed_edge_replay_encoder.h"
#include "stlqpp/training/linkage/replay/fixed_topology_replay_pipeline.h"

#include <cstdint>

namespace stlqpp {

struct FinalReplayIndexBuildRequest final {
  FinalIndexBuildRequest index;
  std::uint32_t round_seed = 0U;
};

// Complete bounded final-replay publication entry point. The checkpoint owns
// the frozen final topology and trained model; the list-major base store owns
// exact real rows. Replay refreshes coefficients against the final model and
// sends each ordered forest window into the one canonical publication
// transaction. No command-line configuration or legacy artifact knowledge is
// reachable from this API.
template <int M>
[[nodiscard]] StatusOr<FixedTopologyReplayPipelineStatistics>
BuildFinalReplayIndex(
    const FinalReplayIndexBuildRequest& request,
    const TrainingCheckpoint<M>& checkpoint,
    const BaseListReader<M>& base_lists,
    const SelectedAtomGramTables<M>& grams,
    const F32GemmBackend& gemm,
    const SpdSolverF64& coefficient_solver,
    const FixedEdgeReplayEncoder<M>& edge_encoder,
    const ParallelForExecutor& parallel);

extern template StatusOr<FixedTopologyReplayPipelineStatistics>
BuildFinalReplayIndex<5>(
    const FinalReplayIndexBuildRequest&,
    const TrainingCheckpoint<5>&,
    const BaseListReader<5>&,
    const SelectedAtomGramTables<5>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const FixedEdgeReplayEncoder<5>&,
    const ParallelForExecutor&);
extern template StatusOr<FixedTopologyReplayPipelineStatistics>
BuildFinalReplayIndex<10>(
    const FinalReplayIndexBuildRequest&,
    const TrainingCheckpoint<10>&,
    const BaseListReader<10>&,
    const SelectedAtomGramTables<10>&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const FixedEdgeReplayEncoder<10>&,
    const ParallelForExecutor&);

}  // namespace stlqpp
