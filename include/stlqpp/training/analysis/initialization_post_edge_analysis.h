#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/index/emit/base_list_reader.h"
#include "stlqpp/training/linkage/initialization/root_to_edge_code_map.h"
#include "stlqpp/training/update/root_only_initialization_transcript.h"
#include "stlqpp/training/update/training_model.h"

#include <cstdint>
#include "stlqpp/runtime/cpu/parallel_for.h"

namespace stlqpp {

struct InitializationPostEdgeAnalysis final {
  std::uint64_t clusters = 0U;
  std::uint64_t real_rows = 0U;
  std::uint64_t linked_real_rows = 0U;
  std::uint64_t depth_sum = 0U;
  std::uint32_t maximum_depth = 0U;
  double squared_error_sum = 0.0;
  double minimum_squared_error = 0.0;
  double maximum_squared_error = 0.0;
  bool has_squared_error = false;
};

// Replays the root-only initialization forest after C_one publication. Root
// rows retain their C_root reconstruction. Linked rows retain their original
// parent, support and FP32 coefficients, map only layer zero into C_one, and
// use the newly solved edge codebooks. This is a read-only quality scan.
template <int M>
[[nodiscard]] StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate(
    const BaseListReader<M>& lists,
    const RootOnlyInitializationTranscript<M>& transcript,
    const TrainingModel<M>& model,
    const RootToEdgeCodeMap& root_to_edge,
    const ParallelForExecutor& parallel);

extern template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<5>(
    const BaseListReader<5>&,
    const RootOnlyInitializationTranscript<5>&,
    const TrainingModel<5>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
extern template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<10>(
    const BaseListReader<10>&,
    const RootOnlyInitializationTranscript<10>&,
    const TrainingModel<10>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);

}  // namespace stlqpp
