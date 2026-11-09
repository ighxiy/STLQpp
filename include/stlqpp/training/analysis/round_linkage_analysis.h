#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/f32_cross_covariance.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/analysis/initialization_post_edge_analysis.h"
#include "stlqpp/training/update/rotation_update.h"
#include "stlqpp/training/update/round_update_transcript.h"
#include "stlqpp/training/update/training_model.h"

#include <cstdint>
#include <filesystem>
#include <optional>

namespace stlqpp {

// Replays one ordinary full/replay transcript against an immutable round
// model snapshot. Exact source rows are gathered in transcript stable-ID
// order and transformed by that snapshot's rotation before error
// accumulation. Passing the topology model measures the pre-C_one forest;
// passing the post-edge model measures the same forest after C_one update.
template <int M>
[[nodiscard]] StatusOr<InitializationPostEdgeAnalysis>
AnalyzeRoundLinkage(
    const std::filesystem::path& list_directory,
    const RoundUpdateTranscript<M>& transcript,
    const TrainingModel<M>& model,
    const F32GemmBackend& gemm,
    const ParallelForExecutor& parallel,
    const F32CrossCovarianceBackend* covariance_backend = nullptr,
    std::optional<RotationCrossCovariance>* covariance_output = nullptr,
    bool compute_metrics = true,
    std::uint32_t covariance_partition_count = 0U);

extern template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeRoundLinkage<5>(
    const std::filesystem::path&,
    const RoundUpdateTranscript<5>&,
    const TrainingModel<5>&,
    const F32GemmBackend&,
    const ParallelForExecutor&,
    const F32CrossCovarianceBackend*,
    std::optional<RotationCrossCovariance>*,
    bool,
    std::uint32_t);
extern template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeRoundLinkage<10>(
    const std::filesystem::path&,
    const RoundUpdateTranscript<10>&,
    const TrainingModel<10>&,
    const F32GemmBackend&,
    const ParallelForExecutor&,
    const F32CrossCovarianceBackend*,
    std::optional<RotationCrossCovariance>*,
    bool,
    std::uint32_t);

}  // namespace stlqpp
