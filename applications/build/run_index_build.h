#pragma once

#include "build_command.h"
#include "build_execution_report.h"
#include "effective_build_recipe.h"

#include "stlqpp/core/status_or.h"

#ifndef STLQPP_HAS_CUDA_RUNTIME
#define STLQPP_HAS_CUDA_RUNTIME 0
#endif

#if STLQPP_HAS_CUDA_RUNTIME
#include "stlqpp/pipeline/cuda_training_topology_context_factory.h"
#endif

#include <optional>

namespace stlqpp::app {

// Execution-only controls for migration and acceptance applications. Product
// construction uses the defaults and therefore always retains the registered
// cross-cluster scheduler behavior.
struct IndexBuildExecutionControls final {
  std::optional<bool> scheduler_enabled;
  std::optional<bool> linkage_scheduler_transport_batching_enabled;
  std::optional<bool> linkage_scheduler_stable;
};

// Shared production mapping for the final-topology resource boundary.
// Migration/performance harnesses consume this function so they cannot drift
// from build_index while bypassing the already accepted base-basic stage.
#if STLQPP_HAS_CUDA_RUNTIME
[[nodiscard]] CudaTrainingTopologyContextOptions
ResolveTrainingTopologyContextOptions(
    const BuildCommand& command,
    const EffectiveBuildRecipe& recipe,
    const IndexBuildExecutionControls& controls = {});

[[nodiscard]] CudaTrainingTopologyContextOptions
ResolveFinalTopologyContextOptions(
    const BuildCommand& command,
    const EffectiveBuildRecipe& recipe,
    const IndexBuildExecutionControls& controls = {});
#endif

[[nodiscard]] StatusOr<BuildExecutionReport> RunIndexBuild(
    const BuildCommand& command,
    EffectiveBuildRecipe recipe,
    IndexBuildExecutionControls controls = {},
    std::filesystem::path namespace_work_base = {});

}  // namespace stlqpp::app
