#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/runtime/cuda/context.h"

#include <cstdint>

namespace stlqpp::detail {

[[nodiscard]] Status LaunchScatterReplayRoots(
    CudaContext& context, const float* roots, const std::uint32_t* columns,
    std::uint32_t dimension, std::uint32_t root_count, float* reconstructions);

[[nodiscard]] Status LaunchGatherFixedEdgeResiduals(
    CudaContext& context,
    const float* exact_nodes,
    const float* window_reconstructions,
    const std::uint32_t* parent_columns,
    std::uint32_t dimension,
    std::uint32_t sample_count,
    float* residuals);

[[nodiscard]] Status LaunchReconstructAndScatterFixedEdges(
    CudaContext& context,
    float* window_reconstructions,
    const float* edge_atoms,
    const std::uint32_t* parent_columns,
    const std::uint32_t* destination_columns,
    const std::uint8_t* codes,
    const float* coefficients,
    std::uint32_t dimension,
    std::uint32_t layer_count,
    std::uint32_t sample_count);

}  // namespace stlqpp::detail
