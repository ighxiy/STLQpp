#include "cuda_fixed_edge_replay_kernels.h"

#include "stlqpp/runtime/cuda/status.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace stlqpp::detail {
namespace {

__global__ void ScatterReplayRoots(const float* roots, const std::uint32_t* columns,
                                   std::uint32_t dimension, float* output) {
  const auto root = blockIdx.x;
  for (auto axis = threadIdx.x; axis < dimension; axis += blockDim.x)
    output[static_cast<std::size_t>(columns[root]) * dimension + axis] =
        roots[static_cast<std::size_t>(root) * dimension + axis];
}

__global__ void GatherFixedEdgeResidualsKernel(
    const float* exact_nodes,
    const float* window_reconstructions,
    const std::uint32_t* parent_columns,
    std::uint32_t dimension,
    std::uint32_t sample_count,
    float* residuals) {
  const std::uint32_t sample = blockIdx.x;
  if (sample >= sample_count) return;
  const float* exact = exact_nodes +
      static_cast<std::size_t>(sample) * dimension;
  const float* parent = window_reconstructions +
      static_cast<std::size_t>(parent_columns[sample]) * dimension;
  float* residual = residuals +
      static_cast<std::size_t>(sample) * dimension;
  for (std::uint32_t axis = threadIdx.x; axis < dimension;
       axis += blockDim.x) {
    residual[axis] = exact[axis] - parent[axis];
  }
}

__global__ void ReconstructAndScatterFixedEdgesKernel(
    float* window_reconstructions,
    const float* edge_atoms,
    const std::uint32_t* parent_columns,
    const std::uint32_t* destination_columns,
    const std::uint8_t* codes,
    const float* coefficients,
    std::uint32_t dimension,
    std::uint32_t layer_count,
    std::uint32_t sample_count) {
  const std::uint64_t flat =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t extent =
      static_cast<std::uint64_t>(sample_count) * dimension;
  if (flat >= extent) return;
  const std::uint32_t sample = static_cast<std::uint32_t>(flat / dimension);
  const std::uint32_t axis = static_cast<std::uint32_t>(flat % dimension);
  const std::size_t sample_base =
      static_cast<std::size_t>(sample) * layer_count;
  float value = window_reconstructions[
      static_cast<std::size_t>(parent_columns[sample]) * dimension + axis];
  for (std::uint32_t layer = 0U; layer < layer_count; ++layer) {
    const std::uint32_t atom =
        layer * 256U + codes[sample_base + layer];
    value += coefficients[sample_base + layer] *
        edge_atoms[static_cast<std::size_t>(atom) * dimension + axis];
  }
  window_reconstructions[
      static_cast<std::size_t>(destination_columns[sample]) * dimension +
      axis] = value;
}

}  // namespace

Status LaunchScatterReplayRoots(CudaContext& context, const float* roots,
    const std::uint32_t* columns, std::uint32_t dimension,
    std::uint32_t root_count, float* reconstructions) {
  ScatterReplayRoots<<<root_count, 256U, 0U,
      static_cast<cudaStream_t>(context.native_stream())>>>(roots, columns, dimension, reconstructions);
  return CudaRuntimeStatus("ScatterReplayRoots", cudaPeekAtLastError());
}

Status LaunchGatherFixedEdgeResiduals(
    CudaContext& context,
    const float* exact_nodes,
    const float* window_reconstructions,
    const std::uint32_t* parent_columns,
    std::uint32_t dimension,
    std::uint32_t sample_count,
    float* residuals) {
  GatherFixedEdgeResidualsKernel<<<
      sample_count, 256U, 0U,
      static_cast<cudaStream_t>(context.native_stream())>>>(
          exact_nodes, window_reconstructions, parent_columns, dimension,
          sample_count, residuals);
  const cudaError_t result = cudaPeekAtLastError();
  return result == cudaSuccess
      ? Status::Ok()
      : CudaRuntimeStatus("GatherFixedEdgeResidualsKernel", result);
}

Status LaunchReconstructAndScatterFixedEdges(
    CudaContext& context,
    float* window_reconstructions,
    const float* edge_atoms,
    const std::uint32_t* parent_columns,
    const std::uint32_t* destination_columns,
    const std::uint8_t* codes,
    const float* coefficients,
    std::uint32_t dimension,
    std::uint32_t layer_count,
    std::uint32_t sample_count) {
  const std::uint64_t extent =
      static_cast<std::uint64_t>(sample_count) * dimension;
  constexpr std::uint32_t kThreads = 256U;
  const std::uint32_t blocks = static_cast<std::uint32_t>(
      (extent + kThreads - 1U) / kThreads);
  ReconstructAndScatterFixedEdgesKernel<<<
      blocks, kThreads, 0U,
      static_cast<cudaStream_t>(context.native_stream())>>>(
          window_reconstructions, edge_atoms, parent_columns,
          destination_columns, codes, coefficients, dimension, layer_count,
          sample_count);
  const cudaError_t result = cudaPeekAtLastError();
  return result == cudaSuccess
      ? Status::Ok()
      : CudaRuntimeStatus(
            "ReconstructAndScatterFixedEdgesKernel", result);
}

}  // namespace stlqpp::detail
