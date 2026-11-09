#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "stlqpp/runtime/cuda/device_buffer.h"
#include "stlqpp/runtime/cuda/gemm_f32.h"
#include "stlqpp/runtime/cuda/status.h"
#include "stlqpp/training/linkage/rsd/detail/linkage_encoding_access.h"
#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"
#include "stlqpp/training/linkage/rsd/linkage_rsd_cuda.h"
#include "stlqpp/training/linkage/rsd/linkage_rsd_device_plan.h"
#include "stlqpp/training/linkage/rsd/ranked_single_deviation.h"

namespace stlqpp {
namespace {

constexpr float kNumericalEpsilon = 1.0e-6F;
// The independent column updates below can expose more rounding error than the
// serial factorization for nearly dependent supports.  Keep those systems on
// the serial path using a scale-independent Schur-pivot test.
constexpr float kCooperativeCholeskyMinimumPivotRatio = 1.0e-5F;
constexpr float kAcceptanceEpsilon = 1.0e-6F;
constexpr std::uint32_t kRsdChunkSamples = 2048U;

template <typename T>
Status UploadModel(CudaDeviceBuffer<T>* destination,
                   ArrayView<const T> source, cudaStream_t stream) {
  if (const Status status = destination->Ensure(source.size()); !status.ok()) {
    return status;
  }
  const cudaError_t result =
      cudaMemcpyAsync(destination->data(), source.data(),
                      source.size() * sizeof(T), cudaMemcpyHostToDevice, stream);
  return result == cudaSuccess
             ? Status::Ok()
             : CudaRuntimeStatus("cudaMemcpyAsync linkage algebra model", result);
}

template <int M>
struct DeviceLinkageAlgebra final {
  std::uint32_t m = M;
  __host__ __device__ constexpr std::uint32_t AtomCount() const {
    return (M == 0 ? m : M) * 256U;
  }

  std::uint32_t dimension = 0U;
  const float* atoms = nullptr;
  const float* squared_norms = nullptr;
  const float* inverse_norms = nullptr;
  const float* gram = nullptr;

  __device__ __forceinline__ float Dot(std::uint32_t row,
                                       std::uint32_t column) const {
    return gram[static_cast<std::size_t>(column) * AtomCount() + row];
  }
};

__device__ __forceinline__ bool Better(float candidate_score,
                                       std::uint32_t candidate_atom,
                                       float incumbent_score,
                                       std::uint32_t incumbent_atom) {
  return candidate_score > incumbent_score ||
         (candidate_score == incumbent_score &&
          candidate_atom < incumbent_atom);
}

__device__ __forceinline__ void InsertTopTwo(float candidate_score,
                                             std::uint32_t candidate_atom,
                                             float* first_score,
                                             std::uint32_t* first_atom,
                                             float* second_score,
                                             std::uint32_t* second_atom) {
  if (Better(candidate_score, candidate_atom, *first_score, *first_atom)) {
    *second_score = *first_score;
    *second_atom = *first_atom;
    *first_score = candidate_score;
    *first_atom = candidate_atom;
  } else if (Better(candidate_score, candidate_atom, *second_score,
                    *second_atom)) {
    *second_score = candidate_score;
    *second_atom = candidate_atom;
  }
}

__device__ __forceinline__ void InsertTopThree(float candidate_score,
                                               std::uint32_t candidate_atom,
                                               float* scores,
                                               std::uint32_t* atoms) {
  if (!Better(candidate_score, candidate_atom, scores[2], atoms[2])) {
    return;
  }
  if (Better(candidate_score, candidate_atom, scores[0], atoms[0])) {
    scores[2] = scores[1];
    atoms[2] = atoms[1];
    scores[1] = scores[0];
    atoms[1] = atoms[0];
    scores[0] = candidate_score;
    atoms[0] = candidate_atom;
  } else if (Better(candidate_score, candidate_atom, scores[1], atoms[1])) {
    scores[2] = scores[1];
    atoms[2] = atoms[1];
    scores[1] = candidate_score;
    atoms[1] = candidate_atom;
  } else {
    scores[2] = candidate_score;
    atoms[2] = candidate_atom;
  }
}

// Both triples are already ordered by Better. Select only the first three
// of their merge; equal score/atom pairs keep the left entry first.
__device__ __forceinline__ void MergeTopThree(
    const float* other_scores, const std::uint32_t* other_atoms,
    float* scores, std::uint32_t* atoms) {
  if (Better(other_scores[0], other_atoms[0], scores[0], atoms[0])) {
    if (Better(other_scores[1], other_atoms[1], scores[0], atoms[0])) {
      if (Better(other_scores[2], other_atoms[2], scores[0], atoms[0])) {
        scores[2] = other_scores[2];
        atoms[2] = other_atoms[2];
      } else {
        scores[2] = scores[0];
        atoms[2] = atoms[0];
      }
      scores[1] = other_scores[1];
      atoms[1] = other_atoms[1];
    } else {
      if (Better(other_scores[1], other_atoms[1], scores[1], atoms[1])) {
        scores[2] = other_scores[1];
        atoms[2] = other_atoms[1];
      } else {
        scores[2] = scores[1];
        atoms[2] = atoms[1];
      }
      scores[1] = scores[0];
      atoms[1] = atoms[0];
    }
    scores[0] = other_scores[0];
    atoms[0] = other_atoms[0];
  } else if (Better(other_scores[0], other_atoms[0], scores[1], atoms[1])) {
    if (Better(other_scores[1], other_atoms[1], scores[1], atoms[1])) {
      scores[2] = other_scores[1];
      atoms[2] = other_atoms[1];
    } else {
      scores[2] = scores[1];
      atoms[2] = atoms[1];
    }
    scores[1] = other_scores[0];
    atoms[1] = other_atoms[0];
  } else if (Better(other_scores[0], other_atoms[0], scores[2], atoms[2])) {
    scores[2] = other_scores[0];
    atoms[2] = other_atoms[0];
  }
}

struct LayerZeroCandidate2 final {
  float error[2];
  std::uint32_t code[2];
};

__device__ __forceinline__ bool BetterLayerZero(float error, std::uint32_t code,
                                                float incumbent_error,
                                                std::uint32_t incumbent_code) {
  return error < incumbent_error ||
         (error == incumbent_error && code < incumbent_code);
}

__device__ __forceinline__ void InsertLayerZero(LayerZeroCandidate2* candidates,
                                                float error,
                                                std::uint32_t code) {
  if (code == candidates->code[0] || code == candidates->code[1]) {
    return;
  }
  if (BetterLayerZero(error, code, candidates->error[0], candidates->code[0])) {
    candidates->error[1] = candidates->error[0];
    candidates->code[1] = candidates->code[0];
    candidates->error[0] = error;
    candidates->code[0] = code;
  } else if (BetterLayerZero(error, code, candidates->error[1],
                             candidates->code[1])) {
    candidates->error[1] = error;
    candidates->code[1] = code;
  }
}

__device__ __forceinline__ LayerZeroCandidate2
MergeLayerZero(LayerZeroCandidate2 left, const LayerZeroCandidate2& right) {
  InsertLayerZero(&left, right.error[0], right.code[0]);
  InsertLayerZero(&left, right.error[1], right.code[1]);
  return left;
}

template <int M>
__global__ __launch_bounds__(256, 1) void SelectLayerZeroCandidatesKernel(
    DeviceLinkageAlgebra<M> algebra, const float* atom_dots,
    const float* squared_norms, std::uint32_t sample_count,
    std::uint8_t* first_codes, std::uint8_t* second_codes) {
  const std::uint32_t sample = blockIdx.x;
  if (sample >= sample_count) {
    return;
  }
  const std::uint32_t code = threadIdx.x;
  LayerZeroCandidate2 local{{CUDART_INF_F, CUDART_INF_F},
                            {0xffffffffU, 0xffffffffU}};
  const float norm2 = algebra.squared_norms[code];
  if (norm2 > 0.0F) {
    const float dot =
        atom_dots[static_cast<std::size_t>(sample) * algebra.AtomCount() +
                  code];
    const float coefficient = dot / norm2;
    const float error = fmaxf(0.0F, squared_norms[sample] - dot * coefficient);
    InsertLayerZero(&local, error, code);
  }
  __shared__ LayerZeroCandidate2 candidates[256];
  candidates[threadIdx.x] = local;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x >> 1U; offset != 0U; offset >>= 1U) {
    if (threadIdx.x < offset) {
      candidates[threadIdx.x] = MergeLayerZero(
          candidates[threadIdx.x], candidates[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    first_codes[sample] = static_cast<std::uint8_t>(candidates[0].code[0]);
    if (second_codes != nullptr) {
      second_codes[sample] = static_cast<std::uint8_t>(candidates[0].code[1]);
    }
  }
}

__device__ __forceinline__ void ReduceTopOne(float best_score,
                                             std::uint32_t best_atom,
                                             float* warp_scores,
                                             std::uint32_t* warp_atoms,
                                             std::uint32_t* selected_atom) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5U;
  constexpr unsigned kWarpMask = 0xffffffffU;
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_score = __shfl_down_sync(kWarpMask, best_score, offset);
    const std::uint32_t other_atom =
        __shfl_down_sync(kWarpMask, best_atom, offset);
    if (lane + static_cast<std::uint32_t>(offset) < 32U &&
        Better(other_score, other_atom, best_score, best_atom)) {
      best_score = other_score;
      best_atom = other_atom;
    }
  }
  if (lane == 0U) {
    warp_scores[warp] = best_score;
    warp_atoms[warp] = best_atom;
  }
  __syncthreads();

  if (warp == 0U) {
    if (lane < 8U) {
      best_score = warp_scores[lane];
      best_atom = warp_atoms[lane];
    } else {
      best_score = -CUDART_INF_F;
      best_atom = 0xffffffffU;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_score = __shfl_down_sync(kWarpMask, best_score, offset);
      const std::uint32_t other_atom =
          __shfl_down_sync(kWarpMask, best_atom, offset);
      if (lane + static_cast<std::uint32_t>(offset) < 32U &&
          Better(other_score, other_atom, best_score, best_atom)) {
        best_score = other_score;
        best_atom = other_atom;
      }
    }
    if (lane == 0U) {
      *selected_atom = best_atom;
    }
  }
  __syncthreads();
}

template <int M>
__device__ __forceinline__ void SelectTopOne(DeviceLinkageAlgebra<M> algebra,
                                             const float* residual_dots,
                                             std::uint32_t available_layers,
                                             float* warp_scores,
                                             std::uint32_t* warp_atoms,
                                             std::uint32_t* selected_atom) {
  float best_score = -CUDART_INF_F;
  std::uint32_t best_atom = 0xffffffffU;
  for (std::uint32_t atom = threadIdx.x; atom < algebra.AtomCount();
       atom += blockDim.x) {
    const std::uint32_t layer = atom >> 8U;
    if ((available_layers & (1U << layer)) == 0U) {
      continue;
    }
    const float score =
        fabsf(residual_dots[atom] * algebra.inverse_norms[atom]);
    if (Better(score, atom, best_score, best_atom)) {
      best_score = score;
      best_atom = atom;
    }
  }
  ReduceTopOne(best_score, best_atom, warp_scores, warp_atoms, selected_atom);
}

template <int M>
__device__ __forceinline__ void UpdateResidualAndSelectTopOne(
    DeviceLinkageAlgebra<M> algebra, float* residual_dots,
    std::uint32_t available_layers, float selected_alpha,
    std::uint32_t selected_atom, float* warp_scores, std::uint32_t* warp_atoms,
    std::uint32_t* next_atom) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  float best_score = -CUDART_INF_F;
  std::uint32_t best_atom = 0xffffffffU;
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    if ((available_layers & (1U << layer)) == 0U) {
      continue;
    }
    const std::uint32_t atom = layer * 256U + threadIdx.x;
    const float updated =
        residual_dots[atom] - selected_alpha * algebra.Dot(atom, selected_atom);
    residual_dots[atom] = updated;
    const float score = fabsf(updated * algebra.inverse_norms[atom]);
    if (Better(score, atom, best_score, best_atom)) {
      best_score = score;
      best_atom = atom;
    }
  }
  ReduceTopOne(best_score, best_atom, warp_scores, warp_atoms, next_atom);
}

__device__ __forceinline__ void ReduceTopOneWithResidual(
    float best_score, std::uint32_t best_atom, float best_residual,
    float* warp_scores, std::uint32_t* warp_atoms, float* warp_residuals,
    std::uint32_t* selected_atom, float* selected_residual) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5U;
  constexpr unsigned kWarpMask = 0xffffffffU;
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_score = __shfl_down_sync(kWarpMask, best_score, offset);
    const std::uint32_t other_atom =
        __shfl_down_sync(kWarpMask, best_atom, offset);
    const float other_residual =
        __shfl_down_sync(kWarpMask, best_residual, offset);
    if (lane + static_cast<std::uint32_t>(offset) < 32U &&
        Better(other_score, other_atom, best_score, best_atom)) {
      best_score = other_score;
      best_atom = other_atom;
      best_residual = other_residual;
    }
  }
  if (lane == 0U) {
    warp_scores[warp] = best_score;
    warp_atoms[warp] = best_atom;
    warp_residuals[warp] = best_residual;
  }
  __syncthreads();

  if (warp == 0U) {
    if (lane < 8U) {
      best_score = warp_scores[lane];
      best_atom = warp_atoms[lane];
      best_residual = warp_residuals[lane];
    } else {
      best_score = -CUDART_INF_F;
      best_atom = 0xffffffffU;
      best_residual = 0.0F;
    }
    for (int offset = 4; offset > 0; offset >>= 1) {
      const float other_score = __shfl_down_sync(kWarpMask, best_score, offset);
      const std::uint32_t other_atom =
          __shfl_down_sync(kWarpMask, best_atom, offset);
      const float other_residual =
          __shfl_down_sync(kWarpMask, best_residual, offset);
      if (lane + static_cast<std::uint32_t>(offset) < 32U &&
          Better(other_score, other_atom, best_score, best_atom)) {
        best_score = other_score;
        best_atom = other_atom;
        best_residual = other_residual;
      }
    }
    if (lane == 0U) {
      *selected_atom = best_atom;
      *selected_residual = best_residual;
    }
  }
  __syncthreads();
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__device__ __forceinline__ void SelectTopOneThreadResiduals(
    DeviceLinkageAlgebra<M> algebra,
    const float (&residual_dots)[Capacity],
    std::uint32_t available_layers, float* warp_scores,
    std::uint32_t* warp_atoms, float* warp_residuals,
    std::uint32_t* selected_atom, float* selected_residual) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  float best_score = -CUDART_INF_F;
  std::uint32_t best_atom = 0xffffffffU;
  float best_residual = 0.0F;
#pragma unroll
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    if ((available_layers & (1U << layer)) == 0U) {
      continue;
    }
    const std::uint32_t atom = layer * 256U + threadIdx.x;
    const float residual = residual_dots[layer];
    const float score = fabsf(residual * algebra.inverse_norms[atom]);
    if (Better(score, atom, best_score, best_atom)) {
      best_score = score;
      best_atom = atom;
      best_residual = residual;
    }
  }
  ReduceTopOneWithResidual(best_score, best_atom, best_residual, warp_scores,
                           warp_atoms, warp_residuals, selected_atom,
                           selected_residual);
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__device__ __forceinline__ void UpdateThreadResidualsAndSelectTopOne(
    DeviceLinkageAlgebra<M> algebra,
    float (&residual_dots)[Capacity],
    std::uint32_t available_layers, float selected_alpha,
    std::uint32_t selected_atom, float* warp_scores, std::uint32_t* warp_atoms,
    float* warp_residuals, std::uint32_t* next_atom, float* next_residual) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  float best_score = -CUDART_INF_F;
  std::uint32_t best_atom = 0xffffffffU;
  float best_residual = 0.0F;
#pragma unroll
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    if ((available_layers & (1U << layer)) == 0U) {
      continue;
    }
    const std::uint32_t atom = layer * 256U + threadIdx.x;
    const float updated = residual_dots[layer] -
                          selected_alpha * algebra.Dot(atom, selected_atom);
    residual_dots[layer] = updated;
    const float score = fabsf(updated * algebra.inverse_norms[atom]);
    if (Better(score, atom, best_score, best_atom)) {
      best_score = score;
      best_atom = atom;
      best_residual = updated;
    }
  }
  ReduceTopOneWithResidual(best_score, best_atom, best_residual, warp_scores,
                           warp_atoms, warp_residuals, next_atom,
                           next_residual);
}

__device__ __forceinline__ void ReduceTopTwo(
    float first_score, std::uint32_t first_atom, float second_score,
    std::uint32_t second_atom, float* warp_first_scores,
    float* warp_second_scores, std::uint32_t* warp_first_atoms,
    std::uint32_t* warp_second_atoms, std::uint32_t* selected_atoms) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5U;
  constexpr unsigned kWarpMask = 0xffffffffU;
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_first_score =
        __shfl_down_sync(kWarpMask, first_score, offset);
    const float other_second_score =
        __shfl_down_sync(kWarpMask, second_score, offset);
    const std::uint32_t other_first_atom =
        __shfl_down_sync(kWarpMask, first_atom, offset);
    const std::uint32_t other_second_atom =
        __shfl_down_sync(kWarpMask, second_atom, offset);
    if (lane + static_cast<std::uint32_t>(offset) < 32U) {
      InsertTopTwo(other_first_score, other_first_atom, &first_score,
                   &first_atom, &second_score, &second_atom);
      InsertTopTwo(other_second_score, other_second_atom, &first_score,
                   &first_atom, &second_score, &second_atom);
    }
  }
  if (lane == 0U) {
    warp_first_scores[warp] = first_score;
    warp_second_scores[warp] = second_score;
    warp_first_atoms[warp] = first_atom;
    warp_second_atoms[warp] = second_atom;
  }
  __syncthreads();

  if (warp == 0U) {
    if (lane < 8U) {
      first_score = warp_first_scores[lane];
      second_score = warp_second_scores[lane];
      first_atom = warp_first_atoms[lane];
      second_atom = warp_second_atoms[lane];
    } else {
      first_score = -CUDART_INF_F;
      second_score = -CUDART_INF_F;
      first_atom = 0xffffffffU;
      second_atom = 0xffffffffU;
    }
    for (int offset = 4; offset > 0; offset >>= 1) {
      const float other_first_score =
          __shfl_down_sync(kWarpMask, first_score, offset);
      const float other_second_score =
          __shfl_down_sync(kWarpMask, second_score, offset);
      const std::uint32_t other_first_atom =
          __shfl_down_sync(kWarpMask, first_atom, offset);
      const std::uint32_t other_second_atom =
          __shfl_down_sync(kWarpMask, second_atom, offset);
      if (lane + static_cast<std::uint32_t>(offset) < 32U) {
        InsertTopTwo(other_first_score, other_first_atom, &first_score,
                     &first_atom, &second_score, &second_atom);
        InsertTopTwo(other_second_score, other_second_atom, &first_score,
                     &first_atom, &second_score, &second_atom);
      }
    }
    if (lane == 0U) {
      selected_atoms[0] = first_atom;
      selected_atoms[1] = second_atom;
    }
  }
  __syncthreads();
}

template <int M>
__device__ __forceinline__ void SelectTopTwo(
    DeviceLinkageAlgebra<M> algebra, const float* residual_dots,
    std::uint32_t available_layers, float* warp_first_scores,
    float* warp_second_scores, std::uint32_t* warp_first_atoms,
    std::uint32_t* warp_second_atoms, std::uint32_t* selected_atoms) {
  float first_score = -CUDART_INF_F;
  float second_score = -CUDART_INF_F;
  std::uint32_t first_atom = 0xffffffffU;
  std::uint32_t second_atom = 0xffffffffU;
  for (std::uint32_t atom = threadIdx.x; atom < algebra.AtomCount();
       atom += blockDim.x) {
    const std::uint32_t layer = atom >> 8U;
    if ((available_layers & (1U << layer)) == 0U) {
      continue;
    }
    const float score =
        fabsf(residual_dots[atom] * algebra.inverse_norms[atom]);
    InsertTopTwo(score, atom, &first_score, &first_atom, &second_score,
                 &second_atom);
  }
  ReduceTopTwo(first_score, first_atom, second_score, second_atom,
               warp_first_scores, warp_second_scores, warp_first_atoms,
               warp_second_atoms, selected_atoms);
}

template <int M>
__device__ __forceinline__ void UpdateResidualAndSelectTopTwo(
    DeviceLinkageAlgebra<M> algebra, float* residual_dots,
    std::uint32_t available_layers, float selected_alpha,
    std::uint32_t selected_atom, float* warp_first_scores,
    float* warp_second_scores, std::uint32_t* warp_first_atoms,
    std::uint32_t* warp_second_atoms, std::uint32_t* selected_atoms) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  float first_score = -CUDART_INF_F;
  float second_score = -CUDART_INF_F;
  std::uint32_t first_atom = 0xffffffffU;
  std::uint32_t second_atom = 0xffffffffU;
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    if ((available_layers & (1U << layer)) == 0U) {
      continue;
    }
    const std::uint32_t atom = layer * 256U + threadIdx.x;
    const float updated =
        residual_dots[atom] - selected_alpha * algebra.Dot(atom, selected_atom);
    residual_dots[atom] = updated;
    const float score = fabsf(updated * algebra.inverse_norms[atom]);
    InsertTopTwo(score, atom, &first_score, &first_atom, &second_score,
                 &second_atom);
  }
  ReduceTopTwo(first_score, first_atom, second_score, second_atom,
               warp_first_scores, warp_second_scores, warp_first_atoms,
               warp_second_atoms, selected_atoms);
}

__device__ __forceinline__ void ReduceTopThree(float* scores,
                                               std::uint32_t* atoms,
                                               float* warp_scores,
                                               std::uint32_t* warp_atoms,
                                               std::uint32_t* selected_atoms) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5U;
  constexpr unsigned kWarpMask = 0xffffffffU;
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_scores[3];
    std::uint32_t other_atoms[3];
#pragma unroll
    for (std::uint32_t rank = 0U; rank < 3U; ++rank) {
      other_scores[rank] = __shfl_down_sync(kWarpMask, scores[rank], offset);
      other_atoms[rank] = __shfl_down_sync(kWarpMask, atoms[rank], offset);
    }
    if (lane + static_cast<std::uint32_t>(offset) < 32U) {
      MergeTopThree(other_scores, other_atoms, scores, atoms);
    }
  }
  if (lane == 0U) {
#pragma unroll
    for (std::uint32_t rank = 0U; rank < 3U; ++rank) {
      warp_scores[warp * 3U + rank] = scores[rank];
      warp_atoms[warp * 3U + rank] = atoms[rank];
    }
  }
  __syncthreads();

  if (warp == 0U) {
    if (lane < 8U) {
#pragma unroll
      for (std::uint32_t rank = 0U; rank < 3U; ++rank) {
        scores[rank] = warp_scores[lane * 3U + rank];
        atoms[rank] = warp_atoms[lane * 3U + rank];
      }
    } else {
#pragma unroll
      for (std::uint32_t rank = 0U; rank < 3U; ++rank) {
        scores[rank] = -CUDART_INF_F;
        atoms[rank] = 0xffffffffU;
      }
    }
    for (int offset = 4; offset > 0; offset >>= 1) {
      float other_scores[3];
      std::uint32_t other_atoms[3];
#pragma unroll
      for (std::uint32_t rank = 0U; rank < 3U; ++rank) {
        other_scores[rank] = __shfl_down_sync(kWarpMask, scores[rank], offset);
        other_atoms[rank] = __shfl_down_sync(kWarpMask, atoms[rank], offset);
      }
      if (lane + static_cast<std::uint32_t>(offset) < 32U) {
        MergeTopThree(other_scores, other_atoms, scores, atoms);
      }
    }
    if (lane == 0U) {
      selected_atoms[0] = atoms[0];
      selected_atoms[1] = atoms[1];
      selected_atoms[2] = atoms[2];
    }
  }
  __syncthreads();
}

template <int M>
__device__ __forceinline__ void SelectTopThree(DeviceLinkageAlgebra<M> algebra,
                                               const float* residual_dots,
                                               std::uint32_t available_layers,
                                               float* warp_scores,
                                               std::uint32_t* warp_atoms,
                                               std::uint32_t* selected_atoms) {
  float scores[3] = {-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F};
  std::uint32_t atoms[3] = {0xffffffffU, 0xffffffffU, 0xffffffffU};
  for (std::uint32_t atom = threadIdx.x; atom < algebra.AtomCount();
       atom += blockDim.x) {
    const std::uint32_t layer = atom >> 8U;
    if ((available_layers & (1U << layer)) == 0U) {
      continue;
    }
    const float score =
        fabsf(residual_dots[atom] * algebra.inverse_norms[atom]);
    InsertTopThree(score, atom, scores, atoms);
  }
  ReduceTopThree(scores, atoms, warp_scores, warp_atoms, selected_atoms);
}

template <int M>
__device__ __forceinline__ void UpdateResidualAndSelectTopThree(
    DeviceLinkageAlgebra<M> algebra, float* residual_dots,
    std::uint32_t available_layers, float selected_alpha,
    std::uint32_t selected_atom, float* warp_scores, std::uint32_t* warp_atoms,
    std::uint32_t* selected_atoms) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  float scores[3] = {-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F};
  std::uint32_t atoms[3] = {0xffffffffU, 0xffffffffU, 0xffffffffU};
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    if ((available_layers & (1U << layer)) == 0U) {
      continue;
    }
    const std::uint32_t atom = layer * 256U + threadIdx.x;
    const float updated =
        residual_dots[atom] - selected_alpha * algebra.Dot(atom, selected_atom);
    residual_dots[atom] = updated;
    const float score = fabsf(updated * algebra.inverse_norms[atom]);
    InsertTopThree(score, atom, scores, atoms);
  }
  ReduceTopThree(scores, atoms, warp_scores, warp_atoms, selected_atoms);
}

template <int M, bool FixedLayerZero>
__host__ __device__ constexpr std::uint32_t BranchResidualOffset(
    std::uint32_t actual_m, std::uint32_t branch_step) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  const std::uint32_t kInitialAvailable =
      static_cast<std::uint32_t>(m) - (FixedLayerZero ? 1U : 0U);
  return branch_step * (2U * kInitialAvailable - branch_step - 1U) * 128U;
}

__host__ __device__ constexpr std::uint32_t BranchResidualValues(
    std::uint32_t m) {
  return m * (m - 1U) / 2U * 256U;
}

__device__ __forceinline__ std::uint32_t PackedLayerRank(std::uint32_t mask,
                                                         std::uint32_t layer) {
  return static_cast<std::uint32_t>(__popc(mask & ((1U << layer) - 1U)));
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>, bool FixedLayerZero>
__global__ __launch_bounds__(256, 1) void BuildRsdRankOneKernel(
    DeviceLinkageAlgebra<M> algebra, const float* atom_dots,
    std::uint32_t sample_count, const std::uint8_t* fixed_layer_zero_codes,
    std::uint8_t* proposal_codes) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  const std::uint32_t kAtomCount = algebra.AtomCount();
  const std::uint32_t kSelectionSteps =
      static_cast<std::uint32_t>(m) - (FixedLayerZero ? 1U : 0U);
  const std::uint32_t sample = blockIdx.x;
  if (sample >= sample_count) {
    return;
  }

  __shared__ std::uint8_t native_codes[Capacity];
  __shared__ float warp_scores[8];
  __shared__ std::uint32_t warp_atoms[8];
  __shared__ float warp_residuals[8];
  __shared__ std::uint32_t selected_atom;
  __shared__ std::uint32_t native_available;
  __shared__ float selected_alpha;
  __shared__ float selected_residual;

  float native_residual_dots[Capacity];

  const float* sample_dots =
      atom_dots + static_cast<std::size_t>(sample) * kAtomCount;
  if (threadIdx.x < static_cast<std::uint32_t>(m)) {
    native_codes[threadIdx.x] = 0U;
  }
  if constexpr (FixedLayerZero) {
    if (threadIdx.x == 0U) {
      selected_atom = fixed_layer_zero_codes[sample];
      const float inverse_norm = algebra.inverse_norms[selected_atom];
      selected_alpha = sample_dots[selected_atom] * inverse_norm * inverse_norm;
      native_available = ((1U << m) - 1U) & ~1U;
      native_codes[0] = static_cast<std::uint8_t>(selected_atom);
    }
  } else if (threadIdx.x == 0U) {
    native_available = (1U << m) - 1U;
  }
  __syncthreads();

#pragma unroll
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    const std::uint32_t atom = layer * 256U + threadIdx.x;
    native_residual_dots[layer] =
        FixedLayerZero ? sample_dots[atom] -
                             selected_alpha * algebra.Dot(atom, selected_atom)
                       : sample_dots[atom];
  }

  SelectTopOneThreadResiduals<M, Capacity>(algebra, native_residual_dots, native_available,
                              warp_scores, warp_atoms, warp_residuals,
                              &selected_atom, &selected_residual);
  for (std::uint32_t step = 0U; step < kSelectionSteps; ++step) {
    if (threadIdx.x == 0U) {
      const float inverse_norm = algebra.inverse_norms[selected_atom];
      selected_alpha = selected_residual * inverse_norm * inverse_norm;
      native_available &= ~(1U << (selected_atom >> 8U));
      native_codes[selected_atom >> 8U] =
          static_cast<std::uint8_t>(selected_atom & 255U);
    }
    __syncthreads();
    if (step + 1U < kSelectionSteps) {
      UpdateThreadResidualsAndSelectTopOne<M, Capacity>(
          algebra, native_residual_dots, native_available, selected_alpha,
          selected_atom, warp_scores, warp_atoms, warp_residuals,
          &selected_atom, &selected_residual);
    }
  }

  if (threadIdx.x < static_cast<std::uint32_t>(m)) {
    proposal_codes[static_cast<std::size_t>(sample) * m + threadIdx.x] =
        native_codes[threadIdx.x];
  }
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>, bool FixedLayerZero>
__global__ __launch_bounds__(256, 1) void BuildRsdSeedsKernel(
    DeviceLinkageAlgebra<M> algebra, const float* atom_dots,
    std::uint32_t sample_count, const std::uint8_t* fixed_layer_zero_codes,
    float* branch_residual_dots, std::uint32_t* branch_remaining_masks,
    std::uint8_t* proposal_codes) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  const std::uint32_t kAtomCount = algebra.AtomCount();
  const std::uint32_t sample = blockIdx.x;
  if (sample >= sample_count) {
    return;
  }

  __shared__ float native_residual_dots[Capacity * 256U];
  __shared__ std::uint8_t native_codes[Capacity];
  __shared__ float warp_first_scores[8];
  __shared__ float warp_second_scores[8];
  __shared__ std::uint32_t warp_first_atoms[8];
  __shared__ std::uint32_t warp_second_atoms[8];
  __shared__ std::uint32_t selected_atoms[2];
  __shared__ std::uint32_t native_available;
  __shared__ std::uint32_t branch_available;
  __shared__ std::uint32_t native_atom;
  __shared__ std::uint32_t branch_atom;
  __shared__ std::uint32_t native_mask;
  __shared__ float native_alpha;
  __shared__ float branch_alpha;

  const std::uint32_t kSelectionSteps =
      static_cast<std::uint32_t>(m) - (FixedLayerZero ? 1U : 0U);
  const std::uint32_t kProposalCount = kSelectionSteps + 1U;

  const float* sample_dots =
      atom_dots + static_cast<std::size_t>(sample) * kAtomCount;
  if constexpr (FixedLayerZero) {
    if (threadIdx.x == 0U) {
      native_atom = fixed_layer_zero_codes[sample];
      const float inverse_norm = algebra.inverse_norms[native_atom];
      native_alpha = sample_dots[native_atom] * inverse_norm * inverse_norm;
      native_available = ((1U << m) - 1U) & ~1U;
    }
  } else if (threadIdx.x == 0U) {
    native_available = (1U << m) - 1U;
  }
  if (threadIdx.x < static_cast<std::uint32_t>(m)) {
    native_codes[threadIdx.x] = 0U;
  }
  __syncthreads();
  if constexpr (FixedLayerZero) {
    if (threadIdx.x == 0U) {
      native_codes[0] = static_cast<std::uint8_t>(native_atom);
    }
  }
  for (std::uint32_t atom = threadIdx.x; atom < kAtomCount;
       atom += blockDim.x) {
    native_residual_dots[atom] =
        FixedLayerZero
            ? sample_dots[atom] - native_alpha * algebra.Dot(atom, native_atom)
            : sample_dots[atom];
  }
  __syncthreads();

  SelectTopTwo(algebra, native_residual_dots, native_available,
               warp_first_scores, warp_second_scores, warp_first_atoms,
               warp_second_atoms, selected_atoms);
  for (std::uint32_t step = 0U; step < kSelectionSteps; ++step) {
    if (threadIdx.x == 0U) {
      native_atom = selected_atoms[0];
      branch_atom = selected_atoms[1];
      const float native_inverse_norm = algebra.inverse_norms[native_atom];
      const float branch_inverse_norm = algebra.inverse_norms[branch_atom];
      native_alpha = native_residual_dots[native_atom] * native_inverse_norm *
                     native_inverse_norm;
      branch_alpha = native_residual_dots[branch_atom] * branch_inverse_norm *
                     branch_inverse_norm;
      branch_available = native_available & ~(1U << (branch_atom >> 8U));
      native_mask = native_available & ~(1U << (native_atom >> 8U));
      branch_remaining_masks[static_cast<std::size_t>(sample) * m + step] =
          branch_available;
    }
    __syncthreads();
    if (threadIdx.x < static_cast<std::uint32_t>(m)) {
      const std::uint32_t layer = threadIdx.x;
      const std::size_t output =
          (static_cast<std::size_t>(sample) * kProposalCount + step + 1U) * m +
          layer;
      proposal_codes[output] =
          layer == (branch_atom >> 8U)
              ? static_cast<std::uint8_t>(branch_atom & 255U)
              : native_codes[layer];
    }
    const std::size_t residual_base =
        static_cast<std::size_t>(sample) * BranchResidualValues(m) +
        BranchResidualOffset<M, FixedLayerZero>(m, step);
    for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
         ++layer) {
      if ((branch_available & (1U << layer)) != 0U) {
        const std::uint32_t atom = layer * 256U + threadIdx.x;
        const std::uint32_t packed_layer =
            PackedLayerRank(branch_available, layer);
        branch_residual_dots[residual_base +
                             static_cast<std::size_t>(packed_layer) * 256U +
                             threadIdx.x] =
            native_residual_dots[atom] -
            branch_alpha * algebra.Dot(atom, branch_atom);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0U) {
      native_available = native_mask;
      native_codes[native_atom >> 8U] =
          static_cast<std::uint8_t>(native_atom & 255U);
    }
    __syncthreads();
    if (step + 1U < kSelectionSteps) {
      UpdateResidualAndSelectTopTwo(
          algebra, native_residual_dots, native_mask, native_alpha, native_atom,
          warp_first_scores, warp_second_scores, warp_first_atoms,
          warp_second_atoms, selected_atoms);
    }
  }

  if (threadIdx.x < static_cast<std::uint32_t>(m)) {
    const std::size_t output =
        static_cast<std::size_t>(sample) * kProposalCount * m + threadIdx.x;
    proposal_codes[output] = native_codes[threadIdx.x];
  }
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>, bool FixedLayerZero>
__global__ __launch_bounds__(256, 1) void BuildRsdRank3SeedsKernel(
    DeviceLinkageAlgebra<M> algebra, const float* atom_dots,
    std::uint32_t sample_count, const std::uint8_t* fixed_layer_zero_codes,
    float* branch_residual_dots, std::uint32_t* branch_remaining_masks,
    std::uint8_t* proposal_codes) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  const std::uint32_t kAtomCount = algebra.AtomCount();
  const std::uint32_t kSelectionSteps =
      static_cast<std::uint32_t>(m) - (FixedLayerZero ? 1U : 0U);
  const std::uint32_t kProposalCount = 1U + 2U * kSelectionSteps;
  const std::uint32_t sample = blockIdx.x;
  if (sample >= sample_count) {
    return;
  }

  __shared__ float native_residual_dots[Capacity * 256U];
  __shared__ std::uint8_t native_codes[Capacity];
  __shared__ float warp_scores[8U * 3U];
  __shared__ std::uint32_t warp_atoms[8U * 3U];
  __shared__ std::uint32_t selected_atoms[3];
  __shared__ std::uint32_t native_available;
  __shared__ std::uint32_t branch_available[2];
  __shared__ std::uint32_t native_atom;
  __shared__ std::uint32_t branch_atom[2];
  __shared__ std::uint32_t native_mask;
  __shared__ float native_alpha;
  __shared__ float branch_alpha[2];

  const float* sample_dots =
      atom_dots + static_cast<std::size_t>(sample) * kAtomCount;
  if constexpr (FixedLayerZero) {
    if (threadIdx.x == 0U) {
      native_atom = fixed_layer_zero_codes[sample];
      const float inverse_norm = algebra.inverse_norms[native_atom];
      native_alpha = sample_dots[native_atom] * inverse_norm * inverse_norm;
      native_available = ((1U << m) - 1U) & ~1U;
    }
  } else if (threadIdx.x == 0U) {
    native_available = (1U << m) - 1U;
  }
  if (threadIdx.x < static_cast<std::uint32_t>(m)) {
    native_codes[threadIdx.x] = 0U;
  }
  __syncthreads();
  if constexpr (FixedLayerZero) {
    if (threadIdx.x == 0U) {
      native_codes[0] = static_cast<std::uint8_t>(native_atom);
    }
  }
  for (std::uint32_t atom = threadIdx.x; atom < kAtomCount;
       atom += blockDim.x) {
    native_residual_dots[atom] =
        FixedLayerZero
            ? sample_dots[atom] - native_alpha * algebra.Dot(atom, native_atom)
            : sample_dots[atom];
  }
  __syncthreads();

  SelectTopThree(algebra, native_residual_dots, native_available, warp_scores,
                 warp_atoms, selected_atoms);
  for (std::uint32_t step = 0U; step < kSelectionSteps; ++step) {
    if (threadIdx.x == 0U) {
      native_atom = selected_atoms[0];
      const float native_inverse_norm = algebra.inverse_norms[native_atom];
      native_alpha = native_residual_dots[native_atom] * native_inverse_norm *
                     native_inverse_norm;
      native_mask = native_available & ~(1U << (native_atom >> 8U));
      for (std::uint32_t branch = 0U; branch < 2U; ++branch) {
        branch_atom[branch] = selected_atoms[branch + 1U];
        const float inverse_norm = algebra.inverse_norms[branch_atom[branch]];
        branch_alpha[branch] = native_residual_dots[branch_atom[branch]] *
                               inverse_norm * inverse_norm;
        branch_available[branch] =
            native_available & ~(1U << (branch_atom[branch] >> 8U));
        branch_remaining_masks[static_cast<std::size_t>(sample) * 2U * m +
                               static_cast<std::size_t>(branch) * m + step] =
            branch_available[branch];
      }
    }
    __syncthreads();

    for (std::uint32_t branch = 0U; branch < 2U; ++branch) {
      const std::uint32_t proposal = 1U + branch * kSelectionSteps + step;
      if (threadIdx.x < static_cast<std::uint32_t>(m)) {
        const std::uint32_t layer = threadIdx.x;
        const std::size_t output =
            (static_cast<std::size_t>(sample) * kProposalCount + proposal) * m +
            layer;
        proposal_codes[output] =
            layer == (branch_atom[branch] >> 8U)
                ? static_cast<std::uint8_t>(branch_atom[branch] & 255U)
                : native_codes[layer];
      }
      const std::size_t residual_base =
          static_cast<std::size_t>(sample) * (2U * BranchResidualValues(m)) +
          static_cast<std::size_t>(branch) * BranchResidualValues(m) +
          BranchResidualOffset<M, FixedLayerZero>(m, step);
      for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
           ++layer) {
        if ((branch_available[branch] & (1U << layer)) != 0U) {
          const std::uint32_t atom = layer * 256U + threadIdx.x;
          const std::uint32_t packed_layer =
              PackedLayerRank(branch_available[branch], layer);
          branch_residual_dots[residual_base +
                               static_cast<std::size_t>(packed_layer) * 256U +
                               threadIdx.x] =
              native_residual_dots[atom] -
              branch_alpha[branch] * algebra.Dot(atom, branch_atom[branch]);
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0U) {
      native_available = native_mask;
      native_codes[native_atom >> 8U] =
          static_cast<std::uint8_t>(native_atom & 255U);
    }
    __syncthreads();
    if (step + 1U < kSelectionSteps) {
      UpdateResidualAndSelectTopThree(algebra, native_residual_dots,
                                      native_mask, native_alpha, native_atom,
                                      warp_scores, warp_atoms, selected_atoms);
    }
  }

  if (threadIdx.x < static_cast<std::uint32_t>(m)) {
    const std::size_t output =
        static_cast<std::size_t>(sample) * kProposalCount * m + threadIdx.x;
    proposal_codes[output] = native_codes[threadIdx.x];
  }
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>, bool FixedLayerZero>
__global__ __launch_bounds__(256, 1) void CompleteRsdBranchSuffixesKernel(
    DeviceLinkageAlgebra<M> algebra, const float* branch_residual_dots,
    const std::uint32_t* branch_remaining_masks, std::uint32_t sample_count,
    std::uint8_t* proposal_codes) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  const std::uint32_t kSelectionSteps =
      static_cast<std::uint32_t>(m) - (FixedLayerZero ? 1U : 0U);
  const std::uint32_t kProposalCount = kSelectionSteps + 1U;
  const std::uint32_t kBranchesWithSuffix = kSelectionSteps - 1U;
  const std::uint32_t alternative = blockIdx.x;
  const std::uint32_t sample = alternative / kBranchesWithSuffix;
  const std::uint32_t branch_step = alternative % kBranchesWithSuffix;
  if (sample >= sample_count) {
    return;
  }

  float residual_dots[Capacity];
  __shared__ float warp_scores[8];
  __shared__ std::uint32_t warp_atoms[8];
  __shared__ float warp_residuals[8];
  __shared__ std::uint32_t available_layers;
  __shared__ std::uint32_t selected_atom;
  __shared__ float selected_residual;
  __shared__ float selected_alpha;

  const std::uint32_t initial_mask =
      branch_remaining_masks[static_cast<std::size_t>(sample) * m +
                             branch_step];
  const std::size_t residual_base =
      static_cast<std::size_t>(sample) * BranchResidualValues(m) +
      BranchResidualOffset<M, FixedLayerZero>(m, branch_step);
#pragma unroll
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    if ((initial_mask & (1U << layer)) != 0U) {
      const std::uint32_t packed_layer = PackedLayerRank(initial_mask, layer);
      residual_dots[layer] =
          branch_residual_dots[residual_base +
                               static_cast<std::size_t>(packed_layer) * 256U +
                               threadIdx.x];
    }
  }
  if (threadIdx.x == 0U) {
    available_layers = initial_mask;
  }
  __syncthreads();

  SelectTopOneThreadResiduals<M, Capacity>(algebra, residual_dots, available_layers,
                              warp_scores, warp_atoms, warp_residuals,
                              &selected_atom, &selected_residual);
  for (std::uint32_t suffix = branch_step + 1U; suffix < kSelectionSteps;
       ++suffix) {
    if (threadIdx.x == 0U) {
      const float inverse_norm = algebra.inverse_norms[selected_atom];
      selected_alpha = selected_residual * inverse_norm * inverse_norm;
      const std::uint32_t layer = selected_atom >> 8U;
      available_layers &= ~(1U << layer);
      const std::size_t output =
          (static_cast<std::size_t>(sample) * kProposalCount + branch_step +
           1U) *
              m +
          layer;
      proposal_codes[output] = static_cast<std::uint8_t>(selected_atom & 255U);
    }
    __syncthreads();
    if (suffix + 1U < kSelectionSteps) {
      UpdateThreadResidualsAndSelectTopOne<M, Capacity>(
          algebra, residual_dots, available_layers, selected_alpha,
          selected_atom, warp_scores, warp_atoms, warp_residuals,
          &selected_atom, &selected_residual);
    }
  }
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>, bool FixedLayerZero>
__global__ __launch_bounds__(256, 1) void CompleteRsdRank3BranchSuffixesKernel(
    DeviceLinkageAlgebra<M> algebra, const float* branch_residual_dots,
    const std::uint32_t* branch_remaining_masks, std::uint32_t sample_count,
    std::uint8_t* proposal_codes) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  const std::uint32_t kSelectionSteps =
      static_cast<std::uint32_t>(m) - (FixedLayerZero ? 1U : 0U);
  const std::uint32_t kProposalCount = 1U + 2U * kSelectionSteps;
  const std::uint32_t kBranchesWithSuffix = kSelectionSteps - 1U;
  const std::uint32_t kAlternativesPerSample = 2U * kBranchesWithSuffix;
  const std::uint32_t alternative = blockIdx.x;
  const std::uint32_t sample = alternative / kAlternativesPerSample;
  const std::uint32_t local = alternative % kAlternativesPerSample;
  const std::uint32_t branch = local / kBranchesWithSuffix;
  const std::uint32_t branch_step = local % kBranchesWithSuffix;
  if (sample >= sample_count) {
    return;
  }

  float residual_dots[Capacity];
  __shared__ float warp_scores[8];
  __shared__ std::uint32_t warp_atoms[8];
  __shared__ float warp_residuals[8];
  __shared__ std::uint32_t available_layers;
  __shared__ std::uint32_t selected_atom;
  __shared__ float selected_residual;
  __shared__ float selected_alpha;

  const std::uint32_t initial_mask =
      branch_remaining_masks[static_cast<std::size_t>(sample) * 2U * m +
                             static_cast<std::size_t>(branch) * m +
                             branch_step];
  const std::size_t residual_base =
      static_cast<std::size_t>(sample) * (2U * BranchResidualValues(m)) +
      static_cast<std::size_t>(branch) * BranchResidualValues(m) +
      BranchResidualOffset<M, FixedLayerZero>(m, branch_step);
#pragma unroll
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    if ((initial_mask & (1U << layer)) != 0U) {
      const std::uint32_t packed_layer = PackedLayerRank(initial_mask, layer);
      residual_dots[layer] =
          branch_residual_dots[residual_base +
                               static_cast<std::size_t>(packed_layer) * 256U +
                               threadIdx.x];
    }
  }
  if (threadIdx.x == 0U) {
    available_layers = initial_mask;
  }
  __syncthreads();

  SelectTopOneThreadResiduals<M, Capacity>(algebra, residual_dots, available_layers,
                              warp_scores, warp_atoms, warp_residuals,
                              &selected_atom, &selected_residual);
  for (std::uint32_t suffix = branch_step + 1U; suffix < kSelectionSteps;
       ++suffix) {
    if (threadIdx.x == 0U) {
      const float inverse_norm = algebra.inverse_norms[selected_atom];
      selected_alpha = selected_residual * inverse_norm * inverse_norm;
      const std::uint32_t layer = selected_atom >> 8U;
      available_layers &= ~(1U << layer);
      const std::uint32_t proposal =
          1U + branch * kSelectionSteps + branch_step;
      const std::size_t output =
          (static_cast<std::size_t>(sample) * kProposalCount + proposal) * m +
          layer;
      proposal_codes[output] = static_cast<std::uint8_t>(selected_atom & 255U);
    }
    __syncthreads();
    if (suffix + 1U < kSelectionSteps) {
      UpdateThreadResidualsAndSelectTopOne<M, Capacity>(
          algebra, residual_dots, available_layers, selected_alpha,
          selected_atom, warp_scores, warp_atoms, warp_residuals,
          &selected_atom, &selected_residual);
    }
  }
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__global__ __launch_bounds__(256, 1) void BuildNativeSupportKernel(
    DeviceLinkageAlgebra<M> algebra, const float* atom_dots,
    std::uint32_t sample_count, std::uint8_t* output_codes) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  const std::uint32_t kAtomCount = algebra.AtomCount();
  const std::uint32_t sample = blockIdx.x;
  if (sample >= sample_count) {
    return;
  }

  __shared__ float residual_dots[Capacity * 256U];
  __shared__ float warp_scores[8];
  __shared__ std::uint32_t warp_atoms[8];
  __shared__ std::uint32_t available_layers;
  __shared__ std::uint32_t selected_atom;
  __shared__ float selected_coefficient;

  const float* sample_dots =
      atom_dots + static_cast<std::size_t>(sample) * kAtomCount;
  for (std::uint32_t atom = threadIdx.x; atom < kAtomCount;
       atom += blockDim.x) {
    residual_dots[atom] = sample_dots[atom];
  }
  if (threadIdx.x == 0U) {
    available_layers = (1U << m) - 1U;
  }
  __syncthreads();

  SelectTopOne(algebra, residual_dots, available_layers, warp_scores,
               warp_atoms, &selected_atom);
  for (std::uint32_t step = 0U; step < static_cast<std::uint32_t>(m); ++step) {
    if (threadIdx.x == 0U) {
      const float inverse_norm = algebra.inverse_norms[selected_atom];
      selected_coefficient =
          residual_dots[selected_atom] * inverse_norm * inverse_norm;
      const std::uint32_t layer = selected_atom >> 8U;
      output_codes[static_cast<std::size_t>(sample) * m + layer] =
          static_cast<std::uint8_t>(selected_atom & 255U);
      available_layers &= ~(1U << layer);
    }
    __syncthreads();
    if (step + 1U < static_cast<std::uint32_t>(m)) {
      UpdateResidualAndSelectTopOne(algebra, residual_dots, available_layers,
                                    selected_coefficient, selected_atom,
                                    warp_scores, warp_atoms, &selected_atom);
    }
  }
}

template <int M>
__device__ __forceinline__ bool CholeskyUpper(std::uint32_t actual_m,
                                              float* matrix) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  for (std::uint32_t column = 0U; column < static_cast<std::uint32_t>(m);
       ++column) {
    float diagonal = matrix[static_cast<std::size_t>(column) * m + column];
    for (std::uint32_t prior = 0U; prior < column; ++prior) {
      const float value = matrix[static_cast<std::size_t>(prior) * m + column];
      diagonal -= value * value;
    }
    if (!(diagonal > 0.0F)) {
      return false;
    }
    diagonal = sqrtf(diagonal);
    matrix[static_cast<std::size_t>(column) * m + column] = diagonal;
    for (std::uint32_t next = column + 1U; next < static_cast<std::uint32_t>(m);
         ++next) {
      float value = matrix[static_cast<std::size_t>(column) * m + next];
      for (std::uint32_t prior = 0U; prior < column; ++prior) {
        value -= matrix[static_cast<std::size_t>(prior) * m + column] *
                 matrix[static_cast<std::size_t>(prior) * m + next];
      }
      matrix[static_cast<std::size_t>(column) * m + next] = value / diagonal;
    }
  }
  return true;
}

template <int M>
__device__ __forceinline__ bool CholeskyUpperWarp(std::uint32_t actual_m,
                                                  float* matrix,
                                                  std::uint32_t lane,
                                                  bool* well_conditioned) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  constexpr unsigned kWarpMask = 0xffffffffU;
  *well_conditioned = true;
  for (std::uint32_t column = 0U; column < static_cast<std::uint32_t>(m);
       ++column) {
    float diagonal = 0.0F;
    bool well_scaled = false;
    if (lane == 0U) {
      const float original_diagonal =
          matrix[static_cast<std::size_t>(column) * m + column];
      diagonal = original_diagonal;
      for (std::uint32_t prior = 0U; prior < column; ++prior) {
        const float value =
            matrix[static_cast<std::size_t>(prior) * m + column];
        diagonal -= value * value;
      }
      well_scaled =
          original_diagonal > 0.0F && diagonal > 0.0F &&
          diagonal >= original_diagonal * kCooperativeCholeskyMinimumPivotRatio;
      if (well_scaled) {
        diagonal = sqrtf(diagonal);
        matrix[static_cast<std::size_t>(column) * m + column] = diagonal;
      }
    }
    const std::uint32_t warp_well_conditioned =
        __shfl_sync(kWarpMask, well_scaled ? 1U : 0U, 0);
    if (warp_well_conditioned == 0U) {
      *well_conditioned = false;
      // Order lane 0's factor reads before the fallback rebuild reuses the
      // shared Gram storage.  The shuffle synchronizes register exchange but
      // is not a shared-memory barrier.
      __syncwarp();
      return false;
    }
    diagonal = __shfl_sync(kWarpMask, diagonal, 0);
    const std::uint32_t next = column + 1U + lane;
    if (next < static_cast<std::uint32_t>(m)) {
      float value = matrix[static_cast<std::size_t>(column) * m + next];
      for (std::uint32_t prior = 0U; prior < column; ++prior) {
        value -= matrix[static_cast<std::size_t>(prior) * m + column] *
                 matrix[static_cast<std::size_t>(prior) * m + next];
      }
      matrix[static_cast<std::size_t>(column) * m + next] = value / diagonal;
    }
    __syncwarp();
  }
  return true;
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__device__ __forceinline__ void SolveUpper(std::uint32_t actual_m,
                                           const float* factor,
                                           const float* rhs, float* solution) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  float forward[Capacity]{};
  for (std::uint32_t row = 0U; row < static_cast<std::uint32_t>(m); ++row) {
    float value = rhs[row];
    for (std::uint32_t prior = 0U; prior < row; ++prior) {
      value -=
          factor[static_cast<std::size_t>(prior) * m + row] * forward[prior];
    }
    forward[row] = value / factor[static_cast<std::size_t>(row) * m + row];
  }
  for (std::uint32_t reverse = static_cast<std::uint32_t>(m); reverse > 0U;
       --reverse) {
    const std::uint32_t row = reverse - 1U;
    float value = forward[row];
    for (std::uint32_t next = row + 1U; next < static_cast<std::uint32_t>(m);
         ++next) {
      value -=
          factor[static_cast<std::size_t>(row) * m + next] * solution[next];
    }
    solution[row] = value / factor[static_cast<std::size_t>(row) * m + row];
  }
}

template <int M>
__device__ __forceinline__ void FillSystem(DeviceLinkageAlgebra<M> algebra,
                                           const std::uint8_t* codes,
                                           const float* atom_dots,
                                           float diagonal_bump, float* gram,
                                           float* rhs) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  for (std::uint32_t entry = 0U; entry < static_cast<std::uint32_t>(m * m);
       ++entry) {
    gram[entry] = 0.0F;
  }
  for (std::uint32_t column = 0U; column < static_cast<std::uint32_t>(m);
       ++column) {
    const std::uint32_t column_atom = column * 256U + codes[column];
    rhs[column] = atom_dots[column_atom];
    for (std::uint32_t row = 0U; row <= column; ++row) {
      const std::uint32_t row_atom = row * 256U + codes[row];
      gram[static_cast<std::size_t>(row) * m + column] =
          algebra.Dot(row_atom, column_atom);
    }
    gram[static_cast<std::size_t>(column) * m + column] += diagonal_bump;
  }
}

template <int M>
__device__ __forceinline__ void FillSystemWarp(DeviceLinkageAlgebra<M> algebra,
                                               const std::uint8_t* codes,
                                               const float* atom_dots,
                                               float diagonal_bump,
                                               std::uint32_t lane, float* gram,
                                               float* rhs) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  for (std::uint32_t entry = lane; entry < static_cast<std::uint32_t>(m * m);
       entry += 32U) {
    gram[entry] = 0.0F;
  }
  __syncwarp();
  if (lane < static_cast<std::uint32_t>(m)) {
    const std::uint32_t column = lane;
    const std::uint32_t column_atom = column * 256U + codes[column];
    rhs[column] = atom_dots[column_atom];
    for (std::uint32_t row = 0U; row <= column; ++row) {
      const std::uint32_t row_atom = row * 256U + codes[row];
      gram[static_cast<std::size_t>(row) * m + column] =
          algebra.Dot(row_atom, column_atom);
    }
    gram[static_cast<std::size_t>(column) * m + column] += diagonal_bump;
  }
  __syncwarp();
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__device__ __forceinline__ void RefitSupportWarpAssembled(
    DeviceLinkageAlgebra<M> algebra, const std::uint8_t* codes,
    const float* atom_dots, float residual_squared_norm, std::uint32_t lane,
    float* gram, float* rhs, float* coefficients, float* squared_error,
    std::uint32_t* solved_output, LeastSquaresCostMode cost_mode) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  float bump = 0.0F;
  for (std::uint32_t attempt = 0U; attempt < 4U; ++attempt) {
    FillSystemWarp<M>(algebra, codes, atom_dots, bump, lane, gram, rhs);
    if (attempt == 0U) {
      bool well_conditioned = false;
      bool solved = CholeskyUpperWarp<M>(m, gram, lane, &well_conditioned);
      if (!well_conditioned) {
        FillSystemWarp<M>(algebra, codes, atom_dots, 0.0F, lane, gram, rhs);
        if (lane == 0U) {
          solved = CholeskyUpper<M>(m, gram);
        }
      }
      if (lane == 0U) {
        *solved_output = solved ? 1U : 0U;
      }
    } else if (lane == 0U) {
      *solved_output = CholeskyUpper<M>(m, gram) ? 1U : 0U;
    }
    __syncwarp();
    if (*solved_output != 0U) {
      break;
    }
    bump = attempt == 0U ? kNumericalEpsilon : bump * 10.0F;
  }
  if (*solved_output == 0U) {
    return;
  }
  if (lane == 0U) {
    SolveUpper<M, Capacity>(m, gram, rhs, coefficients);
    float data_dot = 0.0F;
    float reconstructed_norm = 0.0F;
    for (std::uint32_t column = 0U; column < static_cast<std::uint32_t>(m);
         ++column) {
      const std::uint32_t column_atom = column * 256U + codes[column];
      data_dot += coefficients[column] * atom_dots[column_atom];
      if (cost_mode != LeastSquaresCostMode::kNormalEquation) {
        for (std::uint32_t row = 0U; row < static_cast<std::uint32_t>(m);
             ++row) {
          const std::uint32_t row_atom = row * 256U + codes[row];
          const float gram_value = row <= column
                                       ? algebra.Dot(row_atom, column_atom)
                                       : algebra.Dot(column_atom, row_atom);
          reconstructed_norm +=
              coefficients[column] * coefficients[row] * gram_value;
        }
      }
    }
    if (cost_mode != LeastSquaresCostMode::kNormalEquation) {
      *squared_error =
          residual_squared_norm - 2.0F * data_dot + reconstructed_norm;
    } else {
      *squared_error = residual_squared_norm - data_dot;
      if (bump != 0.0F) {
        float coefficient_norm = 0.0F;
        for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
             ++layer) {
          coefficient_norm += coefficients[layer] * coefficients[layer];
        }
        *squared_error -= bump * coefficient_norm;
      }
    }
    if (!isfinite(*squared_error)) {
      *solved_output = 0U;
    }
  }
  __syncwarp();
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__device__ __forceinline__ bool RefitSupport(DeviceLinkageAlgebra<M> algebra,
                                             const std::uint8_t* codes,
                                             const float* atom_dots,
                                             float residual_squared_norm,
                                             float* coefficients,
                                             float* squared_error,
                                             LeastSquaresCostMode cost_mode) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  float gram[Capacity * Capacity]{};
  float rhs[Capacity]{};
  FillSystem(algebra, codes, atom_dots, 0.0F, gram, rhs);
  bool solved = CholeskyUpper<M>(m, gram);
  float solved_bump = 0.0F;
  float bump = kNumericalEpsilon;
  for (std::uint32_t attempt = 0U; attempt < 3U && !solved; ++attempt) {
    FillSystem(algebra, codes, atom_dots, bump, gram, rhs);
    solved = CholeskyUpper<M>(m, gram);
    solved_bump = bump;
    bump *= 10.0F;
  }
  if (!solved) {
    return false;
  }
  SolveUpper<M, Capacity>(m, gram, rhs, coefficients);

  float data_dot = 0.0F;
  float reconstructed_norm = 0.0F;
  for (std::uint32_t column = 0U; column < static_cast<std::uint32_t>(m);
       ++column) {
    const std::uint32_t column_atom = column * 256U + codes[column];
    data_dot += coefficients[column] * atom_dots[column_atom];
    if (cost_mode != LeastSquaresCostMode::kNormalEquation) {
      for (std::uint32_t row = 0U; row < static_cast<std::uint32_t>(m); ++row) {
        const std::uint32_t row_atom = row * 256U + codes[row];
        const float gram_value = row <= column
                                     ? algebra.Dot(row_atom, column_atom)
                                     : algebra.Dot(column_atom, row_atom);
        reconstructed_norm +=
            coefficients[column] * coefficients[row] * gram_value;
      }
    }
  }
  if (cost_mode != LeastSquaresCostMode::kNormalEquation) {
    *squared_error =
        residual_squared_norm - 2.0F * data_dot + reconstructed_norm;
  } else {
    *squared_error = residual_squared_norm - data_dot;
    float coefficient_norm = 0.0F;
    if (solved_bump != 0.0F) {
      for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
           ++layer) {
        coefficient_norm += coefficients[layer] * coefficients[layer];
      }
      *squared_error -= solved_bump * coefficient_norm;
    }
  }
  return isfinite(*squared_error);
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
// The solve is identical for native supports and RSD proposals. These launch
// parameters describe row addressing and result consumption only: native and
// single-proposal rows have proposals == 1; only the single-proposal path
// copies/clears output codes, and only final outputs count failures.
__global__ void RefitRowsKernel(
    std::uint32_t actual_proposals, DeviceLinkageAlgebra<M> algebra,
    const float* atom_dots, const float* squared_norms,
    const std::uint8_t* row_codes, std::uint32_t sample_count,
    float* row_coefficients, float* row_errors,
    std::uint8_t* output_codes, std::uint32_t* failures,
    LeastSquaresCostMode cost_mode) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  // Proposal count controls row addressing, not the M-wide least-squares solve.
  const std::uint32_t proposals = actual_proposals;
  const std::uint32_t flat = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t proposal_count = sample_count * proposals;
  if (flat >= proposal_count) {
    return;
  }
  const std::uint32_t sample = proposals == 1U ? flat : flat / proposals;
  const std::uint8_t* codes =
      row_codes + static_cast<std::size_t>(flat) * m;
  const float* sample_dots =
      atom_dots + static_cast<std::size_t>(sample) * algebra.AtomCount();
  float* coefficients =
      row_coefficients + static_cast<std::size_t>(flat) * m;
  float error = CUDART_INF_F;
  if (!RefitSupport<M, Capacity>(algebra, codes, sample_dots, squared_norms[sample],
                       coefficients, &error, cost_mode)) {
    for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
         ++layer) {
      coefficients[layer] = 0.0F;
      if (output_codes != nullptr) {
        output_codes[static_cast<std::size_t>(flat) * m + layer] = 0U;
      }
    }
    error = CUDART_INF_F;
    if (failures != nullptr) {
      atomicAdd(failures, 1U);
    }
  } else if (output_codes != nullptr) {
    for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
         ++layer) {
      output_codes[static_cast<std::size_t>(flat) * m + layer] = codes[layer];
    }
  }
  row_errors[flat] = error;
}

template <int M, std::uint32_t ProposalCount>
__global__ void SelectBestRsdProposalKernel(
    std::uint32_t actual_proposals, std::uint32_t actual_m,
    const std::uint8_t* proposal_codes, const float* proposal_coefficients,
    const float* proposal_errors, std::uint32_t sample_count,
    std::uint8_t* output_codes, float* output_coefficients,
    float* output_errors, std::uint32_t* failures) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  const std::uint32_t proposals = M == 0 ? actual_proposals : ProposalCount;
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= sample_count) {
    return;
  }
  const std::size_t proposal_base =
      static_cast<std::size_t>(sample) * proposals;
  std::uint32_t best_proposal = 0U;
  float best_error = proposal_errors[proposal_base];
  for (std::uint32_t proposal = 1U; proposal < proposals; ++proposal) {
    const float error = proposal_errors[proposal_base + proposal];
    if (error < best_error) {
      best_error = error;
      best_proposal = proposal;
    }
  }
  if (!isfinite(best_error)) {
    const std::size_t base = static_cast<std::size_t>(sample) * m;
    for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
         ++layer) {
      output_codes[base + layer] = 0U;
      output_coefficients[base + layer] = 0.0F;
    }
    output_errors[sample] = CUDART_INF_F;
    if (failures != nullptr) {
      atomicAdd(failures, 1U);
    }
    return;
  }
  const std::size_t base = static_cast<std::size_t>(sample) * m;
  const std::size_t selected_base = (proposal_base + best_proposal) * m;
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    output_codes[base + layer] = proposal_codes[selected_base + layer];
    output_coefficients[base + layer] =
        proposal_coefficients[selected_base + layer];
  }
  output_errors[sample] = best_error;
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>, int CostPolicy = -1>
__global__ __launch_bounds__(256, 1) void IcmSweepKernel(
    DeviceLinkageAlgebra<M> algebra, const float* atom_dots,
    const float* squared_norms, const std::uint8_t* layer_orders,
    std::uint32_t sample_count, std::uint8_t* codes, float* coefficients,
    float* errors, LeastSquaresCostMode cost_mode) {
  if constexpr (CostPolicy >= 0) {
    cost_mode = static_cast<LeastSquaresCostMode>(CostPolicy);
  }
  const std::uint32_t m = M == 0 ? algebra.m : M;
  const std::uint32_t sample = blockIdx.x;
  if (sample >= sample_count) {
    return;
  }
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5U;
  __shared__ std::uint8_t shared_codes[Capacity];
  __shared__ float shared_coefficients[Capacity];
  __shared__ float shared_error;
  __shared__ std::uint32_t shared_layer;
  __shared__ std::uint32_t shared_best_code;
  __shared__ float warp_scores[8];
  __shared__ std::uint32_t warp_codes[8];
  __shared__ std::uint8_t trial_codes[Capacity];
  __shared__ float trial_coefficients[Capacity];
  __shared__ float trial_gram[Capacity * Capacity];
  __shared__ float trial_rhs[Capacity];
  __shared__ float trial_error;
  __shared__ std::uint32_t trial_solved;
  const std::size_t base = static_cast<std::size_t>(sample) * m;
  if (threadIdx.x < static_cast<std::uint32_t>(m)) {
    shared_codes[threadIdx.x] = codes[base + threadIdx.x];
    shared_coefficients[threadIdx.x] = coefficients[base + threadIdx.x];
  }
  if (threadIdx.x == 0U) {
    shared_error = errors[sample];
  }
  __syncthreads();
  if (!isfinite(shared_error)) {
    return;
  }

  const float* sample_dots =
      atom_dots + static_cast<std::size_t>(sample) * algebra.AtomCount();
  const std::uint8_t* order = layer_orders + base;
  for (std::uint32_t position = 0U; position < static_cast<std::uint32_t>(m);
       ++position) {
    if (threadIdx.x == 0U) {
      shared_layer = order[position];
    }
    __syncthreads();
    const std::uint32_t layer = shared_layer;
    const std::uint32_t code = threadIdx.x;
    const std::uint32_t atom = layer * 256U + code;
    float remaining_dot = sample_dots[atom];
    for (std::uint32_t other_layer = 0U;
         other_layer < static_cast<std::uint32_t>(m); ++other_layer) {
      if (other_layer == layer) {
        continue;
      }
      const std::uint32_t selected_atom =
          other_layer * 256U + shared_codes[other_layer];
      remaining_dot -=
          shared_coefficients[other_layer] * algebra.Dot(atom, selected_atom);
    }
    float best_score = fabsf(remaining_dot * algebra.inverse_norms[atom]);
    std::uint32_t best_code = code;
    constexpr unsigned kWarpMask = 0xffffffffU;
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_score = __shfl_down_sync(kWarpMask, best_score, offset);
      const std::uint32_t other_code =
          __shfl_down_sync(kWarpMask, best_code, offset);
      if (lane + static_cast<std::uint32_t>(offset) < 32U &&
          Better(other_score, other_code, best_score, best_code)) {
        best_score = other_score;
        best_code = other_code;
      }
    }
    if (lane == 0U) {
      warp_scores[warp] = best_score;
      warp_codes[warp] = best_code;
    }
    __syncthreads();
    if (warp == 0U) {
      if (lane < 8U) {
        best_score = warp_scores[lane];
        best_code = warp_codes[lane];
      } else {
        best_score = -CUDART_INF_F;
        best_code = 0xffffffffU;
      }
      for (int offset = 4; offset > 0; offset >>= 1) {
        const float other_score =
            __shfl_down_sync(kWarpMask, best_score, offset);
        const std::uint32_t other_code =
            __shfl_down_sync(kWarpMask, best_code, offset);
        if (lane + static_cast<std::uint32_t>(offset) < 32U &&
            Better(other_score, other_code, best_score, best_code)) {
          best_score = other_score;
          best_code = other_code;
        }
      }
      if (lane == 0U) {
        shared_best_code = best_code;
      }
    }
    __syncthreads();

    if (shared_best_code != shared_codes[layer]) {
      if (threadIdx.x < static_cast<std::uint32_t>(m)) {
        trial_codes[threadIdx.x] =
            threadIdx.x == layer ? static_cast<std::uint8_t>(shared_best_code)
                                 : shared_codes[threadIdx.x];
      }
      if (threadIdx.x == 0U) {
        trial_error = CUDART_INF_F;
        trial_solved = 0U;
      }
      __syncthreads();
      if (warp == 0U) {
        RefitSupportWarpAssembled<M, Capacity>(
            algebra, trial_codes, sample_dots, squared_norms[sample], lane,
            trial_gram, trial_rhs, trial_coefficients, &trial_error,
            &trial_solved, cost_mode);
      }
      __syncthreads();
      if (threadIdx.x == 0U && trial_solved != 0U &&
          trial_error + kAcceptanceEpsilon < shared_error) {
        shared_codes[layer] = static_cast<std::uint8_t>(shared_best_code);
        for (std::uint32_t copy = 0U; copy < static_cast<std::uint32_t>(m);
             ++copy) {
          shared_coefficients[copy] = trial_coefficients[copy];
        }
        shared_error = trial_error;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x < static_cast<std::uint32_t>(m)) {
    codes[base + threadIdx.x] = shared_codes[threadIdx.x];
    coefficients[base + threadIdx.x] = shared_coefficients[threadIdx.x];
  }
  if (threadIdx.x == 0U) {
    errors[sample] = shared_error;
  }
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
void LaunchRefitRows(
    std::uint32_t proposals, LeastSquaresCostMode cost_mode,
    std::uint32_t grid, std::uint32_t threads, cudaStream_t stream,
    DeviceLinkageAlgebra<M> algebra, const float* atom_dots,
    const float* squared_norms, const std::uint8_t* codes,
    std::uint32_t sample_count, float* coefficients, float* errors,
    std::uint8_t* output_codes, std::uint32_t* failures) {
  RefitRowsKernel<M, Capacity><<<grid, threads, 0U, stream>>>(
      proposals, algebra, atom_dots, squared_norms, codes, sample_count,
      coefficients, errors, output_codes, failures, cost_mode);
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
void LaunchRefitRsdProposals(
    std::uint32_t actual_proposals, LeastSquaresCostMode cost_mode,
    std::uint32_t grid, std::uint32_t threads, cudaStream_t stream,
    DeviceLinkageAlgebra<M> algebra, const float* atom_dots,
    const float* squared_norms, const std::uint8_t* proposal_codes,
    std::uint32_t sample_count, float* proposal_coefficients,
    float* proposal_errors) {
  LaunchRefitRows<M, Capacity>(actual_proposals, cost_mode, grid, threads, stream,
                    algebra, atom_dots, squared_norms, proposal_codes,
                    sample_count, proposal_coefficients, proposal_errors,
                    nullptr, nullptr);
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
void LaunchRefitSingleRsdProposal(
    LeastSquaresCostMode cost_mode, std::uint32_t grid, std::uint32_t threads,
    cudaStream_t stream, DeviceLinkageAlgebra<M> algebra,
    const float* atom_dots, const float* squared_norms,
    const std::uint8_t* proposal_codes, std::uint32_t sample_count,
    std::uint8_t* output_codes, float* output_coefficients,
    float* output_errors, std::uint32_t* failures) {
  LaunchRefitRows<M, Capacity>(1U, cost_mode, grid, threads, stream, algebra, atom_dots,
                    squared_norms, proposal_codes, sample_count,
                    output_coefficients, output_errors, output_codes, failures);
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
void LaunchRefitNative(LeastSquaresCostMode cost_mode, std::uint32_t grid,
                       std::uint32_t threads, cudaStream_t stream,
                       DeviceLinkageAlgebra<M> algebra, const float* atom_dots,
                       const float* squared_norms, std::uint32_t sample_count,
                       const std::uint8_t* codes, float* coefficients,
                       float* errors, std::uint32_t* failures) {
  LaunchRefitRows<M, Capacity>(1U, cost_mode, grid, threads, stream, algebra, atom_dots,
                    squared_norms, codes, sample_count, coefficients, errors,
                    nullptr, failures);
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
void LaunchIcmSweep(LeastSquaresCostMode cost_mode, std::uint32_t sample_count,
                    cudaStream_t stream, DeviceLinkageAlgebra<M> algebra,
                    const float* atom_dots, const float* squared_norms,
                    const std::uint8_t* layer_orders, std::uint8_t* codes,
                    float* coefficients, float* errors) {
  // Runtime-M ICM keeps static cost kernels to preserve its register footprint.
  if constexpr (M == 0) {
    if (cost_mode == LeastSquaresCostMode::kNormalEquation) {
      IcmSweepKernel<M, Capacity, static_cast<int>(LeastSquaresCostMode::kNormalEquation)>
          <<<sample_count, 256U, 0U, stream>>>(
              algebra, atom_dots, squared_norms, layer_orders, sample_count,
              codes, coefficients, errors, cost_mode);
    } else {
      IcmSweepKernel<M, Capacity, static_cast<int>(LeastSquaresCostMode::kExplicitQuadratic)>
          <<<sample_count, 256U, 0U, stream>>>(
              algebra, atom_dots, squared_norms, layer_orders, sample_count,
              codes, coefficients, errors, cost_mode);
    }
  } else {
    IcmSweepKernel<M, Capacity><<<sample_count, 256U, 0U, stream>>>(
        algebra, atom_dots, squared_norms, layer_orders, sample_count, codes,
        coefficients, errors, cost_mode);
  }
}

__global__ void SquaredNormKernel(const float* residuals,
                                  std::uint32_t dimension,
                                  std::uint32_t sample_count,
                                  float* squared_norms) {
  const std::uint32_t sample = blockIdx.x;
  if (sample >= sample_count) {
    return;
  }
  __shared__ float partial[256];
  const float* residual =
      residuals + static_cast<std::size_t>(sample) * dimension;
  float norm2 = 0.0F;
  for (std::uint32_t axis = threadIdx.x; axis < dimension; axis += blockDim.x) {
    norm2 += residual[axis] * residual[axis];
  }
  partial[threadIdx.x] = norm2;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2U; offset > 0U; offset >>= 1U) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    squared_norms[sample] = partial[0];
  }
}

Status SgemmTransposedLeft(cublasHandle_t blas, const float* left,
                           std::uint32_t left_columns, const float* right,
                           std::uint32_t dimension, std::uint32_t right_columns,
                           float* output) {
  const float one = 1.0F;
  const float zero = 0.0F;
  const cublasStatus_t result = cublasSgemm(
      blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(left_columns),
      static_cast<int>(right_columns), static_cast<int>(dimension), &one, left,
      static_cast<int>(dimension), right, static_cast<int>(dimension), &zero,
      output, static_cast<int>(left_columns));
  return result == CUBLAS_STATUS_SUCCESS
             ? Status::Ok()
             : CudaBlasStatus("cublasSgemm linkage projection", result);
}

}  // namespace

template <int M>
struct CudaLinkageRsdDeviceModel<M>::State final {
  int device = 0;
  ModelShape shape;
  std::uint64_t logical_bytes = 0U;
  CudaDeviceBuffer<float> atoms;
  CudaDeviceBuffer<float> squared_norms;
  CudaDeviceBuffer<float> inverse_norms;
  CudaDeviceBuffer<float> gram;

  ~State() { cudaSetDevice(device); }
};

template <int M>
CudaLinkageRsdDeviceModel<M>::CudaLinkageRsdDeviceModel(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
CudaLinkageRsdDeviceModel<M>::~CudaLinkageRsdDeviceModel() = default;

template <int M>
StatusOr<std::shared_ptr<const CudaLinkageRsdDeviceModel<M>>>
CudaLinkageRsdDeviceModel<M>::Create(const LinkageCodebookAlgebra<M>& algebra,
                                     int device) {
  if (device < 0 || algebra.shape().dimension == 0U ||
      algebra.shape().dimension >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    return Status::InvalidArgument(
        "CUDA linkage rsd model received an invalid device or extent");
  }
  if (const cudaError_t result = cudaSetDevice(device); result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice linkage algebra model", result);
  }
  auto state = std::make_unique<State>();
  state->device = device;
  state->shape = algebra.shape();
  const ColumnMajorMatrixF32View atoms = algebra.atoms();
  const std::size_t atom_values =
      static_cast<std::size_t>(atoms.rows) * atoms.columns;
  const std::array<Status, 4> statuses = {
      UploadModel(&state->atoms, {atoms.values, atom_values}, nullptr),
      UploadModel(&state->squared_norms, algebra.squared_norms(), nullptr),
      UploadModel(&state->inverse_norms, algebra.inverse_norms(), nullptr),
      UploadModel(&state->gram, algebra.gram(), nullptr),
  };
  for (const Status& status : statuses) {
    if (!status.ok()) {
      return status;
    }
  }
  // Pageable H2D copies may return after host staging, before device arrival.
  // The immutable model is published to independent nonblocking worker streams.
  if (const cudaError_t result = cudaStreamSynchronize(nullptr);
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaStreamSynchronize linkage model upload", result);
  }
  state->logical_bytes =
      sizeof(float) *
      (state->atoms.capacity() + state->squared_norms.capacity() +
       state->inverse_norms.capacity() + state->gram.capacity());
  return std::shared_ptr<const CudaLinkageRsdDeviceModel<M>>(
      new CudaLinkageRsdDeviceModel<M>(std::move(state)));
}

template <int M>
StatusOr<std::shared_ptr<const CudaLinkageRsdDeviceModel<M>>>
CudaLinkageRsdDeviceModel<M>::Create(const ProductCodebooks<M>& codebooks,
                                     CudaContext& context) {
  const ModelShape& shape = codebooks.shape();
  if (context.device() < 0 || shape.dimension == 0U ||
      shape.dimension >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    return Status::InvalidArgument(
        "CUDA linkage rsd model received an invalid device or extent");
  }
  if (const cudaError_t result = cudaSetDevice(context.device());
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice linkage algebra model", result);
  }
  const ColumnMajorMatrixF32View atoms = codebooks.edge_flat();
  const std::uint32_t kAtoms = codebooks.shape().m * 256U;
  std::vector<float> squared_norms(kAtoms);
  std::vector<float> inverse_norms(kAtoms);
  for (std::uint32_t atom = 0U; atom < kAtoms; ++atom) {
    float sum = 0.0F;
    for (std::uint32_t axis = 0U; axis < shape.dimension; ++axis) {
      const float value = atoms.column(atom)[axis];
      sum += value * value;
    }
    squared_norms[atom] = sum;
    inverse_norms[atom] = sum > 0.0F ? 1.0F / std::sqrt(sum) : 0.0F;
  }
  auto state = std::make_unique<State>();
  state->device = context.device();
  state->shape = shape;
  const std::size_t atom_values =
      static_cast<std::size_t>(shape.dimension) * kAtoms;
  const auto stream = static_cast<cudaStream_t>(context.native_stream());
  const std::array<Status, 3> uploads = {
      UploadModel(&state->atoms, {atoms.values, atom_values}, stream),
      UploadModel(&state->squared_norms,
                  ArrayView<const float>(squared_norms), stream),
      UploadModel(&state->inverse_norms,
                  ArrayView<const float>(inverse_norms), stream),
  };
  for (const Status& status : uploads) {
    if (!status.ok()) return status;
  }
  if (const Status status =
          state->gram.Ensure(static_cast<std::size_t>(kAtoms) * kAtoms);
      !status.ok())
    return status;
  if (const Status status = CudaGemmF32(
          context, CudaGemmOperation::kTranspose, CudaGemmOperation::kNone,
          static_cast<int>(kAtoms), static_cast<int>(kAtoms),
          static_cast<int>(shape.dimension), 1.0F, state->atoms.data(),
          static_cast<int>(shape.dimension), state->atoms.data(),
          static_cast<int>(shape.dimension), 0.0F, state->gram.data(),
          static_cast<int>(kAtoms));
      !status.ok())
    return status;
  if (const Status status = context.Synchronize(); !status.ok()) {
    return status;
  }
  state->logical_bytes =
      sizeof(float) *
      (state->atoms.capacity() + state->squared_norms.capacity() +
       state->inverse_norms.capacity() + state->gram.capacity());
  return std::shared_ptr<const CudaLinkageRsdDeviceModel<M>>(
      new CudaLinkageRsdDeviceModel<M>(std::move(state)));
}

template <int M>
int CudaLinkageRsdDeviceModel<M>::device() const noexcept {
  return state_->device;
}

template <int M>
ModelShape CudaLinkageRsdDeviceModel<M>::shape() const noexcept {
  return state_->shape;
}

template <int M>
std::uint64_t CudaLinkageRsdDeviceModel<M>::logical_bytes() const noexcept {
  return state_->logical_bytes;
}

template <int M>
const float* CudaLinkageRsdDeviceModel<M>::atoms() const noexcept {
  return state_->atoms.data();
}

template <int M>
const float* CudaLinkageRsdDeviceModel<M>::squared_norms() const noexcept {
  return state_->squared_norms.data();
}

template <int M>
const float* CudaLinkageRsdDeviceModel<M>::inverse_norms() const noexcept {
  return state_->inverse_norms.data();
}

template <int M>
const float* CudaLinkageRsdDeviceModel<M>::gram() const noexcept {
  return state_->gram.data();
}

template <int M>
struct CudaLinkageRsdDevicePlan<M>::State final {
  CudaContext* context = nullptr;
  int device = 0;
  ModelShape shape;
  std::uint32_t maximum_rank = 2U;
  std::uint32_t icm_rounds = 1U;
  LeastSquaresCostMode cost_mode = LeastSquaresCostMode::kNormalEquation;
  std::shared_ptr<const CudaLinkageRsdDeviceModel<M>> model;

  CudaDeviceBuffer<float> atom_dots;
  CudaDeviceBuffer<float> residual_squared_norms;
  CudaDeviceBuffer<float> branch_residual_dots;
  CudaDeviceBuffer<std::uint32_t> branch_remaining_masks;
  CudaDeviceBuffer<std::uint8_t> proposal_codes;
  CudaDeviceBuffer<float> proposal_coefficients;
  CudaDeviceBuffer<float> proposal_errors;
  CudaDeviceBuffer<std::uint32_t> failures;
  CudaDeviceBuffer<std::uint8_t> root_first;
  std::uint32_t projected_batch_samples = 0U;

  ~State() { cudaSetDevice(device); }

  [[nodiscard]] DeviceLinkageAlgebra<M> AlgebraView() const noexcept {
    return {
        .m = shape.m,
        .dimension = shape.dimension,
        .atoms = model->atoms(),
        .squared_norms = model->squared_norms(),
        .inverse_norms = model->inverse_norms(),
        .gram = model->gram(),
    };
  }

  [[nodiscard]] Status EnsureWorkspace(std::uint32_t count) {
    const std::size_t rows = count;
    const std::size_t rsd_rows = std::min<std::size_t>(rows, kRsdChunkSamples);
    const std::size_t branch_rank_count = maximum_rank - 1U;
    const std::size_t proposal_count =
        RsdCandidateCount<M>(maximum_rank, shape.m);
    const std::size_t ranked_proposal_count =
        maximum_rank == 1U ? 0U : proposal_count;
    const std::array<Status, 8> statuses = {
        atom_dots.Ensure(rows * shape.m * 256U),
        residual_squared_norms.Ensure(rows),
        branch_residual_dots.Ensure(rsd_rows * branch_rank_count *
                                    BranchResidualValues(shape.m)),
        branch_remaining_masks.Ensure(rsd_rows * branch_rank_count * shape.m),
        proposal_codes.Ensure(rsd_rows * proposal_count * shape.m),
        proposal_coefficients.Ensure(rsd_rows * ranked_proposal_count *
                                     shape.m),
        proposal_errors.Ensure(rsd_rows * ranked_proposal_count),
        failures.Ensure(1U),
    };
    for (const Status& status : statuses) {
      if (!status.ok()) {
        return status;
      }
    }
    return Status::Ok();
  }

  [[nodiscard]] Status EnsureRootCandidateWorkspace(std::uint32_t count) {
    return root_first.Ensure(count);
  }
};

template <int M>
CudaLinkageRsdDevicePlan<M>::CudaLinkageRsdDevicePlan(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
CudaLinkageRsdDevicePlan<M>::~CudaLinkageRsdDevicePlan() = default;

template <int M>
CudaLinkageRsdDevicePlan<M>::CudaLinkageRsdDevicePlan(
    CudaLinkageRsdDevicePlan&& other) noexcept
    : state_(std::move(other.state_)) {}

template <int M>
CudaLinkageRsdDevicePlan<M>& CudaLinkageRsdDevicePlan<M>::operator=(
    CudaLinkageRsdDevicePlan&& other) noexcept {
  state_ = std::move(other.state_);
  return *this;
}

template <int M>
StatusOr<CudaLinkageRsdDevicePlan<M>> CudaLinkageRsdDevicePlan<M>::Create(
    const LinkageCodebookAlgebra<M>& algebra, CudaContext& context,
    std::uint32_t maximum_rank, LeastSquaresCostMode cost_mode,
    std::uint32_t icm_rounds) {
  auto model = CudaLinkageRsdDeviceModel<M>::Create(algebra, context.device());
  if (!model.ok()) return model.status();
  return Create(std::move(model).value(), context, maximum_rank, cost_mode,
                icm_rounds);
}

template <int M>
StatusOr<CudaLinkageRsdDevicePlan<M>> CudaLinkageRsdDevicePlan<M>::Create(
    std::shared_ptr<const CudaLinkageRsdDeviceModel<M>> model,
    CudaContext& context, std::uint32_t maximum_rank,
    LeastSquaresCostMode cost_mode, std::uint32_t icm_rounds) {
  if (model == nullptr || context.device() < 0 ||
      model->device() != context.device() ||
      !IsSupportedRsdRank(maximum_rank)) {
    return Status::InvalidArgument(
        "CUDA linkage rsd plan received an incompatible model or context");
  }
  if (const cudaError_t result = cudaSetDevice(context.device());
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", result);
  }
  auto state = std::make_unique<State>();
  state->context = &context;
  state->device = context.device();
  state->shape = model->shape();
  state->maximum_rank = maximum_rank;
  state->icm_rounds = icm_rounds;
  state->cost_mode = cost_mode;
  state->model = std::move(model);
  return CudaLinkageRsdDevicePlan<M>(std::move(state));
}

template <int M>
Status CudaLinkageRsdDevicePlan<M>::BeginBatch() {
  if (state_ == nullptr) {
    return Status::FailedPrecondition(
        "CUDA linkage rsd plan is not initialized");
  }
  if (const cudaError_t result = cudaSetDevice(state_->device);
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", result);
  }
  state_->projected_batch_samples = 0U;
  if (const Status status = state_->failures.Ensure(1U); !status.ok()) {
    return status;
  }
  const cudaError_t result = cudaMemsetAsync(
      state_->failures.data(), 0, sizeof(std::uint32_t),
      static_cast<cudaStream_t>(state_->context->native_stream()));
  return result == cudaSuccess
             ? Status::Ok()
             : CudaRuntimeStatus("cudaMemsetAsync linkage failures", result);
}

template <int M>
Status CudaLinkageRsdDevicePlan<M>::ReserveWorkspace(
    std::uint32_t sample_count) {
  if (state_ == nullptr || sample_count == 0U) {
    return Status::InvalidArgument(
        "CUDA linkage rsd workspace extent is invalid");
  }
  if (const cudaError_t result = cudaSetDevice(state_->device);
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", result);
  }
  return state_->EnsureWorkspace(sample_count);
}

template <int M>
Status CudaLinkageRsdDevicePlan<M>::BeginProjectedBatch(
    std::uint32_t sample_count) {
  if (const Status status = ReserveWorkspace(sample_count); !status.ok()) {
    return status;
  }
  if (const Status status = BeginBatch(); !status.ok()) {
    return status;
  }
  state_->projected_batch_samples = sample_count;
  return Status::Ok();
}

template <int M>
Status CudaLinkageRsdDevicePlan<M>::EnqueueProjectionSequence(
    CudaMatrixF32View residuals,
    ArrayView<const std::uint32_t> source_sample_offsets) {
  if (state_ == nullptr || state_->projected_batch_samples == 0U ||
      residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows ||
      residuals.columns != state_->projected_batch_samples ||
      source_sample_offsets.size() < 2U || source_sample_offsets[0] != 0U ||
      source_sample_offsets.back() != residuals.columns) {
    return Status::InvalidArgument(
        "CUDA linkage projected batch received invalid source offsets");
  }
  const cublasHandle_t blas =
      static_cast<cublasHandle_t>(state_->context->native_blas());
  const DeviceLinkageAlgebra<M> algebra = state_->AlgebraView();
  for (std::size_t index = 1U; index < source_sample_offsets.size(); ++index) {
    const std::uint32_t begin = source_sample_offsets[index - 1U];
    const std::uint32_t end = source_sample_offsets[index];
    if (end <= begin || end > residuals.columns) {
      return Status::InvalidArgument(
          "CUDA linkage projected batch source offsets are not ordered");
    }
    if (const Status status = SgemmTransposedLeft(
            blas, algebra.atoms, algebra.AtomCount(),
            residuals.values +
                static_cast<std::size_t>(begin) * algebra.dimension,
            algebra.dimension, end - begin,
            state_->atom_dots.data() +
                static_cast<std::size_t>(begin) * algebra.AtomCount());
        !status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

template <int M>
Status CudaLinkageRsdDevicePlan<M>::Enqueue(
    CudaMatrixF32View residuals, const std::uint8_t* device_icm_layer_orders,
    CudaLinkageRsdDeviceOutput<M> output,
    const float* device_residual_squared_norms) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows ||
      device_icm_layer_orders == nullptr || output.codes == nullptr ||
      output.coefficients == nullptr || output.squared_errors == nullptr) {
    return Status::InvalidArgument(
        "CUDA linkage rsd enqueue received an invalid device extent");
  }
  if (const Status status = ReserveWorkspace(residuals.columns); !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  const cublasHandle_t blas =
      static_cast<cublasHandle_t>(state_->context->native_blas());
  const DeviceLinkageAlgebra<M> algebra = state_->AlgebraView();
  const float* squared_norms = device_residual_squared_norms;
  cudaError_t cuda_result = cudaSuccess;
  if (squared_norms == nullptr) {
    SquaredNormKernel<<<residuals.columns, 256U, 0U, stream>>>(
        residuals.values, residuals.rows, residuals.columns,
        state_->residual_squared_norms.data());
    cuda_result = cudaPeekAtLastError();
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("SquaredNormKernel linkage", cuda_result);
    }
    squared_norms = state_->residual_squared_norms.data();
  }
  if (const Status status = SgemmTransposedLeft(
          blas, algebra.atoms, algebra.AtomCount(), residuals.values,
          algebra.dimension, residuals.columns, state_->atom_dots.data());
      !status.ok()) {
    return status;
  }
  return EnqueueProjectedRsdImpl(residuals.columns, device_icm_layer_orders,
                                 output, squared_norms, true, nullptr);
}

template <int M>
Status CudaLinkageRsdDevicePlan<M>::EnqueueRawRsd(
    CudaMatrixF32View residuals, CudaLinkageRsdDeviceOutput<M> output,
    const float* device_residual_squared_norms) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows ||
      output.codes == nullptr || output.coefficients == nullptr ||
      output.squared_errors == nullptr) {
    return Status::InvalidArgument(
        "CUDA raw RSD enqueue received an invalid device extent");
  }
  if (const Status status = ReserveWorkspace(residuals.columns); !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  const cublasHandle_t blas =
      static_cast<cublasHandle_t>(state_->context->native_blas());
  const DeviceLinkageAlgebra<M> algebra = state_->AlgebraView();
  const float* squared_norms = device_residual_squared_norms;
  cudaError_t cuda_result = cudaSuccess;
  if (squared_norms == nullptr) {
    SquaredNormKernel<<<residuals.columns, 256U, 0U, stream>>>(
        residuals.values, residuals.rows, residuals.columns,
        state_->residual_squared_norms.data());
    cuda_result = cudaPeekAtLastError();
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("SquaredNormKernel raw RSD", cuda_result);
    }
    squared_norms = state_->residual_squared_norms.data();
  }
  if (const Status status = SgemmTransposedLeft(
          blas, algebra.atoms, algebra.AtomCount(), residuals.values,
          algebra.dimension, residuals.columns, state_->atom_dots.data());
      !status.ok()) {
    return status;
  }
  return EnqueueProjectedRsdImpl(residuals.columns, nullptr, output,
                                 squared_norms, false, nullptr);
}

template <int M>
Status CudaLinkageRsdDevicePlan<M>::EnqueueRawRsdFixedLayerZero(
    CudaMatrixF32View residuals,
    const std::uint8_t* device_fixed_layer_zero_codes,
    CudaLinkageRsdDeviceOutput<M> output,
    const float* device_residual_squared_norms) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows ||
      device_fixed_layer_zero_codes == nullptr || output.codes == nullptr ||
      output.coefficients == nullptr || output.squared_errors == nullptr) {
    return Status::InvalidArgument(
        "CUDA fixed-layer-zero raw RSD enqueue received an invalid extent");
  }
  if (const Status status = ReserveWorkspace(residuals.columns); !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  const cublasHandle_t blas =
      static_cast<cublasHandle_t>(state_->context->native_blas());
  const DeviceLinkageAlgebra<M> algebra = state_->AlgebraView();
  const float* squared_norms = device_residual_squared_norms;
  cudaError_t cuda_result = cudaSuccess;
  if (squared_norms == nullptr) {
    SquaredNormKernel<<<residuals.columns, 256U, 0U, stream>>>(
        residuals.values, residuals.rows, residuals.columns,
        state_->residual_squared_norms.data());
    cuda_result = cudaPeekAtLastError();
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("SquaredNormKernel fixed-layer-zero raw RSD",
                               cuda_result);
    }
    squared_norms = state_->residual_squared_norms.data();
  }
  if (const Status status = SgemmTransposedLeft(
          blas, algebra.atoms, algebra.AtomCount(), residuals.values,
          algebra.dimension, residuals.columns, state_->atom_dots.data());
      !status.ok()) {
    return status;
  }
  return EnqueueProjectedRsdImpl(residuals.columns, nullptr, output,
                                 squared_norms, false,
                                 device_fixed_layer_zero_codes);
}

template <int M>
Status CudaLinkageRsdDevicePlan<M>::EnqueueRawRsdTopOneRoot(
    CudaMatrixF32View residuals, CudaLinkageRsdDeviceOutput<M> output,
    const float* device_residual_squared_norms) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows ||
      output.codes == nullptr || output.coefficients == nullptr ||
      output.squared_errors == nullptr) {
    return Status::InvalidArgument(
        "CUDA top-one-root raw RSD received an invalid extent");
  }
  if (const Status status = ReserveWorkspace(residuals.columns); !status.ok()) {
    return status;
  }
  if (const Status status =
          state_->EnsureRootCandidateWorkspace(residuals.columns);
      !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  const cublasHandle_t blas =
      static_cast<cublasHandle_t>(state_->context->native_blas());
  const DeviceLinkageAlgebra<M> algebra = state_->AlgebraView();
  const float* squared_norms = device_residual_squared_norms;
  cudaError_t cuda_result = cudaSuccess;
  if (squared_norms == nullptr) {
    SquaredNormKernel<<<residuals.columns, 256U, 0U, stream>>>(
        residuals.values, residuals.rows, residuals.columns,
        state_->residual_squared_norms.data());
    cuda_result = cudaPeekAtLastError();
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("SquaredNormKernel root-constrained raw RSD",
                               cuda_result);
    }
    squared_norms = state_->residual_squared_norms.data();
  }
  if (const Status status = SgemmTransposedLeft(
          blas, algebra.atoms, algebra.AtomCount(), residuals.values,
          algebra.dimension, residuals.columns, state_->atom_dots.data());
      !status.ok()) {
    return status;
  }
  SelectLayerZeroCandidatesKernel<M><<<residuals.columns, 256U, 0U, stream>>>(
      algebra, state_->atom_dots.data(), squared_norms, residuals.columns,
      state_->root_first.data(), nullptr);
  cuda_result = cudaPeekAtLastError();
  if (cuda_result != cudaSuccess) {
    return CudaRuntimeStatus("SelectLayerZeroCandidatesKernel", cuda_result);
  }
  return EnqueueProjectedRsdImpl(residuals.columns, nullptr, output,
                                 squared_norms, false,
                                 state_->root_first.data());
}

template <int M>
Status CudaLinkageRsdDevicePlan<M>::EnqueueProjectedRsd(
    const std::uint8_t* device_icm_layer_orders,
    CudaLinkageRsdDeviceOutput<M> output,
    const float* device_residual_squared_norms) {
  if (state_ == nullptr || state_->projected_batch_samples == 0U ||
      device_icm_layer_orders == nullptr || output.codes == nullptr ||
      output.coefficients == nullptr || output.squared_errors == nullptr ||
      device_residual_squared_norms == nullptr) {
    return Status::InvalidArgument(
        "CUDA linkage projected rsd received an invalid extent");
  }
  const std::uint32_t sample_count = state_->projected_batch_samples;
  state_->projected_batch_samples = 0U;
  return EnqueueProjectedRsdImpl(sample_count, device_icm_layer_orders, output,
                                 device_residual_squared_norms, true, nullptr);
}

template <int M>
Status CudaLinkageRsdDevicePlan<M>::EnqueueProjectedRsdImpl(
    std::uint32_t sample_count, const std::uint8_t* device_icm_layer_orders,
    CudaLinkageRsdDeviceOutput<M> output,
    const float* device_residual_squared_norms, bool run_icm,
    const std::uint8_t* device_fixed_layer_zero_codes,
    bool record_unsolved_supports) {
  if constexpr (M == 0) {
    if (state_->shape.m > kLegacyRuntimeScratchLayers)
      return EnqueueProjectedRsdImplWithCapacity<kWideRuntimeScratchLayers>(
          sample_count, device_icm_layer_orders, output, device_residual_squared_norms,
          run_icm, device_fixed_layer_zero_codes, record_unsolved_supports);
  }
  return EnqueueProjectedRsdImplWithCapacity<kModelScratchLayers<M>>(
      sample_count, device_icm_layer_orders, output, device_residual_squared_norms, run_icm,
      device_fixed_layer_zero_codes, record_unsolved_supports);
}

template <int M>
template <std::uint32_t Capacity>
Status CudaLinkageRsdDevicePlan<M>::EnqueueProjectedRsdImplWithCapacity(
    std::uint32_t sample_count, const std::uint8_t* device_icm_layer_orders,
    CudaLinkageRsdDeviceOutput<M> output,
    const float* device_residual_squared_norms, bool run_icm,
    const std::uint8_t* device_fixed_layer_zero_codes,
    bool record_unsolved_supports) {
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  const DeviceLinkageAlgebra<M> algebra = state_->AlgebraView();
  constexpr std::uint32_t kThreads = 128U;
  cudaError_t cuda_result = cudaSuccess;
  for (std::uint32_t block_offset = 0U; block_offset < sample_count;
       block_offset += kRsdChunkSamples) {
    const std::uint32_t count =
        std::min(kRsdChunkSamples, sample_count - block_offset);
    const float* block_atom_dots =
        state_->atom_dots.data() +
        static_cast<std::size_t>(block_offset) * algebra.AtomCount();
    const bool fixed_layer_zero = device_fixed_layer_zero_codes != nullptr;
    const bool rank1 = state_->maximum_rank == 1U;
    const bool rank3 = state_->maximum_rank == 3U;
    if (rank1 && fixed_layer_zero) {
      BuildRsdRankOneKernel<M, Capacity, true><<<count, 256U, 0U, stream>>>(
          algebra, block_atom_dots, count,
          device_fixed_layer_zero_codes + block_offset,
          state_->proposal_codes.data());
    } else if (rank1) {
      BuildRsdRankOneKernel<M, Capacity, false>
          <<<count, 256U, 0U, stream>>>(algebra, block_atom_dots, count,
                                        nullptr, state_->proposal_codes.data());
    } else if (rank3 && fixed_layer_zero) {
      BuildRsdRank3SeedsKernel<M, Capacity, true><<<count, 256U, 0U, stream>>>(
          algebra, block_atom_dots, count,
          device_fixed_layer_zero_codes + block_offset,
          state_->branch_residual_dots.data(),
          state_->branch_remaining_masks.data(), state_->proposal_codes.data());
    } else if (rank3) {
      BuildRsdRank3SeedsKernel<M, Capacity, false><<<count, 256U, 0U, stream>>>(
          algebra, block_atom_dots, count, nullptr,
          state_->branch_residual_dots.data(),
          state_->branch_remaining_masks.data(), state_->proposal_codes.data());
    } else if (fixed_layer_zero) {
      BuildRsdSeedsKernel<M, Capacity, true><<<count, 256U, 0U, stream>>>(
          algebra, block_atom_dots, count,
          device_fixed_layer_zero_codes + block_offset,
          state_->branch_residual_dots.data(),
          state_->branch_remaining_masks.data(), state_->proposal_codes.data());
    } else {
      BuildRsdSeedsKernel<M, Capacity, false><<<count, 256U, 0U, stream>>>(
          algebra, block_atom_dots, count, nullptr,
          state_->branch_residual_dots.data(),
          state_->branch_remaining_masks.data(), state_->proposal_codes.data());
    }
    cuda_result = cudaPeekAtLastError();
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("BuildRsdSeedsKernel", cuda_result);
    }
    if (rank1) {
      // Rank one is the native global-greedy path and has no deviation
      // branches to complete.
    } else if (rank3 && fixed_layer_zero) {
      if (state_->shape.m > 2) {
        CompleteRsdRank3BranchSuffixesKernel<M, Capacity, true>
            <<<count * 2U * static_cast<std::uint32_t>(state_->shape.m - 2),
               256U, 0U, stream>>>(algebra, state_->branch_residual_dots.data(),
                                   state_->branch_remaining_masks.data(), count,
                                   state_->proposal_codes.data());
      }
    } else if (rank3) {
      CompleteRsdRank3BranchSuffixesKernel<M, Capacity, false>
          <<<count * 2U * static_cast<std::uint32_t>(state_->shape.m - 1), 256U,
             0U, stream>>>(algebra, state_->branch_residual_dots.data(),
                           state_->branch_remaining_masks.data(), count,
                           state_->proposal_codes.data());
    } else if (fixed_layer_zero) {
      if (state_->shape.m > 2) {
        CompleteRsdBranchSuffixesKernel<M, Capacity, true>
            <<<count* static_cast<std::uint32_t>(state_->shape.m - 2), 256U, 0U,
               stream>>>(algebra, state_->branch_residual_dots.data(),
                         state_->branch_remaining_masks.data(), count,
                         state_->proposal_codes.data());
      }
    } else {
      CompleteRsdBranchSuffixesKernel<M, Capacity, false>
          <<<count* static_cast<std::uint32_t>(state_->shape.m - 1), 256U, 0U,
             stream>>>(algebra, state_->branch_residual_dots.data(),
                       state_->branch_remaining_masks.data(), count,
                       state_->proposal_codes.data());
    }
    if (!rank1) {
      cuda_result = cudaPeekAtLastError();
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("CompleteRsdBranchSuffixesKernel",
                                 cuda_result);
      }
    }
    if (rank1) {
      const std::uint32_t refit_grid = (count + kThreads - 1U) / kThreads;
      LaunchRefitSingleRsdProposal<M, Capacity>(
          state_->cost_mode, refit_grid, kThreads, stream, algebra,
          block_atom_dots, device_residual_squared_norms + block_offset,
          state_->proposal_codes.data(), count,
          output.codes +
              static_cast<std::size_t>(block_offset) * state_->shape.m,
          output.coefficients +
              static_cast<std::size_t>(block_offset) * state_->shape.m,
          output.squared_errors + block_offset,
          record_unsolved_supports ? state_->failures.data() : nullptr);
      cuda_result = cudaPeekAtLastError();
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("RefitSingleRsdProposalKernel", cuda_result);
      }
      continue;
    }
    const std::uint32_t proposals_per_sample =
        rank3 ? (fixed_layer_zero
                     ? static_cast<std::uint32_t>(2 * state_->shape.m - 1)
                     : static_cast<std::uint32_t>(2 * state_->shape.m + 1))
              : (fixed_layer_zero
                     ? static_cast<std::uint32_t>(state_->shape.m)
                     : static_cast<std::uint32_t>(state_->shape.m + 1));
    const std::uint32_t proposal_count = count * proposals_per_sample;
    const std::uint32_t proposal_grid =
        (proposal_count + kThreads - 1U) / kThreads;
    LaunchRefitRsdProposals<M, Capacity>(
        proposals_per_sample, state_->cost_mode, proposal_grid, kThreads, stream,
        algebra, block_atom_dots,
        device_residual_squared_norms + block_offset,
        state_->proposal_codes.data(), count,
        state_->proposal_coefficients.data(), state_->proposal_errors.data());
    cuda_result = cudaPeekAtLastError();
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("RefitRsdProposalsKernel", cuda_result);
    }
    const std::uint32_t select_grid = (count + kThreads - 1U) / kThreads;
    if (rank3 && fixed_layer_zero) {
      SelectBestRsdProposalKernel<M, 2U * M - 1U>
          <<<select_grid, kThreads, 0U, stream>>>(
              proposals_per_sample, state_->shape.m,
              state_->proposal_codes.data(),
              state_->proposal_coefficients.data(),
              state_->proposal_errors.data(), count,
              output.codes +
                  static_cast<std::size_t>(block_offset) * state_->shape.m,
              output.coefficients +
                  static_cast<std::size_t>(block_offset) * state_->shape.m,
              output.squared_errors + block_offset,
              record_unsolved_supports ? state_->failures.data() : nullptr);
    } else if (rank3) {
      SelectBestRsdProposalKernel<M, 2U * M + 1U>
          <<<select_grid, kThreads, 0U, stream>>>(
              proposals_per_sample, state_->shape.m,
              state_->proposal_codes.data(),
              state_->proposal_coefficients.data(),
              state_->proposal_errors.data(), count,
              output.codes +
                  static_cast<std::size_t>(block_offset) * state_->shape.m,
              output.coefficients +
                  static_cast<std::size_t>(block_offset) * state_->shape.m,
              output.squared_errors + block_offset,
              record_unsolved_supports ? state_->failures.data() : nullptr);
    } else if (fixed_layer_zero) {
      SelectBestRsdProposalKernel<M, M><<<select_grid, kThreads, 0U, stream>>>(
          proposals_per_sample, state_->shape.m, state_->proposal_codes.data(),
          state_->proposal_coefficients.data(), state_->proposal_errors.data(),
          count,
          output.codes +
              static_cast<std::size_t>(block_offset) * state_->shape.m,
          output.coefficients +
              static_cast<std::size_t>(block_offset) * state_->shape.m,
          output.squared_errors + block_offset,
          record_unsolved_supports ? state_->failures.data() : nullptr);
    } else {
      SelectBestRsdProposalKernel<M, M + 1U>
          <<<select_grid, kThreads, 0U, stream>>>(
              proposals_per_sample, state_->shape.m,
              state_->proposal_codes.data(),
              state_->proposal_coefficients.data(),
              state_->proposal_errors.data(), count,
              output.codes +
                  static_cast<std::size_t>(block_offset) * state_->shape.m,
              output.coefficients +
                  static_cast<std::size_t>(block_offset) * state_->shape.m,
              output.squared_errors + block_offset,
              record_unsolved_supports ? state_->failures.data() : nullptr);
    }
    cuda_result = cudaPeekAtLastError();
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("SelectBestRsdProposalKernel", cuda_result);
    }
  }
  if (run_icm) {
    for (std::uint32_t round = 0U; round < state_->icm_rounds; ++round) {
      LaunchIcmSweep<M, Capacity>(state_->cost_mode, sample_count, stream, algebra,
                        state_->atom_dots.data(), device_residual_squared_norms,
                        device_icm_layer_orders, output.codes,
                        output.coefficients, output.squared_errors);
      cuda_result = cudaPeekAtLastError();
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("IcmSweepKernel linkage", cuda_result);
      }
    }
  }
  return Status::Ok();
}

template <int M>
Status CudaLinkageRsdDevicePlan<M>::EnqueueNative(
    CudaMatrixF32View residuals, const std::uint8_t* device_icm_layer_orders,
    CudaLinkageRsdDeviceOutput<M> output) {
  return EnqueueRefinementImpl(residuals, device_icm_layer_orders, output,
                               true);
}

template <int M>
Status CudaLinkageRsdDevicePlan<M>::EnqueueRefinedSupports(
    CudaMatrixF32View residuals, const std::uint8_t* device_icm_layer_orders,
    CudaLinkageRsdDeviceOutput<M> output) {
  return EnqueueRefinementImpl(residuals, device_icm_layer_orders, output,
                               false);
}

template <int M>
Status CudaLinkageRsdDevicePlan<M>::EnqueueRefinementImpl(
    CudaMatrixF32View residuals, const std::uint8_t* device_icm_layer_orders,
    CudaLinkageRsdDeviceOutput<M> output, bool initialize_native_support) {
  if constexpr (M == 0) {
    if (state_->shape.m > kLegacyRuntimeScratchLayers)
      return EnqueueRefinementImplWithCapacity<kWideRuntimeScratchLayers>(
          residuals, device_icm_layer_orders, output, initialize_native_support);
  }
  return EnqueueRefinementImplWithCapacity<kModelScratchLayers<M>>(
      residuals, device_icm_layer_orders, output, initialize_native_support);
}

template <int M>
template <std::uint32_t Capacity>
Status CudaLinkageRsdDevicePlan<M>::EnqueueRefinementImplWithCapacity(
    CudaMatrixF32View residuals, const std::uint8_t* device_icm_layer_orders,
    CudaLinkageRsdDeviceOutput<M> output, bool initialize_native_support) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows ||
      device_icm_layer_orders == nullptr || output.codes == nullptr ||
      output.coefficients == nullptr || output.squared_errors == nullptr) {
    return Status::InvalidArgument(
        "CUDA linkage refinement received an invalid device extent");
  }
  if (const Status status = ReserveWorkspace(residuals.columns); !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  const cublasHandle_t blas =
      static_cast<cublasHandle_t>(state_->context->native_blas());
  const DeviceLinkageAlgebra<M> algebra = state_->AlgebraView();
  constexpr std::uint32_t kThreads = 128U;
  const std::uint32_t grid = (residuals.columns + kThreads - 1U) / kThreads;
  SquaredNormKernel<<<residuals.columns, 256U, 0U, stream>>>(
      residuals.values, residuals.rows, residuals.columns,
      state_->residual_squared_norms.data());
  cudaError_t cuda_result = cudaPeekAtLastError();
  if (cuda_result != cudaSuccess) {
    return CudaRuntimeStatus("SquaredNormKernel native linkage", cuda_result);
  }
  if (const Status status = SgemmTransposedLeft(
          blas, algebra.atoms, algebra.AtomCount(), residuals.values,
          algebra.dimension, residuals.columns, state_->atom_dots.data());
      !status.ok()) {
    return status;
  }
  if (initialize_native_support) {
    BuildNativeSupportKernel<M, Capacity><<<residuals.columns, 256U, 0U, stream>>>(
        algebra, state_->atom_dots.data(), residuals.columns, output.codes);
    cuda_result = cudaPeekAtLastError();
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("BuildNativeSupportKernel", cuda_result);
    }
  }
  LaunchRefitNative<M, Capacity>(state_->cost_mode, grid, kThreads, stream, algebra,
                       state_->atom_dots.data(),
                       state_->residual_squared_norms.data(), residuals.columns,
                       output.codes, output.coefficients, output.squared_errors,
                       state_->failures.data());
  cuda_result = cudaPeekAtLastError();
  if (cuda_result != cudaSuccess) {
    return CudaRuntimeStatus("RefitNativeKernel", cuda_result);
  }
  for (std::uint32_t round = 0U; round < state_->icm_rounds; ++round) {
    LaunchIcmSweep<M, Capacity>(state_->cost_mode, residuals.columns, stream, algebra,
                      state_->atom_dots.data(),
                      state_->residual_squared_norms.data(),
                      device_icm_layer_orders, output.codes,
                      output.coefficients, output.squared_errors);
    cuda_result = cudaPeekAtLastError();
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("IcmSweepKernel native linkage", cuda_result);
    }
  }
  return Status::Ok();
}

template <int M>
Status CudaLinkageRsdDevicePlan<M>::EnqueueFailureReadback(
    std::uint32_t* host_failures) {
  if (state_ == nullptr || host_failures == nullptr) {
    return Status::InvalidArgument(
        "CUDA linkage failure readback requires initialized storage");
  }
  const cudaError_t result = cudaMemcpyAsync(
      host_failures, state_->failures.data(), sizeof(std::uint32_t),
      cudaMemcpyDeviceToHost,
      static_cast<cudaStream_t>(state_->context->native_stream()));
  return result == cudaSuccess
             ? Status::Ok()
             : CudaRuntimeStatus("cudaMemcpyAsync linkage failures", result);
}

template <int M>
struct CudaLinkageRsdEncoder<M>::State final {
  State(CudaContext* context_in, int device_in, ModelShape shape_in,
        CudaLinkageRsdDevicePlan<M> plan_in) noexcept
      : context(context_in),
        device(device_in),
        shape(shape_in),
        plan(std::move(plan_in)) {}

  CudaContext* context = nullptr;
  int device = 0;
  ModelShape shape;
  CudaLinkageRsdDevicePlan<M> plan;
  CudaDeviceBuffer<float> residuals;
  CudaDeviceBuffer<std::uint8_t> layer_orders;
  CudaDeviceBuffer<std::uint8_t> codes;
  CudaDeviceBuffer<std::uint8_t> resident_support_codes;
  CudaDeviceBuffer<float> resident_squared_norms;
  CudaDeviceBuffer<float> coefficients;
  CudaDeviceBuffer<float> errors;
  CudaDeviceBuffer<std::uint8_t> fixed_layer_zero_codes;
  std::vector<std::uint8_t> host_fixed_layer_zero_codes;

  ~State() { cudaSetDevice(device); }

  [[nodiscard]] Status EnsureWorkspace(std::uint32_t count,
                                       bool need_host_input) {
    const std::size_t rows = count;
    if (need_host_input) {
      if (const Status status = residuals.Ensure(rows * shape.dimension);
          !status.ok()) {
        return status;
      }
    }
    const std::array<Status, 5> statuses = {
        layer_orders.Ensure(rows * shape.m), codes.Ensure(rows * shape.m),
        coefficients.Ensure(rows * shape.m), errors.Ensure(rows),
        fixed_layer_zero_codes.Ensure(rows),
    };
    for (const Status& status : statuses) {
      if (!status.ok()) {
        return status;
      }
    }
    return Status::Ok();
  }
};

template <int M>
CudaLinkageRsdEncoder<M>::CudaLinkageRsdEncoder(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
CudaLinkageRsdEncoder<M>::~CudaLinkageRsdEncoder() = default;

template <int M>
CudaLinkageRsdEncoder<M>::CudaLinkageRsdEncoder(
    CudaLinkageRsdEncoder&& other) noexcept
    : state_(std::move(other.state_)) {}

template <int M>
CudaLinkageRsdEncoder<M>& CudaLinkageRsdEncoder<M>::operator=(
    CudaLinkageRsdEncoder&& other) noexcept {
  state_ = std::move(other.state_);
  return *this;
}

template <int M>
StatusOr<CudaLinkageRsdEncoder<M>> CudaLinkageRsdEncoder<M>::Create(
    const LinkageCodebookAlgebra<M>& algebra, CudaContext& context,
    std::uint32_t maximum_rank, LeastSquaresCostMode cost_mode,
    std::uint32_t icm_rounds) {
  auto plan = CudaLinkageRsdDevicePlan<M>::Create(
      algebra, context, maximum_rank, cost_mode, icm_rounds);
  if (!plan.ok()) {
    return plan.status();
  }
  auto state = std::make_unique<State>(
      &context, context.device(), algebra.shape(), std::move(plan).value());
  return CudaLinkageRsdEncoder<M>(std::move(state));
}

template <int M>
StatusOr<CudaLinkageRsdEncoder<M>> CudaLinkageRsdEncoder<M>::Create(
    const ProductCodebooks<M>& codebooks, CudaContext& context,
    std::uint32_t maximum_rank, LeastSquaresCostMode cost_mode,
    std::uint32_t icm_rounds) {
  auto model = CudaLinkageRsdDeviceModel<M>::Create(codebooks, context);
  if (!model.ok()) return model.status();
  return Create(std::move(model).value(), context, maximum_rank, cost_mode,
                icm_rounds);
}

template <int M>
StatusOr<CudaLinkageRsdEncoder<M>> CudaLinkageRsdEncoder<M>::Create(
    std::shared_ptr<const CudaLinkageRsdDeviceModel<M>> model,
    CudaContext& context, std::uint32_t maximum_rank,
    LeastSquaresCostMode cost_mode, std::uint32_t icm_rounds) {
  if (model == nullptr) {
    return Status::InvalidArgument("CUDA RSD encoder requires a device model");
  }
  const ModelShape shape = model->shape();
  auto plan = CudaLinkageRsdDevicePlan<M>::Create(
      std::move(model), context, maximum_rank, cost_mode, icm_rounds);
  if (!plan.ok()) return plan.status();
  auto state = std::make_unique<State>(
      &context, context.device(), shape, std::move(plan).value());
  return CudaLinkageRsdEncoder<M>(std::move(state));
}

template <int M>
StatusOr<LinkageEncoding<M>> CudaLinkageRsdEncoder<M>::Encode(
    ColumnMajorMatrixF32View residuals,
    ArrayView<const std::uint8_t> icm_layer_orders) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U) {
    return Status::InvalidArgument(
        "CUDA linkage input must be a non-empty contiguous model-space matrix");
  }
  const std::size_t value_count =
      static_cast<std::size_t>(residuals.rows) * residuals.columns;
  for (std::size_t index = 0U; index < value_count; ++index) {
    if (!std::isfinite(residuals.values[index])) {
      return Status::InvalidArgument(
          "CUDA linkage input contains a non-finite value");
    }
  }
  return EncodeImpl(residuals.values, residuals.columns, false,
                    icm_layer_orders, CandidateMode::kRsd,
                    CudaLinkageRsdDevicePlan<M>::kDefaultBlockSamples, {});
}

template <int M>
StatusOr<LinkageEncoding<M>> CudaLinkageRsdEncoder<M>::EncodeRawRsd(
    ColumnMajorMatrixF32View residuals) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U) {
    return Status::InvalidArgument(
        "CUDA raw RSD input must be a non-empty contiguous model-space matrix");
  }
  const std::size_t value_count =
      static_cast<std::size_t>(residuals.rows) * residuals.columns;
  for (std::size_t index = 0U; index < value_count; ++index) {
    if (!std::isfinite(residuals.values[index])) {
      return Status::InvalidArgument(
          "CUDA raw RSD input contains a non-finite value");
    }
  }
  return EncodeImpl(residuals.values, residuals.columns, false, {},
                    CandidateMode::kRawRsd,
                    CudaLinkageRsdDevicePlan<M>::kDefaultBlockSamples, {});
}

template <int M>
StatusOr<LinkageEncoding<M>> CudaLinkageRsdEncoder<M>::EncodeNative(
    ColumnMajorMatrixF32View residuals,
    ArrayView<const std::uint8_t> icm_layer_orders) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U) {
    return Status::InvalidArgument(
        "CUDA native linkage input must be a non-empty contiguous model-space "
        "matrix");
  }
  const std::size_t value_count =
      static_cast<std::size_t>(residuals.rows) * residuals.columns;
  for (std::size_t index = 0U; index < value_count; ++index) {
    if (!std::isfinite(residuals.values[index])) {
      return Status::InvalidArgument(
          "CUDA native linkage input contains a non-finite value");
    }
  }
  return EncodeImpl(residuals.values, residuals.columns, false,
                    icm_layer_orders, CandidateMode::kNative,
                    CudaLinkageRsdDevicePlan<M>::kDefaultBlockSamples, {});
}

template <int M>
StatusOr<LinkageEncoding<M>> CudaLinkageRsdEncoder<M>::RefineSupports(
    ColumnMajorMatrixF32View residuals,
    ArrayView<const std::uint8_t> initial_codes,
    ArrayView<const std::uint8_t> icm_layer_orders) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      initial_codes.size() !=
          static_cast<std::size_t>(residuals.columns) * state_->shape.m) {
    return Status::InvalidArgument(
        "CUDA linkage support-refinement input has inconsistent extents");
  }
  const std::size_t value_count =
      static_cast<std::size_t>(residuals.rows) * residuals.columns;
  for (std::size_t index = 0U; index < value_count; ++index) {
    if (!std::isfinite(residuals.values[index])) {
      return Status::InvalidArgument(
          "CUDA linkage support-refinement input contains a non-finite value");
    }
  }
  return EncodeImpl(residuals.values, residuals.columns, false,
                    icm_layer_orders, CandidateMode::kInitialSupport,
                    CudaLinkageRsdDevicePlan<M>::kDefaultBlockSamples,
                    initial_codes);
}

template <int M>
StatusOr<LinkageEncoding<M>> CudaLinkageRsdEncoder<M>::EncodeDevice(
    CudaMatrixF32View residuals,
    ArrayView<const std::uint8_t> icm_layer_orders) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows) {
    return Status::InvalidArgument(
        "CUDA linkage device input must be a non-empty contiguous model-space "
        "matrix");
  }
  return EncodeImpl(residuals.values, residuals.columns, true, icm_layer_orders,
                    CandidateMode::kRsd,
                    CudaLinkageRsdDevicePlan<M>::kDefaultBlockSamples, {});
}

template <int M>
StatusOr<LinkageEncoding<M>> CudaLinkageRsdEncoder<M>::EncodeRawRsdDevice(
    CudaMatrixF32View residuals) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows) {
    return Status::InvalidArgument(
        "CUDA raw RSD device input must be a non-empty contiguous model-space "
        "matrix");
  }
  return EncodeImpl(residuals.values, residuals.columns, true, {},
                    CandidateMode::kRawRsd,
                    CudaLinkageRsdDevicePlan<M>::kDefaultBlockSamples, {});
}

template <int M>
StatusOr<CudaLinkageRsdSupportView>
CudaLinkageRsdEncoder<M>::EncodeRawRsdSupportDeviceResident(
    CudaMatrixF32View residuals) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows) {
    return Status::InvalidArgument(
        "CUDA raw RSD resident-support input is invalid");
  }
  return EncodeRawRsdSupportDeviceResidentImpl(residuals, {});
}

template <int M>
StatusOr<CudaLinkageRsdSupportView>
CudaLinkageRsdEncoder<M>::EncodeRawRsdFixedLayerZeroSupportDeviceResident(
    CudaMatrixF32View residuals,
    ArrayView<const std::uint32_t> fixed_layer_zero_codes) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows ||
      fixed_layer_zero_codes.size() != residuals.columns) {
    return Status::InvalidArgument(
        "CUDA fixed-layer-zero resident RSD input is invalid");
  }
  for (const std::uint32_t code : fixed_layer_zero_codes) {
    if (code >= 256U) {
      return Status::InvalidArgument(
          "CUDA fixed-layer-zero resident RSD code exceeds 8-bit root domain");
    }
  }
  return EncodeRawRsdSupportDeviceResidentImpl(residuals,
                                               fixed_layer_zero_codes);
}

template <int M>
StatusOr<CudaLinkageRsdSupportView>
CudaLinkageRsdEncoder<M>::EncodeRawRsdTopOneRootSupportDeviceResident(
    CudaMatrixF32View residuals) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows) {
    return Status::InvalidArgument(
        "CUDA top-one-root resident RSD input is invalid");
  }
  return EncodeRawRsdSupportDeviceResidentImpl(residuals, {}, true);
}

template <int M>
StatusOr<LinkageEncoding<M>>
CudaLinkageRsdEncoder<M>::EncodeRawRsdFixedLayerZeroDevice(
    CudaMatrixF32View residuals,
    ArrayView<const std::uint32_t> fixed_layer_zero_codes) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows ||
      fixed_layer_zero_codes.size() != residuals.columns) {
    return Status::InvalidArgument(
        "CUDA fixed-layer-zero raw RSD device input has invalid extents");
  }
  for (const std::uint32_t code : fixed_layer_zero_codes) {
    if (code >= 256U) {
      return Status::InvalidArgument(
          "CUDA fixed-layer-zero raw RSD code exceeds 8-bit root domain");
    }
  }
  return EncodeImpl(residuals.values, residuals.columns, true, {},
                    CandidateMode::kRawRsdFixedLayerZero,
                    CudaLinkageRsdDevicePlan<M>::kDefaultBlockSamples, {},
                    fixed_layer_zero_codes);
}

template <int M>
StatusOr<LinkageEncoding<M>>
CudaLinkageRsdEncoder<M>::EncodeRawRsdTopOneRootDevice(
    CudaMatrixF32View residuals) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows) {
    return Status::InvalidArgument(
        "CUDA root-constrained raw RSD device input is invalid");
  }
  return EncodeImpl(residuals.values, residuals.columns, true, {},
                    CandidateMode::kRawRsdRootTopOne,
                    CudaLinkageRsdDevicePlan<M>::kDefaultBlockSamples, {});
}

template <int M>
auto CudaLinkageRsdEncoder<M>::complete_batch_device_result() const
    -> CompleteBatchDeviceResult {
  return {state_->codes.data(), state_->coefficients.data()};
}

template <int M>
StatusOr<LinkageEncoding<M>>
CudaLinkageRsdEncoder<M>::EncodeDeviceCompleteBatch(
    CudaMatrixF32View residuals,
    ArrayView<const std::uint8_t> icm_layer_orders) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows) {
    return Status::InvalidArgument(
        "CUDA complete linkage batch must be a non-empty contiguous device "
        "matrix");
  }
  return EncodeImpl(residuals.values, residuals.columns, true, icm_layer_orders,
                    CandidateMode::kRsd, residuals.columns, {});
}

template <int M>
StatusOr<LinkageEncoding<M>>
CudaLinkageRsdEncoder<M>::EncodeDeviceCompleteBatchValidated(
    CudaMatrixF32View residuals,
    ArrayView<const std::uint8_t> icm_layer_orders) {
  if (state_ == nullptr || residuals.values == nullptr ||
      residuals.rows != state_->shape.dimension || residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows) {
    return Status::InvalidArgument(
        "CUDA complete linkage batch must be a non-empty contiguous device "
        "matrix");
  }
  return EncodeImpl(residuals.values, residuals.columns, true, icm_layer_orders,
                    CandidateMode::kRsd, residuals.columns, {}, {}, true);
}

template <int M>
StatusOr<LinkageEncoding<M>> CudaLinkageRsdEncoder<M>::EncodeImpl(
    const float* residuals, std::uint32_t sample_count, bool source_is_device,
    ArrayView<const std::uint8_t> icm_layer_orders, CandidateMode mode,
    std::uint32_t maximum_block_samples,
    ArrayView<const std::uint8_t> initial_codes,
    ArrayView<const std::uint32_t> fixed_layer_zero_codes,
    bool orders_prevalidated) {
  if (maximum_block_samples == 0U) {
    return Status::InvalidArgument("CUDA linkage block extent must be nonzero");
  }
  const bool raw_mode = mode == CandidateMode::kRawRsd ||
                        mode == CandidateMode::kRawRsdFixedLayerZero ||
                        mode == CandidateMode::kRawRsdRootTopOne;
  if (raw_mode) {
    if (!icm_layer_orders.empty()) {
      return Status::InvalidArgument(
          "CUDA raw RSD must not receive an ICM layer order");
    }
  } else {
    if (orders_prevalidated) {
      assert(ValidateIcmLayerOrders<M>(sample_count, icm_layer_orders,
                                       state_->shape.m)
                 .ok());
    } else {
      if (const Status status = ValidateIcmLayerOrders<M>(
              sample_count, icm_layer_orders, state_->shape.m);
          !status.ok()) {
        return status;
      }
    }
  }
  if ((mode == CandidateMode::kInitialSupport) != !initial_codes.empty() ||
      (!initial_codes.empty() &&
       initial_codes.size() !=
           static_cast<std::size_t>(sample_count) * state_->shape.m)) {
    return Status::InvalidArgument(
        "CUDA linkage initial support extent disagrees with candidate mode");
  }
  if ((mode == CandidateMode::kRawRsdFixedLayerZero) !=
          !fixed_layer_zero_codes.empty() ||
      (!fixed_layer_zero_codes.empty() &&
       fixed_layer_zero_codes.size() != sample_count)) {
    return Status::InvalidArgument(
        "CUDA fixed-layer-zero raw RSD extent disagrees with candidate mode");
  }
  if (const cudaError_t result = cudaSetDevice(state_->device);
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", result);
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  LinkageEncoding<M> output =
      detail::LinkageEncodingAccess::Create<M>(sample_count, state_->shape.m);
  if (const Status status = state_->plan.BeginBatch(); !status.ok()) {
    return status;
  }

  for (std::uint32_t block_offset = 0U; block_offset < sample_count;
       block_offset += maximum_block_samples) {
    const std::uint32_t count =
        std::min(maximum_block_samples, sample_count - block_offset);
    if (const Status status = state_->EnsureWorkspace(count, !source_is_device);
        !status.ok()) {
      return status;
    }
    const float* device_residuals =
        residuals +
        static_cast<std::size_t>(block_offset) * state_->shape.dimension;
    cudaError_t cuda_result = cudaSuccess;
    if (!source_is_device) {
      const std::size_t block_values =
          static_cast<std::size_t>(count) * state_->shape.dimension;
      cuda_result = cudaMemcpyAsync(state_->residuals.data(), device_residuals,
                                    block_values * sizeof(float),
                                    cudaMemcpyHostToDevice, stream);
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("cudaMemcpyAsync linkage residuals",
                                 cuda_result);
      }
      device_residuals = state_->residuals.data();
    }
    if (!raw_mode) {
      cuda_result = cudaMemcpyAsync(
          state_->layer_orders.data(),
          icm_layer_orders.data() +
              static_cast<std::size_t>(block_offset) * state_->shape.m,
          static_cast<std::size_t>(count) * state_->shape.m,
          cudaMemcpyHostToDevice, stream);
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("cudaMemcpyAsync linkage ICM orders",
                                 cuda_result);
      }
    }
    const CudaMatrixF32View device_block{device_residuals,
                                         state_->shape.dimension, count,
                                         state_->shape.dimension};
    const CudaLinkageRsdDeviceOutput<M> device_output{
        .codes = state_->codes.data(),
        .coefficients = state_->coefficients.data(),
        .squared_errors = state_->errors.data(),
    };
    if (mode == CandidateMode::kInitialSupport) {
      cuda_result = cudaMemcpyAsync(
          state_->codes.data(),
          initial_codes.data() +
              static_cast<std::size_t>(block_offset) * state_->shape.m,
          static_cast<std::size_t>(count) * state_->shape.m,
          cudaMemcpyHostToDevice, stream);
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("cudaMemcpyAsync linkage initial support",
                                 cuda_result);
      }
    }
    if (mode == CandidateMode::kRawRsdFixedLayerZero) {
      state_->host_fixed_layer_zero_codes.resize(count);
      for (std::uint32_t index = 0U; index < count; ++index) {
        state_->host_fixed_layer_zero_codes[index] = static_cast<std::uint8_t>(
            fixed_layer_zero_codes[block_offset + index]);
      }
      cuda_result = cudaMemcpyAsync(state_->fixed_layer_zero_codes.data(),
                                    state_->host_fixed_layer_zero_codes.data(),
                                    count, cudaMemcpyHostToDevice, stream);
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus(
            "cudaMemcpyAsync fixed-layer-zero raw RSD codes", cuda_result);
      }
    }
    Status encode_status =
        Status::Internal("CUDA linkage candidate mode was not dispatched");
    switch (mode) {
      case CandidateMode::kRsd:
        encode_status = state_->plan.Enqueue(
            device_block, state_->layer_orders.data(), device_output);
        break;
      case CandidateMode::kNative:
        encode_status = state_->plan.EnqueueNative(
            device_block, state_->layer_orders.data(), device_output);
        break;
      case CandidateMode::kInitialSupport:
        encode_status = state_->plan.EnqueueRefinedSupports(
            device_block, state_->layer_orders.data(), device_output);
        break;
      case CandidateMode::kRawRsd:
        encode_status = state_->plan.EnqueueRawRsd(device_block, device_output);
        break;
      case CandidateMode::kRawRsdFixedLayerZero:
        encode_status = state_->plan.EnqueueRawRsdFixedLayerZero(
            device_block, state_->fixed_layer_zero_codes.data(), device_output);
        break;
      case CandidateMode::kRawRsdRootTopOne:
        encode_status =
            state_->plan.EnqueueRawRsdTopOneRoot(device_block, device_output);
        break;
    }
    if (!encode_status.ok()) {
      return encode_status;
    }
    cuda_result = cudaMemcpyAsync(
        detail::LinkageEncodingAccess::Codes(&output) +
            static_cast<std::size_t>(block_offset) * state_->shape.m,
        state_->codes.data(), static_cast<std::size_t>(count) * state_->shape.m,
        cudaMemcpyDeviceToHost, stream);
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("cudaMemcpyAsync linkage codes", cuda_result);
    }
    cuda_result = cudaMemcpyAsync(
        detail::LinkageEncodingAccess::Coefficients(&output) +
            static_cast<std::size_t>(block_offset) * state_->shape.m,
        state_->coefficients.data(),
        static_cast<std::size_t>(count) * state_->shape.m * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("cudaMemcpyAsync linkage coefficients",
                               cuda_result);
    }
    cuda_result = cudaMemcpyAsync(
        detail::LinkageEncodingAccess::SquaredErrors(&output) + block_offset,
        state_->errors.data(), static_cast<std::size_t>(count) * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("cudaMemcpyAsync linkage errors", cuda_result);
    }
  }
  std::uint32_t failures = 0U;
  if (const Status status = state_->plan.EnqueueFailureReadback(&failures);
      !status.ok()) {
    return status;
  }
  if (const Status status = state_->context->Synchronize(); !status.ok()) {
    return status;
  }
  if (failures != 0U) {
    return Status::FailedPrecondition(
        "CUDA linkage candidate set contains no solvable support");
  }
  return std::move(output);
}

template <int M>
StatusOr<CudaLinkageRsdSupportView>
CudaLinkageRsdEncoder<M>::EncodeRawRsdSupportDeviceResidentImpl(
    CudaMatrixF32View residuals,
    ArrayView<const std::uint32_t> fixed_layer_zero_codes,
    bool select_top_one_root) {
  if (const cudaError_t result = cudaSetDevice(state_->device);
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", result);
  }
  if (const Status status = state_->resident_support_codes.Ensure(
          static_cast<std::size_t>(residuals.columns) * state_->shape.m);
      !status.ok()) {
    return status;
  }
  if (const Status status =
          state_->resident_squared_norms.Ensure(residuals.columns);
      !status.ok()) {
    return status;
  }
  if (const Status status = state_->plan.BeginBatch(); !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  constexpr std::uint32_t kBlock =
      CudaLinkageRsdDevicePlan<M>::kDefaultBlockSamples;
  for (std::uint32_t block_offset = 0U; block_offset < residuals.columns;
       block_offset += kBlock) {
    const std::uint32_t count =
        std::min(kBlock, residuals.columns - block_offset);
    if (const Status status = state_->EnsureWorkspace(count, false);
        !status.ok()) {
      return status;
    }
    const CudaMatrixF32View block{
        residuals.values +
            static_cast<std::size_t>(block_offset) * residuals.rows,
        residuals.rows, count, residuals.rows};
    float* block_norms = state_->resident_squared_norms.data() + block_offset;
    SquaredNormKernel<<<count, 256U, 0U, stream>>>(block.values, block.rows,
                                                   count, block_norms);
    if (const cudaError_t result = cudaPeekAtLastError();
        result != cudaSuccess) {
      return CudaRuntimeStatus("SquaredNormKernel resident raw RSD", result);
    }
    const CudaLinkageRsdDeviceOutput<M> device_output{
        .codes = state_->resident_support_codes.data() +
                 static_cast<std::size_t>(block_offset) * state_->shape.m,
        .coefficients = state_->coefficients.data(),
        .squared_errors = state_->errors.data(),
    };
    Status encode_status = Status::Ok();
    if (fixed_layer_zero_codes.empty()) {
      encode_status =
          select_top_one_root
              ? state_->plan.EnqueueRawRsdTopOneRoot(block, device_output,
                                                     block_norms)
              : state_->plan.EnqueueRawRsd(block, device_output, block_norms);
    } else {
      state_->host_fixed_layer_zero_codes.resize(count);
      for (std::uint32_t index = 0U; index < count; ++index) {
        state_->host_fixed_layer_zero_codes[index] = static_cast<std::uint8_t>(
            fixed_layer_zero_codes[block_offset + index]);
      }
      const cudaError_t copy_result =
          cudaMemcpyAsync(state_->fixed_layer_zero_codes.data(),
                          state_->host_fixed_layer_zero_codes.data(), count,
                          cudaMemcpyHostToDevice, stream);
      if (copy_result != cudaSuccess) {
        return CudaRuntimeStatus(
            "cudaMemcpyAsync fixed-layer-zero resident RSD codes", copy_result);
      }
      encode_status = state_->plan.EnqueueRawRsdFixedLayerZero(
          block, state_->fixed_layer_zero_codes.data(), device_output,
          block_norms);
    }
    if (!encode_status.ok()) {
      return encode_status;
    }
  }
  std::uint32_t failures = 0U;
  if (const Status status = state_->plan.EnqueueFailureReadback(&failures);
      !status.ok()) {
    return status;
  }
  if (const Status status = state_->context->Synchronize(); !status.ok()) {
    return status;
  }
  if (failures != 0U) {
    return Status::FailedPrecondition(
        "CUDA raw RSD candidate set contains no solvable support");
  }
  return CudaLinkageRsdSupportView{
      .codes = state_->resident_support_codes.data(),
      .squared_norms = state_->resident_squared_norms.data(),
      .sample_count = residuals.columns,
  };
}

template class CudaLinkageRsdDevicePlan<0>;
template class CudaLinkageRsdDevicePlan<5>;
template class CudaLinkageRsdDevicePlan<12>;
template class CudaLinkageRsdDevicePlan<6>;
template class CudaLinkageRsdDevicePlan<10>;
template class CudaLinkageRsdDeviceModel<0>;
template class CudaLinkageRsdDeviceModel<5>;
template class CudaLinkageRsdDeviceModel<12>;
template class CudaLinkageRsdDeviceModel<6>;
template class CudaLinkageRsdDeviceModel<10>;
template class CudaLinkageRsdEncoder<0>;
template class CudaLinkageRsdEncoder<5>;
template class CudaLinkageRsdEncoder<12>;
template class CudaLinkageRsdEncoder<6>;
template class CudaLinkageRsdEncoder<10>;

}  // namespace stlqpp
