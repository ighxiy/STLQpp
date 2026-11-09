#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/runtime/cuda/device_buffer.h"
#include "stlqpp/runtime/cuda/status.h"
#include "stlqpp/training/linkage/root_only/root_only_linkage_device_plan.h"
#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"

namespace stlqpp {
namespace {

constexpr float kAcceptanceEpsilon = 1.0e-6F;

std::uint32_t RootChunkSize(std::uint32_t root_count,
                            std::uint32_t sample_count,
                            std::uint64_t workspace_bytes) noexcept {
  const std::uint64_t budgeted =
      workspace_bytes == 0U
          ? 256U
          : workspace_bytes /
                (sizeof(float) * static_cast<std::uint64_t>(sample_count));
  std::uint32_t chunk = static_cast<std::uint32_t>(std::min<std::uint64_t>(
      root_count, std::max<std::uint64_t>(256U, budgeted)));
  chunk = std::max(1U, (chunk / 256U) * 256U);
  return std::min(chunk, root_count);
}

std::uint32_t RoundAllocatedSampleCount(std::uint32_t count) noexcept {
  if (count <= 8192U) {
    --count;
    count |= count >> 1U;
    count |= count >> 2U;
    count |= count >> 4U;
    count |= count >> 8U;
    count |= count >> 16U;
    return count + 1U;
  }
  if (count <= 65536U) {
    return ((count + 8191U) / 8192U) * 8192U;
  }
  return count;
}

std::uint32_t RoundDownSampleChunk(std::uint64_t count) noexcept {
  if (count > 8192U) {
    return static_cast<std::uint32_t>((count / 8192U) * 8192U);
  }
  if (count >= 256U) {
    return static_cast<std::uint32_t>((count / 256U) * 256U);
  }
  return static_cast<std::uint32_t>(std::max<std::uint64_t>(1U, count));
}

std::uint32_t LargeRootSampleChunkSize(
    std::uint32_t requested_sample_count, std::uint32_t root_count,
    std::uint32_t small_count, std::uint32_t dimension,
    const CudaRootOnlyLinkageModelOptions& options) noexcept {
  std::uint64_t chunk = requested_sample_count;
  if (options.full_root_dot_workspace_bytes != 0U && root_count != 0U) {
    const std::uint64_t full_root_limit =
        options.full_root_dot_workspace_bytes /
        (sizeof(float) * static_cast<std::uint64_t>(root_count));
    if (full_root_limit != 0U && chunk > full_root_limit) {
      chunk = std::min(chunk, full_root_limit);
      if (chunk >= 256U) {
        chunk = (chunk / 256U) * 256U;
      }
      chunk = std::max<std::uint64_t>(1U, chunk);
    }
  }

  if (options.encoder_workspace_bytes != 0U && small_count != 0U) {
    // This is the root-only form of DEV's wrapper estimate.  Root-only has
    // one global-greedy proposal, hence small_rsd == 1.
    const std::uint64_t floats_per_sample =
        5ULL * small_count + dimension +
        std::min<std::uint32_t>(256U, std::max(1U, root_count)) + 64ULL;
    const std::uint64_t bytes_per_sample = floats_per_sample * sizeof(float);
    const std::uint64_t encoder_budget = options.encoder_workspace_bytes / 2U;
    std::uint64_t budget_limit = encoder_budget / bytes_per_sample;
    budget_limit = std::max<std::uint64_t>(1U, budget_limit);
    std::uint32_t candidate =
        RoundDownSampleChunk(std::min(chunk, budget_limit));
    while (candidate > 1U &&
           static_cast<std::uint64_t>(RoundAllocatedSampleCount(candidate)) *
                   bytes_per_sample >
               encoder_budget) {
      if (candidate > 8192U) {
        candidate -= 8192U;
      } else if (candidate >= 256U) {
        candidate -= 256U;
      } else {
        --candidate;
      }
    }
    chunk = std::min<std::uint64_t>(chunk, candidate);
  }
  return static_cast<std::uint32_t>(std::max<std::uint64_t>(1U, chunk));
}

template <typename T>
Status UploadSynchronous(CudaDeviceBuffer<T>* destination,
                         ArrayView<const T> source, const char* operation) {
  if (const Status status = destination->Ensure(source.size()); !status.ok()) {
    return status;
  }
  const cudaError_t result =
      cudaMemcpy(destination->data(), source.data(), source.size() * sizeof(T),
                 cudaMemcpyHostToDevice);
  return result == cudaSuccess ? Status::Ok()
                               : CudaRuntimeStatus(operation, result);
}

template <int M>
struct DeviceRootOnlyAlgebra final {
  std::uint32_t m = M;
  __host__ __device__ constexpr std::uint32_t SmallCount() const {
    return ((M == 0 ? m : M) - 1U) * 256U;
  }

  std::uint32_t dimension = 0U;
  std::uint32_t root_count = 0U;
  const float* roots = nullptr;
  const float* small = nullptr;
  const float* root_norms2 = nullptr;
  const float* root_inverse_norms = nullptr;
  const float* small_norms2 = nullptr;
  const float* small_inverse_norms = nullptr;
  const float* small_gram = nullptr;
  const float* root_small_gram = nullptr;
  const float* root_small_gram_transposed = nullptr;

  __device__ __forceinline__ float SmallDot(std::uint32_t row,
                                            std::uint32_t column) const {
    return small_gram[static_cast<std::size_t>(column) * SmallCount() + row];
  }
  __device__ __forceinline__ float RootSmallDot(std::uint32_t root,
                                                std::uint32_t flat) const {
    return root_small_gram[static_cast<std::size_t>(root) * SmallCount() +
                           flat];
  }
};

__device__ __forceinline__ bool Better(float candidate_score,
                                       std::uint32_t candidate,
                                       float incumbent_score,
                                       std::uint32_t incumbent) {
  return candidate_score > incumbent_score ||
         (candidate_score == incumbent_score && candidate < incumbent);
}

__device__ __forceinline__ int PackedUpperIndex(int row, int column) {
  return row + (column * (column + 1)) / 2;
}

template <int M>
__device__ bool CholeskyUpperPacked(std::uint32_t actual_m, float* matrix) {
  const int m = M == 0 ? actual_m : M;
#pragma unroll
  for (int column = 0; column < m; ++column) {
#pragma unroll
    for (int row = 0; row <= column; ++row) {
      float value = matrix[PackedUpperIndex(row, column)];
#pragma unroll
      for (int prior = 0; prior < row; ++prior) {
        value -= matrix[PackedUpperIndex(prior, row)] *
                 matrix[PackedUpperIndex(prior, column)];
      }
      if (row == column) {
        if (!(value > 0.0F)) {
          return false;
        }
        matrix[PackedUpperIndex(row, column)] = sqrtf(value);
      } else {
        matrix[PackedUpperIndex(row, column)] =
            value / matrix[PackedUpperIndex(row, row)];
      }
    }
  }
  return true;
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__device__ void SolveCholeskyUpperPacked(std::uint32_t actual_m,
                                         const float* factor,
                                         float* right_hand_side) {
  const int m = M == 0 ? actual_m : M;
  float forward[Capacity];
#pragma unroll
  for (int row = 0; row < m; ++row) {
    float value = right_hand_side[row];
#pragma unroll
    for (int prior = 0; prior < row; ++prior) {
      value -= factor[PackedUpperIndex(prior, row)] * forward[prior];
    }
    forward[row] = value / factor[PackedUpperIndex(row, row)];
  }
  for (int row = m - 1; row >= 0; --row) {
    float value = forward[row];
    for (int next = row + 1; next < m; ++next) {
      value -= factor[PackedUpperIndex(row, next)] * right_hand_side[next];
    }
    right_hand_side[row] = value / factor[PackedUpperIndex(row, row)];
  }
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>, bool kDirectRootSmall = false>
__device__ bool Refit(DeviceRootOnlyAlgebra<M> algebra, std::uint32_t root,
                      const std::uint8_t* trailing, float root_dot,
                      float root_norm2, const float* small_dots,
                      float residual_norm2, float* coefficients,
                      float* squared_error) {
  const int m = M == 0 ? algebra.m : M;
  constexpr int kPackedCapacity =
      Capacity * (Capacity + 1) / 2;
  const int kPacked = (m * (m + 1)) / 2;
  float original[kPackedCapacity];
  float factor[kPackedCapacity];
  float rhs[Capacity]{};
  rhs[0] = root_dot;
  original[PackedUpperIndex(0, 0)] = root_norm2;
  for (std::uint32_t column = 1U; column < static_cast<std::uint32_t>(m);
       ++column) {
    const std::uint32_t flat_column =
        (column - 1U) * 256U + trailing[column - 1U];
    rhs[column] = small_dots[flat_column];
    if constexpr (kDirectRootSmall) {
      const float* root_values =
          algebra.roots + static_cast<std::size_t>(root) * algebra.dimension;
      const float* small_values =
          algebra.small +
          static_cast<std::size_t>(flat_column) * algebra.dimension;
      float root_small = 0.0F;
      for (std::uint32_t axis = 0U; axis < algebra.dimension; ++axis) {
        root_small += small_values[axis] * root_values[axis];
      }
      original[PackedUpperIndex(0, static_cast<int>(column))] = root_small;
    } else {
      original[PackedUpperIndex(0, static_cast<int>(column))] =
          algebra.RootSmallDot(root, flat_column);
    }
    for (std::uint32_t row = 1U; row <= column; ++row) {
      const std::uint32_t flat_row = (row - 1U) * 256U + trailing[row - 1U];
      original[PackedUpperIndex(static_cast<int>(row),
                                static_cast<int>(column))] =
          algebra.SmallDot(flat_row, flat_column);
    }
  }

  bool solved = false;
  float bump = 0.0F;
  for (std::uint32_t attempt = 0U; attempt < 4U && !solved; ++attempt) {
    for (std::uint32_t index = 0U; index < static_cast<std::uint32_t>(kPacked);
         ++index) {
      factor[index] = original[index];
    }
    if (bump != 0.0F) {
      for (std::uint32_t diagonal = 0U;
           diagonal < static_cast<std::uint32_t>(m); ++diagonal) {
        factor[PackedUpperIndex(static_cast<int>(diagonal),
                                static_cast<int>(diagonal))] += bump;
      }
    }
    solved = CholeskyUpperPacked<M>(m, factor);
    if (!solved) {
      bump = attempt == 0U ? 1.0e-6F : bump * 10.0F;
    }
  }
  if (!solved) {
    return false;
  }
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    coefficients[layer] = rhs[layer];
  }
  SolveCholeskyUpperPacked<M, Capacity>(m, factor, coefficients);

  float data_dot = 0.0F;
  float reconstructed_norm = 0.0F;
  for (std::uint32_t column = 0U; column < static_cast<std::uint32_t>(m);
       ++column) {
    data_dot += coefficients[column] * rhs[column];
    for (std::uint32_t row = 0U; row < static_cast<std::uint32_t>(m); ++row) {
      const float gram =
          row <= column ? original[PackedUpperIndex(static_cast<int>(row),
                                                    static_cast<int>(column))]
                        : original[PackedUpperIndex(static_cast<int>(column),
                                                    static_cast<int>(row))];
      reconstructed_norm += coefficients[column] * coefficients[row] * gram;
    }
  }
  *squared_error = residual_norm2 - 2.0F * data_dot + reconstructed_norm;
  return isfinite(*squared_error);
}

__global__ void ComputeNormsKernel(const float* residuals,
                                   std::uint32_t dimension,
                                   std::uint32_t sample_count,
                                   float* squared_norms) {
  const std::uint32_t sample = blockIdx.x;
  if (sample >= sample_count) {
    return;
  }
  const std::uint32_t lane = threadIdx.x;
  const float* residual =
      residuals + static_cast<std::size_t>(sample) * dimension;
  float norm2 = 0.0F;
  for (std::uint32_t axis = lane; axis < dimension; axis += blockDim.x) {
    norm2 += residual[axis] * residual[axis];
  }
  __shared__ float partial[256];
  partial[lane] = norm2;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2U; offset > 0U; offset >>= 1U) {
    if (lane < offset) {
      partial[lane] += partial[lane + offset];
    }
    __syncthreads();
  }
  if (lane == 0U) {
    squared_norms[sample] = partial[0];
  }
}

__global__ void InitializeRootBestKernel(std::uint32_t sample_count,
                                         float* best_scores,
                                         std::uint32_t* best_roots,
                                         float* best_dots, float* best_norms2) {
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= sample_count) {
    return;
  }
  best_scores[sample] = -CUDART_INF_F;
  best_roots[sample] = 0xffffffffU;
  best_dots[sample] = 0.0F;
  best_norms2[sample] = 0.0F;
}

template <int M, bool kIcm>
__global__ void UpdateRootBestKernel(
    DeviceRootOnlyAlgebra<M> algebra, std::uint32_t root_offset,
    std::uint32_t root_count, const float* root_dots,
    const std::uint8_t* trailing, const float* coefficients,
    std::uint32_t sample_count, float* best_scores, std::uint32_t* best_roots,
    float* best_dots, float* best_norms2) {
  const int m = M == 0 ? algebra.m : M;
  const std::uint32_t sample = blockIdx.x;
  if (sample >= sample_count) {
    return;
  }
  const std::uint32_t lane = threadIdx.x;
  float lane_score = -CUDART_INF_F;
  std::uint32_t lane_root = 0xffffffffU;
  float lane_dot = 0.0F;
  float lane_norm2 = 0.0F;
  const float* sample_root_dots =
      root_dots + static_cast<std::size_t>(sample) * root_count;
  for (std::uint32_t local = lane; local < root_count; local += blockDim.x) {
    const std::uint32_t root = root_offset + local;
    const float root_dot = sample_root_dots[local];
    float remaining_dot = root_dot;
    if constexpr (kIcm) {
      const std::uint8_t* sample_trailing =
          trailing + static_cast<std::size_t>(sample) * (m - 1U);
      const float* sample_coefficients =
          coefficients + static_cast<std::size_t>(sample) * m;
      float correction = 0.0F;
      for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(m);
           ++layer) {
        const std::uint32_t flat =
            (layer - 1U) * 256U + sample_trailing[layer - 1U];
        correction +=
            sample_coefficients[layer] *
            algebra.root_small_gram_transposed[
                static_cast<std::size_t>(flat) * algebra.root_count + root];
      }
      remaining_dot -= correction;
    }
    const float inverse_norm = algebra.root_inverse_norms[root];
    const float score = fabsf(remaining_dot * inverse_norm);
    if (Better(score, root, lane_score, lane_root)) {
      lane_score = score;
      lane_root = root;
      lane_dot = root_dot;
      lane_norm2 =
          inverse_norm > 0.0F ? 1.0F / (inverse_norm * inverse_norm) : 0.0F;
    }
  }

  __shared__ float scores[256];
  __shared__ std::uint32_t roots[256];
  __shared__ float dots[256];
  __shared__ float norms2[256];
  scores[lane] = lane_score;
  roots[lane] = lane_root;
  dots[lane] = lane_dot;
  norms2[lane] = lane_norm2;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2U; offset > 0U; offset >>= 1U) {
    if (lane < offset && Better(scores[lane + offset], roots[lane + offset],
                                scores[lane], roots[lane])) {
      scores[lane] = scores[lane + offset];
      roots[lane] = roots[lane + offset];
      dots[lane] = dots[lane + offset];
      norms2[lane] = norms2[lane + offset];
    }
    __syncthreads();
  }
  if (lane == 0U &&
      Better(scores[0], roots[0], best_scores[sample], best_roots[sample])) {
    best_scores[sample] = scores[0];
    best_roots[sample] = roots[0];
    best_dots[sample] = dots[0];
    best_norms2[sample] = norms2[0];
  }
}

template <int M>
__global__ void BuildRootProjectedSmallDotsKernel(
    DeviceRootOnlyAlgebra<M> algebra, const float* small_dots,
    const std::uint32_t* roots, const float* root_dots,
    const float* root_norms2, std::uint32_t sample_count,
    float* remaining_dots) {
  const std::uint32_t kSmallCount = algebra.SmallCount();
  const std::uint64_t flat_index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t extent =
      static_cast<std::uint64_t>(sample_count) * kSmallCount;
  if (flat_index >= extent) {
    return;
  }
  const std::uint32_t sample =
      static_cast<std::uint32_t>(flat_index / kSmallCount);
  const std::uint32_t flat =
      static_cast<std::uint32_t>(flat_index % kSmallCount);
  const std::uint32_t root = roots[sample];
  const float root_norm2 = root_norms2[sample];
  const float root_alpha =
      root_norm2 > 0.0F ? root_dots[sample] / root_norm2 : 0.0F;
  remaining_dots[flat_index] =
      small_dots[flat_index] - root_alpha * algebra.RootSmallDot(root, flat);
}

template <int M>
__global__ __launch_bounds__(256, 1) void GlobalGreedyKernel(
    DeviceRootOnlyAlgebra<M> algebra, std::uint32_t sample_count,
    float* remaining_dots, std::uint8_t* trailing) {
  const int m = M == 0 ? algebra.m : M;
  const std::uint32_t kSmallCount = algebra.SmallCount();
  const std::uint32_t sample = blockIdx.x;
  if (sample >= sample_count) {
    return;
  }
  const std::uint32_t lane = threadIdx.x;
  float* sample_dots =
      remaining_dots + static_cast<std::size_t>(sample) * kSmallCount;
  std::uint8_t* sample_codes =
      trailing + static_cast<std::size_t>(sample) * (m - 1U);
  __shared__ float scores[256];
  __shared__ std::uint32_t flats[256];
  __shared__ std::uint32_t selected_flat;
  __shared__ float selected_alpha;
  __shared__ std::uint32_t available_layers;
  if (lane < static_cast<std::uint32_t>(m - 1U)) {
    sample_codes[lane] = 0U;
  }
  if (lane == 0U) {
    available_layers = ((1U << m) - 1U) & ~1U;
  }
  __syncthreads();

  for (std::uint32_t step = 1U; step < static_cast<std::uint32_t>(m); ++step) {
    float lane_score = -CUDART_INF_F;
    std::uint32_t lane_flat = 0xffffffffU;
    for (std::uint32_t flat = lane; flat < kSmallCount; flat += blockDim.x) {
      const std::uint32_t layer = flat / 256U + 1U;
      if ((available_layers & (1U << layer)) == 0U) {
        continue;
      }
      const float score =
          fabsf(sample_dots[flat] * algebra.small_inverse_norms[flat]);
      if (Better(score, flat, lane_score, lane_flat)) {
        lane_score = score;
        lane_flat = flat;
      }
    }
    scores[lane] = lane_score;
    flats[lane] = lane_flat;
    __syncthreads();
    for (std::uint32_t offset = blockDim.x / 2U; offset > 0U; offset >>= 1U) {
      if (lane < offset && Better(scores[lane + offset], flats[lane + offset],
                                  scores[lane], flats[lane])) {
        scores[lane] = scores[lane + offset];
        flats[lane] = flats[lane + offset];
      }
      __syncthreads();
    }
    if (lane == 0U) {
      selected_flat = flats[0];
      const std::uint32_t layer = selected_flat / 256U + 1U;
      sample_codes[layer - 1U] =
          static_cast<std::uint8_t>(selected_flat & 255U);
      available_layers &= ~(1U << layer);
      const float inverse_norm = algebra.small_inverse_norms[selected_flat];
      selected_alpha = sample_dots[selected_flat] * inverse_norm * inverse_norm;
    }
    __syncthreads();
    for (std::uint32_t flat = lane; flat < kSmallCount; flat += blockDim.x) {
      sample_dots[flat] -=
          selected_alpha * algebra.SmallDot(flat, selected_flat);
    }
    __syncthreads();
  }
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__global__ void InitialRefitKernel(
    DeviceRootOnlyAlgebra<M> algebra, const float* small_dots,
    const float* residual_norms2, const std::uint32_t* roots,
    const float* root_dots, const float* root_norms2,
    const std::uint8_t* trailing, std::uint32_t sample_count,
    float* coefficients, float* errors, std::uint32_t* failures) {
  const int m = M == 0 ? algebra.m : M;
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= sample_count) {
    return;
  }
  float* sample_coefficients =
      coefficients + static_cast<std::size_t>(sample) * m;
  if (!Refit<M, Capacity>(
          algebra, roots[sample],
          trailing + static_cast<std::size_t>(sample) * (m - 1U),
          root_dots[sample], root_norms2[sample],
          small_dots + static_cast<std::size_t>(sample) * algebra.SmallCount(),
          residual_norms2[sample], sample_coefficients, errors + sample)) {
    for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
         ++layer) {
      sample_coefficients[layer] = 0.0F;
    }
    errors[sample] = residual_norms2[sample];
    atomicAdd(failures, 1U);
  }
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__global__ void AcceptRootIcmKernel(
    DeviceRootOnlyAlgebra<M> algebra, const float* small_dots,
    const float* residual_norms2, const float* residuals,
    std::uint32_t dimension, const std::uint32_t* candidate_roots,
    std::uint32_t sample_count, std::uint32_t* roots, float* root_dots,
    float* root_norms2, const std::uint8_t* trailing, float* coefficients,
    float* errors) {
  const int m = M == 0 ? algebra.m : M;
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= sample_count || candidate_roots[sample] == roots[sample]) {
    return;
  }
  float trial_coefficients[Capacity]{};
  float trial_error = 0.0F;
  const float* residual =
      residuals + static_cast<std::size_t>(sample) * dimension;
  const float* root_vector =
      algebra.roots +
      static_cast<std::size_t>(candidate_roots[sample]) * dimension;
  float candidate_dot = 0.0F;
  float candidate_norm2 = 0.0F;
  for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
    const float value = root_vector[axis];
    candidate_norm2 += value * value;
    candidate_dot += residual[axis] * value;
  }
  if (!Refit<M, Capacity, true>(
          algebra, candidate_roots[sample],
          trailing + static_cast<std::size_t>(sample) * (m - 1U), candidate_dot,
          candidate_norm2,
          small_dots + static_cast<std::size_t>(sample) * algebra.SmallCount(),
          residual_norms2[sample], trial_coefficients, &trial_error) ||
      !(trial_error + kAcceptanceEpsilon < errors[sample])) {
    return;
  }
  roots[sample] = candidate_roots[sample];
  root_dots[sample] = candidate_dot;
  root_norms2[sample] = candidate_norm2;
  errors[sample] = trial_error;
  float* destination = coefficients + static_cast<std::size_t>(sample) * m;
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    destination[layer] = trial_coefficients[layer];
  }
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__global__ __launch_bounds__(256, 1) void SmallIcmKernel(
    DeviceRootOnlyAlgebra<M> algebra, std::uint32_t layer,
    const float* small_dots, const float* residual_norms2,
    const std::uint32_t* roots, const float* root_dots,
    const float* root_norms2, std::uint32_t sample_count,
    std::uint8_t* trailing, float* coefficients, float* errors) {
  const int m = M == 0 ? algebra.m : M;
  const std::uint32_t sample = blockIdx.x;
  if (sample >= sample_count) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x);
  std::uint8_t* sample_codes =
      trailing + static_cast<std::size_t>(sample) * (m - 1U);
  float* sample_coefficients =
      coefficients + static_cast<std::size_t>(sample) * m;
  const float* sample_dots =
      small_dots + static_cast<std::size_t>(sample) * algebra.SmallCount();

  __shared__ std::uint8_t shared_codes[Capacity];
  __shared__ float shared_coefficients[Capacity];
  if (lane < m - 1) {
    shared_codes[lane + 1] = sample_codes[lane];
  }
  if (lane < m) {
    shared_coefficients[lane] = sample_coefficients[lane];
  }
  if (lane == 0) {
    shared_codes[0] = 0U;
  }
  __syncthreads();

  const int old_code = static_cast<int>(sample_codes[layer - 1U]);
  const int atom = (static_cast<int>(layer) - 1) * 256 + lane;
  float remaining_dot = sample_dots[atom];
  remaining_dot -=
      shared_coefficients[0] *
      algebra.RootSmallDot(roots[sample], static_cast<std::uint32_t>(atom));
#pragma unroll
  for (int other = 1; other < m; ++other) {
    if (other == static_cast<int>(layer)) {
      continue;
    }
    const int selected =
        (other - 1) * 256 + static_cast<int>(shared_codes[other]);
    remaining_dot -= shared_coefficients[other] *
                     algebra.SmallDot(static_cast<std::uint32_t>(atom),
                                      static_cast<std::uint32_t>(selected));
  }
  __shared__ float scores[256];
  __shared__ int codes[256];
  scores[lane] = fabsf(remaining_dot * algebra.small_inverse_norms[atom]);
  codes[lane] = lane;
  __syncthreads();
  for (int offset = static_cast<int>(blockDim.x) / 2; offset > 0;
       offset >>= 1) {
    if (lane < offset) {
      const float other_score = scores[lane + offset];
      const int other_code = codes[lane + offset];
      const int current_code = codes[lane];
      if (other_score > scores[lane] ||
          (other_score == scores[lane] && other_code < current_code)) {
        scores[lane] = other_score;
        codes[lane] = other_code;
      }
    }
    __syncthreads();
  }
  const int best_code = codes[0];
  if (lane != 0 || best_code == old_code) {
    return;
  }

  int trial_codes[Capacity];
  trial_codes[0] = 0;
#pragma unroll
  for (int other = 1; other < m; ++other) {
    trial_codes[other] = static_cast<int>(sample_codes[other - 1]);
  }
  trial_codes[layer] = best_code;

  constexpr int kPackedCapacity =
      Capacity * (Capacity + 1) / 2;
  const int kPacked = (m * (m + 1)) / 2;
  float original[kPackedCapacity];
  float factor[kPackedCapacity];
  float original_rhs[Capacity];
  float solution[Capacity];

  original_rhs[0] = root_dots[sample];
  solution[0] = root_dots[sample];
  original[PackedUpperIndex(0, 0)] = root_norms2[sample];
#pragma unroll
  for (int column = 1; column < m; ++column) {
    const int flat_column = (column - 1) * 256 + trial_codes[column];
    const float dot = sample_dots[flat_column];
    original_rhs[column] = dot;
    solution[column] = dot;
    original[PackedUpperIndex(0, column)] = algebra.RootSmallDot(
        roots[sample], static_cast<std::uint32_t>(flat_column));
#pragma unroll
    for (int row = 1; row <= column; ++row) {
      const int flat_row = (row - 1) * 256 + trial_codes[row];
      original[PackedUpperIndex(row, column)] =
          algebra.SmallDot(static_cast<std::uint32_t>(flat_row),
                           static_cast<std::uint32_t>(flat_column));
    }
  }

  bool solved = false;
  float bump = 0.0F;
#pragma unroll
  for (int attempt = 0; attempt < 4 && !solved; ++attempt) {
#pragma unroll
    for (int index = 0; index < kPacked; ++index) {
      factor[index] = original[index];
    }
    if (bump > 0.0F) {
#pragma unroll
      for (int diagonal = 0; diagonal < m; ++diagonal) {
        factor[PackedUpperIndex(diagonal, diagonal)] += bump;
      }
    }
    solved = CholeskyUpperPacked<M>(m, factor);
    if (!solved) {
      bump = attempt == 0 ? kAcceptanceEpsilon : bump * 10.0F;
    }
  }
  if (!solved) {
    return;
  }

  SolveCholeskyUpperPacked<M, Capacity>(m, factor, solution);
  float data_dot = 0.0F;
  float reconstructed_norm = 0.0F;
#pragma unroll
  for (int column = 0; column < m; ++column) {
    const float alpha = solution[column];
    data_dot += alpha * original_rhs[column];
#pragma unroll
    for (int row = 0; row < m; ++row) {
      const float gram = row <= column
                             ? original[PackedUpperIndex(row, column)]
                             : original[PackedUpperIndex(column, row)];
      reconstructed_norm += alpha * solution[row] * gram;
    }
  }
  const float trial_error =
      residual_norms2[sample] - 2.0F * data_dot + reconstructed_norm;
  if (trial_error + kAcceptanceEpsilon < errors[sample]) {
    sample_codes[layer - 1U] = static_cast<std::uint8_t>(best_code);
#pragma unroll
    for (int other = 0; other < m; ++other) {
      sample_coefficients[other] = solution[other];
    }
    errors[sample] = trial_error;
  }
}

std::uint64_t ModelByteCount(RuntimeRootCodebookView codebooks) {
  const std::uint64_t roots = codebooks.shape.nlist;
  const std::uint64_t small = codebooks.root_small_flat.columns;
  return static_cast<std::uint64_t>(sizeof(float)) *
         ((roots + small) * codebooks.shape.dimension +
          2U * (roots + small) + small * small + 2U * roots * small);
}

}  // namespace

template <int M>
struct CudaRootOnlyLinkageModel<M>::State final {
  int device = 0;
  std::uint32_t m = M;
  std::uint32_t dimension = 0U;
  std::uint32_t root_count = 0U;
  std::uint64_t root_dot_workspace_bytes = 0U;
  CudaRootOnlyLinkageModelOptions options;
  std::uint64_t device_bytes = 0U;
  CudaDeviceBuffer<float> roots;
  CudaDeviceBuffer<float> small;
  CudaDeviceBuffer<float> root_norms2;
  CudaDeviceBuffer<float> root_inverse_norms;
  CudaDeviceBuffer<float> small_norms2;
  CudaDeviceBuffer<float> small_inverse_norms;
  CudaDeviceBuffer<float> small_gram;
  CudaDeviceBuffer<float> root_small_gram;
  CudaDeviceBuffer<float> root_small_gram_transposed;
};

template <int M>
CudaRootOnlyLinkageModel<M>::CudaRootOnlyLinkageModel(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
CudaRootOnlyLinkageModel<M>::~CudaRootOnlyLinkageModel() {
  if (state_) {
    cudaSetDevice(state_->device);
  }
}

template <int M>
StatusOr<std::shared_ptr<CudaRootOnlyLinkageModel<M>>>
CudaRootOnlyLinkageModel<M>::Create(const BasicCodebookAlgebra<M>& algebra,
                                    int device, bool allow_tf32,
                                    CudaRootOnlyLinkageModelOptions options) {
  return CreateImpl({algebra.shape(), algebra.root0(), algebra.small_flat()},
                    algebra.small_gram(), device, allow_tf32, options);
}

template <int M>
StatusOr<std::shared_ptr<CudaRootOnlyLinkageModel<M>>>
CudaRootOnlyLinkageModel<M>::Create(const RootCodebooks<M>& codebooks,
                                    const F32GemmBackend& gemm, int device,
                                    bool allow_tf32,
                                    CudaRootOnlyLinkageModelOptions options) {
  // Initialization uses the independent small Gram, not the virtual-root
  // dense C_all contract. The root/small cross table is built on device below.
  const auto small = codebooks.root_small_flat();
  std::vector<float> gram(static_cast<std::size_t>(small.columns) * small.columns);
  TransposedLeftGemmF32 operation;
  operation.left = small;
  operation.right = small.values;
  operation.query_count = small.columns;
  operation.right_leading_dimension = small.rows;
  operation.output = gram.data();
  operation.output_leading_dimension = small.columns;
  const auto status = gemm.MultiplyTransposedLeft(operation);
  if (!status.ok()) return status;
  return CreateImpl(codebooks.runtime_view(), gram, device, allow_tf32, options);
}

template <int M>
StatusOr<std::shared_ptr<CudaRootOnlyLinkageModel<M>>>
CudaRootOnlyLinkageModel<M>::CreateImpl(
    RuntimeRootCodebookView codebooks, ArrayView<const float> small_gram,
    int device, bool allow_tf32, CudaRootOnlyLinkageModelOptions options) {
  auto build_context = CudaContext::Create(device, allow_tf32);
  if (!build_context.ok()) {
    return build_context.status();
  }
  auto state = std::make_unique<State>();
  state->device = device;
  state->m = codebooks.shape.m;
  state->dimension = codebooks.shape.dimension;
  state->root_count = codebooks.shape.nlist;
  state->root_dot_workspace_bytes = options.root_dot_workspace_bytes;
  state->options = options;
  state->device_bytes = ModelByteCount(codebooks);
  std::vector<float> root_norms2(state->root_count, 0.0F);
  std::vector<float> root_inverse_norms(state->root_count, 0.0F);
  for (std::uint32_t root = 0U; root < state->root_count; ++root) {
    const float* values = codebooks.root0.column(root);
    float norm2 = 0.0F;
    for (std::uint32_t axis = 0U; axis < state->dimension; ++axis) {
      const float value = values[axis];
      norm2 += value * value;
    }
    root_norms2[root] = norm2;
    root_inverse_norms[root] = norm2 > 0.0F ? 1.0F / std::sqrt(norm2) : 0.0F;
  }
  const std::uint32_t small_count = codebooks.root_small_flat.columns;
  std::vector<float> small_norms2(small_count, 0.0F);
  std::vector<float> small_inverse_norms(small_count, 0.0F);
  for (std::uint32_t flat = 0U; flat < small_count; ++flat) {
    const float* values = codebooks.root_small_flat.column(flat);
    float norm2 = 0.0F;
    for (std::uint32_t axis = 0U; axis < state->dimension; ++axis) {
      const float value = values[axis];
      norm2 += value * value;
    }
    small_norms2[flat] = norm2;
    small_inverse_norms[flat] = norm2 > 0.0F ? 1.0F / std::sqrt(norm2) : 0.0F;
  }
  const std::array<Status, 7> uploads = {
      UploadSynchronous(&state->roots,
                        {codebooks.root0.values,
                         static_cast<std::size_t>(state->dimension) *
                             state->root_count},
                        "cudaMemcpy root-only roots"),
      UploadSynchronous(&state->small,
                        {codebooks.root_small_flat.values,
                         static_cast<std::size_t>(state->dimension) * small_count},
                        "cudaMemcpy root-only small atoms"),
      UploadSynchronous(&state->root_norms2,
                        ArrayView<const float>(root_norms2),
                        "cudaMemcpy root-only root norms"),
      UploadSynchronous(&state->root_inverse_norms,
                        ArrayView<const float>(root_inverse_norms),
                        "cudaMemcpy root-only root inverse norms"),
      UploadSynchronous(&state->small_norms2,
                        ArrayView<const float>(small_norms2),
                        "cudaMemcpy root-only small norms"),
      UploadSynchronous(&state->small_inverse_norms,
                        ArrayView<const float>(small_inverse_norms),
                        "cudaMemcpy root-only small inverse norms"),
      UploadSynchronous(&state->small_gram, small_gram,
                        "cudaMemcpy root-only small Gram"),
  };
  for (const Status& status : uploads) {
    if (!status.ok()) {
      return status;
    }
  }
  if (const Status status = state->root_small_gram.Ensure(
          static_cast<std::size_t>(state->root_count) *
          small_count);
      !status.ok()) {
    return status;
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const cublasStatus_t cross_status = cublasSgemm(
      static_cast<cublasHandle_t>(build_context.value().native_blas()),
      CUBLAS_OP_T, CUBLAS_OP_N,
      static_cast<int>(small_count),
      static_cast<int>(state->root_count), static_cast<int>(state->dimension),
      &alpha, state->small.data(), static_cast<int>(state->dimension),
      state->roots.data(), static_cast<int>(state->dimension), &beta,
      state->root_small_gram.data(),
      static_cast<int>(small_count));
  if (cross_status != CUBLAS_STATUS_SUCCESS) {
    return CudaBlasStatus("cublasSgemm root-only root-small Gram",
                          cross_status);
  }
  if (const Status status = state->root_small_gram_transposed.Ensure(
          static_cast<std::size_t>(state->root_count) * small_count);
      !status.ok()) return status;
  const cublasStatus_t transpose_status = cublasSgeam(
      static_cast<cublasHandle_t>(build_context.value().native_blas()),
      CUBLAS_OP_T, CUBLAS_OP_T, state->root_count, small_count,
      &alpha, state->root_small_gram.data(), small_count,
      &beta, state->root_small_gram.data(), small_count,
      state->root_small_gram_transposed.data(), state->root_count);
  if (transpose_status != CUBLAS_STATUS_SUCCESS)
    return CudaBlasStatus("transpose root-only cross Gram", transpose_status);
  if (const Status status = build_context.value().Synchronize(); !status.ok()) {
    return status;
  }
  return std::shared_ptr<CudaRootOnlyLinkageModel<M>>(
      new CudaRootOnlyLinkageModel<M>(std::move(state)));
}

template <int M>
int CudaRootOnlyLinkageModel<M>::device() const noexcept {
  return state_->device;
}

template <int M>
std::uint32_t CudaRootOnlyLinkageModel<M>::dimension() const noexcept {
  return state_->dimension;
}

template <int M>
std::uint64_t CudaRootOnlyLinkageModel<M>::device_bytes() const noexcept {
  return state_->device_bytes;
}

template <int M>
std::uint32_t CudaRootOnlyLinkageModel<M>::m() const noexcept {
  return state_->m;
}

template <int M>
struct CudaRootOnlyLinkageDevicePlan<M>::State final {
  State(std::shared_ptr<const CudaRootOnlyLinkageModel<M>> model_in,
        CudaContext* context_in)
      : model(std::move(model_in)), context(context_in) {}

  [[nodiscard]] DeviceRootOnlyAlgebra<M> Algebra() const noexcept {
    const auto& source = *model->state_;
    return {
        .m = source.m,
        .dimension = source.dimension,
        .root_count = source.root_count,
        .roots = source.roots.data(),
        .small = source.small.data(),
        .root_norms2 = source.root_norms2.data(),
        .root_inverse_norms = source.root_inverse_norms.data(),
        .small_norms2 = source.small_norms2.data(),
        .small_inverse_norms = source.small_inverse_norms.data(),
        .small_gram = source.small_gram.data(),
        .root_small_gram = source.root_small_gram.data(),
        .root_small_gram_transposed = source.root_small_gram_transposed.data(),
    };
  }

  std::shared_ptr<const CudaRootOnlyLinkageModel<M>> model;
  CudaContext* context = nullptr;
  std::uint32_t icm_rounds = 1U;
  CudaDeviceBuffer<float> small_dots;
  CudaDeviceBuffer<float> residual_norms2;
  CudaDeviceBuffer<float> root_dot_chunk;
  CudaDeviceBuffer<float> root_best_scores;
  CudaDeviceBuffer<float> selected_root_dots;
  CudaDeviceBuffer<float> selected_root_norms2;
  CudaDeviceBuffer<float> remaining_small_dots;
  CudaDeviceBuffer<std::uint32_t> candidate_roots;
  CudaDeviceBuffer<float> candidate_root_dots;
  CudaDeviceBuffer<float> candidate_root_norms2;
  CudaDeviceBuffer<float> candidate_root_scores;
  CudaDeviceBuffer<std::uint32_t> failures;
};

template <int M>
CudaRootOnlyLinkageDevicePlan<M>::CudaRootOnlyLinkageDevicePlan(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
CudaRootOnlyLinkageDevicePlan<M>::~CudaRootOnlyLinkageDevicePlan() = default;

template <int M>
CudaRootOnlyLinkageDevicePlan<M>::CudaRootOnlyLinkageDevicePlan(
    CudaRootOnlyLinkageDevicePlan&&) noexcept = default;

template <int M>
CudaRootOnlyLinkageDevicePlan<M>& CudaRootOnlyLinkageDevicePlan<M>::operator=(
    CudaRootOnlyLinkageDevicePlan&&) noexcept = default;

template <int M>
StatusOr<CudaRootOnlyLinkageDevicePlan<M>>
CudaRootOnlyLinkageDevicePlan<M>::Create(
    std::shared_ptr<const CudaRootOnlyLinkageModel<M>> model,
    CudaContext& context, std::uint32_t icm_rounds) {
  if (!model || model->device() != context.device()) {
    return Status::InvalidArgument(
        "root-only model and CUDA context must share one device");
  }
  auto state = std::make_unique<State>(std::move(model), &context);
  state->icm_rounds = icm_rounds;
  return CudaRootOnlyLinkageDevicePlan<M>(std::move(state));
}

template <int M>
Status CudaRootOnlyLinkageDevicePlan<M>::BeginBatch() {
  if (const Status status = state_->failures.Ensure(1U); !status.ok()) {
    return status;
  }
  const cudaError_t result = cudaMemsetAsync(
      state_->failures.data(), 0, sizeof(std::uint32_t),
      static_cast<cudaStream_t>(state_->context->native_stream()));
  return result == cudaSuccess
             ? Status::Ok()
             : CudaRuntimeStatus("cudaMemsetAsync root-only failures", result);
}

template <int M>
std::uint32_t CudaRootOnlyLinkageDevicePlan<M>::sample_chunk_size(
    std::uint32_t requested_sample_count) const noexcept {
  const auto& model = *state_->model->state_;
  return LargeRootSampleChunkSize(requested_sample_count, model.root_count,
                                  ((state_->model->state_->m - 1U) * 256U),
                                  model.dimension, model.options);
}

template <int M>
Status CudaRootOnlyLinkageDevicePlan<M>::ReserveWorkspace(
    std::uint32_t sample_count) {
  if (state_ == nullptr || sample_count == 0U) {
    return Status::InvalidArgument(
        "root-only CUDA workspace extent is invalid");
  }
  if (const cudaError_t result = cudaSetDevice(state_->model->device());
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice root-only workspace", result);
  }
  const auto& model = *state_->model->state_;
  const std::uint32_t kSmallCount = ((state_->model->state_->m - 1U) * 256U);
  const std::array<Status, 11> storage = {
      state_->small_dots.Ensure(static_cast<std::size_t>(sample_count) *
                                kSmallCount),
      state_->residual_norms2.Ensure(sample_count),
      state_->root_best_scores.Ensure(sample_count),
      state_->selected_root_dots.Ensure(sample_count),
      state_->selected_root_norms2.Ensure(sample_count),
      state_->remaining_small_dots.Ensure(
          static_cast<std::size_t>(sample_count) * kSmallCount),
      state_->candidate_roots.Ensure(sample_count),
      state_->candidate_root_dots.Ensure(sample_count),
      state_->candidate_root_norms2.Ensure(sample_count),
      state_->candidate_root_scores.Ensure(sample_count),
      state_->failures.Ensure(1U),
  };
  for (const Status& status : storage) {
    if (!status.ok()) {
      return status;
    }
  }
  const std::uint32_t root_chunk = RootChunkSize(
      model.root_count, sample_count, model.root_dot_workspace_bytes);
  const bool cache_full = static_cast<std::uint64_t>(model.root_count) *
      sample_count * sizeof(float) <= model.options.full_root_dot_workspace_bytes;
  return state_->root_dot_chunk.Ensure(
      static_cast<std::size_t>(cache_full ? model.root_count : root_chunk) *
      sample_count);
}

template <int M>
Status CudaRootOnlyLinkageDevicePlan<M>::Enqueue(
    CudaMatrixF32View residuals, ArrayView<const std::uint8_t> icm_layer_order,
    CudaRootOnlyLinkageDeviceOutput<M> output) {
  if constexpr (M == 0) {
    if (state_->model->state_->m > kLegacyRuntimeScratchLayers)
      return EnqueueWithCapacity<kWideRuntimeScratchLayers>(residuals, icm_layer_order, output);
  }
  return EnqueueWithCapacity<kModelScratchLayers<M>>(residuals, icm_layer_order, output);
}

template <int M>
template <std::uint32_t Capacity>
Status CudaRootOnlyLinkageDevicePlan<M>::EnqueueWithCapacity(
    CudaMatrixF32View residuals, ArrayView<const std::uint8_t> icm_layer_order,
    CudaRootOnlyLinkageDeviceOutput<M> output) {
  const auto& model = *state_->model->state_;
  if (residuals.values == nullptr || residuals.rows != model.dimension ||
      residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows ||
      output.first_codes == nullptr || output.trailing_codes == nullptr ||
      output.coefficients == nullptr || output.squared_errors == nullptr) {
    return Status::InvalidArgument(
        "root-only device evaluation has an invalid matrix or output");
  }
  if (const Status status = ValidateIcmLayerOrders<M>(1U, icm_layer_order,
                                                      state_->model->state_->m);
      !status.ok()) {
    return status;
  }
  const std::uint32_t samples = residuals.columns;
  const std::uint32_t kSmallCount = ((state_->model->state_->m - 1U) * 256U);
  if (const Status status = ReserveWorkspace(samples); !status.ok()) {
    return status;
  }

  const std::uint32_t root_chunk =
      RootChunkSize(model.root_count, samples, model.root_dot_workspace_bytes);
  const bool cache_full = static_cast<std::uint64_t>(model.root_count) *
      samples * sizeof(float) <= model.options.full_root_dot_workspace_bytes;

  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  const cublasHandle_t blas =
      static_cast<cublasHandle_t>(state_->context->native_blas());
  constexpr std::uint32_t kThreads = 256U;
  const std::uint32_t sample_grid = (samples + kThreads - 1U) / kThreads;
  ComputeNormsKernel<<<samples, kThreads, 0U, stream>>>(
      residuals.values, model.dimension, samples,
      state_->residual_norms2.data());
  InitializeRootBestKernel<<<sample_grid, kThreads, 0U, stream>>>(
      samples, state_->root_best_scores.data(), output.first_codes,
      state_->selected_root_dots.data(), state_->selected_root_norms2.data());

  const float alpha = 1.0F;
  const float beta = 0.0F;
  cublasStatus_t blas_status = cublasSgemm(
      blas, CUBLAS_OP_T, CUBLAS_OP_N, kSmallCount, samples, model.dimension,
      &alpha, model.small.data(), model.dimension, residuals.values,
      model.dimension, &beta, state_->small_dots.data(), kSmallCount);
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return Status::Internal("cublasSgemm root-only small projection failed");
  }

  const DeviceRootOnlyAlgebra<M> algebra = state_->Algebra();
  auto scan_roots = [&](bool icm) -> Status {
    for (std::uint32_t begin = 0U; begin < model.root_count;
         begin += root_chunk) {
      const std::uint32_t count =
          std::min(root_chunk, model.root_count - begin);
      // Cache chunks in their original SGEMM layout, retaining numerical
      // dimensions and reusing the unchanged input projection for every ICM.
      float* root_dots = state_->root_dot_chunk.data() +
          (cache_full ? static_cast<std::size_t>(begin) * samples : 0U);
      if (!icm || !cache_full) {
        blas_status =
          cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, count, samples,
                      model.dimension, &alpha,
                      model.roots.data() +
                          static_cast<std::size_t>(begin) * model.dimension,
                      model.dimension, residuals.values, model.dimension, &beta,
                      root_dots, count);
      if (blas_status != CUBLAS_STATUS_SUCCESS) {
        return Status::Internal("cublasSgemm root-only root projection failed");
      }
      }
      if (icm) {
        UpdateRootBestKernel<M, true><<<samples, kThreads, 0U, stream>>>(
            algebra, begin, count, root_dots,
            output.trailing_codes, output.coefficients, samples,
            state_->candidate_root_scores.data(),
            state_->candidate_roots.data(), state_->candidate_root_dots.data(),
            state_->candidate_root_norms2.data());
      } else {
        UpdateRootBestKernel<M, false><<<samples, kThreads, 0U, stream>>>(
            algebra, begin, count, root_dots, nullptr,
            nullptr, samples, state_->root_best_scores.data(),
            output.first_codes, state_->selected_root_dots.data(),
            state_->selected_root_norms2.data());
      }
    }
    const cudaError_t launch = cudaPeekAtLastError();
    return launch == cudaSuccess
               ? Status::Ok()
               : CudaRuntimeStatus("UpdateRootBestKernel", launch);
  };

  if (const Status status = scan_roots(false); !status.ok()) {
    return status;
  }
  const std::uint64_t small_extent =
      static_cast<std::uint64_t>(samples) * kSmallCount;
  const std::uint32_t small_grid =
      static_cast<std::uint32_t>((small_extent + kThreads - 1U) / kThreads);
  BuildRootProjectedSmallDotsKernel<M><<<small_grid, kThreads, 0U, stream>>>(
      algebra, state_->small_dots.data(), output.first_codes,
      state_->selected_root_dots.data(), state_->selected_root_norms2.data(),
      samples, state_->remaining_small_dots.data());
  GlobalGreedyKernel<M><<<samples, kThreads, 0U, stream>>>(
      algebra, samples, state_->remaining_small_dots.data(),
      output.trailing_codes);
  InitialRefitKernel<M, Capacity><<<sample_grid, kThreads, 0U, stream>>>(
      algebra, state_->small_dots.data(), state_->residual_norms2.data(),
      output.first_codes, state_->selected_root_dots.data(),
      state_->selected_root_norms2.data(), output.trailing_codes, samples,
      output.coefficients, output.squared_errors, state_->failures.data());
  for (std::uint32_t round = 0U; round < state_->icm_rounds; ++round) {
    for (std::uint32_t position = 0U;
         position < static_cast<std::uint32_t>(state_->model->state_->m);
         ++position) {
      const std::uint32_t layer = icm_layer_order[position];
      if (layer == 0U) {
        InitializeRootBestKernel<<<sample_grid, kThreads, 0U, stream>>>(
            samples, state_->candidate_root_scores.data(),
            state_->candidate_roots.data(), state_->candidate_root_dots.data(),
            state_->candidate_root_norms2.data());
        if (const Status status = scan_roots(true); !status.ok()) {
          return status;
        }
        AcceptRootIcmKernel<M, Capacity><<<sample_grid, kThreads, 0U, stream>>>(
            algebra, state_->small_dots.data(), state_->residual_norms2.data(),
            residuals.values, model.dimension, state_->candidate_roots.data(),
            samples, output.first_codes, state_->selected_root_dots.data(),
            state_->selected_root_norms2.data(), output.trailing_codes,
            output.coefficients, output.squared_errors);
      } else {
        SmallIcmKernel<M, Capacity><<<samples, kThreads, 0U, stream>>>(
            algebra, layer, state_->small_dots.data(),
            state_->residual_norms2.data(), output.first_codes,
            state_->selected_root_dots.data(),
            state_->selected_root_norms2.data(), samples, output.trailing_codes,
            output.coefficients, output.squared_errors);
      }
    }
  }
  const cudaError_t launch = cudaPeekAtLastError();
  return launch == cudaSuccess
             ? Status::Ok()
             : CudaRuntimeStatus("root-only linkage kernels", launch);
}

template <int M>
Status CudaRootOnlyLinkageDevicePlan<M>::EnqueueFailureReadback(
    std::uint32_t* host_failures) {
  if (host_failures == nullptr) {
    return Status::InvalidArgument(
        "root-only failure readback requires host storage");
  }
  const cudaError_t result = cudaMemcpyAsync(
      host_failures, state_->failures.data(), sizeof(std::uint32_t),
      cudaMemcpyDeviceToHost,
      static_cast<cudaStream_t>(state_->context->native_stream()));
  return result == cudaSuccess
             ? Status::Ok()
             : CudaRuntimeStatus("cudaMemcpyAsync root-only failures", result);
}

template class CudaRootOnlyLinkageModel<0>;
template class CudaRootOnlyLinkageModel<5>;
template class CudaRootOnlyLinkageModel<12>;
template class CudaRootOnlyLinkageModel<6>;
template class CudaRootOnlyLinkageModel<10>;
template class CudaRootOnlyLinkageDevicePlan<0>;
template class CudaRootOnlyLinkageDevicePlan<5>;
template class CudaRootOnlyLinkageDevicePlan<12>;
template class CudaRootOnlyLinkageDevicePlan<6>;
template class CudaRootOnlyLinkageDevicePlan<10>;

}  // namespace stlqpp
