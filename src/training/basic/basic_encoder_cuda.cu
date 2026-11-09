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
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "stlqpp/runtime/cuda/device_buffer.h"
#include "stlqpp/runtime/cuda/gemm_f32.h"
#include "stlqpp/runtime/cuda/status.h"
#include "stlqpp/training/basic/basic_encoder_cuda.h"
#include "stlqpp/training/basic/detail/basic_encoding_access.h"
#include "stlqpp/training/basic/detail/cuda_layer_order.cuh"
#include "stlqpp/training/linkage/rsd/linkage_rsd_device_plan.h"

namespace stlqpp {
namespace {

constexpr std::uint32_t kRootBlock = 1024U;
constexpr float kNumericalEpsilon = 1.0e-6F;
constexpr float kAcceptanceEpsilon = 1.0e-6F;

__global__ void StoreRootSmallGramRootMajor(const float* flat_major,
                                            std::uint32_t root_begin,
                                            std::uint32_t root_count,
                                            std::uint32_t small_count,
                                            float* root_major) {
  __shared__ float tile[32][33];
  const std::uint32_t source_root = blockIdx.x * 32U + threadIdx.x;
  for (std::uint32_t offset = 0U; offset < 32U; offset += 8U) {
    const std::uint32_t source_flat = blockIdx.y * 32U + threadIdx.y + offset;
    if (source_root < root_count && source_flat < small_count) {
      tile[threadIdx.y + offset][threadIdx.x] =
          flat_major[static_cast<std::size_t>(source_flat) * root_count +
                     source_root];
    }
  }
  __syncthreads();
  const std::uint32_t target_flat = blockIdx.y * 32U + threadIdx.x;
  for (std::uint32_t offset = 0U; offset < 32U; offset += 8U) {
    const std::uint32_t target_root = blockIdx.x * 32U + threadIdx.y + offset;
    if (target_root < root_count && target_flat < small_count) {
      root_major[(static_cast<std::size_t>(root_begin) + target_root) *
                     small_count +
                 target_flat] = tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

template <typename T>
Status Upload(CudaDeviceBuffer<T>& destination, ArrayView<const T> source,
              cudaStream_t stream) {
  if (const Status status = destination.Ensure(source.size()); !status.ok()) {
    return status;
  }
  const cudaError_t result = cudaMemcpyAsync(destination.data(), source.data(),
                                             source.size() * sizeof(T),
                                             cudaMemcpyHostToDevice, stream);
  return result == cudaSuccess
             ? Status::Ok()
             : CudaRuntimeStatus("cudaMemcpyAsync H2D", result);
}

template <int M>
struct DeviceAlgebra final {
  std::uint32_t m = M;
  std::uint32_t dimension = 0U;
  std::uint32_t nlist = 0U;
  std::uint32_t small_count = 0U;
  const float* root0 = nullptr;
  const float* small = nullptr;
  const float* root_norm2 = nullptr;
  const float* small_norm2 = nullptr;
  const float* small_inverse_norm = nullptr;
  const float* small_gram = nullptr;
  const float* root_small_gram = nullptr;
};

__global__ void SquaredNormKernel(const float* samples, std::uint32_t dimension,
                                  std::uint32_t count, float* norms) {
  const std::uint32_t sample = blockIdx.x;
  if (sample >= count) {
    return;
  }
  const float* vector = samples + static_cast<std::size_t>(sample) * dimension;
  float norm = 0.0F;
  for (std::uint32_t component = threadIdx.x; component < dimension;
       component += blockDim.x) {
    norm += vector[component] * vector[component];
  }
  __shared__ float partial[256];
  partial[threadIdx.x] = norm;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x >> 1U; offset != 0U; offset >>= 1U) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    norms[sample] = partial[0];
  }
}

template <int M>
__global__ void UnpackResidentSupportsKernel(std::uint32_t actual_m,
                                             const std::uint8_t* support_codes,
                                             std::uint32_t sample_count,
                                             std::uint32_t* roots,
                                             std::uint8_t* small_codes) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= sample_count) {
    return;
  }
  const std::uint8_t* input =
      support_codes + static_cast<std::size_t>(sample) * m;
  roots[sample] = input[0];
  std::uint8_t* output =
      small_codes + static_cast<std::size_t>(sample) * (m - 1U);
  for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    output[layer - 1U] = input[layer];
  }
}

template <int M>
__device__ __forceinline__ void FillSupportSystem(
    const DeviceAlgebra<M>& algebra, std::uint32_t root,
    const std::uint8_t* codes, float root_dot, const float* small_dots,
    float diagonal_bump, float* gram, float* rhs) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  for (std::uint32_t entry = 0U; entry < m * m; ++entry) {
    gram[entry] = 0.0F;
  }
  rhs[0] = root_dot;
  gram[0] = algebra.root_norm2[root] + diagonal_bump;
  for (std::uint32_t column = 1U; column < m; ++column) {
    const std::uint32_t flat_column = (column - 1U) * 256U + codes[column - 1U];
    rhs[column] = small_dots[flat_column];
    gram[column] = algebra.root_small_gram[static_cast<std::size_t>(root) *
                                               algebra.small_count +
                                           flat_column];
    for (std::uint32_t row = 1U; row <= column; ++row) {
      const std::uint32_t flat_row = (row - 1U) * 256U + codes[row - 1U];
      gram[static_cast<std::size_t>(row) * m + column] =
          algebra.small_gram[static_cast<std::size_t>(flat_column) *
                                 algebra.small_count +
                             flat_row];
    }
    gram[static_cast<std::size_t>(column) * m + column] += diagonal_bump;
  }
}

template <int M>
__device__ __forceinline__ void FillSupportSystemWarp(
    const DeviceAlgebra<M>& algebra, std::uint32_t root,
    const std::uint8_t* codes, float root_dot, const float* small_dots,
    float diagonal_bump, std::uint32_t lane, float* gram, float* rhs) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  for (std::uint32_t entry = lane; entry < m * m; entry += 32U) {
    gram[entry] = 0.0F;
  }
  // Publish the caller's support codes and finish all zero writes before
  // another lane reads the codes or overwrites its Gram entries.
  __syncwarp();
  if (lane == 0U) {
    rhs[0] = root_dot;
    gram[0] = algebra.root_norm2[root] + diagonal_bump;
  }
  if (lane < m - 1U) {
    const std::uint32_t column = lane + 1U;
    const std::uint32_t flat_column = (column - 1U) * 256U + codes[column - 1U];
    rhs[column] = small_dots[flat_column];
    gram[column] = algebra.root_small_gram[static_cast<std::size_t>(root) *
                                               algebra.small_count +
                                           flat_column];
    for (std::uint32_t row = 1U; row <= column; ++row) {
      const std::uint32_t flat_row = (row - 1U) * 256U + codes[row - 1U];
      gram[static_cast<std::size_t>(row) * m + column] =
          algebra.small_gram[static_cast<std::size_t>(flat_column) *
                                 algebra.small_count +
                             flat_row];
    }
    gram[static_cast<std::size_t>(column) * m + column] += diagonal_bump;
  }
  __syncwarp();
}

template <int M>
__device__ __forceinline__ bool CholeskyUpper(std::uint32_t actual_m,
                                              float* matrix) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  for (std::uint32_t column = 0U; column < m; ++column) {
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
    for (std::uint32_t next = column + 1U; next < m; ++next) {
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
                                                  std::uint32_t lane) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  for (std::uint32_t column = 0U; column < m; ++column) {
    float diagonal = 0.0F;
    bool positive = false;
    if (lane == 0U) {
      diagonal = matrix[static_cast<std::size_t>(column) * m + column];
      for (std::uint32_t prior = 0U; prior < column; ++prior) {
        const float value =
            matrix[static_cast<std::size_t>(prior) * m + column];
        diagonal -= value * value;
      }
      positive = diagonal > 0.0F;
      if (positive) {
        diagonal = sqrtf(diagonal);
        matrix[static_cast<std::size_t>(column) * m + column] = diagonal;
      }
    }
    const std::uint32_t warp_positive =
        __shfl_sync(0xffffffffU, positive ? 1U : 0U, 0);
    if (warp_positive == 0U) {
      return false;
    }
    diagonal = __shfl_sync(0xffffffffU, diagonal, 0);
    const std::uint32_t next = column + 1U + lane;
    if (next < m) {
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
  for (std::uint32_t row = 0U; row < m; ++row) {
    float value = rhs[row];
    for (std::uint32_t prior = 0U; prior < row; ++prior) {
      value -=
          factor[static_cast<std::size_t>(prior) * m + row] * forward[prior];
    }
    forward[row] = value / factor[static_cast<std::size_t>(row) * m + row];
  }
  for (std::uint32_t reverse = m; reverse > 0U; --reverse) {
    const std::uint32_t row = reverse - 1U;
    float value = forward[row];
    for (std::uint32_t next = row + 1U; next < m; ++next) {
      value -=
          factor[static_cast<std::size_t>(row) * m + next] * solution[next];
    }
    solution[row] = value / factor[static_cast<std::size_t>(row) * m + row];
  }
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__device__ __forceinline__ bool SolveSupportWithScratch(
    const DeviceAlgebra<M>& algebra, std::uint32_t root,
    const std::uint8_t* codes, float root_dot, const float* small_dots,
    float* gram, float* rhs, float* coefficients,
    float* applied_diagonal_bump) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  FillSupportSystem<M>(algebra, root, codes, root_dot, small_dots, 0.0F, gram,
                       rhs);
  bool solved = CholeskyUpper<M>(m, gram);
  float solved_bump = 0.0F;
  float bump = kNumericalEpsilon;
  for (std::uint32_t attempt = 0U; attempt < 3U && !solved; ++attempt) {
    FillSupportSystem<M>(algebra, root, codes, root_dot, small_dots, bump, gram,
                         rhs);
    solved = CholeskyUpper<M>(m, gram);
    solved_bump = bump;
    bump *= 10.0F;
  }
  if (!solved) {
    return false;
  }
  SolveUpper<M, Capacity>(m, gram, rhs, coefficients);
  *applied_diagonal_bump = solved_bump;
  return true;
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__device__ __forceinline__ bool SolveSupportWarp(
    const DeviceAlgebra<M>& algebra, std::uint32_t root,
    const std::uint8_t* codes, float root_dot, const float* small_dots,
    std::uint32_t lane, float* gram, float* rhs, float* coefficients,
    float* applied_diagonal_bump) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  float bump = 0.0F;
  for (std::uint32_t attempt = 0U; attempt < 4U; ++attempt) {
    FillSupportSystemWarp<M>(algebra, root, codes, root_dot, small_dots, bump,
                             lane, gram, rhs);
    if (CholeskyUpperWarp<M>(m, gram, lane)) {
      if (lane == 0U) {
        SolveUpper<M, Capacity>(m, gram, rhs, coefficients);
        *applied_diagonal_bump = bump;
      }
      return true;
    }
    bump = attempt == 0U ? kNumericalEpsilon : bump * 10.0F;
  }
  return false;
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__device__ __forceinline__ bool SolveSupport(
    const DeviceAlgebra<M>& algebra, std::uint32_t root,
    const std::uint8_t* codes, float root_dot, const float* small_dots,
    float* coefficients, float* applied_diagonal_bump) {
  float gram[Capacity * Capacity]{};
  float rhs[Capacity]{};
  return SolveSupportWithScratch<M, Capacity>(algebra, root, codes, root_dot, small_dots,
                                    gram, rhs, coefficients,
                                    applied_diagonal_bump);
}

template <int M>
__device__ __forceinline__ float NormalEquationSupportCost(
    std::uint32_t actual_m, const std::uint8_t* codes, float root_dot,
    const float* small_dots, const float* coefficients, float sample_norm,
    float applied_diagonal_bump) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  float data_dot = 0.0F;
  for (std::uint32_t row = 0U; row < m; ++row) {
    const float row_dot =
        row == 0U ? root_dot : small_dots[(row - 1U) * 256U + codes[row - 1U]];
    data_dot += coefficients[row] * row_dot;
  }
  float squared_error = sample_norm - data_dot;
  if (applied_diagonal_bump != 0.0F) {
    float coefficient_norm = 0.0F;
    for (std::uint32_t row = 0U; row < m; ++row) {
      coefficient_norm += coefficients[row] * coefficients[row];
    }
    squared_error -= applied_diagonal_bump * coefficient_norm;
  }
  return squared_error;
}

template <int M>
__device__ __forceinline__ float ExplicitSupportCost(
    const DeviceAlgebra<M>& algebra, std::uint32_t root,
    const std::uint8_t* codes, float root_dot, const float* small_dots,
    const float* coefficients, float sample_norm) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  float linear = 0.0F;
  float quadratic = 0.0F;
  for (std::uint32_t row = 0U; row < m; ++row) {
    const float row_coefficient = coefficients[row];
    const float row_dot =
        row == 0U ? root_dot : small_dots[(row - 1U) * 256U + codes[row - 1U]];
    linear += row_coefficient * row_dot;
    for (std::uint32_t column = 0U; column < m; ++column) {
      float gram = 0.0F;
      if (row == 0U && column == 0U) {
        gram = algebra.root_norm2[root];
      } else if (row == 0U || column == 0U) {
        const std::uint32_t small_layer = row == 0U ? column : row;
        const std::uint32_t flat =
            (small_layer - 1U) * 256U + codes[small_layer - 1U];
        gram = algebra.root_small_gram[static_cast<std::size_t>(root) *
                                           algebra.small_count +
                                       flat];
      } else {
        const std::uint32_t flat_row = (row - 1U) * 256U + codes[row - 1U];
        const std::uint32_t flat_column =
            (column - 1U) * 256U + codes[column - 1U];
        gram = algebra.small_gram[static_cast<std::size_t>(flat_column) *
                                      algebra.small_count +
                                  flat_row];
      }
      quadratic += row_coefficient * coefficients[column] * gram;
    }
  }
  return sample_norm - 2.0F * linear + quadratic;
}

template <int M, LeastSquaresCostMode CostMode>
__device__ __forceinline__ float SupportCost(
    const DeviceAlgebra<M>& algebra, std::uint32_t root,
    const std::uint8_t* codes, float root_dot, const float* small_dots,
    const float* coefficients, float sample_norm, float applied_diagonal_bump) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  if constexpr (CostMode == LeastSquaresCostMode::kNormalEquation) {
    return NormalEquationSupportCost<M>(m, codes, root_dot, small_dots,
                                        coefficients, sample_norm,
                                        applied_diagonal_bump);
  }
  return ExplicitSupportCost<M>(algebra, root, codes, root_dot, small_dots,
                                coefficients, sample_norm);
}

__device__ __forceinline__ bool BetterRsdAtom(float score, std::uint32_t atom,
                                              float other_score,
                                              std::uint32_t other_atom) {
  return score > other_score || (score == other_score && atom < other_atom);
}

template <int M>
__device__ void SelectBasicRsdAtoms(const DeviceAlgebra<M>& algebra,
                                    const float* residual,
                                    std::uint32_t available,
                                    std::uint32_t rank_count, float* scores,
                                    std::uint32_t* atoms,
                                    std::uint32_t* selected) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  float local_scores[3] = {-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F};
  std::uint32_t local_atoms[3] = {0xffffffffU, 0xffffffffU, 0xffffffffU};
  for (std::uint32_t layer = 0U; layer < m - 1U; ++layer) {
    if ((available & (1U << layer)) == 0U) continue;
    const std::uint32_t atom = layer * 256U + threadIdx.x;
    const float score =
        fabsf(residual[atom] * algebra.small_inverse_norm[atom]);
    for (std::uint32_t rank = 0U; rank < rank_count; ++rank) {
      if (!BetterRsdAtom(score, atom, local_scores[rank], local_atoms[rank])) {
        continue;
      }
      for (std::uint32_t move = rank_count - 1U; move > rank; --move) {
        local_scores[move] = local_scores[move - 1U];
        local_atoms[move] = local_atoms[move - 1U];
      }
      local_scores[rank] = score;
      local_atoms[rank] = atom;
      break;
    }
  }
  // Merge disjoint lane-local top-K lists within each warp. Comparison is
  // still (score descending, atom ascending); no arithmetic is reassociated.
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5U;
  for (std::uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    float incoming_scores[3];
    std::uint32_t incoming_atoms[3];
    for (std::uint32_t rank = 0U; rank < rank_count; ++rank) {
      incoming_scores[rank] = __shfl_down_sync(
          0xffffffffU, local_scores[rank], offset);
      incoming_atoms[rank] = __shfl_down_sync(
          0xffffffffU, local_atoms[rank], offset);
    }
    if (lane < offset) {
      for (std::uint32_t other = 0U; other < rank_count; ++other) {
        for (std::uint32_t rank = 0U; rank < rank_count; ++rank) {
          if (!BetterRsdAtom(incoming_scores[other], incoming_atoms[other],
                             local_scores[rank], local_atoms[rank])) continue;
          for (std::uint32_t move = rank_count - 1U; move > rank; --move) {
            local_scores[move] = local_scores[move - 1U];
            local_atoms[move] = local_atoms[move - 1U];
          }
          local_scores[rank] = incoming_scores[other];
          local_atoms[rank] = incoming_atoms[other];
          break;
        }
      }
    }
  }
  if (lane == 0U) {
    for (std::uint32_t rank = 0U; rank < rank_count; ++rank) {
      scores[rank * 8U + warp] = local_scores[rank];
      atoms[rank * 8U + warp] = local_atoms[rank];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    float best_scores[3] = {-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F};
    std::uint32_t best_atoms[3] = {0xffffffffU, 0xffffffffU, 0xffffffffU};
    for (std::uint32_t lane = 0U; lane < 8U; ++lane) {
      for (std::uint32_t local = 0U; local < rank_count; ++local) {
        const float score = scores[local * 8U + lane];
        const std::uint32_t atom = atoms[local * 8U + lane];
        for (std::uint32_t rank = 0U; rank < rank_count; ++rank) {
          if (!BetterRsdAtom(score, atom, best_scores[rank], best_atoms[rank]))
            continue;
          for (std::uint32_t move = rank_count - 1U; move > rank; --move) {
            best_scores[move] = best_scores[move - 1U];
            best_atoms[move] = best_atoms[move - 1U];
          }
          best_scores[rank] = score;
          best_atoms[rank] = atom;
          break;
        }
      }
    }
    for (std::uint32_t rank = 0U; rank < rank_count; ++rank) {
      selected[rank] = best_atoms[rank];
    }
  }
  __syncthreads();
}

template <int M>
__device__ void ApplyBasicRsdAtom(const DeviceAlgebra<M>& algebra,
                                  std::uint32_t atom, std::uint32_t* available,
                                  float* residual, std::uint8_t* codes,
                                  float* coefficient) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  if (threadIdx.x == 0U) {
    const std::uint32_t layer = atom / 256U;
    codes[layer] = static_cast<std::uint8_t>(atom & 255U);
    *available &= ~(1U << layer);
    const float norm2 = algebra.small_norm2[atom];
    *coefficient = norm2 > 0.0F ? residual[atom] / norm2 : 0.0F;
  }
  __syncthreads();
  for (std::uint32_t layer = 0U; layer < m - 1U; ++layer) {
    if ((*available & (1U << layer)) == 0U) continue;
    const std::uint32_t other = layer * 256U + threadIdx.x;
    residual[other] -=
        *coefficient *
        algebra
            .small_gram[static_cast<std::size_t>(atom) * algebra.small_count +
                        other];
  }
  __syncthreads();
}

// Cost policy specializes only scoring; shared kernels retain their M-specific
// numerical flow. Static policies remain available where registers matter.
template <int M, int CostPolicy = -1>
__device__ __forceinline__ float ConfiguredSupportCost(
    LeastSquaresCostMode cost_mode, const DeviceAlgebra<M>& algebra,
    std::uint32_t root, const std::uint8_t* codes, float root_dot,
    const float* small_dots, const float* coefficients, float sample_norm,
    float applied_diagonal_bump) {
  if constexpr (CostPolicy >= 0) {
    return SupportCost<M, static_cast<LeastSquaresCostMode>(CostPolicy)>(
        algebra, root, codes, root_dot, small_dots, coefficients, sample_norm,
        applied_diagonal_bump);
  } else {
    if (cost_mode == LeastSquaresCostMode::kNormalEquation) {
      return SupportCost<M, LeastSquaresCostMode::kNormalEquation>(
          algebra, root, codes, root_dot, small_dots, coefficients, sample_norm,
          applied_diagonal_bump);
    }
    return SupportCost<M, LeastSquaresCostMode::kExplicitQuadratic>(
        algebra, root, codes, root_dot, small_dots, coefficients, sample_norm,
        applied_diagonal_bump);
  }
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>, int CostPolicy = -1>
__global__ void BuildBasicRsdSupportsKernel(
    DeviceAlgebra<M> algebra, const float* small_dots, const float* root_dots,
    const float* sample_norms, std::uint32_t count, const std::uint32_t* roots,
    std::uint32_t rank_count, LeastSquaresCostMode cost_mode,
    std::uint8_t* output_codes, std::uint32_t* failures) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  const std::uint32_t sample = blockIdx.x;
  if (sample >= count) return;
  const std::uint32_t kSmallCount = (m - 1U) * 256U;
  __shared__ float native[(Capacity - 1U) * 256U];
  __shared__ float branch[(Capacity - 1U) * 256U];
  __shared__ float local_scores[3U * 8U];
  __shared__ std::uint32_t local_atoms[3U * 8U];
  __shared__ std::uint32_t selected[3];
  __shared__ std::uint32_t native_selected[3];
  __shared__ std::uint8_t native_codes[Capacity - 1U];
  __shared__ std::uint8_t branch_codes[Capacity - 1U];
  __shared__ std::uint8_t best_codes[Capacity - 1U];
  __shared__ std::uint32_t native_available;
  __shared__ std::uint32_t branch_available;
  __shared__ float coefficient;
  __shared__ float best_error;
  __shared__ std::uint32_t best_index;
  const std::uint32_t root = roots[sample];
  const float* dots =
      small_dots + static_cast<std::size_t>(sample) * kSmallCount;
  const float root_dot = root_dots[sample];
  if (!(algebra.root_norm2[root] > 0.0F)) {
    if (threadIdx.x == 0U) atomicAdd(failures, 1U);
    return;
  }
  const float root_coefficient = root_dot / algebra.root_norm2[root];
  for (std::uint32_t atom = threadIdx.x; atom < kSmallCount;
       atom += blockDim.x) {
    native[atom] =
        dots[atom] -
        root_coefficient *
            algebra
                .root_small_gram[static_cast<std::size_t>(root) * kSmallCount +
                                 atom];
  }
  if (threadIdx.x == 0U) {
    native_available = (1U << (m - 1U)) - 1U;
    best_error = CUDART_INF_F;
    best_index = 0xffffffffU;
    for (std::uint32_t layer = 0U; layer < m - 1U; ++layer) {
      native_codes[layer] = 0U;
    }
  }
  __syncthreads();
  for (std::uint32_t step = 0U; step < m - 1U; ++step) {
    SelectBasicRsdAtoms<M>(algebra, native, native_available, rank_count,
                           local_scores, local_atoms, selected);
    if (threadIdx.x == 0U) {
      for (std::uint32_t rank = 0U; rank < rank_count; ++rank) {
        native_selected[rank] = selected[rank];
      }
    }
    __syncthreads();
    const std::uint32_t native_atom = native_selected[0];
    for (std::uint32_t rank = 2U; rank <= rank_count; ++rank) {
      const std::uint32_t branch_atom = native_selected[rank - 1U];
      for (std::uint32_t atom = threadIdx.x; atom < kSmallCount;
           atom += blockDim.x) {
        branch[atom] = native[atom];
      }
      if (threadIdx.x == 0U) {
        for (std::uint32_t layer = 0U; layer < m - 1U; ++layer) {
          branch_codes[layer] = native_codes[layer];
        }
        branch_available = native_available;
      }
      __syncthreads();
      ApplyBasicRsdAtom<M>(algebra, branch_atom, &branch_available, branch,
                           branch_codes, &coefficient);
      for (std::uint32_t suffix = step + 1U; suffix < m - 1U; ++suffix) {
        SelectBasicRsdAtoms<M>(algebra, branch, branch_available, 1U,
                               local_scores, local_atoms, selected);
        ApplyBasicRsdAtom<M>(algebra, selected[0], &branch_available, branch,
                             branch_codes, &coefficient);
      }
      if (threadIdx.x == 0U) {
        float coefficients[Capacity]{};
        float bump = 0.0F;
        if (SolveSupport<M, Capacity>(algebra, root, branch_codes, root_dot, dots,
                            coefficients, &bump)) {
          const float error = ConfiguredSupportCost<M, CostPolicy>(
              cost_mode, algebra, root, branch_codes, root_dot, dots, coefficients,
              sample_norms[sample], bump);
          const std::uint32_t index = 1U + (rank - 2U) * (m - 1U) + step;
          if (isfinite(error) && (error < best_error || (error == best_error &&
                                                         index < best_index))) {
            best_error = error;
            best_index = index;
            for (std::uint32_t layer = 0U; layer < m - 1U; ++layer) {
              best_codes[layer] = branch_codes[layer];
            }
          }
        }
      }
      __syncthreads();
    }
    ApplyBasicRsdAtom<M>(algebra, native_atom, &native_available, native,
                         native_codes, &coefficient);
  }
  if (threadIdx.x == 0U) {
    float coefficients[Capacity]{};
    float bump = 0.0F;
    if (SolveSupport<M, Capacity>(algebra, root, native_codes, root_dot, dots,
                        coefficients, &bump)) {
      const float error = ConfiguredSupportCost<M, CostPolicy>(
          cost_mode, algebra, root, native_codes, root_dot, dots, coefficients,
          sample_norms[sample], bump);
      if (isfinite(error) &&
          (error < best_error || (error == best_error && 0U < best_index))) {
        best_error = error;
        best_index = 0U;
        for (std::uint32_t layer = 0U; layer < m - 1U; ++layer) {
          best_codes[layer] = native_codes[layer];
        }
      }
    }
    if (best_index == 0xffffffffU) {
      atomicAdd(failures, 1U);
    } else {
      std::uint8_t* output =
          output_codes + static_cast<std::size_t>(sample) * (m - 1U);
      for (std::uint32_t layer = 0U; layer < m - 1U; ++layer) {
        output[layer] = best_codes[layer];
      }
    }
  }
}

__global__ void InitializeRsdTopOneRootsKernel(std::uint32_t count,
                                               float* errors,
                                               std::uint32_t* roots) {
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample < count) {
    errors[sample] = CUDART_INF_F;
    roots[sample] = 0U;
  }
}

__global__ void UpdateRsdTopOneRootsKernel(
    std::uint32_t count, std::uint32_t root_offset, std::uint32_t root_count,
    const float* scores, const float* sample_norms, const float* root_norms,
    float* errors, std::uint32_t* roots) {
  const std::uint32_t sample = blockIdx.x;
  if (sample >= count) return;
  float best_error = CUDART_INF_F;
  std::uint32_t best_root = 0xffffffffU;
  const float* sample_scores =
      scores + static_cast<std::size_t>(sample) * root_count;
  for (std::uint32_t local = threadIdx.x; local < root_count;
       local += blockDim.x) {
    const std::uint32_t root = root_offset + local;
    const float norm2 = root_norms[root];
    if (!(norm2 > 0.0F)) continue;
    const float dot = sample_scores[local];
    const float error = fmaxf(0.0F, sample_norms[sample] - dot * (dot / norm2));
    if (error < best_error || (error == best_error && root < best_root)) {
      best_error = error;
      best_root = root;
    }
  }
  __shared__ float lane_errors[256];
  __shared__ std::uint32_t lane_roots[256];
  lane_errors[threadIdx.x] = best_error;
  lane_roots[threadIdx.x] = best_root;
  __syncthreads();
  for (std::uint32_t width = blockDim.x >> 1U; width != 0U; width >>= 1U) {
    if (threadIdx.x < width) {
      const float other_error = lane_errors[threadIdx.x + width];
      const std::uint32_t other_root = lane_roots[threadIdx.x + width];
      if (other_error < lane_errors[threadIdx.x] ||
          (other_error == lane_errors[threadIdx.x] &&
           other_root < lane_roots[threadIdx.x])) {
        lane_errors[threadIdx.x] = other_error;
        lane_roots[threadIdx.x] = other_root;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U &&
      (lane_errors[0] < errors[sample] ||
       (lane_errors[0] == errors[sample] && lane_roots[0] < roots[sample]))) {
    errors[sample] = lane_errors[0];
    roots[sample] = lane_roots[0];
  }
}

template <int M>
__global__ void RootDotKernel(DeviceAlgebra<M> algebra, const float* samples,
                              std::uint32_t count, const std::uint32_t* roots,
                              float* root_dots) {
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= count) {
    return;
  }
  const float* input =
      samples + static_cast<std::size_t>(sample) * algebra.dimension;
  const float* root = algebra.root0 + static_cast<std::size_t>(roots[sample]) *
                                          algebra.dimension;
  float dot = 0.0F;
  for (std::uint32_t component = 0U; component < algebra.dimension;
       ++component) {
    dot += root[component] * input[component];
  }
  root_dots[sample] = dot;
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>, int CostPolicy = -1>
__global__ void RefitKernel(DeviceAlgebra<M> algebra, const float* small_dots,
                            const float* sample_norms, const float* root_dots,
                            std::uint32_t count, const std::uint32_t* roots,
                            const std::uint8_t* codes, float* coefficients,
                            float* errors, std::uint8_t* active,
                            std::uint32_t* failures, LeastSquaresCostMode cost_mode) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= count) {
    return;
  }
  const std::uint8_t* sample_codes =
      codes + static_cast<std::size_t>(sample) * (m - 1U);
  float* sample_coefficients =
      coefficients + static_cast<std::size_t>(sample) * m;
  const float* dots =
      small_dots + static_cast<std::size_t>(sample) * algebra.small_count;
  float applied_diagonal_bump = 0.0F;
  if (!SolveSupport<M, Capacity>(algebra, roots[sample], sample_codes, root_dots[sample],
                       dots, sample_coefficients, &applied_diagonal_bump)) {
    errors[sample] = CUDART_INF_F;
    active[sample] = 0U;
    atomicAdd(failures, 1U);
    return;
  }
  errors[sample] = ConfiguredSupportCost<M, CostPolicy>(
      cost_mode, algebra, roots[sample], sample_codes, root_dots[sample], dots,
      sample_coefficients, sample_norms[sample], applied_diagonal_bump);
  active[sample] = 1U;
}

__device__ __forceinline__ bool BetterScore(float candidate_score,
                                            std::uint32_t candidate_code,
                                            float incumbent_score,
                                            std::uint32_t incumbent_code) {
  return candidate_score > incumbent_score ||
         (candidate_score == incumbent_score &&
          candidate_code < incumbent_code);
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>, LeastSquaresCostMode CostMode>
__global__ __launch_bounds__(256) void IcmSweepsWarpRowsKernel(
    DeviceAlgebra<M> algebra, const float* small_dots,
    const float* sample_norms, const float* root_dots, std::uint32_t count,
    detail::CudaLayerOrder<M> packed_order0,
    detail::CudaLayerOrder<M> packed_order1, const std::uint32_t* roots,
    std::uint8_t* codes, float* coefficients, float* errors,
    std::uint8_t* active, bool absolute_score, std::uint32_t sweep_count) {
  const std::uint32_t m = M == 0 ? algebra.m : M;
  constexpr std::uint32_t kWarpsPerBlock = 8U;
  constexpr std::uint32_t kSharedGramValues = M == 10 ? M * M : 1U;
  constexpr std::uint32_t kSharedRhsValues = M == 10 ? M : 1U;
  constexpr std::uint32_t kSharedBackupValues = M == 10 ? M : 1U;
  __shared__ std::uint8_t shared_codes[kWarpsPerBlock]
                                      [Capacity - 1U];
  __shared__ float shared_coefficients[kWarpsPerBlock][Capacity];
  __shared__ float shared_gram[kWarpsPerBlock][kSharedGramValues];
  __shared__ float shared_rhs[kWarpsPerBlock][kSharedRhsValues];
  __shared__ float shared_backup[kWarpsPerBlock][kSharedBackupValues];
  const std::uint32_t warp = threadIdx.x >> 5U;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t sample = blockIdx.x * kWarpsPerBlock + warp;
  if (sample >= count || active[sample] == 0U) {
    return;
  }
  std::uint8_t* local_codes = shared_codes[warp];
  float* local_coefficients = shared_coefficients[warp];
  if (lane == 0U) {
    for (std::uint32_t layer = 1U; layer < m; ++layer) {
      local_codes[layer - 1U] =
          codes[static_cast<std::size_t>(sample) * (m - 1U) + layer - 1U];
    }
    for (std::uint32_t layer = 0U; layer < m; ++layer) {
      local_coefficients[layer] =
          coefficients[static_cast<std::size_t>(sample) * m + layer];
    }
  }
  __syncwarp();
  const float* dots =
      small_dots + static_cast<std::size_t>(sample) * algebra.small_count;
  // The caller keeps the original one/two-sweep launch grouping and order
  // packing. Sweep count does not specialize the M-wide numerical body.
  for (std::uint32_t sweep = 0U; sweep < sweep_count; ++sweep) {
    const detail::CudaLayerOrder<M> packed_order =
        sweep == 0U ? packed_order0 : packed_order1;
    bool changed = false;
    for (std::uint32_t order_index = 0U; order_index < m; ++order_index) {
      const std::uint32_t layer =
          detail::ReadCudaLayerOrder<M>(packed_order, order_index);
      if (layer == 0U) {
        continue;
      }
      float lane_score = -CUDART_INF_F;
      std::uint32_t lane_code = lane;
      for (std::uint32_t code = lane; code < 256U; code += 32U) {
        const std::uint32_t flat = (layer - 1U) * 256U + code;
        float residual_dot =
            dots[flat] -
            local_coefficients[0] *
                algebra
                    .root_small_gram[static_cast<std::size_t>(roots[sample]) *
                                         algebra.small_count +
                                     flat];
        for (std::uint32_t other = 1U; other < m; ++other) {
          if (other == layer) {
            continue;
          }
          const std::uint32_t other_flat =
              (other - 1U) * 256U + local_codes[other - 1U];
          residual_dot -=
              local_coefficients[other] *
              algebra.small_gram[static_cast<std::size_t>(other_flat) *
                                     algebra.small_count +
                                 flat];
        }
        float score = residual_dot * algebra.small_inverse_norm[flat];
        if (absolute_score) {
          score = fabsf(score);
        }
        if (BetterScore(score, code, lane_score, lane_code)) {
          lane_score = score;
          lane_code = code;
        }
      }
      for (std::uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
        const float other_score =
            __shfl_down_sync(0xffffffffU, lane_score, static_cast<int>(offset));
        const std::uint32_t other_code =
            __shfl_down_sync(0xffffffffU, lane_code, static_cast<int>(offset));
        if (BetterScore(other_score, other_code, lane_score, lane_code)) {
          lane_score = other_score;
          lane_code = other_code;
        }
      }
      if (lane == 0U) {
        const std::uint8_t old_code = local_codes[layer - 1U];
        if constexpr (M != 10) {
          if (lane_code != old_code) {
            float backup[Capacity]{};
            for (std::uint32_t index = 0U; index < m; ++index) {
              backup[index] = local_coefficients[index];
            }
            local_codes[layer - 1U] = static_cast<std::uint8_t>(lane_code);
            float applied_diagonal_bump = 0.0F;
            if (SolveSupport<M, Capacity>(algebra, roots[sample], local_codes,
                                root_dots[sample], dots, local_coefficients,
                                &applied_diagonal_bump)) {
              const float candidate_error = SupportCost<M, CostMode>(
                  algebra, roots[sample], local_codes, root_dots[sample], dots,
                  local_coefficients, sample_norms[sample],
                  applied_diagonal_bump);
              if (candidate_error + kAcceptanceEpsilon < errors[sample]) {
                errors[sample] = candidate_error;
                changed = true;
              } else {
                local_codes[layer - 1U] = old_code;
                for (std::uint32_t index = 0U; index < m; ++index) {
                  local_coefficients[index] = backup[index];
                }
              }
            } else {
              local_codes[layer - 1U] = old_code;
              for (std::uint32_t index = 0U; index < m; ++index) {
                local_coefficients[index] = backup[index];
              }
            }
          }
        }
      }
      if constexpr (M == 10) {
        std::uint32_t old_code = 0U;
        bool try_candidate = false;
        if (lane == 0U) {
          old_code = local_codes[layer - 1U];
          try_candidate = lane_code != old_code;
          if (try_candidate) {
            for (std::uint32_t index = 0U; index < m; ++index) {
              shared_backup[warp][index] = local_coefficients[index];
            }
            local_codes[layer - 1U] = static_cast<std::uint8_t>(lane_code);
          }
        }
        old_code = __shfl_sync(0xffffffffU, old_code, 0);
        const std::uint32_t warp_try =
            __shfl_sync(0xffffffffU, try_candidate ? 1U : 0U, 0);
        bool solved = false;
        float applied_diagonal_bump = 0.0F;
        if (warp_try != 0U) {
          solved = SolveSupportWarp<M, Capacity>(
              algebra, roots[sample], local_codes, root_dots[sample], dots,
              lane, shared_gram[warp], shared_rhs[warp], local_coefficients,
              &applied_diagonal_bump);
        }
        if (lane == 0U && warp_try != 0U) {
          if (solved) {
            const float candidate_error = SupportCost<M, CostMode>(
                algebra, roots[sample], local_codes, root_dots[sample], dots,
                local_coefficients, sample_norms[sample],
                applied_diagonal_bump);
            if (candidate_error + kAcceptanceEpsilon < errors[sample]) {
              errors[sample] = candidate_error;
              changed = true;
            } else {
              local_codes[layer - 1U] = static_cast<std::uint8_t>(old_code);
              for (std::uint32_t index = 0U; index < m; ++index) {
                local_coefficients[index] = shared_backup[warp][index];
              }
            }
          } else {
            local_codes[layer - 1U] = static_cast<std::uint8_t>(old_code);
            for (std::uint32_t index = 0U; index < m; ++index) {
              local_coefficients[index] = shared_backup[warp][index];
            }
          }
        }
      }
      __syncwarp();
    }
    const std::uint32_t warp_changed =
        __shfl_sync(0xffffffffU, changed ? 1U : 0U, 0);
    if (warp_changed == 0U) {
      if (lane == 0U) {
        active[sample] = 0U;
      }
      break;
    }
  }
  if (lane == 0U) {
    for (std::uint32_t layer = 1U; layer < m; ++layer) {
      codes[static_cast<std::size_t>(sample) * (m - 1U) + layer - 1U] =
          local_codes[layer - 1U];
    }
    for (std::uint32_t layer = 0U; layer < m; ++layer) {
      coefficients[static_cast<std::size_t>(sample) * m + layer] =
          local_coefficients[layer];
    }
  }
}

__device__ __forceinline__ std::uint64_t SplitMix64Next(std::uint64_t* state) {
  std::uint64_t value = (*state += 0x9e3779b97f4a7c15ULL);
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

__device__ __forceinline__ std::uint32_t UniformBelow(std::uint64_t* state,
                                                      std::uint32_t bound) {
  const std::uint64_t value = SplitMix64Next(state);
  const std::uint64_t high = value >> 32U;
  const std::uint64_t low = value & 0xffffffffULL;
  const std::uint64_t middle = high * bound + ((low * bound) >> 32U);
  return static_cast<std::uint32_t>(middle >> 32U);
}

template <int M>
__global__ void PerturbKernel(std::uint32_t actual_m, std::uint32_t count,
                              std::uint32_t round, std::uint32_t seed,
                              std::uint32_t perturbations,
                              std::uint64_t global_offset,
                              std::uint8_t* codes) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= count) {
    return;
  }
  const std::uint64_t global_sample = global_offset + sample;
  std::uint64_t random =
      static_cast<std::uint64_t>(seed) ^
      (static_cast<std::uint64_t>(round + 1U) * 0x9e3779b97f4a7c15ULL) ^
      ((global_sample + 1U) * 0xbf58476d1ce4e5b9ULL);
  std::uint8_t* sample_codes =
      codes + static_cast<std::size_t>(sample) * (m - 1U);
  for (std::uint32_t index = 0U; index < perturbations; ++index) {
    const std::uint32_t layer =
        1U + UniformBelow(&random, static_cast<std::uint32_t>(m - 1U));
    const std::uint8_t old_code = sample_codes[layer - 1U];
    const std::uint32_t raw = UniformBelow(&random, 255U);
    sample_codes[layer - 1U] =
        static_cast<std::uint8_t>(raw >= old_code ? raw + 1U : raw);
  }
}

template <int M>
__global__ void PerturbExplicitIdsKernel(
    std::uint32_t actual_m, std::uint32_t count, std::uint32_t round,
    std::uint32_t seed, std::uint32_t perturbations,
    const std::uint64_t* global_sample_ids, std::uint8_t* codes) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= count) {
    return;
  }
  const std::uint64_t global_sample = global_sample_ids[sample];
  std::uint64_t random =
      static_cast<std::uint64_t>(seed) ^
      (static_cast<std::uint64_t>(round + 1U) * 0x9e3779b97f4a7c15ULL) ^
      ((global_sample + 1U) * 0xbf58476d1ce4e5b9ULL);
  std::uint8_t* sample_codes =
      codes + static_cast<std::size_t>(sample) * (m - 1U);
  for (std::uint32_t index = 0U; index < perturbations; ++index) {
    const std::uint32_t layer =
        1U + UniformBelow(&random, static_cast<std::uint32_t>(m - 1U));
    const std::uint8_t old_code = sample_codes[layer - 1U];
    const std::uint32_t raw = UniformBelow(&random, 255U);
    sample_codes[layer - 1U] =
        static_cast<std::uint8_t>(raw >= old_code ? raw + 1U : raw);
  }
}

template <int M>
__global__ void AcceptKernel(std::uint32_t actual_m, std::uint32_t count,
                             const std::uint8_t* candidate_codes,
                             const float* candidate_coefficients,
                             const float* candidate_errors,
                             std::uint8_t* current_codes,
                             float* current_coefficients,
                             float* current_errors) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= count || !(candidate_errors[sample] + kAcceptanceEpsilon <
                           current_errors[sample])) {
    return;
  }
  for (std::uint32_t layer = 0U; layer < m - 1U; ++layer) {
    current_codes[static_cast<std::size_t>(sample) * (m - 1U) + layer] =
        candidate_codes[static_cast<std::size_t>(sample) * (m - 1U) + layer];
  }
  for (std::uint32_t layer = 0U; layer < m; ++layer) {
    current_coefficients[static_cast<std::size_t>(sample) * m + layer] =
        candidate_coefficients[static_cast<std::size_t>(sample) * m + layer];
  }
  current_errors[sample] = candidate_errors[sample];
}

template <int M>
std::vector<detail::CudaLayerOrder<M>> MakeLayerOrders(std::uint32_t actual_m,
                                                       BasicSchedule schedule,
                                                       std::uint32_t seed) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  std::vector<detail::CudaLayerOrder<M>> output;
  const std::uint32_t ils_rounds = EffectiveBasicIlsRounds<M>(schedule, m);
  output.reserve(static_cast<std::size_t>(ils_rounds + 1U) *
                 schedule.icm_sweeps());
  std::mt19937 random(seed);
  for (std::uint32_t phase = 0U; phase <= ils_rounds; ++phase) {
    auto order = MakeModelLayers<std::uint8_t, M>(m);
    std::iota(order.begin(), order.end(), static_cast<std::uint8_t>(0U));
    for (std::uint32_t sweep = 0U; sweep < schedule.icm_sweeps(); ++sweep) {
      std::shuffle(order.begin(), order.end(), random);
      output.push_back(detail::PackCudaLayerOrder<M>(order.data(), m));
    }
  }
  return std::move(output);
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>, LeastSquaresCostMode CostMode>
cudaError_t LaunchSweeps(DeviceAlgebra<M> algebra, const float* small_dots,
                         const float* sample_norms, const float* root_dots,
                         std::uint32_t count, detail::CudaLayerOrder<M> order0,
                         detail::CudaLayerOrder<M> order1,
                         const std::uint32_t* roots, std::uint8_t* codes,
                         float* coefficients, float* errors,
                         std::uint8_t* active, bool absolute_score,
                         std::uint32_t sweep_count, cudaStream_t stream) {
  constexpr std::uint32_t kWarpsPerBlock = 8U;
  IcmSweepsWarpRowsKernel<M, Capacity, CostMode>
      <<<(count + kWarpsPerBlock - 1U) / kWarpsPerBlock, 256U, 0U, stream>>>(
          algebra, small_dots, sample_norms, root_dots, count, order0, order1,
          roots, codes, coefficients, errors, active, absolute_score,
          sweep_count);
  return cudaPeekAtLastError();
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
cudaError_t LaunchConfiguredSweeps(
    LeastSquaresCostMode cost_mode, DeviceAlgebra<M> algebra,
    const float* small_dots, const float* sample_norms, const float* root_dots,
    std::uint32_t count, detail::CudaLayerOrder<M> order0,
    detail::CudaLayerOrder<M> order1, const std::uint32_t* roots,
    std::uint8_t* codes, float* coefficients, float* errors,
    std::uint8_t* active, bool absolute_score, std::uint32_t sweep_count,
    cudaStream_t stream) {
  if (cost_mode == LeastSquaresCostMode::kNormalEquation) {
    return LaunchSweeps<M, Capacity, LeastSquaresCostMode::kNormalEquation>(
        algebra, small_dots, sample_norms, root_dots, count, order0, order1,
        roots, codes, coefficients, errors, active, absolute_score,
        sweep_count, stream);
  }
  return LaunchSweeps<M, Capacity, LeastSquaresCostMode::kExplicitQuadratic>(
      algebra, small_dots, sample_norms, root_dots, count, order0, order1,
      roots, codes, coefficients, errors, active, absolute_score, sweep_count,
      stream);
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
cudaError_t LaunchConfiguredRefit(
    LeastSquaresCostMode cost_mode, DeviceAlgebra<M> algebra,
    const float* small_dots, const float* sample_norms, const float* root_dots,
    std::uint32_t count, const std::uint32_t* roots, const std::uint8_t* codes,
    float* coefficients, float* errors, std::uint8_t* active,
    std::uint32_t* failures, cudaStream_t stream) {
  constexpr std::uint32_t kThreads = 128U;
  const std::uint32_t grid = (count + kThreads - 1U) / kThreads;
  // Small-M normal cost kernels retain their lower register footprint.
  if constexpr (M == 5 || M == 6) {
    if (cost_mode == LeastSquaresCostMode::kNormalEquation) {
      RefitKernel<M, Capacity, static_cast<int>(LeastSquaresCostMode::kNormalEquation)>
          <<<grid, kThreads, 0U, stream>>>(
              algebra, small_dots, sample_norms, root_dots, count, roots, codes,
              coefficients, errors, active, failures, cost_mode);
    } else {
      RefitKernel<M, Capacity, static_cast<int>(LeastSquaresCostMode::kExplicitQuadratic)>
          <<<grid, kThreads, 0U, stream>>>(
              algebra, small_dots, sample_norms, root_dots, count, roots, codes,
              coefficients, errors, active, failures, cost_mode);
    }
  } else {
    RefitKernel<M, Capacity><<<grid, kThreads, 0U, stream>>>(
        algebra, small_dots, sample_norms, root_dots, count, roots, codes,
        coefficients, errors, active, failures, cost_mode);
  }
  return cudaPeekAtLastError();
}

Status GemmTransposedLeft(CudaContext& context, const float* left,
                          std::uint32_t left_columns, const float* right,
                          std::uint32_t dimension, std::uint32_t right_columns,
                          float* output) {
  return CudaGemmF32(
      context, CudaGemmOperation::kTranspose, CudaGemmOperation::kNone,
      static_cast<int>(left_columns), static_cast<int>(right_columns),
      static_cast<int>(dimension), 1.0F, left, static_cast<int>(dimension),
      right, static_cast<int>(dimension), 0.0F, output,
      static_cast<int>(left_columns));
}

Status GemmRootTransposedLeft(CudaContext& context, const float* left,
                              std::uint32_t left_columns, const float* right,
                              std::uint32_t dimension,
                              std::uint32_t right_columns, float* output) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  return CudaBlasStatus(
      "cublasSgemm root",
      cublasSgemm(static_cast<cublasHandle_t>(context.native_blas()),
                  CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(left_columns),
                  static_cast<int>(right_columns), static_cast<int>(dimension),
                  &alpha, left, static_cast<int>(dimension), right,
                  static_cast<int>(dimension), &beta, output,
                  static_cast<int>(left_columns)));
}

template <int M>
Status RunRsdTopOneRoot(DeviceAlgebra<M> algebra, CudaContext& context,
                        const float* samples, const float* sample_norms,
                        std::uint32_t count, float* root_scores,
                        float* root_errors, std::uint32_t* output_roots,
                        cudaStream_t stream) {
  constexpr std::uint32_t kThreads = 128U;
  const std::uint32_t grid = (count + kThreads - 1U) / kThreads;
  InitializeRsdTopOneRootsKernel<<<grid, kThreads, 0U, stream>>>(
      count, root_errors, output_roots);
  if (cudaError_t result = cudaPeekAtLastError(); result != cudaSuccess) {
    return CudaRuntimeStatus("InitializeRsdTopOneRootsKernel", result);
  }
  for (std::uint32_t root_offset = 0U; root_offset < algebra.nlist;
       root_offset += kRootBlock) {
    const std::uint32_t root_count =
        std::min(kRootBlock, algebra.nlist - root_offset);
    if (const Status status = GemmRootTransposedLeft(
            context,
            algebra.root0 +
                static_cast<std::size_t>(root_offset) * algebra.dimension,
            root_count, samples, algebra.dimension, count, root_scores);
        !status.ok())
      return status;
    UpdateRsdTopOneRootsKernel<<<count, 256U, 0U, stream>>>(
        count, root_offset, root_count, root_scores, sample_norms,
        algebra.root_norm2, root_errors, output_roots);
    if (cudaError_t result = cudaPeekAtLastError(); result != cudaSuccess) {
      return CudaRuntimeStatus("UpdateRsdTopOneRootsKernel", result);
    }
  }
  return Status::Ok();
}



}  // namespace

namespace detail {

// Device allocations and transfers depend on model extents, not on the
// compile-time width of the numerical kernels using them.
struct CudaBasicEncoderStorage final {
  CudaContext* context = nullptr;
  int device = 0;
  ModelShape shape;
  // Type-erased ownership only; numerical kernels use the typed float views.
  // Keep the RSD model alive even if its originating encoder is destroyed.
  std::shared_ptr<const void> shared_model_owner;
  const float* shared_atoms = nullptr;
  CudaDeviceBuffer<float> root0;
  CudaDeviceBuffer<float> small;
  CudaDeviceBuffer<float> root_norm2;
  CudaDeviceBuffer<float> small_norm2;
  CudaDeviceBuffer<float> small_inverse_norm;
  CudaDeviceBuffer<float> small_gram;
  CudaDeviceBuffer<float> root_small_gram;

  CudaDeviceBuffer<float> samples;
  CudaDeviceBuffer<float> small_dots;
  CudaDeviceBuffer<float> sample_norms;
  CudaDeviceBuffer<float> root_scores;
  CudaDeviceBuffer<float> root_errors;
  CudaDeviceBuffer<std::uint32_t> roots;
  CudaDeviceBuffer<std::uint8_t> codes;
  CudaDeviceBuffer<float> coefficients;
  CudaDeviceBuffer<float> errors;
  CudaDeviceBuffer<float> root_dots;
  CudaDeviceBuffer<std::uint8_t> active;
  CudaDeviceBuffer<std::uint32_t> failures;
  CudaDeviceBuffer<std::uint8_t> candidate_codes;
  CudaDeviceBuffer<float> candidate_coefficients;
  CudaDeviceBuffer<float> candidate_errors;
  CudaDeviceBuffer<std::uint64_t> global_sample_ids;

  ~CudaBasicEncoderStorage() { cudaSetDevice(device); }

  static StatusOr<std::unique_ptr<CudaBasicEncoderStorage>> Create(
      const RuntimeBasicCodebookAlgebra& algebra, CudaContext& context);
  static StatusOr<std::unique_ptr<CudaBasicEncoderStorage>> Create(
      RuntimeRootCodebookView codebooks, CudaContext& context,
      std::shared_ptr<const void> shared_model_owner = {},
      const float* shared_atoms = nullptr);

  Status PrepareDeviceGram();

  Status Download(std::uint32_t count, std::uint32_t* output_roots,
                  std::uint8_t* output_codes, float* output_coefficients,
                  float* output_errors);

  template <int M>
  DeviceAlgebra<M> AlgebraView() const noexcept {
    DeviceAlgebra<M> view;
    view.m = shape.m;
    view.dimension = shape.dimension;
    view.nlist = shape.nlist;
    view.small_count = static_cast<std::uint32_t>((shape.m - 1) * 256);
    view.root0 = shared_atoms != nullptr ? shared_atoms : root0.data();
    view.small = shared_atoms != nullptr
        ? shared_atoms + static_cast<std::size_t>(shape.nlist) * shape.dimension
        : small.data();
    view.root_norm2 = root_norm2.data();
    view.small_norm2 = small_norm2.data();
    view.small_inverse_norm = small_inverse_norm.data();
    view.small_gram = small_gram.data();
    view.root_small_gram = root_small_gram.data();
    return view;
  }

  Status EnsureCommonWorkspace(std::uint32_t count, bool need_host_input,
                               bool need_explicit_sample_ids) {
    const std::size_t rows = count;
    const std::size_t small_count =
        static_cast<std::size_t>((shape.m - 1) * 256);
    if (need_host_input) {
      if (const Status status = samples.Ensure(rows * shape.dimension);
          !status.ok()) {
        return status;
      }
    }
    const std::array<Status, 8> statuses = {
        small_dots.Ensure(rows * small_count),
        sample_norms.Ensure(rows),
        roots.Ensure(rows),
        codes.Ensure(rows * (shape.m - 1U)),
        coefficients.Ensure(rows * shape.m),
        errors.Ensure(rows),
        root_dots.Ensure(rows),
        active.Ensure(rows),
    };
    for (const Status& status : statuses) {
      if (!status.ok()) {
        return status;
      }
    }
    if (need_explicit_sample_ids) {
      if (const Status status = global_sample_ids.Ensure(rows); !status.ok()) {
        return status;
      }
    }
    return failures.Ensure(1U);
  }



  Status EnsureRsdRootWorkspace(std::uint32_t count) {
    const std::size_t rows = count;
    const std::array<Status, 2> statuses = {
        root_scores.Ensure(rows * kRootBlock),
        root_errors.Ensure(rows),
    };
    for (const Status& status : statuses) {
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }

  Status EnsureIlsWorkspace(std::uint32_t count) {
    const std::size_t rows = count;
    if (const Status status = candidate_codes.Ensure(rows * (shape.m - 1U));
        !status.ok())
      return status;
    if (const Status status = candidate_coefficients.Ensure(rows * shape.m);
        !status.ok())
      return status;
    return candidate_errors.Ensure(rows);
  }
};

}  // namespace detail

template <int M>
CudaBasicEncoder<M>::CudaBasicEncoder(
    std::unique_ptr<detail::CudaBasicEncoderStorage> state) noexcept
    : state_(std::move(state)) {}

template <int M>
CudaBasicEncoder<M>::~CudaBasicEncoder() = default;

template <int M>
CudaBasicEncoder<M>::CudaBasicEncoder(CudaBasicEncoder&& other) noexcept
    : state_(std::move(other.state_)) {}

template <int M>
CudaBasicEncoder<M>& CudaBasicEncoder<M>::operator=(
    CudaBasicEncoder&& other) noexcept {
  state_ = std::move(other.state_);
  return *this;
}

template <int M>
std::uint64_t CudaBasicEncoder<M>::model_resident_bytes() const noexcept {
  if (state_ == nullptr) {
    return 0U;
  }
  const std::size_t values =
      state_->root0.capacity() + state_->small.capacity() +
      state_->root_norm2.capacity() + state_->small_norm2.capacity() +
      state_->small_inverse_norm.capacity() + state_->small_gram.capacity() +
      state_->root_small_gram.capacity();
  return static_cast<std::uint64_t>(values) * sizeof(float);
}

StatusOr<std::unique_ptr<detail::CudaBasicEncoderStorage>>
detail::CudaBasicEncoderStorage::Create(
    const RuntimeBasicCodebookAlgebra& algebra, CudaContext& context) {
  if (context.device() < 0 ||
      algebra.shape().dimension >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      algebra.shape().nlist >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    return Status::InvalidArgument(
        "CUDA basic encoder received an invalid context or model extent");
  }
  const cudaError_t set_device = cudaSetDevice(context.device());
  if (set_device != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", set_device);
  }
  auto state = std::make_unique<CudaBasicEncoderStorage>();
  state->context = &context;
  state->device = context.device();
  state->shape = algebra.shape();
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context.native_stream());
  const std::size_t root_values =
      static_cast<std::size_t>(algebra.root0().rows) * algebra.root0().columns;
  const std::size_t small_values =
      static_cast<std::size_t>(algebra.small_flat().rows) *
      algebra.small_flat().columns;
  const std::array<Status, 7> statuses = {
      Upload(state->root0, {algebra.root0().values, root_values}, stream),
      Upload(state->small, {algebra.small_flat().values, small_values}, stream),
      Upload(state->root_norm2, algebra.root_norms2(), stream),
      Upload(state->small_norm2, algebra.small_norms2(), stream),
      Upload(state->small_inverse_norm, algebra.small_inverse_norms(), stream),
      Upload(state->small_gram, algebra.small_gram(), stream),
      Upload(state->root_small_gram, algebra.root_small_gram(), stream),
  };
  for (const Status& status : statuses) {
    if (!status.ok()) {
      return status;
    }
  }
  if (const Status status = context.Synchronize(); !status.ok()) {
    return status;
  }
  return std::move(state);
}

StatusOr<std::unique_ptr<detail::CudaBasicEncoderStorage>>
detail::CudaBasicEncoderStorage::Create(RuntimeRootCodebookView codebooks,
                                        CudaContext& context,
                                        std::shared_ptr<const void> model_owner,
                                        const float* model_atoms) {
  const ModelShape& shape = codebooks.shape;
  if (context.device() < 0 || shape.dimension == 0U ||
      shape.dimension >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      shape.nlist == 0U ||
      shape.nlist >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    return Status::InvalidArgument(
        "CUDA basic encoder received an invalid context or model extent");
  }
  if (const cudaError_t result = cudaSetDevice(context.device());
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", result);
  }
  auto state = std::make_unique<CudaBasicEncoderStorage>();
  state->context = &context;
  state->device = context.device();
  state->shape = shape;
  state->shared_model_owner = std::move(model_owner);
  state->shared_atoms = model_atoms;
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context.native_stream());
  const ColumnMajorMatrixF32View root = codebooks.root0;
  const ColumnMajorMatrixF32View small = codebooks.root_small_flat;
  const std::size_t root_values =
      static_cast<std::size_t>(root.rows) * root.columns;
  const std::size_t small_values =
      static_cast<std::size_t>(small.rows) * small.columns;
  std::vector<float> root_norm2;
  std::vector<float> root_inverse_norm;
  std::vector<float> small_norm2;
  std::vector<float> small_inverse_norm;
  ComputeBasicCodebookNorms(root, small, &root_norm2, &root_inverse_norm,
                            &small_norm2, &small_inverse_norm);
  if (model_atoms == nullptr) {
    if (const Status status =
            Upload(state->root0, {root.values, root_values}, stream);
        !status.ok()) return status;
    if (const Status status =
            Upload(state->small, {small.values, small_values}, stream);
        !status.ok()) return status;
  }
  const std::array<Status, 3> uploads = {
      Upload(state->root_norm2, ArrayView<const float>(root_norm2), stream),
      Upload(state->small_norm2, ArrayView<const float>(small_norm2), stream),
      Upload(state->small_inverse_norm,
             ArrayView<const float>(small_inverse_norm), stream),
  };
  for (const Status& status : uploads) {
    if (!status.ok()) return status;
  }
  if (const Status status = state->PrepareDeviceGram(); !status.ok()) {
    return status;
  }
  return std::move(state);
}

Status detail::CudaBasicEncoderStorage::PrepareDeviceGram() {
  const auto algebra = AlgebraView<0>();
  const auto small_count = algebra.small_count;
  const auto root_count_total = shape.nlist;
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context->native_stream());
  if (const Status status = small_gram.Ensure(
          static_cast<std::size_t>(small_count) * small_count);
      !status.ok())
    return status;
  if (const Status status = root_small_gram.Ensure(
          static_cast<std::size_t>(root_count_total) * small_count);
      !status.ok())
    return status;
  if (const Status status = CudaGemmF32(
          *context, CudaGemmOperation::kTranspose, CudaGemmOperation::kNone,
          static_cast<int>(small_count), static_cast<int>(small_count),
          static_cast<int>(shape.dimension), 1.0F, algebra.small,
          static_cast<int>(shape.dimension), algebra.small,
          static_cast<int>(shape.dimension), 0.0F, small_gram.data(),
          static_cast<int>(small_count));
      !status.ok())
    return status;
  // Keep the established C_root^T C_small GEMM shape. Only its layout change
  // is performed on the device. Bound the temporary for large nlist models.
  constexpr std::uint32_t kRootTile = 4096U;
  CudaDeviceBuffer<float> flat_major_root_small;
  if (const Status status = flat_major_root_small.Ensure(
          static_cast<std::size_t>(std::min(root_count_total, kRootTile)) *
          small_count);
      !status.ok())
    return status;
  for (std::uint32_t root_begin = 0U; root_begin < root_count_total;
       root_begin += kRootTile) {
    const std::uint32_t root_count =
        std::min(kRootTile, root_count_total - root_begin);
    if (const Status status = CudaGemmF32(
            *context, CudaGemmOperation::kTranspose, CudaGemmOperation::kNone,
            static_cast<int>(root_count), static_cast<int>(small_count),
            static_cast<int>(shape.dimension), 1.0F,
            algebra.root0 +
                static_cast<std::size_t>(root_begin) * shape.dimension,
            static_cast<int>(shape.dimension), algebra.small,
            static_cast<int>(shape.dimension), 0.0F,
            flat_major_root_small.data(), static_cast<int>(root_count));
        !status.ok())
      return status;
    const dim3 grid((root_count + 31U) / 32U, (small_count + 31U) / 32U);
    const dim3 block(32U, 8U);
    StoreRootSmallGramRootMajor<<<grid, block, 0U, stream>>>(
        flat_major_root_small.data(), root_begin, root_count, small_count,
        root_small_gram.data());
    if (const cudaError_t result = cudaPeekAtLastError();
        result != cudaSuccess) {
      return CudaRuntimeStatus("StoreRootSmallGramRootMajor", result);
    }
  }
  return context->Synchronize();
}

Status detail::CudaBasicEncoderStorage::Download(std::uint32_t count,
                                                 std::uint32_t* output_roots,
                                                 std::uint8_t* output_codes,
                                                 float* output_coefficients,
                                                 float* output_errors) {
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context->native_stream());
  cudaError_t cuda_result =
      cudaMemcpyAsync(output_roots, roots.data(),
                      static_cast<std::size_t>(count) * sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost, stream);
  if (cuda_result != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpyAsync roots D2H", cuda_result);
  }
  cuda_result =
      cudaMemcpyAsync(output_codes, codes.data(),
                      static_cast<std::size_t>(count) * (shape.m - 1U),
                      cudaMemcpyDeviceToHost, stream);
  if (cuda_result != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpyAsync codes D2H", cuda_result);
  }
  cuda_result =
      cudaMemcpyAsync(output_coefficients, coefficients.data(),
                      static_cast<std::size_t>(count) * shape.m * sizeof(float),
                      cudaMemcpyDeviceToHost, stream);
  if (cuda_result != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpyAsync coefficients D2H", cuda_result);
  }
  cuda_result = cudaMemcpyAsync(output_errors, errors.data(),
                                static_cast<std::size_t>(count) * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
  if (cuda_result != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpyAsync errors D2H", cuda_result);
  }
  if (const Status status = context->Synchronize(); !status.ok()) {
    return status;
  }
  return Status::Ok();
}

template <int M>
StatusOr<CudaBasicEncoder<M>> CudaBasicEncoder<M>::Create(
    const BasicCodebookAlgebra<M>& algebra, CudaContext& context) {
  auto state = detail::CudaBasicEncoderStorage::Create(algebra, context);
  if (!state.ok()) return state.status();
  return CudaBasicEncoder<M>(std::move(state).value());
}

template <int M>
StatusOr<CudaBasicEncoder<M>> CudaBasicEncoder<M>::Create(
    const RootCodebooks<M>& codebooks, CudaContext& context) {
  auto state = detail::CudaBasicEncoderStorage::Create(codebooks.runtime_view(),
                                                       context);
  if (!state.ok()) return state.status();
  return CudaBasicEncoder<M>(std::move(state).value());
}

template <int M>
StatusOr<CudaBasicEncoder<M>> CudaBasicEncoder<M>::Create(
    const RootCodebooks<M>& codebooks,
    std::shared_ptr<const CudaLinkageRsdDeviceModel<M>> basic_rsd_model,
    CudaContext& context) {
  if (basic_rsd_model == nullptr || basic_rsd_model->shape().nlist != 256U ||
      basic_rsd_model->device() != context.device() ||
      codebooks.shape().nlist != 256U ||
      basic_rsd_model->shape().m != codebooks.shape().m ||
      basic_rsd_model->shape().dimension != codebooks.shape().dimension) {
    return Status::InvalidArgument(
        "shared basic RSD model requires IVF256 on the context device");
  }
  const float* atoms = basic_rsd_model->atoms();
  auto state = detail::CudaBasicEncoderStorage::Create(
      codebooks.runtime_view(), context, std::move(basic_rsd_model), atoms);
  if (!state.ok()) return state.status();
  return CudaBasicEncoder<M>(std::move(state).value());
}

template <int M>
StatusOr<BasicEncoding<M>> CudaBasicEncoder<M>::Encode(
    ColumnMajorMatrixF32View samples, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t global_sample_offset) {
  if (state_ == nullptr || samples.values == nullptr ||
      samples.rows != state_->shape.dimension || samples.columns == 0U) {
    return Status::InvalidArgument(
        "CUDA basic input must be a non-empty contiguous model-space matrix");
  }
  const std::size_t input_values =
      static_cast<std::size_t>(samples.rows) * samples.columns;
  for (std::size_t index = 0U; index < input_values; ++index) {
    if (!std::isfinite(samples.values[index])) {
      return Status::InvalidArgument(
          "CUDA basic input contains a non-finite value");
    }
  }
  return EncodeImpl(samples.values, samples.columns, false, schedule, seed,
                    global_sample_offset, {}, {}, {}, {}, {}, {}, true);
}

template <int M>
StatusOr<BasicEncoding<M>> CudaBasicEncoder<M>::EncodeDevice(
    CudaMatrixF32View samples, BasicSchedule schedule, std::uint32_t seed,
    std::uint64_t global_sample_offset) {
  if (state_ == nullptr || samples.values == nullptr ||
      samples.rows != state_->shape.dimension || samples.columns == 0U ||
      samples.leading_dimension != samples.rows) {
    return Status::InvalidArgument(
        "CUDA basic device input must be a non-empty contiguous model-space "
        "matrix");
  }
  return EncodeImpl(samples.values, samples.columns, true, schedule, seed,
                    global_sample_offset, {}, {}, {}, {}, {}, {}, true);
}

template <int M>
StatusOr<BasicEncoding<M>> CudaBasicEncoder<M>::EncodeFixedRoots(
    ColumnMajorMatrixF32View samples,
    ArrayView<const std::uint32_t> fixed_roots,
    ArrayView<const std::uint64_t> global_sample_ids, BasicSchedule schedule,
    std::uint32_t seed) {
  if (state_ == nullptr || samples.values == nullptr ||
      samples.rows != state_->shape.dimension || samples.columns == 0U ||
      fixed_roots.size() != samples.columns ||
      global_sample_ids.size() != samples.columns) {
    return Status::InvalidArgument(
        "CUDA fixed-root input has inconsistent model-space extents");
  }
  const std::size_t input_values =
      static_cast<std::size_t>(samples.rows) * samples.columns;
  for (std::size_t index = 0U; index < input_values; ++index) {
    if (!std::isfinite(samples.values[index])) {
      return Status::InvalidArgument(
          "CUDA fixed-root input contains a non-finite value");
    }
  }
  for (const std::uint32_t root : fixed_roots) {
    if (root >= state_->shape.nlist) {
      return Status::InvalidArgument("CUDA fixed-root code exceeds nlist");
    }
  }
  return EncodeImpl(samples.values, samples.columns, false, schedule, seed, 0U,
                    fixed_roots, global_sample_ids, {}, {}, {}, {}, true);
}

template <int M>
StatusOr<BasicEncoding<M>> CudaBasicEncoder<M>::EncodeFixedRootsDevice(
    CudaMatrixF32View samples, ArrayView<const std::uint32_t> fixed_roots,
    std::uint64_t global_sample_offset, BasicSchedule schedule,
    std::uint32_t seed) {
  if (state_ == nullptr || samples.values == nullptr ||
      samples.rows != state_->shape.dimension || samples.columns == 0U ||
      samples.leading_dimension != samples.rows ||
      fixed_roots.size() != samples.columns) {
    return Status::InvalidArgument(
        "CUDA fixed-root device input has inconsistent extents");
  }
  for (const std::uint32_t root : fixed_roots) {
    if (root >= state_->shape.nlist) {
      return Status::InvalidArgument("CUDA fixed-root code exceeds nlist");
    }
  }
  return EncodeImpl(samples.values, samples.columns, true, schedule, seed,
                    global_sample_offset, fixed_roots, {}, {}, {}, {}, {},
                    true);
}

template <int M>
StatusOr<BasicEncoding<M>> CudaBasicEncoder<M>::RefineFixedRootSupports(
    ColumnMajorMatrixF32View samples,
    ArrayView<const std::uint32_t> fixed_roots,
    ArrayView<const std::uint8_t> initial_small_codes, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t global_sample_offset) {
  if (state_ == nullptr || samples.values == nullptr ||
      samples.rows != state_->shape.dimension || samples.columns == 0U ||
      fixed_roots.size() != samples.columns ||
      initial_small_codes.size() !=
          static_cast<std::size_t>(samples.columns) * (state_->shape.m - 1U)) {
    return Status::InvalidArgument(
        "CUDA fixed-root support input has inconsistent extents");
  }
  const std::size_t input_values =
      static_cast<std::size_t>(samples.rows) * samples.columns;
  for (std::size_t index = 0U; index < input_values; ++index) {
    if (!std::isfinite(samples.values[index])) {
      return Status::InvalidArgument(
          "CUDA fixed-root support input contains a non-finite value");
    }
  }
  for (const std::uint32_t root : fixed_roots) {
    if (root >= state_->shape.nlist) {
      return Status::InvalidArgument(
          "CUDA fixed-root support code exceeds nlist");
    }
  }
  return EncodeImpl(samples.values, samples.columns, false, schedule, seed,
                    global_sample_offset, fixed_roots, {}, initial_small_codes,
                    {}, {}, {}, true);
}

template <int M>
StatusOr<BasicEncoding<M>> CudaBasicEncoder<M>::RefineFixedRootSupportsDevice(
    CudaMatrixF32View samples, ArrayView<const std::uint32_t> fixed_roots,
    ArrayView<const std::uint8_t> initial_small_codes, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t global_sample_offset) {
  if (state_ == nullptr || samples.values == nullptr ||
      samples.rows != state_->shape.dimension || samples.columns == 0U ||
      samples.leading_dimension != samples.rows ||
      fixed_roots.size() != samples.columns ||
      initial_small_codes.size() !=
          static_cast<std::size_t>(samples.columns) * (state_->shape.m - 1U)) {
    return Status::InvalidArgument(
        "CUDA fixed-root support device input has inconsistent extents");
  }
  for (const std::uint32_t root : fixed_roots) {
    if (root >= state_->shape.nlist) {
      return Status::InvalidArgument(
          "CUDA fixed-root support device code exceeds nlist");
    }
  }
  return EncodeImpl(samples.values, samples.columns, true, schedule, seed,
                    global_sample_offset, fixed_roots, {}, initial_small_codes,
                    {}, {}, {}, true);
}

template <int M>
StatusOr<BasicEncoding<M>> CudaBasicEncoder<M>::RefineDeviceResidentSupports(
    CudaMatrixF32View samples, const std::uint8_t* device_support_codes,
    BasicSchedule schedule, std::uint32_t seed,
    std::uint64_t global_sample_offset, const float* device_sample_norms) {
  if (state_ == nullptr || samples.values == nullptr ||
      samples.rows != state_->shape.dimension || samples.columns == 0U ||
      samples.leading_dimension != samples.rows ||
      device_support_codes == nullptr) {
    return Status::InvalidArgument(
        "CUDA resident basic support input has inconsistent extents");
  }
  return EncodeImpl(samples.values, samples.columns, true, schedule, seed,
                    global_sample_offset, {}, {}, {}, {}, {}, {}, true,
                    device_support_codes, device_sample_norms);
}

template <int M>
StatusOr<BasicEncoding<M>> CudaBasicEncoder<M>::EncodeRsdDevice(
    CudaMatrixF32View samples, ArrayView<const std::uint32_t> fixed_roots,
    std::uint32_t rsd_rank, LeastSquaresCostMode rsd_cost_mode,
    BasicSchedule schedule, std::uint32_t seed,
    std::uint64_t global_sample_offset) {
  if (state_ == nullptr || samples.values == nullptr ||
      samples.rows != state_->shape.dimension ||
      samples.leading_dimension != samples.rows || samples.columns == 0U ||
      (!fixed_roots.empty() && fixed_roots.size() != samples.columns) ||
      rsd_rank < 1U || rsd_rank > 3U) {
    return Status::InvalidArgument("CUDA basic RSD request is invalid");
  }
  for (std::uint32_t root : fixed_roots) {
    if (root >= state_->shape.nlist) {
      return Status::InvalidArgument("CUDA basic RSD root exceeds codebook");
    }
  }
  return EncodeImpl(samples.values, samples.columns, true, schedule, seed,
                    global_sample_offset, fixed_roots, {}, {}, {}, {}, {}, true,
                    nullptr, nullptr, rsd_rank, rsd_cost_mode);
}

template <int M>
StatusOr<BasicEncoding<M>>
CudaBasicEncoder<M>::RefineFixedRootProjectedSupports(
    std::uint32_t sample_count, ArrayView<const std::uint32_t> fixed_roots,
    ArrayView<const std::uint8_t> initial_small_codes,
    ArrayView<const float> root_dots, ArrayView<const float> small_dots,
    ArrayView<const float> sample_norms2, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t global_sample_offset) {
  if (state_ == nullptr || sample_count == 0U ||
      fixed_roots.size() != sample_count ||
      initial_small_codes.size() !=
          static_cast<std::size_t>(sample_count) * (state_->shape.m - 1U) ||
      root_dots.size() != sample_count ||
      small_dots.size() != static_cast<std::size_t>(sample_count) *
                               (state_->shape.m - 1U) * 256U ||
      sample_norms2.size() != sample_count) {
    return Status::InvalidArgument(
        "CUDA projected fixed-root input has inconsistent extents");
  }
  for (const std::uint32_t root : fixed_roots) {
    if (root >= state_->shape.nlist) {
      return Status::InvalidArgument(
          "CUDA projected fixed-root code exceeds nlist");
    }
  }
  return EncodeImpl(nullptr, sample_count, false, schedule, seed,
                    global_sample_offset, fixed_roots, {}, initial_small_codes,
                    root_dots, small_dots, sample_norms2, true);
}

template <int M>
StatusOr<BasicEncoding<M>> CudaBasicEncoder<M>::EncodeImpl(
    const float* samples, std::uint32_t sample_count, bool source_is_device,
    BasicSchedule schedule, std::uint32_t seed,
    std::uint64_t global_sample_offset,
    ArrayView<const std::uint32_t> fixed_roots,
    ArrayView<const std::uint64_t> global_sample_ids,
    ArrayView<const std::uint8_t> initial_small_codes,
    ArrayView<const float> projected_root_dots,
    ArrayView<const float> projected_small_dots,
    ArrayView<const float> projected_sample_norms2, bool refine,
    const std::uint8_t* device_initial_support_codes,
    const float* device_sample_norms, std::uint32_t rsd_rank,
    LeastSquaresCostMode rsd_cost_mode) {
  if constexpr (M == 0) {
    if (state_->shape.m > kLegacyRuntimeScratchLayers)
      return EncodeImplWithCapacity<kWideRuntimeScratchLayers>(
          samples, sample_count, source_is_device, schedule, seed, global_sample_offset,
          fixed_roots, global_sample_ids, initial_small_codes, projected_root_dots,
          projected_small_dots, projected_sample_norms2, refine, device_initial_support_codes,
          device_sample_norms, rsd_rank, rsd_cost_mode);
  }
  return EncodeImplWithCapacity<kModelScratchLayers<M>>(
      samples, sample_count, source_is_device, schedule, seed, global_sample_offset,
      fixed_roots, global_sample_ids, initial_small_codes, projected_root_dots,
      projected_small_dots, projected_sample_norms2, refine, device_initial_support_codes,
      device_sample_norms, rsd_rank, rsd_cost_mode);
}

template <int M>
template <std::uint32_t Capacity>
StatusOr<BasicEncoding<M>> CudaBasicEncoder<M>::EncodeImplWithCapacity(
    const float* samples, std::uint32_t sample_count, bool source_is_device,
    BasicSchedule schedule, std::uint32_t seed,
    std::uint64_t global_sample_offset,
    ArrayView<const std::uint32_t> fixed_roots,
    ArrayView<const std::uint64_t> global_sample_ids,
    ArrayView<const std::uint8_t> initial_small_codes,
    ArrayView<const float> projected_root_dots,
    ArrayView<const float> projected_small_dots,
    ArrayView<const float> projected_sample_norms2, bool refine,
    const std::uint8_t* device_initial_support_codes,
    const float* device_sample_norms, std::uint32_t rsd_rank,
    LeastSquaresCostMode rsd_cost_mode) {
  const bool use_device_initial_supports =
      device_initial_support_codes != nullptr;
  const bool use_fixed_roots =
      !fixed_roots.empty() || use_device_initial_supports;
  const bool use_explicit_sample_ids = !global_sample_ids.empty();
  const bool use_initial_supports =
      !initial_small_codes.empty() || use_device_initial_supports;
  const bool use_projected = !projected_root_dots.empty();
  if (device_sample_norms != nullptr &&
      (!source_is_device || !use_device_initial_supports || use_projected)) {
    return Status::InvalidArgument(
        "CUDA resident basic norms require device-resident supports");
  }
  const cudaError_t set_device = cudaSetDevice(state_->device);
  if (set_device != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", set_device);
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  BasicEncoding<M> output =
      detail::BasicEncodingAccess::Create<M>(sample_count, state_->shape.m);
  const std::vector<detail::CudaLayerOrder<M>> layer_orders =
      refine ? MakeLayerOrders<M>(state_->shape.m, schedule, seed)
             : std::vector<detail::CudaLayerOrder<M>>{};

  // Frozen DEV's closed-loop fixed-root refresh presents one complete source
  // block to projection/RSD/refinement.  Unlike ordinary basic encoding it
  // does not tile that block to 8192 columns; cuBLAS TF32 kernel selection is
  // shape-sensitive, so tiling here changes the replay supports and every LS
  // coefficient. Keep ordinary encoding bounded by the registered variable-
  // root tile and retain
  // the source-owned block boundary for fixed-root replay.
  const std::uint32_t encoding_block =
      use_fixed_roots ? sample_count : kVariableRootTileRows;
  for (std::uint32_t block_offset = 0U; block_offset < sample_count;
       block_offset += encoding_block) {
    const std::uint32_t count =
        std::min(encoding_block, sample_count - block_offset);
    if (const Status status = state_->EnsureCommonWorkspace(
            count, !source_is_device && !use_projected,
            use_explicit_sample_ids);
        !status.ok()) {
      return status;
    }
    if (!use_initial_supports && !use_fixed_roots) {
      if (const Status status = state_->EnsureRsdRootWorkspace(count);
          !status.ok())
        return status;
    }
    if (refine && EffectiveBasicIlsRounds<M>(schedule, state_->shape.m) != 0U) {
      if (const Status status = state_->EnsureIlsWorkspace(count);
          !status.ok()) {
        return status;
      }
    }
    const float* device_samples =
        use_projected ? nullptr
                      : samples + static_cast<std::size_t>(block_offset) *
                                      state_->shape.dimension;
    cudaError_t cuda_result = cudaSuccess;
    if (!source_is_device && !use_projected) {
      const std::size_t block_values =
          static_cast<std::size_t>(state_->shape.dimension) * count;
      cuda_result = cudaMemcpyAsync(state_->samples.data(), device_samples,
                                    block_values * sizeof(float),
                                    cudaMemcpyHostToDevice, stream);
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("cudaMemcpyAsync basic samples", cuda_result);
      }
      device_samples = state_->samples.data();
    }
    if (use_fixed_roots) {
      if (use_device_initial_supports) {
        constexpr std::uint32_t kUnpackThreads = 128U;
        const std::uint32_t unpack_grid =
            (count + kUnpackThreads - 1U) / kUnpackThreads;
        UnpackResidentSupportsKernel<M>
            <<<unpack_grid, kUnpackThreads, 0U, stream>>>(
                state_->shape.m,
                device_initial_support_codes +
                    static_cast<std::size_t>(block_offset) * state_->shape.m,
                count, state_->roots.data(), state_->codes.data());
        if (cuda_result = cudaPeekAtLastError(); cuda_result != cudaSuccess) {
          return CudaRuntimeStatus("UnpackResidentSupportsKernel", cuda_result);
        }
      } else {
        cuda_result = cudaMemcpyAsync(
            state_->roots.data(), fixed_roots.data() + block_offset,
            static_cast<std::size_t>(count) * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice, stream);
        if (cuda_result != cudaSuccess) {
          return CudaRuntimeStatus("cudaMemcpyAsync fixed roots H2D",
                                   cuda_result);
        }
      }
      if (use_explicit_sample_ids) {
        cuda_result = cudaMemcpyAsync(
            state_->global_sample_ids.data(),
            global_sample_ids.data() + block_offset,
            static_cast<std::size_t>(count) * sizeof(std::uint64_t),
            cudaMemcpyHostToDevice, stream);
        if (cuda_result != cudaSuccess) {
          return CudaRuntimeStatus("cudaMemcpyAsync explicit sample IDs H2D",
                                   cuda_result);
        }
      }
      if (use_initial_supports && !use_device_initial_supports) {
        cuda_result = cudaMemcpyAsync(
            state_->codes.data(),
            initial_small_codes.data() +
                static_cast<std::size_t>(block_offset) * (state_->shape.m - 1U),
            static_cast<std::size_t>(count) * (state_->shape.m - 1U),
            cudaMemcpyHostToDevice, stream);
        if (cuda_result != cudaSuccess) {
          return CudaRuntimeStatus("cudaMemcpyAsync initial support H2D",
                                   cuda_result);
        }
      }
    }
    if (use_projected) {
      cuda_result = cudaMemcpyAsync(
          state_->root_dots.data(), projected_root_dots.data() + block_offset,
          static_cast<std::size_t>(count) * sizeof(float),
          cudaMemcpyHostToDevice, stream);
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("cudaMemcpyAsync projected root dots H2D",
                                 cuda_result);
      }
      cuda_result = cudaMemcpyAsync(
          state_->small_dots.data(),
          projected_small_dots.data() + static_cast<std::size_t>(block_offset) *
                                            (state_->shape.m - 1U) * 256U,
          static_cast<std::size_t>(count) * (state_->shape.m - 1U) * 256U *
              sizeof(float),
          cudaMemcpyHostToDevice, stream);
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("cudaMemcpyAsync projected small dots H2D",
                                 cuda_result);
      }
      cuda_result =
          cudaMemcpyAsync(state_->sample_norms.data(),
                          projected_sample_norms2.data() + block_offset,
                          static_cast<std::size_t>(count) * sizeof(float),
                          cudaMemcpyHostToDevice, stream);
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("cudaMemcpyAsync projected sample norm2 H2D",
                                 cuda_result);
      }
    }
    constexpr std::uint32_t kThreads = 128U;
    const std::uint32_t grid = (count + kThreads - 1U) / kThreads;
    const DeviceAlgebra<M> algebra = state_->AlgebraView<M>();
    const float* block_sample_norms = device_sample_norms != nullptr
                                          ? device_sample_norms + block_offset
                                          : state_->sample_norms.data();
    if (!use_projected) {
      if (device_sample_norms == nullptr) {
        SquaredNormKernel<<<count, 256U, 0U, stream>>>(
            device_samples, state_->shape.dimension, count,
            state_->sample_norms.data());
        if (cuda_result = cudaPeekAtLastError(); cuda_result != cudaSuccess) {
          return CudaRuntimeStatus("SquaredNormKernel", cuda_result);
        }
      }
      if (const Status status = GemmTransposedLeft(
              *state_->context, algebra.small, algebra.small_count,
              device_samples, algebra.dimension, count,
              state_->small_dots.data());
          !status.ok()) {
        return status;
      }
    }
    if (!use_initial_supports) {
      const auto search_rank = rsd_rank == 0U ? 2U : rsd_rank;
      const auto search_cost_mode = rsd_rank == 0U
          ? schedule.least_squares_cost_mode() : rsd_cost_mode;
      if (!use_fixed_roots) {
        if (const Status status = RunRsdTopOneRoot<M>(
                algebra, *state_->context, device_samples, block_sample_norms,
                count, state_->root_scores.data(), state_->root_errors.data(),
                state_->roots.data(), stream);
            !status.ok())
          return status;
      }
      RootDotKernel<M><<<grid, kThreads, 0U, stream>>>(
          algebra, device_samples, count, state_->roots.data(),
          state_->root_dots.data());
      if (cuda_result = cudaPeekAtLastError(); cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("RSD RootDotKernel", cuda_result);
      }
      cuda_result = cudaMemsetAsync(state_->failures.data(), 0,
                                    sizeof(std::uint32_t), stream);
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("cudaMemsetAsync RSD failures", cuda_result);
      }
      // M5's two costs have different occupancy limits. Keep those numeric
      // variants; other M paths share the complete RSD candidate construction.
      if constexpr (M == 5) {
        if (search_cost_mode == LeastSquaresCostMode::kNormalEquation) {
          BuildBasicRsdSupportsKernel<M, Capacity, static_cast<int>(LeastSquaresCostMode::kNormalEquation)>
              <<<count, 256U, 0U, stream>>>(
                  algebra, state_->small_dots.data(), state_->root_dots.data(),
                  block_sample_norms, count, state_->roots.data(), search_rank,
                  search_cost_mode, state_->codes.data(), state_->failures.data());
        } else {
          BuildBasicRsdSupportsKernel<M, Capacity, static_cast<int>(LeastSquaresCostMode::kExplicitQuadratic)>
              <<<count, 256U, 0U, stream>>>(
                  algebra, state_->small_dots.data(), state_->root_dots.data(),
                  block_sample_norms, count, state_->roots.data(), search_rank,
                  search_cost_mode, state_->codes.data(), state_->failures.data());
        }
      } else {
        BuildBasicRsdSupportsKernel<M, Capacity><<<count, 256U, 0U, stream>>>(
            algebra, state_->small_dots.data(), state_->root_dots.data(),
            block_sample_norms, count, state_->roots.data(), search_rank,
            search_cost_mode, state_->codes.data(), state_->failures.data());
      }
      if (cuda_result = cudaPeekAtLastError(); cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("BuildBasicRsdSupportsKernel", cuda_result);
      }
      std::uint32_t rsd_failures = 0U;
      cuda_result =
          cudaMemcpyAsync(&rsd_failures, state_->failures.data(),
                          sizeof(rsd_failures), cudaMemcpyDeviceToHost, stream);
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("cudaMemcpyAsync RSD failures", cuda_result);
      }
      if (const Status status = state_->context->Synchronize(); !status.ok()) {
        return status;
      }
      if (rsd_failures != 0U) {
        return Status::FailedPrecondition(
            "CUDA basic RSD contains a singular candidate set");
      }
    } else if (use_fixed_roots) {
      if (!use_projected) {
        RootDotKernel<M><<<grid, kThreads, 0U, stream>>>(
            algebra, device_samples, count, state_->roots.data(),
            state_->root_dots.data());
        if (cuda_result = cudaPeekAtLastError(); cuda_result != cudaSuccess) {
          return CudaRuntimeStatus("fixed-root RootDotKernel", cuda_result);
        }
      }
    }
    cuda_result = cudaMemsetAsync(state_->failures.data(), 0,
                                  sizeof(std::uint32_t), stream);
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("cudaMemsetAsync failures", cuda_result);
    }
    cuda_result = LaunchConfiguredRefit<M, Capacity>(
        schedule.least_squares_cost_mode(), algebra, state_->small_dots.data(),
        block_sample_norms, state_->root_dots.data(), count,
        state_->roots.data(), state_->codes.data(), state_->coefficients.data(),
        state_->errors.data(), state_->active.data(), state_->failures.data(),
        stream);
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("RefitKernel", cuda_result);
    }
    std::uint32_t failures = 0U;
    cuda_result =
        cudaMemcpyAsync(&failures, state_->failures.data(), sizeof(failures),
                        cudaMemcpyDeviceToHost, stream);
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("cudaMemcpyAsync refit failures", cuda_result);
    }
    cuda_result = cudaMemcpyAsync(
        detail::BasicEncodingAccess::InitialErrors(&output) + block_offset,
        state_->errors.data(), static_cast<std::size_t>(count) * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    if (cuda_result != cudaSuccess) {
      return CudaRuntimeStatus("cudaMemcpyAsync initial errors D2H", cuda_result);
    }
    if (const Status status = state_->context->Synchronize(); !status.ok()) {
      return status;
    }
    if (failures != 0U) {
      return Status::FailedPrecondition(
          "initial CUDA basic support is singular after Cholesky retries");
    }

    std::size_t order_index = 0U;
    auto run_sweeps = [&](detail::CudaLayerOrder<M> order0,
                          detail::CudaLayerOrder<M> order1, bool pair,
                          std::uint8_t* codes, float* coefficients,
                          float* errors) -> Status {
      const bool absolute_score =
          schedule.score_rule() == BasicScoreRule::kAbsolute;
      const cudaError_t result =
          LaunchConfiguredSweeps<M, Capacity>(
              schedule.least_squares_cost_mode(), algebra,
              state_->small_dots.data(), block_sample_norms,
              state_->root_dots.data(), count, order0, order1,
              state_->roots.data(), codes, coefficients, errors,
              state_->active.data(), absolute_score, pair ? 2U : 1U, stream);
      return result == cudaSuccess
                 ? Status::Ok()
                 : CudaRuntimeStatus("IcmSweepsWarpRowsKernel", result);
    };
    auto run_phase = [&](std::uint8_t* codes, float* coefficients,
                         float* errors) -> Status {
      std::uint32_t sweep = 0U;
      for (; sweep + 1U < schedule.icm_sweeps(); sweep += 2U) {
        const detail::CudaLayerOrder<M> order0 = layer_orders[order_index++];
        const detail::CudaLayerOrder<M> order1 = layer_orders[order_index++];
        if (const Status status =
                run_sweeps(order0, order1, true, codes, coefficients, errors);
            !status.ok()) {
          return status;
        }
      }
      if (sweep < schedule.icm_sweeps()) {
        if (const Status status =
                run_sweeps(layer_orders[order_index++], {}, false, codes,
                           coefficients, errors);
            !status.ok()) {
          return status;
        }
      }
      return Status::Ok();
    };
    if (refine) {
      if (const Status status =
              run_phase(state_->codes.data(), state_->coefficients.data(),
                        state_->errors.data());
          !status.ok()) {
        return status;
      }
    }
    for (std::uint32_t round = 0U;
         refine &&
         round < EffectiveBasicIlsRounds<M>(schedule, state_->shape.m);
         ++round) {
      cuda_result = cudaMemcpyAsync(
          state_->candidate_codes.data(), state_->codes.data(),
          static_cast<std::size_t>(count) * (state_->shape.m - 1U),
          cudaMemcpyDeviceToDevice, stream);
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("cudaMemcpyAsync candidate codes",
                                 cuda_result);
      }
      if (use_explicit_sample_ids) {
        PerturbExplicitIdsKernel<M><<<grid, kThreads, 0U, stream>>>(
            state_->shape.m, count, round, seed,
            EffectiveBasicPerturbations<M>(schedule, state_->shape.m),
            state_->global_sample_ids.data(), state_->candidate_codes.data());
      } else {
        PerturbKernel<M><<<grid, kThreads, 0U, stream>>>(
            state_->shape.m, count, round, seed,
            EffectiveBasicPerturbations<M>(schedule, state_->shape.m),
            global_sample_offset + block_offset,
            state_->candidate_codes.data());
      }
      if (cuda_result = cudaPeekAtLastError(); cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("PerturbKernel", cuda_result);
      }
      cuda_result = cudaMemsetAsync(state_->failures.data(), 0,
                                    sizeof(std::uint32_t), stream);
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("cudaMemsetAsync candidate failures",
                                 cuda_result);
      }
      cuda_result = LaunchConfiguredRefit<M, Capacity>(
          schedule.least_squares_cost_mode(), algebra,
          state_->small_dots.data(), block_sample_norms,
          state_->root_dots.data(), count, state_->roots.data(),
          state_->candidate_codes.data(), state_->candidate_coefficients.data(),
          state_->candidate_errors.data(), state_->active.data(),
          state_->failures.data(), stream);
      if (cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("candidate RefitKernel", cuda_result);
      }
      if (const Status status = run_phase(state_->candidate_codes.data(),
                                          state_->candidate_coefficients.data(),
                                          state_->candidate_errors.data());
          !status.ok()) {
        return status;
      }
      AcceptKernel<M><<<grid, kThreads, 0U, stream>>>(
          state_->shape.m, count, state_->candidate_codes.data(),
          state_->candidate_coefficients.data(),
          state_->candidate_errors.data(), state_->codes.data(),
          state_->coefficients.data(), state_->errors.data());
      if (cuda_result = cudaPeekAtLastError(); cuda_result != cudaSuccess) {
        return CudaRuntimeStatus("AcceptKernel", cuda_result);
      }
    }

    if (const Status status = state_->Download(
            count, detail::BasicEncodingAccess::Roots(&output) + block_offset,
            detail::BasicEncodingAccess::Codes(&output) +
                static_cast<std::size_t>(block_offset) * (state_->shape.m - 1U),
            detail::BasicEncodingAccess::Coefficients(&output) +
                static_cast<std::size_t>(block_offset) * state_->shape.m,
            detail::BasicEncodingAccess::Errors(&output) + block_offset);
        !status.ok()) {
      return status;
    }
  }
  return std::move(output);
}

template class CudaBasicEncoder<0>;
template class CudaBasicEncoder<5>;
template class CudaBasicEncoder<12>;
template class CudaBasicEncoder<6>;
template class CudaBasicEncoder<10>;

}  // namespace stlqpp
