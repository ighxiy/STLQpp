#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "stlqpp/runtime/cuda/device_buffer.h"
#include "stlqpp/runtime/cuda/status.h"
#include "stlqpp/training/basic/cuda_fixed_root_support_refiner.h"
#include "stlqpp/training/basic/detail/basic_encoding_access.h"
#include "stlqpp/training/basic/forced_root_global_greedy.h"

namespace stlqpp {
namespace {

constexpr float kAcceptanceEpsilon = 1.0e-6F;

template <int M>
struct DeviceFixedRootAlgebra final {
  std::uint32_t m = M;
  std::uint32_t small_count = 0U;
  const float* root_norm2 = nullptr;
  const float* small_inverse_norm = nullptr;
  const float* small_gram = nullptr;
  const float* fixed_root_small_gram = nullptr;
};

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
                                         const float* factor, float* rhs) {
  const int m = M == 0 ? actual_m : M;
  float forward[Capacity];
#pragma unroll
  for (int row = 0; row < m; ++row) {
    float value = rhs[row];
#pragma unroll
    for (int prior = 0; prior < row; ++prior) {
      value -= factor[PackedUpperIndex(prior, row)] * forward[prior];
    }
    forward[row] = value / factor[PackedUpperIndex(row, row)];
  }
  for (int row = m - 1; row >= 0; --row) {
    float value = forward[row];
    for (int next = row + 1; next < m; ++next) {
      value -= factor[PackedUpperIndex(row, next)] * rhs[next];
    }
    rhs[row] = value / factor[PackedUpperIndex(row, row)];
  }
}

template <int M>
__device__ __forceinline__ std::uint32_t FlatCode(const std::uint8_t* codes,
                                                  int layer) {
  return static_cast<std::uint32_t>(layer - 1) * 256U +
         static_cast<std::uint32_t>(codes[layer - 1]);
}

template <int M>
__device__ __forceinline__ float GramValue(DeviceFixedRootAlgebra<M> algebra,
                                           std::uint32_t root,
                                           const std::uint8_t* codes, int row,
                                           int column) {
  if (row == 0 && column == 0) {
    return algebra.root_norm2[root];
  }
  if (row == 0 || column == 0) {
    const int small_layer = row == 0 ? column : row;
    return algebra.fixed_root_small_gram[FlatCode<M>(codes, small_layer)];
  }
  const std::uint32_t flat_row = FlatCode<M>(codes, row);
  const std::uint32_t flat_column = FlatCode<M>(codes, column);
  return algebra
      .small_gram[static_cast<std::size_t>(flat_column) * algebra.small_count +
                  flat_row];
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__device__ bool SolveAndCost(DeviceFixedRootAlgebra<M> algebra,
                             std::uint32_t root, const std::uint8_t* codes,
                             float root_dot, const float* small_dots,
                             float sample_norm2, float* coefficients,
                             float* cost, LeastSquaresCostMode cost_mode) {
  const int m = M == 0 ? algebra.m : M;
  const int kPacked = (m * (m + 1)) / 2;
  float original[Capacity * (Capacity + 1) / 2];
  float factor[Capacity * (Capacity + 1) / 2];
  float dots[Capacity];
  float solution[Capacity];

#pragma unroll
  for (int column = 0; column < m; ++column) {
    const float dot =
        column == 0 ? root_dot : small_dots[FlatCode<M>(codes, column)];
    dots[column] = dot;
    solution[column] = dot;
#pragma unroll
    for (int row = 0; row <= column; ++row) {
      original[PackedUpperIndex(row, column)] =
          GramValue<M>(algebra, root, codes, row, column);
    }
  }

  bool solved = false;
  float bump = 0.0F;
  for (int attempt = 0; attempt < 4 && !solved; ++attempt) {
#pragma unroll
    for (int entry = 0; entry < kPacked; ++entry) {
      factor[entry] = original[entry];
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
#pragma unroll
    for (int layer = 0; layer < m; ++layer) {
      coefficients[layer] = 0.0F;
    }
    *cost = sample_norm2;
    return false;
  }

  SolveCholeskyUpperPacked<M, Capacity>(m, factor, solution);
#pragma unroll
  for (int layer = 0; layer < m; ++layer) {
    coefficients[layer] = solution[layer];
  }

  float linear = 0.0F;
#pragma unroll
  for (int row = 0; row < m; ++row) {
    linear += solution[row] * dots[row];
  }
  // Cost selection does not specialize the packed solve and retry flow.
  if (cost_mode == LeastSquaresCostMode::kNormalEquation) {
    *cost = sample_norm2 - linear;
    if (bump > 0.0F) {
      float coefficient_norm = 0.0F;
#pragma unroll
      for (int layer = 0; layer < m; ++layer) {
        coefficient_norm += solution[layer] * solution[layer];
      }
      *cost -= bump * coefficient_norm;
    }
  } else {
    float quadratic = 0.0F;
#pragma unroll
    for (int row = 0; row < m; ++row) {
#pragma unroll
      for (int column = 0; column < m; ++column) {
        const float gram = column <= row
                               ? original[PackedUpperIndex(column, row)]
                               : original[PackedUpperIndex(row, column)];
        quadratic += solution[row] * solution[column] * gram;
      }
    }
    *cost = sample_norm2 - 2.0F * linear + quadratic;
  }
  return true;
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__global__ void SolveAllKernel(DeviceFixedRootAlgebra<M> algebra,
                               std::uint32_t root, const std::uint8_t* codes,
                               const float* root_dots, const float* small_dots,
                               const float* sample_norms2, float* coefficients,
                               float* costs, std::uint32_t count,
                               LeastSquaresCostMode cost_mode) {
  const int m = M == 0 ? algebra.m : M;
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= count) {
    return;
  }
  SolveAndCost<M, Capacity>(
      algebra, root, codes + static_cast<std::size_t>(sample) * (m - 1),
      root_dots[sample],
      small_dots + static_cast<std::size_t>(sample) * algebra.small_count,
      sample_norms2[sample],
      coefficients + static_cast<std::size_t>(sample) * m, costs + sample,
      cost_mode);
}

template <int M>
__global__ void GreedyInitializeSmallKernel(
    std::uint32_t actual_m, std::uint32_t small_count,
    const std::uint32_t* small_offsets, const std::uint32_t* layer_sizes,
    const float* small_inverse_norm, const float* small_gram,
    float* remaining_small_dots, std::uint8_t* codes, std::uint32_t count) {
  const int m = M == 0 ? actual_m : M;
  const std::uint32_t sample = blockIdx.x;
  if (sample >= count) {
    return;
  }
  const int thread = threadIdx.x;
  float* remaining =
      remaining_small_dots + static_cast<std::size_t>(sample) * small_count;
  std::uint8_t* sample_codes =
      codes + static_cast<std::size_t>(sample) * (m - 1);
  if (thread < m - 1) {
    sample_codes[thread] = 0U;
  }

  __shared__ std::uint32_t available;
  __shared__ std::uint32_t selected_flat;
  __shared__ float selected_coefficient;
  __shared__ float scores[256];
  __shared__ std::uint32_t flats[256];
  if (thread == 0) {
    available = ((1U << m) - 1U) & ~1U;
  }
  __syncthreads();

  for (int step = 1; step < m; ++step) {
    float best_score = -CUDART_INF_F;
    std::uint32_t best_flat = 0U;
    const std::uint32_t available_now = available;
    for (std::uint32_t flat = static_cast<std::uint32_t>(thread);
         flat < small_count; flat += blockDim.x) {
      int layer = 1;
#pragma unroll
      for (int candidate_layer = 1; candidate_layer < m; ++candidate_layer) {
        const std::uint32_t begin = small_offsets[candidate_layer];
        const std::uint32_t end = begin + layer_sizes[candidate_layer];
        if (flat >= begin && flat < end) {
          layer = candidate_layer;
          break;
        }
      }
      if ((available_now & (1U << layer)) == 0U) {
        continue;
      }
      const float adjusted = remaining[flat] * small_inverse_norm[flat];
      const float score = fabsf(adjusted);
      if (score > best_score || (score == best_score && flat < best_flat)) {
        best_score = score;
        best_flat = flat;
      }
    }

    scores[thread] = best_score;
    flats[thread] = best_flat;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (thread < offset) {
        const float other_score = scores[thread + offset];
        const std::uint32_t other_flat = flats[thread + offset];
        if (other_score > scores[thread] ||
            (other_score == scores[thread] && other_flat < flats[thread])) {
          scores[thread] = other_score;
          flats[thread] = other_flat;
        }
      }
      __syncthreads();
    }

    if (thread == 0) {
      selected_flat = flats[0];
      int layer = 1;
#pragma unroll
      for (int candidate_layer = 1; candidate_layer < m; ++candidate_layer) {
        const std::uint32_t begin = small_offsets[candidate_layer];
        const std::uint32_t end = begin + layer_sizes[candidate_layer];
        if (selected_flat >= begin && selected_flat < end) {
          layer = candidate_layer;
          break;
        }
      }
      const float inverse_norm = small_inverse_norm[selected_flat];
      const float adjusted = remaining[selected_flat] * inverse_norm;
      selected_coefficient = adjusted * inverse_norm;
      sample_codes[layer - 1] =
          static_cast<std::uint8_t>(selected_flat - small_offsets[layer]);
      available &= ~(1U << layer);
    }
    __syncthreads();

    const std::uint32_t selected = selected_flat;
    const float coefficient = selected_coefficient;
    for (std::uint32_t flat = static_cast<std::uint32_t>(thread);
         flat < small_count; flat += blockDim.x) {
      remaining[flat] -=
          coefficient *
          small_gram[static_cast<std::size_t>(selected) * small_count + flat];
    }
    __syncthreads();
  }
}

__global__ void InitializeActivityKernel(std::uint8_t* active,
                                         std::uint8_t* changed,
                                         std::uint32_t count) {
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample < count) {
    active[sample] = 1U;
    changed[sample] = 0U;
  }
}

__global__ void ClearChangedKernel(std::uint8_t* changed, std::uint32_t count) {
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample < count) {
    changed[sample] = 0U;
  }
}

__global__ void UpdateActivityKernel(std::uint8_t* active,
                                     const std::uint8_t* changed,
                                     std::uint32_t count) {
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample < count && active[sample] != 0U && changed[sample] == 0U) {
    active[sample] = 0U;
  }
}

template <int M, std::uint32_t Capacity = kModelScratchLayers<M>>
__global__
__launch_bounds__(256, (M != 0 && M <= 10) ? 2 : 1) void IcmLayerKernel(
    DeviceFixedRootAlgebra<M> algebra, std::uint32_t root, int layer,
    const float* root_dots, const float* small_dots, const float* sample_norms2,
    const std::uint8_t* active, std::uint8_t* changed, std::uint8_t* codes,
    float* coefficients, float* costs, std::uint32_t count,
    LeastSquaresCostMode cost_mode) {
  const int m = M == 0 ? algebra.m : M;
  const std::uint32_t sample = blockIdx.x;
  if (sample >= count || active[sample] == 0U) {
    return;
  }
  const int thread = threadIdx.x;
  std::uint8_t* sample_codes =
      codes + static_cast<std::size_t>(sample) * (m - 1);
  float* sample_coefficients =
      coefficients + static_cast<std::size_t>(sample) * m;
  const float* dots =
      small_dots + static_cast<std::size_t>(sample) * algebra.small_count;

  __shared__ std::uint8_t shared_codes[Capacity - 1];
  __shared__ float shared_coefficients[Capacity];
  if (thread < m - 1) {
    shared_codes[thread] = sample_codes[thread];
  }
  if (thread < m) {
    shared_coefficients[thread] = sample_coefficients[thread];
  }
  __syncthreads();

  const int old_code = static_cast<int>(shared_codes[layer - 1]);
  float best_score = -CUDART_INF_F;
  int best_code = old_code;
  for (int code = thread; code < 256; code += blockDim.x) {
    const std::uint32_t flat = static_cast<std::uint32_t>(layer - 1) * 256U +
                               static_cast<std::uint32_t>(code);
    float residual_dot = dots[flat] - shared_coefficients[0] *
                                          algebra.fixed_root_small_gram[flat];
#pragma unroll
    for (int other = 1; other < m; ++other) {
      if (other == layer) {
        continue;
      }
      const std::uint32_t other_flat =
          static_cast<std::uint32_t>(other - 1) * 256U +
          static_cast<std::uint32_t>(shared_codes[other - 1]);
      residual_dot -= shared_coefficients[other] *
                      algebra.small_gram[static_cast<std::size_t>(other_flat) *
                                             algebra.small_count +
                                         flat];
    }
    const float score = fabsf(residual_dot * algebra.small_inverse_norm[flat]);
    if (score > best_score || (score == best_score && code < best_code)) {
      best_score = score;
      best_code = code;
    }
  }

  __shared__ float block_scores[256];
  __shared__ int block_codes[256];
  block_scores[thread] = best_score;
  block_codes[thread] = best_code;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (thread < offset) {
      const float other_score = block_scores[thread + offset];
      const int other_code = block_codes[thread + offset];
      if (other_score > block_scores[thread] ||
          (other_score == block_scores[thread] &&
           other_code < block_codes[thread])) {
        block_scores[thread] = other_score;
        block_codes[thread] = other_code;
      }
    }
    __syncthreads();
  }

  const int new_code = block_codes[0];
  if (thread != 0 || new_code == old_code) {
    return;
  }
  float old_coefficients[Capacity];
#pragma unroll
  for (int index = 0; index < m; ++index) {
    old_coefficients[index] = sample_coefficients[index];
  }
  sample_codes[layer - 1] = static_cast<std::uint8_t>(new_code);
  float candidate_cost = 0.0F;
  if (!SolveAndCost<M, Capacity>(algebra, root, sample_codes, root_dots[sample], dots,
                       sample_norms2[sample], sample_coefficients,
                       &candidate_cost, cost_mode) ||
      !(candidate_cost + kAcceptanceEpsilon < costs[sample])) {
    sample_codes[layer - 1] = static_cast<std::uint8_t>(old_code);
#pragma unroll
    for (int index = 0; index < m; ++index) {
      sample_coefficients[index] = old_coefficients[index];
    }
    return;
  }
  costs[sample] = candidate_cost;
  changed[sample] = 1U;
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
  return static_cast<std::uint32_t>(
      __umul64hi(value, static_cast<std::uint64_t>(bound)));
}

template <int M>
__global__ void CopyAndPerturbKernel(
    std::uint32_t actual_m, const std::uint8_t* source,
    std::uint8_t* destination, const std::uint64_t* sample_ids,
    std::uint32_t seed, std::uint32_t round_one_based,
    std::uint32_t perturbations, std::uint32_t count) {
  const int m = M == 0 ? actual_m : M;
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= count) {
    return;
  }
  const std::uint8_t* source_codes =
      source + static_cast<std::size_t>(sample) * (m - 1);
  std::uint8_t* destination_codes =
      destination + static_cast<std::size_t>(sample) * (m - 1);
#pragma unroll
  for (int layer = 0; layer < m - 1; ++layer) {
    destination_codes[layer] = source_codes[layer];
  }

  std::uint64_t random = static_cast<std::uint64_t>(seed);
  random ^= static_cast<std::uint64_t>(round_one_based) * 0x9e3779b97f4a7c15ULL;
  random ^= (sample_ids[sample] + 1ULL) * 0xbf58476d1ce4e5b9ULL;
  for (std::uint32_t index = 0U; index < perturbations; ++index) {
    const std::uint32_t layer =
        1U + UniformBelow(&random, static_cast<std::uint32_t>(m - 1));
    const std::uint8_t old_code = destination_codes[layer - 1U];
    const std::uint32_t raw = UniformBelow(&random, 255U);
    destination_codes[layer - 1U] =
        static_cast<std::uint8_t>(raw >= old_code ? raw + 1U : raw);
  }
}

template <int M>
__global__ void AcceptKernel(std::uint32_t actual_m,
                             const std::uint8_t* candidate_codes,
                             const float* candidate_coefficients,
                             const float* candidate_costs, std::uint8_t* codes,
                             float* coefficients, float* costs,
                             std::uint32_t count) {
  const int m = M == 0 ? actual_m : M;
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= count ||
      !(candidate_costs[sample] + kAcceptanceEpsilon < costs[sample])) {
    return;
  }
  costs[sample] = candidate_costs[sample];
#pragma unroll
  for (int layer = 0; layer < m - 1; ++layer) {
    codes[static_cast<std::size_t>(sample) * (m - 1) + layer] =
        candidate_codes[static_cast<std::size_t>(sample) * (m - 1) + layer];
  }
#pragma unroll
  for (int layer = 0; layer < m; ++layer) {
    coefficients[static_cast<std::size_t>(sample) * m + layer] =
        candidate_coefficients[static_cast<std::size_t>(sample) * m + layer];
  }
}

template <int M>
std::vector<ModelLayers<std::uint8_t, M>> MakeLayerOrders(
    std::uint32_t actual_m, BasicSchedule schedule, std::uint32_t seed) {
  const int m = M == 0 ? actual_m : M;
  std::vector<ModelLayers<std::uint8_t, M>> orders;
  const std::uint32_t ils_rounds = EffectiveBasicIlsRounds<M>(schedule, m);
  orders.reserve(static_cast<std::size_t>(ils_rounds + 1U) *
                 schedule.icm_sweeps());
  std::mt19937 random(seed);
  for (std::uint32_t phase = 0U; phase <= ils_rounds; ++phase) {
    for (std::uint32_t sweep = 0U; sweep < schedule.icm_sweeps(); ++sweep) {
      auto order = MakeModelLayers<std::uint8_t, M>(m);
      std::iota(order.begin(), order.end(), static_cast<std::uint8_t>(0U));
      std::shuffle(order.begin(), order.end(), random);
      orders.push_back(order);
    }
  }
  return orders;
}

}  // namespace

template <int M>
struct CudaFixedRootDeviceModel<M>::State final {
  int device = 0;
  ModelShape shape;
  CudaDeviceBuffer<float> root_atoms;
  CudaDeviceBuffer<float> small_atoms;
  CudaDeviceBuffer<float> root_norm2;
  CudaDeviceBuffer<float> small_inverse_norm;
  CudaDeviceBuffer<float> small_gram;
  CudaDeviceBuffer<std::uint32_t> small_offsets;
  CudaDeviceBuffer<std::uint32_t> layer_sizes;
  std::vector<float> host_root_norm2;

  ~State() { cudaSetDevice(device); }
};

template <int M>
struct CudaFixedRootSupportRefiner<M>::State final {
  CudaContext* context = nullptr;
  int device = 0;
  ModelShape shape;
  std::shared_ptr<const CudaFixedRootDeviceModel<M>> model;
  CudaDeviceBuffer<float> fixed_root_small_gram;
  CudaDeviceBuffer<float> samples;
  CudaDeviceBuffer<std::uint8_t> codes;
  CudaDeviceBuffer<std::uint8_t> candidate_codes;
  CudaDeviceBuffer<std::uint64_t> sample_ids;
  CudaDeviceBuffer<float> root_dots;
  CudaDeviceBuffer<float> small_dots;
  CudaDeviceBuffer<float> remaining_small_dots;
  CudaDeviceBuffer<float> sample_norms2;
  CudaDeviceBuffer<float> coefficients;
  CudaDeviceBuffer<float> candidate_coefficients;
  CudaDeviceBuffer<float> costs;
  CudaDeviceBuffer<float> candidate_costs;
  CudaDeviceBuffer<std::uint8_t> active;
  CudaDeviceBuffer<std::uint8_t> changed;

  ~State() { cudaSetDevice(device); }

  [[nodiscard]] DeviceFixedRootAlgebra<M> Algebra() const noexcept {
    return {shape.m,
            static_cast<std::uint32_t>((shape.m - 1) * 256),
            model->root_norm2(),
            model->small_inverse_norm(),
            model->small_gram(),
            fixed_root_small_gram.data()};
  }

  [[nodiscard]] Status EnsureWorkspace(std::uint32_t count) {
    const std::size_t rows = count;
    const std::size_t code_count = rows * (shape.m - 1);
    const std::size_t coefficient_count = rows * shape.m;
    const std::size_t small_dot_count = rows * (shape.m - 1) * 256U;
    const std::size_t sample_value_count = rows * shape.dimension;
    const std::array<Status, 15> statuses = {
        samples.Ensure(sample_value_count),
        codes.Ensure(code_count),
        candidate_codes.Ensure(code_count),
        sample_ids.Ensure(rows),
        root_dots.Ensure(rows),
        small_dots.Ensure(small_dot_count),
        remaining_small_dots.Ensure(small_dot_count),
        sample_norms2.Ensure(rows),
        coefficients.Ensure(coefficient_count),
        candidate_coefficients.Ensure(coefficient_count),
        costs.Ensure(rows),
        candidate_costs.Ensure(rows),
        active.Ensure(rows),
        changed.Ensure(rows),
        fixed_root_small_gram.Ensure((shape.m - 1) * 256U)};
    for (const Status& status : statuses) {
      if (!status.ok()) {
        return status;
      }
    }
    return Status::Ok();
  }
};

template <int M>
CudaFixedRootSupportRefiner<M>::CudaFixedRootSupportRefiner(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
CudaFixedRootSupportRefiner<M>::~CudaFixedRootSupportRefiner() = default;

template <int M>
CudaFixedRootSupportRefiner<M>::CudaFixedRootSupportRefiner(
    CudaFixedRootSupportRefiner&& other) noexcept
    : state_(std::move(other.state_)) {}

template <int M>
CudaFixedRootSupportRefiner<M>& CudaFixedRootSupportRefiner<M>::operator=(
    CudaFixedRootSupportRefiner&& other) noexcept {
  state_ = std::move(other.state_);
  return *this;
}

template <int M>
CudaFixedRootDeviceModel<M>::CudaFixedRootDeviceModel(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
CudaFixedRootDeviceModel<M>::~CudaFixedRootDeviceModel() = default;

template <int M>
StatusOr<std::shared_ptr<const CudaFixedRootDeviceModel<M>>>
CudaFixedRootDeviceModel<M>::Create(
    const BasicCodebookAlgebra<M>& algebra, CudaContext& context,
    CudaFixedRootModelPreparation* preparation) {
  return CreateImpl({algebra.shape(), algebra.root0(), algebra.small_flat()},
                    algebra.small_gram(), context, preparation);
}

template <int M>
StatusOr<std::shared_ptr<const CudaFixedRootDeviceModel<M>>>
CudaFixedRootDeviceModel<M>::Create(
    const RootCodebooks<M>& codebooks, const F32GemmBackend& gemm,
    CudaContext& context) {
  auto gram = BuildVirtualRootSmallGram(codebooks.runtime_view(), gemm);
  if (!gram.ok()) return gram.status();
  return CreateImpl(codebooks.runtime_view(), gram.value(), context, nullptr);
}

template <int M>
StatusOr<std::shared_ptr<const CudaFixedRootDeviceModel<M>>>
CudaFixedRootDeviceModel<M>::CreateImpl(
    RuntimeRootCodebookView codebooks, ArrayView<const float> small_gram,
    CudaContext& context, CudaFixedRootModelPreparation* preparation) {
  using Clock = std::chrono::steady_clock;
  const auto elapsed = [](Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
  };
  auto state = std::make_unique<State>();
  state->device = context.device();
  state->shape = codebooks.shape;
  if (const cudaError_t result = cudaSetDevice(state->device);
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", result);
  }
  const std::size_t root_count = codebooks.shape.nlist;
  const std::uint32_t m = codebooks.shape.m;
  const std::size_t small_count = static_cast<std::size_t>((m - 1) * 256);
  const ColumnMajorMatrixF32View root_atoms = codebooks.root0;
  const ColumnMajorMatrixF32View small_atoms = codebooks.root_small_flat;
  const std::size_t root_values =
      static_cast<std::size_t>(root_atoms.rows) * root_atoms.columns;
  const std::size_t small_values =
      static_cast<std::size_t>(small_atoms.rows) * small_atoms.columns;
  const auto allocation_begin = Clock::now();
  const std::array<Status, 7> allocations = {
      state->root_atoms.Ensure(root_values),
      state->small_atoms.Ensure(small_values),
      state->root_norm2.Ensure(root_count),
      state->small_inverse_norm.Ensure(small_count),
      state->small_gram.Ensure(small_count * small_count),
      state->small_offsets.Ensure(m),
      state->layer_sizes.Ensure(m)};
  for (const Status& status : allocations) {
    if (!status.ok()) {
      return status;
    }
  }
  if (preparation != nullptr) {
    preparation->allocation_seconds += elapsed(allocation_begin);
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context.native_stream());
  const auto host_begin = Clock::now();
  std::vector<float> fixed_root_norm2(root_count);
  for (std::size_t root = 0U; root < root_count; ++root) {
    const float* atom = root_atoms.column(static_cast<std::uint32_t>(root));
    double norm2 = 0.0;
    for (std::uint32_t axis = 0U; axis < root_atoms.rows; ++axis) {
      const double value = static_cast<double>(atom[axis]);
      norm2 += value * value;
    }
    fixed_root_norm2[root] = static_cast<float>(norm2);
  }
  state->host_root_norm2 = fixed_root_norm2;
  std::vector<float> small_inverse_norm(small_count);
  for (std::size_t flat = 0U; flat < small_count; ++flat) {
    const float* atom = small_atoms.column(static_cast<std::uint32_t>(flat));
    float norm2 = 0.0F;
    for (std::uint32_t axis = 0U; axis < small_atoms.rows; ++axis) {
      const float value = atom[axis];
      norm2 += value * value;
    }
    small_inverse_norm[flat] = norm2 > 0.0F ? 1.0F / std::sqrt(norm2) : 0.0F;
  }
  auto small_offsets = MakeModelLayers<std::uint32_t, M>(m);
  auto layer_sizes = MakeModelLayers<std::uint32_t, M>(m);
  layer_sizes[0] = codebooks.shape.nlist;
  for (int layer = 1; layer < m; ++layer) {
    small_offsets[layer] = static_cast<std::uint32_t>((layer - 1) * 256);
    layer_sizes[layer] = 256U;
  }
  if (preparation != nullptr) {
    preparation->host_preparation_seconds += elapsed(host_begin);
  }
  const auto submission_begin = Clock::now();
  const std::array<cudaError_t, 7> copies = {
      cudaMemcpyAsync(state->root_atoms.data(), root_atoms.values,
                      root_values * sizeof(float), cudaMemcpyHostToDevice,
                      stream),
      cudaMemcpyAsync(state->small_atoms.data(), small_atoms.values,
                      small_values * sizeof(float), cudaMemcpyHostToDevice,
                      stream),
      cudaMemcpyAsync(state->root_norm2.data(), fixed_root_norm2.data(),
                      root_count * sizeof(float), cudaMemcpyHostToDevice,
                      stream),
      cudaMemcpyAsync(state->small_inverse_norm.data(),
                      small_inverse_norm.data(), small_count * sizeof(float),
                      cudaMemcpyHostToDevice, stream),
      cudaMemcpyAsync(state->small_gram.data(), small_gram.data(),
                      small_count * small_count * sizeof(float),
                      cudaMemcpyHostToDevice, stream),
      cudaMemcpyAsync(state->small_offsets.data(), small_offsets.data(),
                      m * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                      stream),
      cudaMemcpyAsync(state->layer_sizes.data(), layer_sizes.data(),
                      m * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                      stream)};
  for (const cudaError_t result : copies) {
    if (result != cudaSuccess) {
      return CudaRuntimeStatus("cudaMemcpyAsync fixed-root model", result);
    }
  }
  if (preparation != nullptr) {
    preparation->upload_submission_seconds += elapsed(submission_begin);
  }
  const auto wait_begin = Clock::now();
  if (const Status status = context.Synchronize(); !status.ok()) {
    return status;
  }
  if (preparation != nullptr) {
    preparation->upload_wait_seconds += elapsed(wait_begin);
  }
  return std::shared_ptr<const CudaFixedRootDeviceModel<M>>(
      new CudaFixedRootDeviceModel<M>(std::move(state)));
}

template <int M>
int CudaFixedRootDeviceModel<M>::device() const noexcept {
  return state_->device;
}

template <int M>
ModelShape CudaFixedRootDeviceModel<M>::shape() const noexcept {
  return state_->shape;
}

template <int M>
std::uint64_t CudaFixedRootDeviceModel<M>::logical_bytes() const noexcept {
  const std::uint64_t root_count = state_->shape.nlist;
  const std::uint64_t small_count = (state_->shape.m - 1U) * 256U;
  const std::uint64_t dimension = state_->shape.dimension;
  return sizeof(float) *
             (dimension * (root_count + small_count) + root_count +
              small_count + small_count * small_count + small_count) +
         sizeof(std::uint32_t) * 2U * state_->shape.m;
}

template <int M>
const float* CudaFixedRootDeviceModel<M>::root_atoms() const noexcept {
  return state_->root_atoms.data();
}

template <int M>
const float* CudaFixedRootDeviceModel<M>::small_atoms() const noexcept {
  return state_->small_atoms.data();
}

template <int M>
const float* CudaFixedRootDeviceModel<M>::root_norm2() const noexcept {
  return state_->root_norm2.data();
}

template <int M>
const float* CudaFixedRootDeviceModel<M>::small_inverse_norm() const noexcept {
  return state_->small_inverse_norm.data();
}

template <int M>
const float* CudaFixedRootDeviceModel<M>::small_gram() const noexcept {
  return state_->small_gram.data();
}

template <int M>
const std::uint32_t* CudaFixedRootDeviceModel<M>::small_offsets()
    const noexcept {
  return state_->small_offsets.data();
}

template <int M>
const std::uint32_t* CudaFixedRootDeviceModel<M>::layer_sizes() const noexcept {
  return state_->layer_sizes.data();
}

template <int M>
float CudaFixedRootDeviceModel<M>::host_root_norm2(
    std::uint32_t root) const noexcept {
  return state_->host_root_norm2[root];
}

template <int M>
StatusOr<CudaFixedRootSupportRefiner<M>> CudaFixedRootSupportRefiner<M>::Create(
    const BasicCodebookAlgebra<M>& algebra, CudaContext& context,
    CudaFixedRootModelPreparation* preparation) {
  auto model =
      CudaFixedRootDeviceModel<M>::Create(algebra, context, preparation);
  if (!model.ok()) {
    return model.status();
  }
  return Create(std::move(model).value(), context);
}

template <int M>
StatusOr<CudaFixedRootSupportRefiner<M>> CudaFixedRootSupportRefiner<M>::Create(
    std::shared_ptr<const CudaFixedRootDeviceModel<M>> model,
    CudaContext& context) {
  if (model == nullptr || model->device() != context.device()) {
    return Status::InvalidArgument(
        "fixed-root CUDA model and lane device do not match");
  }
  auto state = std::make_unique<State>();
  state->context = &context;
  state->device = context.device();
  state->shape = model->shape();
  state->model = std::move(model);
  return CudaFixedRootSupportRefiner<M>(std::move(state));
}

template <int M>
StatusOr<BasicEncoding<M>> CudaFixedRootSupportRefiner<M>::Refine(
    ColumnMajorMatrixF32View samples, std::uint32_t fixed_root,
    ArrayView<const std::uint64_t> sample_ids, BasicSchedule schedule,
    std::uint32_t seed) const {
  if constexpr (M == 0) {
    if (state_->shape.m > kLegacyRuntimeScratchLayers)
      return RefineWithCapacity<kWideRuntimeScratchLayers>(samples, fixed_root, sample_ids, schedule, seed);
  }
  return RefineWithCapacity<kModelScratchLayers<M>>(samples, fixed_root, sample_ids, schedule, seed);
}

template <int M>
template <std::uint32_t Capacity>
StatusOr<BasicEncoding<M>> CudaFixedRootSupportRefiner<M>::RefineWithCapacity(
    ColumnMajorMatrixF32View samples, std::uint32_t fixed_root,
    ArrayView<const std::uint64_t> sample_ids, BasicSchedule schedule,
    std::uint32_t seed) const {
  const std::uint32_t count = samples.columns;
  const std::size_t small_count = (state_->shape.m - 1U) * 256U;
  if (state_ == nullptr || samples.values == nullptr || count == 0U ||
      samples.rows != state_->shape.dimension ||
      fixed_root >= state_->shape.nlist || sample_ids.size() != count) {
    return Status::InvalidArgument(
        "fixed-root support input does not match the model");
  }
  if (const cudaError_t result = cudaSetDevice(state_->device);
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", result);
  }
  if (const Status status = state_->EnsureWorkspace(count); !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  const cublasHandle_t blas =
      static_cast<cublasHandle_t>(state_->context->native_blas());
  const std::size_t code_count =
      static_cast<std::size_t>(count) * (state_->shape.m - 1);
  const std::size_t sample_values =
      static_cast<std::size_t>(samples.rows) * count;
  const std::vector<float> sample_norms2 =
      ComputeVirtualRootSampleNorms2(samples);
  const std::array<cudaError_t, 3> copies = {
      cudaMemcpyAsync(state_->samples.data(), samples.values,
                      sample_values * sizeof(float), cudaMemcpyHostToDevice,
                      stream),
      cudaMemcpyAsync(state_->sample_norms2.data(), sample_norms2.data(),
                      static_cast<std::size_t>(count) * sizeof(float),
                      cudaMemcpyHostToDevice, stream),
      cudaMemcpyAsync(state_->sample_ids.data(), sample_ids.data(),
                      static_cast<std::size_t>(count) * sizeof(std::uint64_t),
                      cudaMemcpyHostToDevice, stream)};
  for (const cudaError_t result : copies) {
    if (result != cudaSuccess) {
      return CudaRuntimeStatus("cudaMemcpyAsync fixed-root inputs", result);
    }
  }

  const int dimension = static_cast<int>(samples.rows);
  const int small_count_i = static_cast<int>(small_count);
  const int count_i = static_cast<int>(count);
  const float one = 1.0F;
  const float zero = 0.0F;
  cublasStatus_t blas_result = cublasSgemm(
      blas, CUBLAS_OP_T, CUBLAS_OP_N, small_count_i, count_i, dimension, &one,
      state_->model->small_atoms(), dimension, state_->samples.data(),
      dimension, &zero, state_->small_dots.data(), small_count_i);
  if (blas_result != CUBLAS_STATUS_SUCCESS) {
    return CudaBlasStatus("cublasSgemm fixed-root small projection",
                          blas_result);
  }
  const cudaError_t residual_copy = cudaMemcpyAsync(
      state_->remaining_small_dots.data(), state_->small_dots.data(),
      static_cast<std::size_t>(count) * small_count * sizeof(float),
      cudaMemcpyDeviceToDevice, stream);
  if (residual_copy != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpyAsync fixed-root residual dots",
                             residual_copy);
  }

  const float* root_atom = state_->model->root_atoms() +
                           static_cast<std::size_t>(fixed_root) * samples.rows;
  blas_result =
      cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, 1, count_i, dimension, &one,
                  root_atom, dimension, state_->samples.data(), dimension,
                  &zero, state_->root_dots.data(), 1);
  if (blas_result != CUBLAS_STATUS_SUCCESS) {
    return CudaBlasStatus("cublasSgemm fixed-root root projection",
                          blas_result);
  }
  blas_result = cublasSgemm(
      blas, CUBLAS_OP_T, CUBLAS_OP_N, small_count_i, 1, dimension, &one,
      state_->model->small_atoms(), dimension, root_atom, dimension, &zero,
      state_->fixed_root_small_gram.data(), small_count_i);
  if (blas_result != CUBLAS_STATUS_SUCCESS) {
    return CudaBlasStatus("cublasSgemm fixed-root root-small Gram",
                          blas_result);
  }
  const float root_norm2 = state_->model->host_root_norm2(fixed_root);
  if (root_norm2 > 0.0F) {
    const float subtract_inverse_norm = -1.0F / root_norm2;
    blas_result = cublasSgemm(
        blas, CUBLAS_OP_N, CUBLAS_OP_N, small_count_i, count_i, 1,
        &subtract_inverse_norm, state_->fixed_root_small_gram.data(),
        small_count_i, state_->root_dots.data(), 1, &one,
        state_->remaining_small_dots.data(), small_count_i);
    if (blas_result != CUBLAS_STATUS_SUCCESS) {
      return CudaBlasStatus("cublasSgemm fixed-root residual update",
                            blas_result);
    }
  }

  GreedyInitializeSmallKernel<M><<<count, 256U, 0U, stream>>>(
      state_->shape.m, static_cast<std::uint32_t>(small_count),
      state_->model->small_offsets(), state_->model->layer_sizes(),
      state_->model->small_inverse_norm(), state_->model->small_gram(),
      state_->remaining_small_dots.data(), state_->codes.data(), count);
  const DeviceFixedRootAlgebra<M> algebra = state_->Algebra();
  constexpr std::uint32_t kThreads = 128U;
  const std::uint32_t grid = (count + kThreads - 1U) / kThreads;
  const auto launch_solve_all = [&](const std::uint8_t* codes,
                                    float* coefficients, float* costs) {
    SolveAllKernel<M, Capacity><<<grid, kThreads, 0U, stream>>>(
        algebra, fixed_root, codes, state_->root_dots.data(),
        state_->small_dots.data(), state_->sample_norms2.data(), coefficients,
        costs, count, schedule.least_squares_cost_mode());
  };
  launch_solve_all(state_->codes.data(), state_->coefficients.data(),
                   state_->costs.data());
  if (cudaError_t result = cudaPeekAtLastError(); result != cudaSuccess) {
    return CudaRuntimeStatus("fixed-root initial solve", result);
  }

  const auto orders = MakeLayerOrders<M>(state_->shape.m, schedule, seed);
  std::size_t order_index = 0U;
  auto run_icm = [&](std::uint8_t* codes, float* coefficients,
                     float* costs) -> Status {
    InitializeActivityKernel<<<grid, kThreads, 0U, stream>>>(
        state_->active.data(), state_->changed.data(), count);
    for (std::uint32_t sweep = 0U; sweep < schedule.icm_sweeps(); ++sweep) {
      ClearChangedKernel<<<grid, kThreads, 0U, stream>>>(state_->changed.data(),
                                                         count);
      for (const std::uint8_t layer : orders[order_index++]) {
        if (layer == 0U) {
          continue;
        }
        IcmLayerKernel<M, Capacity><<<count, 256U, 0U, stream>>>(
            algebra, fixed_root, static_cast<int>(layer),
            state_->root_dots.data(), state_->small_dots.data(),
            state_->sample_norms2.data(), state_->active.data(),
            state_->changed.data(), codes, coefficients, costs, count,
            schedule.least_squares_cost_mode());
      }
      UpdateActivityKernel<<<grid, kThreads, 0U, stream>>>(
          state_->active.data(), state_->changed.data(), count);
    }
    const cudaError_t result = cudaPeekAtLastError();
    return result == cudaSuccess ? Status::Ok()
                                 : CudaRuntimeStatus("fixed-root ICM", result);
  };
  if (const Status status =
          run_icm(state_->codes.data(), state_->coefficients.data(),
                  state_->costs.data());
      !status.ok()) {
    return status;
  }

  for (std::uint32_t round = 0U;
       round < EffectiveBasicIlsRounds<M>(schedule, state_->shape.m); ++round) {
    CopyAndPerturbKernel<M><<<grid, kThreads, 0U, stream>>>(
        state_->shape.m, state_->codes.data(), state_->candidate_codes.data(),
        state_->sample_ids.data(), seed, round + 1U,
        EffectiveBasicPerturbations<M>(schedule, state_->shape.m), count);
    launch_solve_all(state_->candidate_codes.data(),
                     state_->candidate_coefficients.data(),
                     state_->candidate_costs.data());
    if (const Status status = run_icm(state_->candidate_codes.data(),
                                      state_->candidate_coefficients.data(),
                                      state_->candidate_costs.data());
        !status.ok()) {
      return status;
    }
    AcceptKernel<M><<<grid, kThreads, 0U, stream>>>(
        state_->shape.m, state_->candidate_codes.data(),
        state_->candidate_coefficients.data(), state_->candidate_costs.data(),
        state_->codes.data(), state_->coefficients.data(), state_->costs.data(),
        count);
  }
  if (cudaError_t result = cudaPeekAtLastError(); result != cudaSuccess) {
    return CudaRuntimeStatus("fixed-root ILS", result);
  }

  BasicEncoding<M> output =
      detail::BasicEncodingAccess::Create<M>(count, state_->shape.m);
  std::fill(detail::BasicEncodingAccess::Roots(&output),
            detail::BasicEncodingAccess::Roots(&output) + count, fixed_root);
  const std::array<cudaError_t, 3> downloads = {
      cudaMemcpyAsync(detail::BasicEncodingAccess::Codes(&output),
                      state_->codes.data(), code_count, cudaMemcpyDeviceToHost,
                      stream),
      cudaMemcpyAsync(
          detail::BasicEncodingAccess::Coefficients(&output),
          state_->coefficients.data(),
          static_cast<std::size_t>(count) * state_->shape.m * sizeof(float),
          cudaMemcpyDeviceToHost, stream),
      cudaMemcpyAsync(detail::BasicEncodingAccess::Errors(&output),
                      state_->costs.data(),
                      static_cast<std::size_t>(count) * sizeof(float),
                      cudaMemcpyDeviceToHost, stream)};
  for (const cudaError_t result : downloads) {
    if (result != cudaSuccess) {
      return CudaRuntimeStatus("cudaMemcpyAsync fixed-root outputs", result);
    }
  }
  if (const Status status = state_->context->Synchronize(); !status.ok()) {
    return status;
  }
  return std::move(output);
}

template <int M>
Status CudaFixedRootSupportRefiner<M>::ReserveWorkspace(
    std::uint32_t maximum_sample_count) {
  if (state_ == nullptr || maximum_sample_count == 0U) {
    return Status::InvalidArgument(
        "fixed-root workspace reservation requires a nonzero bound");
  }
  if (const cudaError_t result = cudaSetDevice(state_->device);
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", result);
  }
  return state_->EnsureWorkspace(maximum_sample_count);
}

template <int M>
std::uint64_t CudaFixedRootSupportRefiner<M>::model_resident_bytes()
    const noexcept {
  if (state_ == nullptr) {
    return 0U;
  }
  return state_->model->logical_bytes();
}

template <int M>
std::uint64_t CudaFixedRootSupportRefiner<M>::workspace_resident_bytes()
    const noexcept {
  if (state_ == nullptr) {
    return 0U;
  }
  return sizeof(float) *
             (state_->samples.capacity() + state_->root_dots.capacity() +
              state_->small_dots.capacity() +
              state_->remaining_small_dots.capacity() +
              state_->sample_norms2.capacity() +
              state_->coefficients.capacity() +
              state_->candidate_coefficients.capacity() +
              state_->costs.capacity() + state_->candidate_costs.capacity() +
              state_->fixed_root_small_gram.capacity()) +
         sizeof(std::uint8_t) *
             (state_->codes.capacity() + state_->candidate_codes.capacity() +
              state_->active.capacity() + state_->changed.capacity()) +
         sizeof(std::uint64_t) * state_->sample_ids.capacity();
}

template class CudaFixedRootSupportRefiner<0>;
template class CudaFixedRootSupportRefiner<5>;
template class CudaFixedRootSupportRefiner<12>;
template class CudaFixedRootSupportRefiner<6>;
template class CudaFixedRootSupportRefiner<10>;
template class CudaFixedRootDeviceModel<0>;
template class CudaFixedRootDeviceModel<5>;
template class CudaFixedRootDeviceModel<12>;
template class CudaFixedRootDeviceModel<6>;
template class CudaFixedRootDeviceModel<10>;

}  // namespace stlqpp
