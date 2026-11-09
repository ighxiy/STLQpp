#include "stlqpp/training/initialization/cuda_spherical_accumulator.h"

#include "stlqpp/runtime/cuda/device_buffer.h"
#include "stlqpp/runtime/cuda/status.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

constexpr std::uint32_t kThreads = 256U;
constexpr float kCenterFinalizationEpsilon = 1.0e-6F;

__global__ void FillWeightsKernel(float* weights,
                                  std::uint64_t count,
                                  float value) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < count; index += stride) {
    weights[index] = value;
  }
}

__global__ void FillKeysAndValuesKernel(
    const std::uint32_t* assignments,
    std::uint32_t count,
    std::uint32_t* keys,
    std::uint32_t* values) {
  const std::uint32_t index =
      static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t column = index; column < count; column += stride) {
    keys[column] = assignments[column];
    values[column] = column;
  }
}

__global__ void ReduceSegmentsAccumulateKernel(
    const float* residuals,
    std::uint32_t dimension,
    const std::uint32_t* sorted_values,
    const std::uint32_t* unique_keys,
    const std::uint32_t* segment_offsets,
    const std::uint32_t* segment_counts,
    const std::uint32_t* segment_count,
    const float* weights,
    float* sums,
    float* center_weights) {
  const std::uint32_t segment = blockIdx.x;
  if (segment >= segment_count[0]) {
    return;
  }
  const std::uint32_t center = unique_keys[segment];
  const std::uint32_t begin = segment_offsets[segment];
  const std::uint32_t length = segment_counts[segment];
  assert(length != 0U);

  float local_weight = 0.0F;
  for (std::uint32_t index = threadIdx.x; index < length;
       index += blockDim.x) {
    local_weight += weights[sorted_values[begin + index]];
  }
  __shared__ float weight_scratch[kThreads];
  weight_scratch[threadIdx.x] = local_weight;
  __syncthreads();
  for (std::uint32_t stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      weight_scratch[threadIdx.x] +=
          weight_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    center_weights[center] += weight_scratch[0];
  }

  for (std::uint32_t axis = threadIdx.x; axis < dimension;
       axis += blockDim.x) {
    float value = 0.0F;
    for (std::uint32_t index = 0U; index < length; ++index) {
      const std::uint32_t column = sorted_values[begin + index];
      value += weights[column] *
          residuals[static_cast<std::size_t>(column) * dimension + axis];
    }
    sums[static_cast<std::size_t>(center) * dimension + axis] += value;
  }
}

__global__ void CostHistogramKernel(const float* best_dots,
                                    std::uint32_t count,
                                    std::uint32_t* histogram,
                                    std::uint32_t bins) {
  const std::uint32_t index =
      static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t column = index; column < count; column += stride) {
    const float cost = fminf(2.0F, fmaxf(0.0F, 1.0F - best_dots[column]));
    std::uint32_t bin = static_cast<std::uint32_t>(
        cost * (0.5F * static_cast<float>(bins)));
    bin = min(bins - 1U, bin);
    const unsigned int matching = __match_any_sync(0xffffffffU, bin);
    const int leader = __ffs(matching) - 1;
    if ((static_cast<int>(threadIdx.x) & 31) == leader) {
      atomicAdd(histogram + bin,
                static_cast<std::uint32_t>(__popc(matching)));
    }
  }
}

__global__ void UpdateWeightsAndTotalsKernel(
    const float* best_dots,
    std::uint32_t count,
    float threshold,
    float annealing_factor,
    float minimum_weight,
    float* weights,
    double* totals) {
  double weight_sum = 0.0;
  double cost_sum = 0.0;
  const std::uint32_t index =
      static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t column = index; column < count; column += stride) {
    const float cost = 1.0F - best_dots[column];
    float weight = weights[column];
    weight = cost > threshold
        ? fmaxf(minimum_weight, weight * (1.0F - annealing_factor))
        : fminf(1.0F, weight * (1.0F + 0.5F * annealing_factor));
    weights[column] = weight;
    weight_sum += static_cast<double>(weight);
    cost_sum += static_cast<double>(weight) * static_cast<double>(cost);
  }
  using Reduction = cub::BlockReduce<double, kThreads>;
  __shared__ typename Reduction::TempStorage temporary;
  const double block_weight = Reduction(temporary).Sum(weight_sum);
  __syncthreads();
  const double block_cost = Reduction(temporary).Sum(cost_sum);
  if (threadIdx.x == 0U) {
    atomicAdd(totals, block_weight);
    atomicAdd(totals + 1U, block_cost);
  }
}

__global__ void AccumulateTotalsKernel(const float* best_dots,
                                       const float* weights,
                                       std::uint32_t count,
                                       double* totals) {
  double weight_sum = 0.0;
  double cost_sum = 0.0;
  const std::uint32_t index =
      static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t column = index; column < count; column += stride) {
    const float weight = weights[column];
    weight_sum += static_cast<double>(weight);
    cost_sum += static_cast<double>(weight) *
        static_cast<double>(1.0F - best_dots[column]);
  }
  using Reduction = cub::BlockReduce<double, kThreads>;
  __shared__ typename Reduction::TempStorage temporary;
  const double block_weight = Reduction(temporary).Sum(weight_sum);
  __syncthreads();
  const double block_cost = Reduction(temporary).Sum(cost_sum);
  if (threadIdx.x == 0U) {
    atomicAdd(totals, block_weight);
    atomicAdd(totals + 1U, block_cost);
  }
}

__global__ void AccumulateMseProxyKernel(
    const float* best_dots,
    const float* source_squared_norms,
    const float* projected_squared_norms,
    std::uint32_t count,
    double* metric_totals) {
  double mse_sum = 0.0;
  double row_count = 0.0;
  const std::uint32_t index =
      static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t column = index; column < count; column += stride) {
    const double best_dot = static_cast<double>(best_dots[column]);
    const double source_norm =
        static_cast<double>(source_squared_norms[column]);
    const double projected_norm =
        static_cast<double>(projected_squared_norms[column]);
    mse_sum += source_norm * projected_norm *
        (1.0 - best_dot * best_dot);
    row_count += 1.0;
  }
  using Reduction = cub::BlockReduce<double, kThreads>;
  __shared__ typename Reduction::TempStorage temporary;
  const double block_mse = Reduction(temporary).Sum(mse_sum);
  __syncthreads();
  const double block_count = Reduction(temporary).Sum(row_count);
  if (threadIdx.x == 0U) {
    atomicAdd(metric_totals, block_mse);
    atomicAdd(metric_totals + 1U, block_count);
  }
}

struct FinalizeCentersContext final {
  const float* sums = nullptr;
  const float* center_weights = nullptr;
  const float* fallback = nullptr;
  std::uint32_t dimension = 0U;
  float* output = nullptr;
};

void FinalizeOneCenter(void* opaque,
                       std::uint32_t,
                       std::uint32_t center) noexcept {
  auto& context = *static_cast<FinalizeCentersContext*>(opaque);
  const std::size_t offset =
      static_cast<std::size_t>(center) * context.dimension;
  float* output = context.output + offset;
  const float weight = context.center_weights[center];
  if (weight <= kCenterFinalizationEpsilon) {
    std::copy(context.fallback + offset,
              context.fallback + offset + context.dimension, output);
    return;
  }
  const float inverse_weight = 1.0F / weight;
  float squared_norm = 0.0F;
  for (std::uint32_t axis = 0U; axis < context.dimension; ++axis) {
    const float value = context.sums[offset + axis] * inverse_weight;
    output[axis] = value;
    squared_norm += value * value;
  }
  const float inverse_norm =
      1.0F /
      std::sqrt(std::max(squared_norm, kCenterFinalizationEpsilon));
  for (std::uint32_t axis = 0U; axis < context.dimension; ++axis) {
    output[axis] *= inverse_norm;
  }
}

Status CubStatus(const char* operation, cudaError_t result) {
  return CudaRuntimeStatus(operation, result);
}

}  // namespace

struct CudaSphericalAccumulator::State final {
  CudaContext* context = nullptr;
  const ParallelForExecutor* parallel = nullptr;
  int device = 0;
  CudaSphericalAccumulatorOptions options;
  std::uint64_t next_row = 0U;
  bool iteration_open = false;
  CudaDeviceBuffer<float> weights;
  CudaDeviceBuffer<float> sums;
  CudaDeviceBuffer<float> center_weights;
  CudaDeviceBuffer<std::uint32_t> histogram;
  CudaDeviceBuffer<double> totals;
  CudaDeviceBuffer<double> metric_totals;
  CudaDeviceBuffer<std::uint32_t> keys_in;
  CudaDeviceBuffer<std::uint32_t> keys_out;
  CudaDeviceBuffer<std::uint32_t> values_in;
  CudaDeviceBuffer<std::uint32_t> values_out;
  CudaDeviceBuffer<std::uint32_t> unique_keys;
  CudaDeviceBuffer<std::uint32_t> segment_counts;
  CudaDeviceBuffer<std::uint32_t> segment_offsets;
  CudaDeviceBuffer<std::uint32_t> segment_count;
  CudaDeviceBuffer<std::uint8_t> cub_scratch;

  ~State() { cudaSetDevice(device); }
};

CudaSphericalAccumulator::CudaSphericalAccumulator(
    std::unique_ptr<State> state) noexcept : state_(std::move(state)) {}

CudaSphericalAccumulator::~CudaSphericalAccumulator() = default;
CudaSphericalAccumulator::CudaSphericalAccumulator(
    CudaSphericalAccumulator&& other) noexcept = default;
CudaSphericalAccumulator& CudaSphericalAccumulator::operator=(
    CudaSphericalAccumulator&& other) noexcept = default;

StatusOr<CudaSphericalAccumulator> CudaSphericalAccumulator::Create(
    CudaContext& context,
    const ParallelForExecutor& parallel,
    CudaSphericalAccumulatorOptions options) {
  if (context.device() < 0 || options.source_rows == 0U ||
      options.dimension == 0U || options.center_count == 0U ||
      options.quantile_bins == 0U || !std::isfinite(options.initial_weight) ||
      options.initial_weight <= 0.0F || options.initial_weight > 1.0F ||
      options.source_rows > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument(
        "CUDA spherical accumulator options are invalid");
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(context.device()));
      !status.ok()) {
    return status;
  }
  auto state = std::make_unique<State>();
  state->context = &context;
  state->parallel = &parallel;
  state->device = context.device();
  state->options = options;
  const std::size_t center_values =
      static_cast<std::size_t>(options.dimension) * options.center_count;
  if (const Status status = state->weights.Ensure(
          static_cast<std::size_t>(options.source_rows)); !status.ok()) {
    return status;
  }
  if (const Status status = state->sums.Ensure(center_values); !status.ok()) {
    return status;
  }
  if (const Status status = state->center_weights.Ensure(
          options.center_count); !status.ok()) {
    return status;
  }
  if (const Status status = state->histogram.Ensure(options.quantile_bins);
      !status.ok()) {
    return status;
  }
  if (const Status status = state->totals.Ensure(2U); !status.ok()) {
    return status;
  }
  if (options.collect_metrics) {
    if (const Status status = state->metric_totals.Ensure(2U); !status.ok()) {
      return status;
    }
  }
  const std::uint64_t blocks64 = std::min<std::uint64_t>(
      65'535U, (options.source_rows + kThreads - 1U) / kThreads);
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context.native_stream());
  FillWeightsKernel<<<static_cast<unsigned int>(blocks64), kThreads, 0U,
                      stream>>>(
      state->weights.data(), options.source_rows, options.initial_weight);
  if (const Status status = CudaRuntimeStatus(
          "FillWeightsKernel", cudaPeekAtLastError()); !status.ok()) {
    return status;
  }
  if (const Status status = context.Synchronize(); !status.ok()) {
    return status;
  }
  return CudaSphericalAccumulator(std::move(state));
}

Status CudaSphericalAccumulator::BeginIteration() {
  if (state_ == nullptr || state_->iteration_open) {
    return Status::FailedPrecondition(
        "CUDA spherical iteration is already open or uninitialized");
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(state_->device)); !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  const std::size_t center_values =
      static_cast<std::size_t>(state_->options.dimension) *
      state_->options.center_count;
  if (const Status status = CudaRuntimeStatus(
          "cudaMemsetAsync spherical sums",
          cudaMemsetAsync(state_->sums.data(), 0,
                          center_values * sizeof(float), stream));
      !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaMemsetAsync spherical center weights",
          cudaMemsetAsync(state_->center_weights.data(), 0,
                          state_->options.center_count * sizeof(float),
                          stream)); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaMemsetAsync spherical histogram",
          cudaMemsetAsync(state_->histogram.data(), 0,
                          state_->options.quantile_bins *
                              sizeof(std::uint32_t),
                          stream)); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaMemsetAsync spherical totals",
          cudaMemsetAsync(state_->totals.data(), 0, 2U * sizeof(double),
                          stream)); !status.ok()) {
    return status;
  }
  if (state_->options.collect_metrics) {
    if (const Status status = CudaRuntimeStatus(
            "cudaMemsetAsync spherical metric totals",
            cudaMemsetAsync(state_->metric_totals.data(), 0,
                            2U * sizeof(double), stream)); !status.ok()) {
      return status;
    }
  }
  state_->next_row = 0U;
  state_->iteration_open = true;
  return Status::Ok();
}

Status CudaSphericalAccumulator::Accumulate(
    const CudaSphericalDeviceAssignment& assigned,
    std::uint64_t first_id,
    bool update_weights,
    float effective_threshold,
    float annealing_factor,
    float minimum_weight,
    const float* source_squared_norms,
    const float* projected_squared_norms) {
  if (state_ == nullptr || !state_->iteration_open ||
      first_id != state_->next_row || assigned.count == 0U ||
      assigned.codes == nullptr || assigned.best_dots == nullptr ||
      assigned.residuals.values == nullptr ||
      assigned.residuals.rows != state_->options.dimension ||
      assigned.residuals.columns != assigned.count ||
      assigned.residuals.leading_dimension != state_->options.dimension ||
      assigned.count > state_->options.source_rows - first_id ||
      !std::isfinite(effective_threshold) ||
      !std::isfinite(annealing_factor) || annealing_factor < 0.0F ||
      annealing_factor > 1.0F || !std::isfinite(minimum_weight) ||
      minimum_weight <= 0.0F || minimum_weight > 1.0F) {
    return Status::InvalidArgument(
        "CUDA spherical accumulation block is invalid or out of order");
  }
  assert(source_squared_norms != nullptr);
  assert(projected_squared_norms != nullptr);
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  const std::uint32_t blocks = std::min<std::uint32_t>(
      4'096U, (assigned.count + kThreads - 1U) / kThreads);
  if (state_->options.collect_metrics) {
    AccumulateMseProxyKernel<<<blocks, kThreads, 0U, stream>>>(
        assigned.best_dots, source_squared_norms, projected_squared_norms,
        assigned.count, state_->metric_totals.data());
  }
  float* block_weights = state_->weights.data() + first_id;
  if (update_weights) {
    CostHistogramKernel<<<blocks, kThreads, 0U, stream>>>(
        assigned.best_dots, assigned.count, state_->histogram.data(),
        state_->options.quantile_bins);
    UpdateWeightsAndTotalsKernel<<<blocks, kThreads, 0U, stream>>>(
        assigned.best_dots, assigned.count, effective_threshold,
        annealing_factor, minimum_weight, block_weights,
        state_->totals.data());
  } else {
    AccumulateTotalsKernel<<<blocks, kThreads, 0U, stream>>>(
        assigned.best_dots, block_weights, assigned.count,
        state_->totals.data());
  }
  if (const Status status = CudaRuntimeStatus(
          "spherical annealing/statistics kernels", cudaPeekAtLastError());
      !status.ok()) {
    return status;
  }

  if (const Status status = state_->keys_in.Ensure(assigned.count);
      !status.ok()) {
    return status;
  }
  if (const Status status = state_->keys_out.Ensure(assigned.count);
      !status.ok()) {
    return status;
  }
  if (const Status status = state_->values_in.Ensure(assigned.count);
      !status.ok()) {
    return status;
  }
  if (const Status status = state_->values_out.Ensure(assigned.count);
      !status.ok()) {
    return status;
  }
  const std::uint32_t centers = state_->options.center_count;
  if (const Status status = state_->unique_keys.Ensure(centers); !status.ok()) {
    return status;
  }
  if (const Status status = state_->segment_counts.Ensure(centers);
      !status.ok()) {
    return status;
  }
  if (const Status status = state_->segment_offsets.Ensure(centers);
      !status.ok()) {
    return status;
  }
  if (const Status status = state_->segment_count.Ensure(1U); !status.ok()) {
    return status;
  }
  FillKeysAndValuesKernel<<<blocks, kThreads, 0U, stream>>>(
      assigned.codes, assigned.count, state_->keys_in.data(),
      state_->values_in.data());
  if (const Status status = CudaRuntimeStatus(
          "FillKeysAndValuesKernel", cudaPeekAtLastError()); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaMemsetAsync spherical segment counts",
          cudaMemsetAsync(state_->segment_counts.data(), 0,
                          centers * sizeof(std::uint32_t), stream));
      !status.ok()) {
    return status;
  }
  std::size_t sort_bytes = 0U;
  std::size_t run_length_bytes = 0U;
  std::size_t scan_bytes = 0U;
  if (const Status status = CubStatus(
          "query spherical radix sort scratch",
          cub::DeviceRadixSort::SortPairs(
              nullptr, sort_bytes, state_->keys_in.data(),
              state_->keys_out.data(), state_->values_in.data(),
              state_->values_out.data(), assigned.count, 0, 32, stream));
      !status.ok()) {
    return status;
  }
  if (const Status status = CubStatus(
          "query spherical run-length scratch",
          cub::DeviceRunLengthEncode::Encode(
              nullptr, run_length_bytes, state_->keys_out.data(),
              state_->unique_keys.data(), state_->segment_counts.data(),
              state_->segment_count.data(), assigned.count, stream));
      !status.ok()) {
    return status;
  }
  if (const Status status = CubStatus(
          "query spherical scan scratch",
          cub::DeviceScan::ExclusiveSum(
              nullptr, scan_bytes, state_->segment_counts.data(),
              state_->segment_offsets.data(), centers, stream));
      !status.ok()) {
    return status;
  }
  if (const Status status = state_->cub_scratch.Ensure(
          std::max(sort_bytes, std::max(run_length_bytes, scan_bytes)));
      !status.ok()) {
    return status;
  }
  if (const Status status = CubStatus(
          "spherical radix sort",
          cub::DeviceRadixSort::SortPairs(
              state_->cub_scratch.data(), sort_bytes,
              state_->keys_in.data(), state_->keys_out.data(),
              state_->values_in.data(), state_->values_out.data(),
              assigned.count, 0, 32, stream)); !status.ok()) {
    return status;
  }
  if (const Status status = CubStatus(
          "spherical run-length encode",
          cub::DeviceRunLengthEncode::Encode(
              state_->cub_scratch.data(), run_length_bytes,
              state_->keys_out.data(), state_->unique_keys.data(),
              state_->segment_counts.data(), state_->segment_count.data(),
              assigned.count, stream)); !status.ok()) {
    return status;
  }
  if (const Status status = CubStatus(
          "spherical segment offset scan",
          cub::DeviceScan::ExclusiveSum(
              state_->cub_scratch.data(), scan_bytes,
              state_->segment_counts.data(), state_->segment_offsets.data(),
              centers, stream)); !status.ok()) {
    return status;
  }
  ReduceSegmentsAccumulateKernel<<<centers, kThreads, 0U, stream>>>(
      assigned.residuals.values, state_->options.dimension,
      state_->values_out.data(), state_->unique_keys.data(),
      state_->segment_offsets.data(), state_->segment_counts.data(),
      state_->segment_count.data(), block_weights, state_->sums.data(),
      state_->center_weights.data());
  if (const Status status = CudaRuntimeStatus(
          "ReduceSegmentsAccumulateKernel", cudaPeekAtLastError());
      !status.ok()) {
    return status;
  }
  state_->next_row += assigned.count;
  return Status::Ok();
}

StatusOr<CudaSphericalIterationResult>
CudaSphericalAccumulator::FinishIteration(
    ColumnMajorMatrixF32View empty_cluster_fallback,
    float quantile) {
  if (state_ == nullptr || !state_->iteration_open ||
      state_->next_row != state_->options.source_rows ||
      empty_cluster_fallback.values == nullptr ||
      empty_cluster_fallback.rows != state_->options.dimension ||
      empty_cluster_fallback.columns != state_->options.center_count ||
      !std::isfinite(quantile) || quantile < 0.0F || quantile > 1.0F) {
    return Status::FailedPrecondition(
        "CUDA spherical iteration is incomplete or has invalid centers");
  }
  const std::size_t center_values =
      static_cast<std::size_t>(state_->options.dimension) *
      state_->options.center_count;
  std::vector<float> sums(center_values);
  std::vector<float> center_weights(state_->options.center_count);
  std::vector<std::uint32_t> histogram(state_->options.quantile_bins);
  double totals[2] = {0.0, 0.0};
  double metric_totals[2] = {0.0, 0.0};
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync spherical sums D2H",
          cudaMemcpyAsync(sums.data(), state_->sums.data(),
                          sums.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, stream)); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync spherical center weights D2H",
          cudaMemcpyAsync(center_weights.data(), state_->center_weights.data(),
                          center_weights.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, stream)); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync spherical histogram D2H",
          cudaMemcpyAsync(histogram.data(), state_->histogram.data(),
                          histogram.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, stream)); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync spherical totals D2H",
          cudaMemcpyAsync(totals, state_->totals.data(), sizeof(totals),
                          cudaMemcpyDeviceToHost, stream)); !status.ok()) {
    return status;
  }
  if (state_->options.collect_metrics) {
    if (const Status status = CudaRuntimeStatus(
            "cudaMemcpyAsync spherical metric totals D2H",
            cudaMemcpyAsync(metric_totals, state_->metric_totals.data(),
                            sizeof(metric_totals), cudaMemcpyDeviceToHost,
                            stream)); !status.ok()) {
      return status;
    }
  }
  if (const Status status = state_->context->Synchronize(); !status.ok()) {
    return status;
  }

  CudaSphericalIterationResult output;
  output.centers.resize(center_values);
  FinalizeCentersContext finalize{
      .sums = sums.data(),
      .center_weights = center_weights.data(),
      .fallback = empty_cluster_fallback.values,
      .dimension = state_->options.dimension,
      .output = output.centers.data(),
  };
  state_->parallel->Run(
      state_->options.center_count, &finalize, FinalizeOneCenter);

  const std::uint64_t target = static_cast<std::uint64_t>(
      quantile * static_cast<float>(std::max<std::uint64_t>(
                     1U, state_->options.source_rows - 1U)));
  std::uint64_t cumulative = 0U;
  std::uint32_t selected_bin = state_->options.quantile_bins - 1U;
  for (std::uint32_t bin = 0U; bin < state_->options.quantile_bins; ++bin) {
    cumulative += histogram[bin];
    if (cumulative > target) {
      selected_bin = bin;
      break;
    }
  }
  output.quantile_threshold =
      2.0F * (static_cast<float>(selected_bin) + 0.5F) /
      static_cast<float>(state_->options.quantile_bins);
  output.total_weight = totals[0];
  output.mean_cost = totals[0] > 0.0 ? totals[1] / totals[0] : 0.0;
  output.mse_proxy = !state_->options.collect_metrics
      ? -1.0
      : metric_totals[1] > 0.0
          ? metric_totals[0] / metric_totals[1]
          : 0.0;
  state_->iteration_open = false;
  return output;
}

}  // namespace stlqpp
