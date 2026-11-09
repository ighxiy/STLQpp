#include "stlqpp/training/initialization/cuda_hierarchical_spherical_scorer.h"

#include "stlqpp/runtime/cuda/device_buffer.h"
#include "stlqpp/runtime/cuda/gemm_f32.h"
#include "stlqpp/runtime/cuda/status.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

namespace stlqpp {
namespace {

constexpr std::uint32_t kCoarseThreads = 256U;
constexpr std::uint32_t kFineThreads = 128U;
constexpr std::uint32_t kFineWarps = 4U;

__global__ void SelectCoarseKernel(const float* scores,
                                   std::uint32_t leading_scores,
                                   std::uint32_t coarse_count,
                                   std::uint32_t columns,
                                   std::uint32_t* selected) {
  const std::uint32_t column = blockIdx.x;
  if (column >= columns) {
    return;
  }
  const float* values = scores +
      static_cast<std::size_t>(column) * leading_scores;
  float best = -FLT_MAX;
  std::uint32_t best_id = 0U;
  for (std::uint32_t coarse = threadIdx.x; coarse < coarse_count;
       coarse += blockDim.x) {
    const float value = values[coarse];
    if (value > best || (value == best && coarse < best_id)) {
      best = value;
      best_id = coarse;
    }
  }
  constexpr unsigned int kFullWarp = 0xffffffffU;
  for (std::uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    const float candidate = __shfl_down_sync(kFullWarp, best, offset);
    const std::uint32_t candidate_id =
        __shfl_down_sync(kFullWarp, best_id, offset);
    if (candidate > best ||
        (candidate == best && candidate_id < best_id)) {
      best = candidate;
      best_id = candidate_id;
    }
  }
  __shared__ float warp_best[8];
  __shared__ std::uint32_t warp_ids[8];
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5U;
  if (lane == 0U) {
    warp_best[warp] = best;
    warp_ids[warp] = best_id;
  }
  __syncthreads();
  if (warp == 0U) {
    const std::uint32_t warp_count = blockDim.x >> 5U;
    best = lane < warp_count ? warp_best[lane] : -FLT_MAX;
    best_id = lane < warp_count ? warp_ids[lane] : 0U;
    for (std::uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
      const float candidate = __shfl_down_sync(kFullWarp, best, offset);
      const std::uint32_t candidate_id =
          __shfl_down_sync(kFullWarp, best_id, offset);
      if (candidate > best ||
          (candidate == best && candidate_id < best_id)) {
        best = candidate;
        best_id = candidate_id;
      }
    }
    if (lane == 0U) {
      selected[column] = best_id;
    }
  }
}

__device__ inline void UpdateTop2(float value,
                                  std::uint32_t id,
                                  float* first_value,
                                  std::uint32_t* first_id,
                                  float* second_value,
                                  std::uint32_t* second_id) {
  if (value > *first_value ||
      (value == *first_value && id < *first_id)) {
    *second_value = *first_value;
    *second_id = *first_id;
    *first_value = value;
    *first_id = id;
  } else if (id != *first_id &&
             (value > *second_value ||
              (value == *second_value && id < *second_id))) {
    *second_value = value;
    *second_id = id;
  }
}

__global__ void SelectTop2CoarseKernel(const float* scores,
                                       std::uint32_t leading_scores,
                                       std::uint32_t coarse_count,
                                       std::uint32_t columns,
                                       std::uint32_t* selected) {
  const std::uint32_t column = blockIdx.x;
  if (column >= columns) {
    return;
  }
  const float* values = scores +
      static_cast<std::size_t>(column) * leading_scores;
  float first_value = -FLT_MAX;
  float second_value = -FLT_MAX;
  std::uint32_t first_id = 0U;
  std::uint32_t second_id = 0U;
  for (std::uint32_t coarse = threadIdx.x; coarse < coarse_count;
       coarse += blockDim.x) {
    UpdateTop2(values[coarse], coarse, &first_value, &first_id,
               &second_value, &second_id);
  }

  extern __shared__ unsigned char shared[];
  float* first_values = reinterpret_cast<float*>(shared);
  float* second_values = first_values + blockDim.x;
  auto* first_ids = reinterpret_cast<std::uint32_t*>(
      second_values + blockDim.x);
  std::uint32_t* second_ids = first_ids + blockDim.x;
  first_values[threadIdx.x] = first_value;
  second_values[threadIdx.x] = second_value;
  first_ids[threadIdx.x] = first_id;
  second_ids[threadIdx.x] = second_id;
  __syncthreads();

  for (std::uint32_t stride = blockDim.x / 2U; stride != 0U;
       stride >>= 1U) {
    if (threadIdx.x < stride) {
      first_value = first_values[threadIdx.x];
      second_value = second_values[threadIdx.x];
      first_id = first_ids[threadIdx.x];
      second_id = second_ids[threadIdx.x];
      UpdateTop2(first_values[threadIdx.x + stride],
                 first_ids[threadIdx.x + stride], &first_value, &first_id,
                 &second_value, &second_id);
      UpdateTop2(second_values[threadIdx.x + stride],
                 second_ids[threadIdx.x + stride], &first_value, &first_id,
                 &second_value, &second_id);
      first_values[threadIdx.x] = first_value;
      second_values[threadIdx.x] = second_value;
      first_ids[threadIdx.x] = first_id;
      second_ids[threadIdx.x] = second_id;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    selected[2U * column] = first_ids[0U];
    selected[2U * column + 1U] = second_ids[0U];
  }
}

template <std::uint32_t TopCoarse>
__global__ void ScoreFineGroupKernel(
    const float* residuals,
    std::uint32_t dimension,
    std::uint32_t columns,
    const float* fine,
    std::uint32_t center_count,
    const std::uint32_t* coarse_ids,
    std::uint32_t fine_capacity,
    std::uint32_t* output_codes,
    float* output_dots) {
  const std::uint32_t column = blockIdx.x;
  if (column >= columns) {
    return;
  }
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5U;
  const float* residual = residuals +
      static_cast<std::size_t>(column) * dimension;
  __shared__ float warp_dots[kFineWarps];
  __shared__ std::uint32_t warp_codes[kFineWarps];
  float best = -FLT_MAX;
  std::uint32_t best_code = 0U;

  for (std::uint32_t selected = 0U; selected < TopCoarse; ++selected) {
    const std::uint32_t coarse =
        coarse_ids[TopCoarse * column + selected];
    const std::uint32_t first = coarse * fine_capacity;
    const std::uint32_t group_size = first < center_count
        ? min(fine_capacity, center_count - first)
        : 0U;
    for (std::uint32_t base = 0U; base < group_size;
         base += kFineWarps) {
      const std::uint32_t local = base + warp;
      float dot = -FLT_MAX;
      std::uint32_t code = 0U;
      if (local < group_size) {
        code = first + local;
        const float* center = fine +
            static_cast<std::size_t>(code) * dimension;
        float sum = 0.0F;
        for (std::uint32_t axis = lane; axis < dimension; axis += 32U) {
          sum += residual[axis] * center[axis];
        }
        constexpr unsigned int kFullWarp = 0xffffffffU;
        for (std::uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
          sum += __shfl_down_sync(kFullWarp, sum, offset);
        }
        dot = __shfl_sync(kFullWarp, sum, 0);
      }
      if (lane == 0U) {
        warp_dots[warp] = dot;
        warp_codes[warp] = code;
      }
      __syncthreads();
      if (threadIdx.x == 0U) {
        for (std::uint32_t candidate = 0U; candidate < kFineWarps;
             ++candidate) {
          const float value = warp_dots[candidate];
          const std::uint32_t code_id = warp_codes[candidate];
          if (value > best || (value == best && code_id < best_code)) {
            best = value;
            best_code = code_id;
          }
        }
      }
      __syncthreads();
    }
  }
  if (threadIdx.x == 0U) {
    output_codes[column] = best_code;
    output_dots[column] = best;
  }
}

}  // namespace

struct CudaHierarchicalSphericalScorer::State final {
  CudaContext* context = nullptr;
  const ParallelForExecutor* parallel = nullptr;
  int device = 0;
  std::uint32_t dimension = 0U;
  std::uint32_t top_coarse = 0U;
  std::size_t maximum_score_elements = 0U;
  SphericalHierarchicalLayout layout;
  bool centers_ready = false;
  CudaDeviceBuffer<float> fine;
  CudaDeviceBuffer<float> coarse;
  CudaDeviceBuffer<float> scores;
  CudaDeviceBuffer<std::uint32_t> selected_coarse;
  CudaDeviceBuffer<std::uint32_t> codes;
  CudaDeviceBuffer<float> best_dots;

  ~State() { cudaSetDevice(device); }
};

CudaHierarchicalSphericalScorer::CudaHierarchicalSphericalScorer(
    std::unique_ptr<State> state) noexcept : state_(std::move(state)) {}

CudaHierarchicalSphericalScorer::~CudaHierarchicalSphericalScorer() = default;
CudaHierarchicalSphericalScorer::CudaHierarchicalSphericalScorer(
    CudaHierarchicalSphericalScorer&&) noexcept = default;
CudaHierarchicalSphericalScorer&
CudaHierarchicalSphericalScorer::operator=(
    CudaHierarchicalSphericalScorer&&) noexcept = default;

StatusOr<CudaHierarchicalSphericalScorer>
CudaHierarchicalSphericalScorer::Create(
    CudaContext& context,
    const ParallelForExecutor& parallel,
    std::uint32_t dimension,
    std::uint32_t center_count,
    std::uint32_t top_coarse,
    std::size_t maximum_score_bytes,
    SphericalHierarchicalSplit split) {
  const SphericalHierarchicalLayout layout =
      MakeSphericalHierarchicalLayout(center_count, split);
  if (context.device() < 0 || dimension == 0U || top_coarse == 0U ||
      top_coarse > 2U ||
      layout.coarse_count == 0U ||
      maximum_score_bytes / sizeof(float) < layout.coarse_count) {
    return Status::InvalidArgument(
        "CUDA hierarchical spherical scorer options are invalid");
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(context.device())); !status.ok()) {
    return status;
  }
  auto state = std::make_unique<State>();
  state->context = &context;
  state->parallel = &parallel;
  state->device = context.device();
  state->dimension = dimension;
  state->top_coarse = top_coarse;
  state->maximum_score_elements = maximum_score_bytes / sizeof(float);
  state->layout = layout;
  return CudaHierarchicalSphericalScorer(std::move(state));
}

Status CudaHierarchicalSphericalScorer::SetFineCenters(
    ColumnMajorMatrixF32View fine) {
  if (state_ == nullptr || fine.values == nullptr ||
      fine.rows != state_->dimension ||
      fine.columns != state_->layout.center_count) {
    return Status::InvalidArgument(
        "CUDA hierarchical fine center table is invalid");
  }
  auto coarse = BuildSphericalCoarseCenters(
      fine, state_->layout, *state_->parallel);
  if (!coarse.ok()) {
    return coarse.status();
  }
  const std::size_t fine_values =
      static_cast<std::size_t>(fine.rows) * fine.columns;
  if (const Status status = state_->fine.Ensure(fine_values); !status.ok()) {
    return status;
  }
  if (const Status status = state_->coarse.Ensure(coarse.value().size());
      !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync hierarchical fine centers H2D",
          cudaMemcpyAsync(state_->fine.data(), fine.values,
                          fine_values * sizeof(float),
                          cudaMemcpyHostToDevice, stream)); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync hierarchical coarse centers H2D",
          cudaMemcpyAsync(state_->coarse.data(), coarse.value().data(),
                          coarse.value().size() * sizeof(float),
                          cudaMemcpyHostToDevice, stream)); !status.ok()) {
    return status;
  }
  state_->centers_ready = true;
  return Status::Ok();
}

StatusOr<CudaHierarchicalSphericalScoreView>
CudaHierarchicalSphericalScorer::Score(CudaMatrixF32View residuals) {
  if (state_ == nullptr || !state_->centers_ready ||
      residuals.values == nullptr || residuals.rows != state_->dimension ||
      residuals.columns == 0U ||
      residuals.leading_dimension != residuals.rows) {
    return Status::FailedPrecondition(
        "CUDA hierarchical spherical scoring state is invalid");
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(state_->device)); !status.ok()) {
    return status;
  }
  if (const Status status = state_->selected_coarse.Ensure(
          static_cast<std::size_t>(residuals.columns) * state_->top_coarse);
      !status.ok()) {
    return status;
  }
  if (const Status status = state_->codes.Ensure(residuals.columns);
      !status.ok()) {
    return status;
  }
  if (const Status status = state_->best_dots.Ensure(residuals.columns);
      !status.ok()) {
    return status;
  }
  const std::uint32_t maximum_columns = static_cast<std::uint32_t>(
      std::min<std::size_t>(
          std::numeric_limits<std::uint32_t>::max(),
          state_->maximum_score_elements / state_->layout.coarse_count));
  if (const Status status = state_->scores.Ensure(
          static_cast<std::size_t>(state_->layout.coarse_count) *
          std::min(maximum_columns, residuals.columns)); !status.ok()) {
    return status;
  }
  const float one = 1.0F;
  const float zero = 0.0F;
  for (std::uint32_t first = 0U; first < residuals.columns;
       first += maximum_columns) {
    const std::uint32_t columns =
        std::min(maximum_columns, residuals.columns - first);
    if (const Status status = CudaGemmF32(
            *state_->context, CudaGemmOperation::kTranspose,
            CudaGemmOperation::kNone,
            static_cast<int>(state_->layout.coarse_count),
            static_cast<int>(columns), static_cast<int>(state_->dimension),
            one, state_->coarse.data(), static_cast<int>(state_->dimension),
            residuals.values +
                static_cast<std::size_t>(first) * state_->dimension,
            static_cast<int>(state_->dimension), zero,
            state_->scores.data(),
            static_cast<int>(state_->layout.coarse_count)); !status.ok()) {
      return status;
    }
    const cudaStream_t stream =
        static_cast<cudaStream_t>(state_->context->native_stream());
    if (state_->top_coarse == 1U) {
      SelectCoarseKernel<<<columns, kCoarseThreads, 0U, stream>>>(
          state_->scores.data(), state_->layout.coarse_count,
          state_->layout.non_empty_coarse_count(), columns,
          state_->selected_coarse.data() + first);
    } else {
      constexpr std::size_t kTop2SharedBytes =
          kCoarseThreads * (2U * sizeof(float) +
                            2U * sizeof(std::uint32_t));
      SelectTop2CoarseKernel<<<
          columns, kCoarseThreads, kTop2SharedBytes, stream>>>(
          state_->scores.data(), state_->layout.coarse_count,
          state_->layout.non_empty_coarse_count(), columns,
          state_->selected_coarse.data() + 2U * first);
    }
    if (const Status status = CudaRuntimeStatus(
            "SelectCoarseKernel", cudaPeekAtLastError()); !status.ok()) {
      return status;
    }
    if (state_->top_coarse == 1U) {
      ScoreFineGroupKernel<1U><<<columns, kFineThreads, 0U, stream>>>(
          residuals.values +
              static_cast<std::size_t>(first) * state_->dimension,
          state_->dimension, columns, state_->fine.data(),
          state_->layout.center_count,
          state_->selected_coarse.data() + first,
          state_->layout.fine_capacity, state_->codes.data() + first,
          state_->best_dots.data() + first);
    } else {
      ScoreFineGroupKernel<2U><<<columns, kFineThreads, 0U, stream>>>(
          residuals.values +
              static_cast<std::size_t>(first) * state_->dimension,
          state_->dimension, columns, state_->fine.data(),
          state_->layout.center_count,
          state_->selected_coarse.data() + 2U * first,
          state_->layout.fine_capacity, state_->codes.data() + first,
          state_->best_dots.data() + first);
    }
    if (const Status status = CudaRuntimeStatus(
            "ScoreFineGroupKernel", cudaPeekAtLastError()); !status.ok()) {
      return status;
    }
  }
  return CudaHierarchicalSphericalScoreView{
      .codes = state_->codes.data(),
      .best_dots = state_->best_dots.data(),
      .count = residuals.columns,
  };
}

const SphericalHierarchicalLayout&
CudaHierarchicalSphericalScorer::layout() const noexcept {
  return state_->layout;
}

}  // namespace stlqpp
