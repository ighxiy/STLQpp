#include "stlqpp/training/initialization/cuda_spherical_assignment.h"

#include "stlqpp/runtime/cuda/device_buffer.h"
#include "stlqpp/runtime/cuda/gemm_f32.h"
#include "stlqpp/runtime/cuda/status.h"
#include "stlqpp/training/initialization/cuda_hierarchical_spherical_scorer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

constexpr std::uint32_t kWarpSize = 32U;
constexpr std::uint32_t kWarpsPerBlock = 8U;
constexpr std::uint32_t kVectorThreads = kWarpSize * kWarpsPerBlock;

template <std::uint32_t WarpsPerBlock, typename Scalar>
__global__ void NormalizeColumnsWarpKernel(const Scalar* input,
                                           std::uint32_t rows,
                                           std::uint32_t columns,
                                           float* output) {
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1U);
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t column = blockIdx.x * WarpsPerBlock + warp;
  if (column >= columns) {
    return;
  }
  const std::size_t offset = static_cast<std::size_t>(column) * rows;
  float local_norm = 0.0F;
  for (std::uint32_t row = lane; row < rows; row += kWarpSize) {
    const float value = static_cast<float>(input[offset + row]);
    output[offset + row] = value;
    local_norm += value * value;
  }
  constexpr unsigned int kFullWarp = 0xffffffffU;
  for (std::uint32_t step = kWarpSize / 2U; step != 0U; step >>= 1U) {
    local_norm += __shfl_down_sync(kFullWarp, local_norm, step);
  }
  const float inverse_norm = rsqrtf(fmaxf(
      __shfl_sync(kFullWarp, local_norm, 0), 1.0e-12F));
  for (std::uint32_t row = lane; row < rows; row += kWarpSize) {
    output[offset + row] *= inverse_norm;
  }
}

__device__ __forceinline__ int LoadCode(const std::uint8_t* bytes,
                                        int offset,
                                        int width,
                                        int column) {
  const std::uint8_t* value = bytes + static_cast<std::size_t>(offset) +
      static_cast<std::size_t>(column) * static_cast<std::size_t>(width);
  if (width == 1) {
    return static_cast<int>(value[0]);
  }
  if (width == 2) {
    const std::uint16_t decoded =
        static_cast<std::uint16_t>(value[0]) |
        (static_cast<std::uint16_t>(value[1]) << 8);
    return static_cast<int>(decoded);
  }
  const std::uint32_t decoded =
      static_cast<std::uint32_t>(value[0]) |
      (static_cast<std::uint32_t>(value[1]) << 8) |
      (static_cast<std::uint32_t>(value[2]) << 16) |
      (static_cast<std::uint32_t>(value[3]) << 24);
  return static_cast<int>(decoded);
}

template <int WarpsPerBlock>
__global__ void ProjectPriorAndNormalizeWarpKernel(
    float* residuals,
    int rows,
    int columns,
    const float* prior_codebooks,
    const int* metadata,
    const std::uint8_t* code_bytes,
    int prior_layer_count,
    float* projected_squared_norms) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int column = static_cast<int>(blockIdx.x) * WarpsPerBlock + warp;
  if (column >= columns) {
    return;
  }
  float* residual = residuals + static_cast<std::size_t>(column) *
      static_cast<std::size_t>(rows);
  const int* codebook_offsets = metadata;
  const int* code_offsets = codebook_offsets + 2 * prior_layer_count;
  const int* code_widths = code_offsets + prior_layer_count;

  constexpr unsigned int kFullWarp = 0xffffffffU;
  constexpr int kMaximumValuesPerLane = 8;
  const int values_per_lane = (rows + 31) / 32;
  if (values_per_lane > kMaximumValuesPerLane) {
    for (int layer = 0; layer < prior_layer_count; ++layer) {
      const int code = LoadCode(
          code_bytes, code_offsets[layer], code_widths[layer], column);
      const float* center = prior_codebooks + codebook_offsets[layer] +
          static_cast<std::size_t>(code) * static_cast<std::size_t>(rows);
      float dot = 0.0F;
      for (int row = lane; row < rows; row += 32) {
        dot += residual[row] * center[row];
      }
      for (int step = 16; step > 0; step >>= 1) {
        dot += __shfl_down_sync(kFullWarp, dot, step);
      }
      dot = __shfl_sync(kFullWarp, dot, 0);
      for (int row = lane; row < rows; row += 32) {
        residual[row] -= dot * center[row];
      }
    }
    float squared_norm = 0.0F;
    for (int row = lane; row < rows; row += 32) {
      squared_norm += residual[row] * residual[row];
    }
    for (int step = 16; step > 0; step >>= 1) {
      squared_norm += __shfl_down_sync(kFullWarp, squared_norm, step);
    }
    squared_norm = __shfl_sync(kFullWarp, squared_norm, 0);
    if (projected_squared_norms != nullptr && lane == 0) {
      projected_squared_norms[column] = squared_norm;
    }
    const float inverse_norm = rsqrtf(fmaxf(squared_norm, 1.0e-12F));
    for (int row = lane; row < rows; row += 32) {
      residual[row] *= inverse_norm;
    }
    return;
  }

  float values[kMaximumValuesPerLane];
#pragma unroll
  for (int local = 0; local < kMaximumValuesPerLane; ++local) {
    values[local] = 0.0F;
  }
  for (int local = 0; local < values_per_lane; ++local) {
    const int row = lane + 32 * local;
    if (row < rows) {
      values[local] = residual[row];
    }
  }
  for (int layer = 0; layer < prior_layer_count; ++layer) {
    const int code = LoadCode(
        code_bytes, code_offsets[layer], code_widths[layer], column);
    const float* center = prior_codebooks + codebook_offsets[layer] +
        static_cast<std::size_t>(code) * static_cast<std::size_t>(rows);
    float center_values[kMaximumValuesPerLane];
#pragma unroll
    for (int local = 0; local < kMaximumValuesPerLane; ++local) {
      center_values[local] = 0.0F;
    }
    float dot = 0.0F;
    for (int local = 0; local < values_per_lane; ++local) {
      const int row = lane + 32 * local;
      const float center_value = row < rows ? center[row] : 0.0F;
      center_values[local] = center_value;
      dot += values[local] * center_value;
    }
    for (int step = 16; step > 0; step >>= 1) {
      dot += __shfl_down_sync(kFullWarp, dot, step);
    }
    dot = __shfl_sync(kFullWarp, dot, 0);
    for (int local = 0; local < values_per_lane; ++local) {
      values[local] -= dot * center_values[local];
    }
  }
  float squared_norm = 0.0F;
  for (int local = 0; local < values_per_lane; ++local) {
    squared_norm += values[local] * values[local];
  }
  for (int step = 16; step > 0; step >>= 1) {
    squared_norm += __shfl_down_sync(kFullWarp, squared_norm, step);
  }
  squared_norm = __shfl_sync(kFullWarp, squared_norm, 0);
  if (projected_squared_norms != nullptr && lane == 0) {
    projected_squared_norms[column] = squared_norm;
  }
  const float inverse_norm = rsqrtf(fmaxf(squared_norm, 1.0e-12F));
  for (int local = 0; local < values_per_lane; ++local) {
    values[local] *= inverse_norm;
  }
  for (int local = 0; local < values_per_lane; ++local) {
    const int row = lane + 32 * local;
    if (row < rows) {
      residual[row] = values[local];
    }
  }
}

__global__ void FillFloatKernel(float* values,
                                std::uint32_t count,
                                float value) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    values[index] = value;
  }
}

__global__ void InitializeBestKernel(std::uint32_t columns,
                                     float* best_dots,
                                     std::uint32_t* best_codes) {
  const std::uint32_t column =
      static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column < columns) {
    best_dots[column] = -FLT_MAX;
    best_codes[column] = 0U;
  }
}

__global__ void MergeScoreTileKernel(const float* scores,
                                     std::uint32_t tile_entries,
                                     std::uint32_t columns,
                                     std::uint32_t first_entry,
                                     float* best_dots,
                                     std::uint32_t* best_codes) {
  const std::uint32_t column =
      static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= columns) {
    return;
  }
  const float* score = scores +
      static_cast<std::size_t>(column) * tile_entries;
  float best = best_dots[column];
  std::uint32_t code = best_codes[column];
  for (std::uint32_t local = 0U; local < tile_entries; ++local) {
    if (score[local] > best) {
      best = score[local];
      code = first_entry + local;
    }
  }
  best_dots[column] = best;
  best_codes[column] = code;
}

Status ValidateBook(ColumnMajorMatrixF32View book,
                    std::uint32_t dimension,
                    std::uint32_t entries) {
  if (book.values == nullptr || book.rows != dimension ||
      book.columns != entries) {
    return Status::InvalidArgument(
        "CUDA spherical assignment codebook shape is invalid");
  }
  const std::size_t count =
      static_cast<std::size_t>(dimension) * entries;
  for (std::size_t index = 0U; index < count; ++index) {
    if (!std::isfinite(book.values[index])) {
      return Status::InvalidArgument(
          "CUDA spherical assignment codebook is non-finite");
    }
  }
  return Status::Ok();
}

}  // namespace

template <typename Scalar>
struct CudaSphericalAssignment<Scalar>::State final {
  CudaContext* context = nullptr;
  int device = 0;
  ModelShape shape;
  const RvqAssignmentStore* assignments = nullptr;
  std::uint32_t prior_layer_count = 0U;
  std::size_t maximum_score_bytes = 0U;
  std::uint32_t center_count = 0U;
  std::uint32_t expected_center_count = 0U;
  const std::uint32_t* last_codes = nullptr;
  const float* last_best_dots = nullptr;
  std::uint32_t last_count = 0U;
  std::vector<int> codebook_offsets;
  std::vector<int> codebook_entries;
  CudaDeviceBuffer<float> prior_codebooks;
  CudaDeviceBuffer<float> centers;
  CudaDeviceBuffer<Scalar> raw;
  CudaDeviceBuffer<float> residuals;
  CudaDeviceBuffer<std::uint8_t> codes;
  CudaDeviceBuffer<int> metadata;
  std::array<CudaDeviceBuffer<std::uint8_t>, 2> staged_codes;
  std::array<CudaDeviceBuffer<int>, 2> staged_metadata;
  std::array<CudaDeviceBuffer<float>, 2> projected_squared_norms;
  std::array<std::uint32_t, 2> staged_counts{0U, 0U};
  CudaDeviceBuffer<float> scores;
  CudaDeviceBuffer<float> best_dots;
  CudaDeviceBuffer<std::uint32_t> best_codes;
  std::unique_ptr<CudaHierarchicalSphericalScorer> hierarchical;

  ~State() { cudaSetDevice(device); }
};

template <typename Scalar>
CudaSphericalAssignment<Scalar>::CudaSphericalAssignment(
    std::unique_ptr<State> state) noexcept : state_(std::move(state)) {}

template <typename Scalar>
CudaSphericalAssignment<Scalar>::~CudaSphericalAssignment() = default;

template <typename Scalar>
CudaSphericalAssignment<Scalar>::CudaSphericalAssignment(
    CudaSphericalAssignment&& other) noexcept = default;

template <typename Scalar>
CudaSphericalAssignment<Scalar>&
CudaSphericalAssignment<Scalar>::operator=(
    CudaSphericalAssignment&& other) noexcept = default;

template <typename Scalar>
StatusOr<CudaSphericalAssignment<Scalar>>
CudaSphericalAssignment<Scalar>::Create(
    CudaContext& context,
    const ModelShape& shape,
    ArrayView<const ColumnMajorMatrixF32View> prior_codebooks,
    std::uint32_t prior_layer_count,
    const RvqAssignmentStore* assignments,
    CudaSphericalAssignmentOptions options) {
  if (const Status status = shape.Validate(); !status.ok()) {
    return status;
  }
  if (context.device() < 0 || prior_codebooks.size() != shape.m ||
      prior_layer_count >= shape.m ||
      options.maximum_score_bytes < sizeof(float) ||
      (prior_layer_count != 0U && assignments == nullptr) ||
      options.hierarchical_top_coarse == 0U ||
      options.hierarchical_top_coarse > 2U ||
      (options.use_registered_hierarchy && options.parallel == nullptr)) {
    return Status::InvalidArgument(
        "CUDA spherical assignment request is invalid");
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(context.device()));
      !status.ok()) {
    return status;
  }

  auto state = std::make_unique<State>();
  state->context = &context;
  state->device = context.device();
  state->shape = shape;
  state->assignments = assignments;
  state->prior_layer_count = prior_layer_count;
  state->maximum_score_bytes = options.maximum_score_bytes;
  state->expected_center_count =
      prior_layer_count == 0U ? shape.nlist : 256U;

  if (options.use_registered_hierarchy) {
    auto hierarchical = CudaHierarchicalSphericalScorer::Create(
        context, *options.parallel, shape.dimension,
        state->expected_center_count, options.hierarchical_top_coarse,
        options.maximum_score_bytes, options.hierarchical_split);
    if (!hierarchical.ok()) {
      return hierarchical.status();
    }
    state->hierarchical =
        std::make_unique<CudaHierarchicalSphericalScorer>(
            std::move(hierarchical).value());
  }

  std::vector<float> concatenated;
  for (std::uint32_t layer = 0U; layer < prior_layer_count; ++layer) {
    const std::uint32_t entries = layer == 0U ? shape.nlist : 256U;
    if (const Status status =
            ValidateBook(prior_codebooks[layer], shape.dimension, entries);
        !status.ok()) {
      return status;
    }
    if (!assignments->complete(layer)) {
      return Status::FailedPrecondition(
          "CUDA spherical assignment needs closed prior codes");
    }
    state->codebook_offsets.push_back(
        static_cast<int>(concatenated.size()));
    state->codebook_entries.push_back(static_cast<int>(entries));
    const std::size_t count =
        static_cast<std::size_t>(shape.dimension) * entries;
    concatenated.insert(concatenated.end(), prior_codebooks[layer].values,
                        prior_codebooks[layer].values + count);
  }
  if (!concatenated.empty()) {
    if (const Status status = state->prior_codebooks.Ensure(
            concatenated.size()); !status.ok()) {
      return status;
    }
    const cudaStream_t stream =
        static_cast<cudaStream_t>(context.native_stream());
    if (const Status status = CudaRuntimeStatus(
            "cudaMemcpyAsync prior RVQ codebooks H2D",
            cudaMemcpyAsync(state->prior_codebooks.data(),
                            concatenated.data(),
                            concatenated.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream));
        !status.ok()) {
      return status;
    }
    if (const Status status = context.Synchronize(); !status.ok()) {
      return status;
    }
  }
  return CudaSphericalAssignment<Scalar>(std::move(state));
}

template <typename Scalar>
Status CudaSphericalAssignment<Scalar>::SetCenters(
    ColumnMajorMatrixF32View centers) {
  if (state_ == nullptr || centers.values == nullptr ||
      centers.rows != state_->shape.dimension || centers.columns == 0U ||
      (state_->hierarchical != nullptr &&
       centers.columns != state_->expected_center_count) ||
      centers.columns >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    return Status::InvalidArgument(
        "CUDA spherical assignment centers are invalid");
  }
  if (const Status status = ValidateBook(
          centers, state_->shape.dimension, centers.columns); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(state_->device)); !status.ok()) {
    return status;
  }
  state_->last_codes = nullptr;
  state_->last_best_dots = nullptr;
  state_->last_count = 0U;
  if (state_->hierarchical != nullptr) {
    const Status status = state_->hierarchical->SetFineCenters(centers);
    if (status.ok()) {
      state_->center_count = centers.columns;
    }
    return status;
  }
  const std::size_t count =
      static_cast<std::size_t>(centers.rows) * centers.columns;
  if (const Status status = state_->centers.Ensure(count); !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync spherical centers H2D",
          cudaMemcpyAsync(state_->centers.data(), centers.values,
                          count * sizeof(float), cudaMemcpyHostToDevice,
                          stream));
      !status.ok()) {
    return status;
  }
  state_->center_count = centers.columns;
  return Status::Ok();
}

template <typename Scalar>
StatusOr<CudaMatrixF32View>
CudaSphericalAssignment<Scalar>::PrepareResidualsDevice(
    const BasicInputBlock<Scalar>& input) {
  if (state_ == nullptr || input.count == 0U ||
      input.dimension != state_->shape.dimension ||
      input.values.size() !=
          static_cast<std::size_t>(input.dimension) * input.count ||
      (state_->assignments != nullptr &&
       (input.first_id > state_->assignments->row_count() ||
        input.count > state_->assignments->row_count() - input.first_id))) {
    return Status::InvalidArgument(
        "CUDA spherical assignment input block is invalid");
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(state_->device)); !status.ok()) {
    return status;
  }
  const std::size_t value_count =
      static_cast<std::size_t>(input.dimension) * input.count;
  if (const Status status = state_->raw.Ensure(value_count); !status.ok()) {
    return status;
  }
  if (const Status status = state_->residuals.Ensure(value_count);
      !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync spherical input H2D",
          cudaMemcpyAsync(state_->raw.data(), input.values.data(),
                          value_count * sizeof(Scalar),
                          cudaMemcpyHostToDevice, stream));
      !status.ok()) {
    return status;
  }
  const std::uint32_t grid =
      (input.count + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  NormalizeColumnsWarpKernel<kWarpsPerBlock, Scalar><<<
      grid, kVectorThreads, 0U, stream>>>(
      state_->raw.data(), input.dimension, input.count,
      state_->residuals.data());
  if (const Status status = CudaRuntimeStatus(
          "NormalizeColumnsWarpKernel", cudaPeekAtLastError());
      !status.ok()) {
    return status;
  }
  return ApplyPriorProjection(input.first_id, input.count);
}

template <typename Scalar>
StatusOr<CudaMatrixF32View>
CudaSphericalAssignment<Scalar>::PrepareNormalizedResidualsDevice(
    CudaMatrixF32View normalized,
    std::uint64_t first_id) {
  if (state_ == nullptr || normalized.values == nullptr ||
      normalized.rows != state_->shape.dimension ||
      normalized.columns == 0U ||
      normalized.leading_dimension != normalized.rows ||
      (state_->assignments != nullptr &&
       (first_id > state_->assignments->row_count() ||
        normalized.columns > state_->assignments->row_count() - first_id))) {
    return Status::InvalidArgument(
        "CUDA cached spherical input block is invalid");
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(state_->device)); !status.ok()) {
    return status;
  }
  const std::size_t value_count =
      static_cast<std::size_t>(normalized.rows) * normalized.columns;
  if (const Status status = state_->residuals.Ensure(value_count);
      !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync cached spherical source D2D",
          cudaMemcpyAsync(state_->residuals.data(), normalized.values,
                          value_count * sizeof(float),
                          cudaMemcpyDeviceToDevice, stream)); !status.ok()) {
    return status;
  }
  return ApplyPriorProjection(first_id, normalized.columns);
}

template <typename Scalar>
StatusOr<CudaMatrixF32View>
CudaSphericalAssignment<Scalar>::ApplyPriorProjection(
    std::uint64_t first_id,
    std::uint32_t count) {
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());

  if (state_->prior_layer_count != 0U) {
    std::vector<int> code_offsets(state_->prior_layer_count);
    std::vector<int> code_widths(state_->prior_layer_count);
    std::vector<RvqAssignmentBlock> blocks(state_->prior_layer_count);
    std::size_t code_bytes = 0U;
    for (std::uint32_t layer = 0U;
         layer < state_->prior_layer_count; ++layer) {
      auto block = state_->assignments->ReadBlock(
          layer, first_id, count);
      if (!block.ok()) {
        return block.status();
      }
      blocks[layer] = block.value();
      code_offsets[layer] = static_cast<int>(code_bytes);
      code_widths[layer] = static_cast<int>(block.value().width);
      code_bytes += static_cast<std::size_t>(count) *
                    code_widths[layer];
    }
    if (const Status status = state_->codes.Ensure(code_bytes);
        !status.ok()) {
      return status;
    }
    for (std::uint32_t layer = 0U;
         layer < state_->prior_layer_count; ++layer) {
      const std::size_t bytes = static_cast<std::size_t>(count) *
          code_widths[layer];
      if (const Status status = CudaRuntimeStatus(
              "cudaMemcpyAsync prior RVQ codes H2D",
              cudaMemcpyAsync(state_->codes.data() + code_offsets[layer],
                              blocks[layer].values, bytes,
                              cudaMemcpyHostToDevice, stream));
          !status.ok()) {
        return status;
      }
    }
    std::vector<int> metadata;
    metadata.reserve(4U * state_->prior_layer_count);
    metadata.insert(metadata.end(), state_->codebook_offsets.begin(),
                    state_->codebook_offsets.end());
    metadata.insert(metadata.end(), state_->codebook_entries.begin(),
                    state_->codebook_entries.end());
    metadata.insert(metadata.end(), code_offsets.begin(), code_offsets.end());
    metadata.insert(metadata.end(), code_widths.begin(), code_widths.end());
    if (const Status status = state_->metadata.Ensure(metadata.size());
        !status.ok()) {
      return status;
    }
    if (const Status status = CudaRuntimeStatus(
            "cudaMemcpyAsync prior RVQ metadata H2D",
            cudaMemcpyAsync(state_->metadata.data(), metadata.data(),
                            metadata.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream));
        !status.ok()) {
      return status;
    }
    const std::uint32_t grid =
        (count + kWarpsPerBlock - 1U) / kWarpsPerBlock;
    ProjectPriorAndNormalizeWarpKernel<kWarpsPerBlock><<<
        grid, kVectorThreads, 0U, stream>>>(
        state_->residuals.data(), state_->shape.dimension, count,
        state_->prior_codebooks.data(), state_->metadata.data(),
        state_->codes.data(), state_->prior_layer_count, nullptr);
    if (const Status status = CudaRuntimeStatus(
            "ProjectPriorAndNormalizeWarpKernel", cudaPeekAtLastError());
        !status.ok()) {
      return status;
    }
  }

  return CudaMatrixF32View{
      state_->residuals.data(), state_->shape.dimension, count,
      state_->shape.dimension};
}

template <typename Scalar>
Status CudaSphericalAssignment<Scalar>::StagePriorCodes(
    std::uint64_t first_id,
    std::uint32_t count,
    std::uint8_t slot,
    void* native_upload_stream) {
  assert(slot < 2U);
  assert(native_upload_stream != nullptr);
  const cudaStream_t upload =
      static_cast<cudaStream_t>(native_upload_stream);
  if (const Status status = state_->projected_squared_norms[slot].Ensure(count);
      !status.ok()) {
    return status;
  }
  if (state_->prior_layer_count == 0U) {
    constexpr std::uint32_t kThreads = 256U;
    const std::uint32_t blocks = (count + kThreads - 1U) / kThreads;
    FillFloatKernel<<<blocks, kThreads, 0U, upload>>>(
        state_->projected_squared_norms[slot].data(), count, 1.0F);
    if (const Status status = CudaRuntimeStatus(
            "FillFloatKernel root projected norm2", cudaPeekAtLastError());
        !status.ok()) {
      return status;
    }
    state_->staged_counts[slot] = count;
    return Status::Ok();
  }
  std::array<int, kMaximumModelLayers> code_offsets{};
  std::array<int, kMaximumModelLayers> code_widths{};
  std::array<RvqAssignmentBlock, kMaximumModelLayers> blocks{};
  std::size_t total_code_bytes = 0U;
  for (std::uint32_t layer = 0U;
       layer < state_->prior_layer_count; ++layer) {
    auto block = state_->assignments->ReadBlock(layer, first_id, count);
    if (!block.ok()) {
      return block.status();
    }
    blocks[layer] = block.value();
    code_offsets[layer] = static_cast<int>(total_code_bytes);
    code_widths[layer] = static_cast<int>(block.value().width);
    total_code_bytes += static_cast<std::size_t>(count) *
                        static_cast<std::size_t>(code_widths[layer]);
  }
  auto& device_codes = state_->staged_codes[slot];
  if (const Status status = device_codes.Ensure(total_code_bytes);
      !status.ok()) {
    return status;
  }
  for (std::uint32_t layer = 0U;
       layer < state_->prior_layer_count; ++layer) {
    const std::size_t bytes = static_cast<std::size_t>(count) *
                              static_cast<std::size_t>(code_widths[layer]);
    if (const Status status = CudaRuntimeStatus(
            "cudaMemcpyAsync staged prior RVQ codes H2D",
            cudaMemcpyAsync(device_codes.data() + code_offsets[layer],
                            blocks[layer].values, bytes,
                            cudaMemcpyHostToDevice, upload)); !status.ok()) {
      return status;
    }
  }
  std::array<int, 4 * kMaximumModelLayers> metadata{};
  for (std::uint32_t layer = 0U;
       layer < state_->prior_layer_count; ++layer) {
    metadata[layer] = state_->codebook_offsets[layer];
    metadata[state_->prior_layer_count + layer] =
        state_->codebook_entries[layer];
    metadata[2U * state_->prior_layer_count + layer] =
        code_offsets[layer];
    metadata[3U * state_->prior_layer_count + layer] =
        code_widths[layer];
  }
  const std::size_t metadata_count = 4U * state_->prior_layer_count;
  auto& device_metadata = state_->staged_metadata[slot];
  if (const Status status = device_metadata.Ensure(metadata_count);
      !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync staged prior RVQ metadata H2D",
          cudaMemcpyAsync(device_metadata.data(), metadata.data(),
                          metadata_count * sizeof(int),
                          cudaMemcpyHostToDevice, upload)); !status.ok()) {
    return status;
  }
  state_->staged_counts[slot] = count;
  return Status::Ok();
}

template <typename Scalar>
StatusOr<CudaSphericalResidualBlock>
CudaSphericalAssignment<Scalar>::ProjectPriorInPlace(
    CudaMatrixF32View normalized,
    std::uint8_t slot) {
  assert(slot < 2U);
  assert(state_->staged_counts[slot] == normalized.columns);
  if (state_->prior_layer_count == 0U) {
    return CudaSphericalResidualBlock{
        .residuals = normalized,
        .squared_norms = state_->projected_squared_norms[slot].data(),
    };
  }
  const cudaStream_t compute =
      static_cast<cudaStream_t>(state_->context->native_stream());
  const std::uint32_t grid =
      (normalized.columns + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  ProjectPriorAndNormalizeWarpKernel<kWarpsPerBlock><<<
      grid, kVectorThreads, 0U, compute>>>(
      const_cast<float*>(normalized.values), normalized.rows,
      normalized.columns, state_->prior_codebooks.data(),
      state_->staged_metadata[slot].data(),
      state_->staged_codes[slot].data(), state_->prior_layer_count,
      state_->projected_squared_norms[slot].data());
  if (const Status status = CudaRuntimeStatus(
          "ProjectPriorAndNormalizeWarpKernel streaming",
          cudaPeekAtLastError()); !status.ok()) {
    return status;
  }
  return CudaSphericalResidualBlock{
      .residuals = normalized,
      .squared_norms = state_->projected_squared_norms[slot].data(),
  };
}

template <typename Scalar>
StatusOr<CudaSphericalDeviceAssignment>
CudaSphericalAssignment<Scalar>::AssignDevice(
    const BasicInputBlock<Scalar>& input) {
  if (state_ == nullptr || state_->center_count == 0U) {
    return Status::FailedPrecondition(
        "CUDA spherical assignment has no current centers");
  }
  auto residuals = PrepareResidualsDevice(input);
  if (!residuals.ok()) {
    return residuals.status();
  }
  return AssignPreparedDevice(residuals.value());
}

template <typename Scalar>
StatusOr<CudaSphericalDeviceAssignment>
CudaSphericalAssignment<Scalar>::AssignNormalizedDevice(
    CudaMatrixF32View normalized,
    std::uint64_t first_id) {
  if (state_ == nullptr || state_->center_count == 0U) {
    return Status::FailedPrecondition(
        "CUDA spherical assignment has no current centers");
  }
  auto residuals = PrepareNormalizedResidualsDevice(normalized, first_id);
  if (!residuals.ok()) {
    return residuals.status();
  }
  return AssignPreparedDevice(residuals.value());
}

template <typename Scalar>
StatusOr<CudaSphericalDeviceAssignment>
CudaSphericalAssignment<Scalar>::AssignPreparedDevice(
    CudaMatrixF32View residuals) {
  if (state_->hierarchical != nullptr) {
    auto scored = state_->hierarchical->Score(residuals);
    if (!scored.ok()) {
      return scored.status();
    }
    state_->last_codes = scored.value().codes;
    state_->last_best_dots = scored.value().best_dots;
    state_->last_count = scored.value().count;
    return CudaSphericalDeviceAssignment{
        .residuals = residuals,
        .codes = scored.value().codes,
        .best_dots = scored.value().best_dots,
        .count = scored.value().count,
    };
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());

  constexpr std::uint32_t kThreads = 256U;
  const std::uint32_t count = residuals.columns;
  if (const Status status = state_->best_dots.Ensure(count); !status.ok()) {
    return status;
  }
  if (const Status status = state_->best_codes.Ensure(count); !status.ok()) {
    return status;
  }
  const std::uint32_t blocks = (count + kThreads - 1U) / kThreads;
  InitializeBestKernel<<<blocks, kThreads, 0U, stream>>>(
      count, state_->best_dots.data(), state_->best_codes.data());
  if (const Status status = CudaRuntimeStatus(
          "InitializeBestKernel", cudaPeekAtLastError()); !status.ok()) {
    return status;
  }

  const std::size_t maximum_score_elements =
      state_->maximum_score_bytes / sizeof(float);
  const std::uint32_t tile_capacity = static_cast<std::uint32_t>(
      std::min<std::size_t>(
          state_->center_count,
          std::max<std::size_t>(1U,
              maximum_score_elements / count)));
  if (const Status status = state_->scores.Ensure(
          static_cast<std::size_t>(tile_capacity) * count);
      !status.ok()) {
    return status;
  }
  const int dimension = static_cast<int>(residuals.rows);
  const int columns = static_cast<int>(count);
  const float one = 1.0F;
  const float zero = 0.0F;
  for (std::uint32_t first = 0U; first < state_->center_count;
       first += tile_capacity) {
    const std::uint32_t tile =
        std::min(tile_capacity, state_->center_count - first);
    if (const Status status = CudaGemmF32(
            *state_->context, CudaGemmOperation::kTranspose,
            CudaGemmOperation::kNone, static_cast<int>(tile), columns,
            dimension, one,
            state_->centers.data() +
                static_cast<std::size_t>(first) * residuals.rows,
            dimension, residuals.values, dimension, zero,
            state_->scores.data(), static_cast<int>(tile)); !status.ok()) {
      return status;
    }
    MergeScoreTileKernel<<<blocks, kThreads, 0U, stream>>>(
        state_->scores.data(), tile, count, first,
        state_->best_dots.data(), state_->best_codes.data());
    if (const Status status = CudaRuntimeStatus(
            "MergeScoreTileKernel", cudaPeekAtLastError()); !status.ok()) {
      return status;
    }
  }

  state_->last_codes = state_->best_codes.data();
  state_->last_best_dots = state_->best_dots.data();
  state_->last_count = count;
  return CudaSphericalDeviceAssignment{
      .residuals = residuals,
      .codes = state_->best_codes.data(),
      .best_dots = state_->best_dots.data(),
      .count = count,
  };
}

template <typename Scalar>
StatusOr<CudaSphericalAssignmentResult>
CudaSphericalAssignment<Scalar>::Assign(
    const BasicInputBlock<Scalar>& input) {
  auto device = AssignDevice(input);
  if (!device.ok()) {
    return device.status();
  }
  CudaSphericalAssignmentResult output;
  auto codes = CopyCodes(device.value());
  if (!codes.ok()) {
    return codes.status();
  }
  output.codes = std::move(codes).value();
  output.best_dots.resize(device.value().count);
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync spherical dots D2H",
          cudaMemcpyAsync(output.best_dots.data(), device.value().best_dots,
                          device.value().count * sizeof(float),
                          cudaMemcpyDeviceToHost, stream)); !status.ok()) {
    return status;
  }
  if (const Status status = state_->context->Synchronize(); !status.ok()) {
    return status;
  }
  return output;
}

template <typename Scalar>
StatusOr<std::vector<std::uint32_t>>
CudaSphericalAssignment<Scalar>::CopyCodes(
    const CudaSphericalDeviceAssignment& assigned) {
  if (state_ == nullptr || assigned.codes != state_->last_codes ||
      assigned.best_dots != state_->last_best_dots ||
      assigned.count == 0U || assigned.count != state_->last_count) {
    return Status::InvalidArgument(
        "CUDA spherical code view is not owned by this engine");
  }
  std::vector<std::uint32_t> output(assigned.count);
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync spherical codes D2H",
          cudaMemcpyAsync(output.data(), assigned.codes,
                          assigned.count * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, stream)); !status.ok()) {
    return status;
  }
  if (const Status status = state_->context->Synchronize(); !status.ok()) {
    return status;
  }
  return output;
}

template class CudaSphericalAssignment<std::uint8_t>;
template class CudaSphericalAssignment<float>;

}  // namespace stlqpp
