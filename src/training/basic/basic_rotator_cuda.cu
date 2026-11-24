#include "stlqpp/training/basic/basic_rotator_cuda.h"

#include "stlqpp/runtime/cuda/device_buffer.h"
#include "stlqpp/runtime/cuda/gemm_f32.h"
#include "stlqpp/runtime/cuda/status.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace stlqpp {
namespace {

__global__ void ConvertU8ToF32Kernel(const std::uint8_t* input,
                                     std::size_t count,
                                     float* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = static_cast<float>(input[index]);
  }
}

}  // namespace

template <typename Scalar>
struct CudaBasicRotator<Scalar>::State final {
  CudaContext* context = nullptr;
  int device = 0;
  std::uint32_t dimension = 0U;
  CudaDeviceBuffer<float> rotation;
  CudaDeviceBuffer<std::uint8_t> raw_u8;
  CudaDeviceBuffer<float> converted;
  CudaDeviceBuffer<float> rotated;

  ~State() { cudaSetDevice(device); }
};

template <typename Scalar>
CudaBasicRotator<Scalar>::CudaBasicRotator(
    std::unique_ptr<State> state) noexcept : state_(std::move(state)) {}

template <typename Scalar>
CudaBasicRotator<Scalar>::~CudaBasicRotator() = default;

template <typename Scalar>
CudaBasicRotator<Scalar>::CudaBasicRotator(
    CudaBasicRotator&& other) noexcept = default;

template <typename Scalar>
CudaBasicRotator<Scalar>& CudaBasicRotator<Scalar>::operator=(
    CudaBasicRotator&& other) noexcept = default;

template <typename Scalar>
StatusOr<CudaBasicRotator<Scalar>> CudaBasicRotator<Scalar>::Create(
    ColumnMajorMatrixF32View rotation,
    CudaContext& context) {
  static_assert(std::is_same_v<Scalar, std::uint8_t> ||
                std::is_same_v<Scalar, float>);
  if (context.device() < 0 || rotation.values == nullptr ||
      rotation.rows == 0U || rotation.rows != rotation.columns ||
      rotation.rows >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    return Status::InvalidArgument(
        "CUDA basic rotation requires an initialized context and square matrix");
  }
  const std::size_t value_count =
      static_cast<std::size_t>(rotation.rows) * rotation.columns;
  for (std::size_t index = 0U; index < value_count; ++index) {
    if (!std::isfinite(rotation.values[index])) {
      return Status::InvalidArgument(
          "CUDA basic rotation matrix contains a non-finite value");
    }
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(context.device()));
      !status.ok()) {
    return status;
  }
  auto state = std::make_unique<State>();
  state->context = &context;
  state->device = context.device();
  state->dimension = rotation.rows;
  if (const Status status = state->rotation.Ensure(value_count);
      !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context.native_stream());
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync basic rotation H2D",
          cudaMemcpyAsync(state->rotation.data(), rotation.values,
                          value_count * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
      !status.ok()) {
    return status;
  }
  if (const Status status = context.Synchronize(); !status.ok()) {
    return status;
  }
  return CudaBasicRotator<Scalar>(std::move(state));
}

template <typename Scalar>
StatusOr<CudaBasicRotator<Scalar>> CudaBasicRotator<Scalar>::SourceSpace(
    std::uint32_t dimension, CudaContext& context) {
  auto state = std::make_unique<State>();
  state->context = &context;
  state->device = context.device();
  state->dimension = dimension;
  return CudaBasicRotator<Scalar>(std::move(state));
}

template <typename Scalar>
std::uint32_t CudaBasicRotator<Scalar>::dimension() const noexcept {
  return state_ == nullptr ? 0U : state_->dimension;
}

template <typename Scalar>
StatusOr<CudaMatrixF32View> CudaBasicRotator<Scalar>::Rotate(
    ColumnMajorMatrixView<Scalar> samples) {
  if (state_ == nullptr || samples.values == nullptr ||
      samples.rows != state_->dimension || samples.columns == 0U ||
      samples.columns >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    return Status::InvalidArgument(
        "CUDA basic input must be a non-empty contiguous matrix with model dimension");
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(state_->device));
      !status.ok()) {
    return status;
  }
  const std::size_t value_count =
      static_cast<std::size_t>(samples.rows) * samples.columns;
  if (const Status status = state_->converted.Ensure(value_count);
      !status.ok()) {
    return status;
  }

  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  if constexpr (std::is_same_v<Scalar, float>) {
    if (const Status status = CudaRuntimeStatus(
            "cudaMemcpyAsync F32 basic tile H2D",
            cudaMemcpyAsync(state_->converted.data(), samples.values,
                            value_count * sizeof(float),
                            cudaMemcpyHostToDevice, stream));
        !status.ok()) {
      return status;
    }
  } else {
    if (const Status status = state_->raw_u8.Ensure(value_count);
        !status.ok()) {
      return status;
    }
    if (const Status status = CudaRuntimeStatus(
            "cudaMemcpyAsync U8 basic tile H2D",
            cudaMemcpyAsync(state_->raw_u8.data(), samples.values,
                            value_count * sizeof(std::uint8_t),
                            cudaMemcpyHostToDevice, stream));
        !status.ok()) {
      return status;
    }
    constexpr std::uint32_t kThreads = 256U;
    const std::size_t block_count =
        (value_count + kThreads - 1U) / kThreads;
    ConvertU8ToF32Kernel<<<static_cast<unsigned int>(block_count), kThreads,
                           0U, stream>>>(
        state_->raw_u8.data(), value_count, state_->converted.data());
    if (const Status status = CudaRuntimeStatus(
            "ConvertU8ToF32Kernel", cudaPeekAtLastError());
        !status.ok()) {
      return status;
    }
  }

  if (!state_->rotation.data())
    return CudaMatrixF32View{state_->converted.data(), samples.rows,
                             samples.columns, samples.rows};
  if (const Status status = state_->rotated.Ensure(value_count);
      !status.ok()) {
    return status;
  }
  const int dimension = static_cast<int>(state_->dimension);
  const int columns = static_cast<int>(samples.columns);
  if (const Status status = CudaGemmF32(
          *state_->context, CudaGemmOperation::kNone,
          CudaGemmOperation::kNone, dimension, columns, dimension, 1.0F,
          state_->rotation.data(), dimension, state_->converted.data(),
          dimension, 0.0F, state_->rotated.data(), dimension);
      !status.ok()) {
    return status;
  }
  return CudaMatrixF32View{
      state_->rotated.data(), state_->dimension, samples.columns,
      state_->dimension};
}

template <typename Scalar>
StatusOr<std::vector<float>> CudaBasicRotator<Scalar>::CopyToHost(
    CudaMatrixF32View rotated) {
  if (state_ == nullptr || rotated.values != (state_->rotation.data()
                                  ? state_->rotated.data() : state_->converted.data()) ||
      rotated.rows != state_->dimension || rotated.columns == 0U ||
      rotated.leading_dimension != state_->dimension) {
    return Status::InvalidArgument(
        "CUDA retained tile is not owned by this rotator");
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(state_->device));
      !status.ok()) {
    return status;
  }
  const std::size_t value_count =
      static_cast<std::size_t>(rotated.rows) * rotated.columns;
  std::vector<float> host(value_count);
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync retained rotation D2H",
          cudaMemcpyAsync(host.data(), rotated.values,
                          value_count * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
      !status.ok()) {
    return status;
  }
  if (const Status status = state_->context->Synchronize(); !status.ok()) {
    return status;
  }
  return host;
}

template class CudaBasicRotator<std::uint8_t>;
template class CudaBasicRotator<float>;

}  // namespace stlqpp
