#include "stlqpp/runtime/cuda/host_f32_gemm.h"

#include "stlqpp/runtime/cuda/gemm_f32.h"
#include "stlqpp/runtime/cuda/status.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace stlqpp {

Status CudaHostF32Gemm::MultiplyLeft(
    const LeftGemmF32& operation) const {
  if (context_ == nullptr || operation.left.values == nullptr ||
      operation.left.rows == 0U || operation.left.columns == 0U ||
      operation.right == nullptr || operation.query_count == 0U ||
      operation.right_leading_dimension < operation.left.columns ||
      operation.output == nullptr ||
      operation.output_leading_dimension < operation.left.rows ||
      operation.left.rows >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      operation.left.columns >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      operation.query_count >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      operation.right_leading_dimension >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      operation.output_leading_dimension >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    return Status::InvalidArgument(
        "CUDA host GEMM received an invalid non-transposed-left operation");
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(context_->device()));
      !status.ok()) {
    return status;
  }
  const std::size_t left_values =
      static_cast<std::size_t>(operation.left.rows) *
      operation.left.columns;
  const std::size_t right_values =
      operation.right_leading_dimension * operation.query_count;
  const std::size_t output_values =
      static_cast<std::size_t>(operation.output_leading_dimension) *
      operation.query_count;
  if (const Status status = left_.Ensure(left_values); !status.ok()) {
    return status;
  }
  if (const Status status = right_.Ensure(right_values); !status.ok()) {
    return status;
  }
  if (const Status status = output_.Ensure(output_values); !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context_->native_stream());
  cudaError_t result = cudaMemcpyAsync(
      left_.data(), operation.left.values, left_values * sizeof(float),
      cudaMemcpyHostToDevice, stream);
  if (result != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpyAsync algebra left H2D", result);
  }
  result = cudaMemcpy2DAsync(
      right_.data(), operation.right_leading_dimension * sizeof(float),
      operation.right, operation.right_leading_dimension * sizeof(float),
      static_cast<std::size_t>(operation.left.columns) * sizeof(float),
      operation.query_count, cudaMemcpyHostToDevice, stream);
  if (result != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpy2DAsync algebra right H2D", result);
  }
  if (const Status status = CudaGemmF32(
          *context_, CudaGemmOperation::kNone,
          CudaGemmOperation::kNone,
          static_cast<int>(operation.left.rows),
          static_cast<int>(operation.query_count),
          static_cast<int>(operation.left.columns), 1.0F, left_.data(),
          static_cast<int>(operation.left.rows), right_.data(),
          static_cast<int>(operation.right_leading_dimension), 0.0F,
          output_.data(),
          static_cast<int>(operation.output_leading_dimension));
      !status.ok()) {
    return status;
  }
  result = cudaMemcpy2DAsync(
      operation.output,
      static_cast<std::size_t>(operation.output_leading_dimension) *
          sizeof(float),
      output_.data(),
      static_cast<std::size_t>(operation.output_leading_dimension) *
          sizeof(float),
      static_cast<std::size_t>(operation.left.rows) * sizeof(float),
      operation.query_count, cudaMemcpyDeviceToHost, stream);
  if (result != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpy2DAsync algebra output D2H", result);
  }
  return context_->Synchronize();
}

Status CudaHostF32Gemm::MultiplyTransposedLeft(
    const TransposedLeftGemmF32& operation) const {
  if (context_ == nullptr || operation.left.values == nullptr ||
      operation.left.rows == 0U || operation.left.columns == 0U ||
      operation.right == nullptr || operation.query_count == 0U ||
      operation.right_leading_dimension < operation.left.rows ||
      operation.output == nullptr ||
      operation.output_leading_dimension < operation.left.columns ||
      operation.left.rows >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      operation.left.columns >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      operation.query_count >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      operation.right_leading_dimension >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      operation.output_leading_dimension >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    return Status::InvalidArgument(
        "CUDA host GEMM received an invalid transposed-left operation");
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(context_->device()));
      !status.ok()) {
    return status;
  }
  const std::size_t left_values =
      static_cast<std::size_t>(operation.left.rows) *
      operation.left.columns;
  const std::size_t right_values =
      operation.right_leading_dimension * operation.query_count;
  const std::size_t output_values =
      static_cast<std::size_t>(operation.output_leading_dimension) *
      operation.query_count;
  if (const Status status = left_.Ensure(left_values); !status.ok()) {
    return status;
  }
  if (const Status status = right_.Ensure(right_values); !status.ok()) {
    return status;
  }
  if (const Status status = output_.Ensure(output_values); !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context_->native_stream());
  cudaError_t result = cudaMemcpyAsync(
      left_.data(), operation.left.values, left_values * sizeof(float),
      cudaMemcpyHostToDevice, stream);
  if (result != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpyAsync algebra left H2D", result);
  }
  result = cudaMemcpy2DAsync(
      right_.data(), operation.right_leading_dimension * sizeof(float),
      operation.right, operation.right_leading_dimension * sizeof(float),
      static_cast<std::size_t>(operation.left.rows) * sizeof(float),
      operation.query_count, cudaMemcpyHostToDevice, stream);
  if (result != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpy2DAsync algebra right H2D", result);
  }
  if (const Status status = CudaGemmF32(
          *context_, CudaGemmOperation::kTranspose,
          CudaGemmOperation::kNone,
          static_cast<int>(operation.left.columns),
          static_cast<int>(operation.query_count),
          static_cast<int>(operation.left.rows), 1.0F, left_.data(),
          static_cast<int>(operation.left.rows), right_.data(),
          static_cast<int>(operation.right_leading_dimension), 0.0F,
          output_.data(),
          static_cast<int>(operation.output_leading_dimension));
      !status.ok()) {
    return status;
  }
  result = cudaMemcpy2DAsync(
      operation.output,
      static_cast<std::size_t>(operation.output_leading_dimension) *
          sizeof(float),
      output_.data(),
      static_cast<std::size_t>(operation.output_leading_dimension) *
          sizeof(float),
      static_cast<std::size_t>(operation.left.columns) * sizeof(float),
      operation.query_count, cudaMemcpyDeviceToHost, stream);
  if (result != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpy2DAsync algebra output D2H", result);
  }
  return context_->Synchronize();
}

}  // namespace stlqpp
