#include "stlqpp/runtime/cuda/context.h"

#include "stlqpp/runtime/cuda/status.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <utility>

namespace stlqpp {

struct CudaContext::State final {
  int device = 0;
  bool allow_tf32 = true;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;

  ~State() {
    cudaSetDevice(device);
    if (blas != nullptr) {
      cublasDestroy(blas);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

CudaContext::CudaContext() = default;
CudaContext::~CudaContext() = default;
CudaContext::CudaContext(CudaContext&& other) noexcept = default;
CudaContext& CudaContext::operator=(CudaContext&& other) noexcept = default;

CudaContext::CudaContext(std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

StatusOr<CudaContext> CudaContext::Create(
    int device, bool allow_tf32, CudaBlasAtomics atomics) {
  int device_count = 0;
  cudaError_t cuda_result = cudaGetDeviceCount(&device_count);
  if (cuda_result != cudaSuccess) {
    return CudaRuntimeStatus("cudaGetDeviceCount", cuda_result);
  }
  if (device < 0 || device >= device_count) {
    return Status::InvalidArgument("CUDA device is outside the visible set");
  }
  cuda_result = cudaSetDevice(device);
  if (cuda_result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", cuda_result);
  }
  auto state = std::make_unique<State>();
  state->device = device;
  state->allow_tf32 = allow_tf32;
  cuda_result = cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking);
  if (cuda_result != cudaSuccess) {
    return CudaRuntimeStatus("cudaStreamCreateWithFlags", cuda_result);
  }
  cublasStatus_t blas_result = cublasCreate(&state->blas);
  if (blas_result != CUBLAS_STATUS_SUCCESS) {
    return CudaBlasStatus("cublasCreate", blas_result);
  }
  blas_result = cublasSetStream(state->blas, state->stream);
  if (blas_result != CUBLAS_STATUS_SUCCESS) {
    return CudaBlasStatus("cublasSetStream", blas_result);
  }
  blas_result = cublasSetMathMode(
      state->blas,
      allow_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
  if (blas_result != CUBLAS_STATUS_SUCCESS) {
    return CudaBlasStatus("cublasSetMathMode", blas_result);
  }
  // Keep the established linkage policy by default. DEV's RVQ context uses
  // allowed atomics with TF32; its owner requests that policy explicitly.
  blas_result = cublasSetAtomicsMode(
      state->blas, atomics == CudaBlasAtomics::kAllowed
          ? CUBLAS_ATOMICS_ALLOWED : CUBLAS_ATOMICS_NOT_ALLOWED);
  if (blas_result != CUBLAS_STATUS_SUCCESS) {
    return CudaBlasStatus("cublasSetAtomicsMode", blas_result);
  }
  return CudaContext(std::move(state));
}

int CudaContext::device() const noexcept {
  return state_ == nullptr ? -1 : state_->device;
}

bool CudaContext::allow_tf32() const noexcept {
  return state_ != nullptr && state_->allow_tf32;
}

void* CudaContext::native_stream() const noexcept {
  return state_ == nullptr ? nullptr : state_->stream;
}

void* CudaContext::native_blas() const noexcept {
  return state_ == nullptr ? nullptr : state_->blas;
}

Status CudaContext::Synchronize() const {
  if (state_ == nullptr) {
    return Status::FailedPrecondition("CUDA context is not initialized");
  }
  const cudaError_t result = cudaStreamSynchronize(state_->stream);
  return CudaRuntimeStatus("cudaStreamSynchronize", result);
}

}  // namespace stlqpp
