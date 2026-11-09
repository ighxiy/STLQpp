#include "stlqpp/runtime/cuda/status.h"

#include <string>

namespace stlqpp {

Status CudaRuntimeStatus(const char* operation, cudaError_t result) {
  if (result == cudaSuccess) {
    return Status::Ok();
  }
  const std::string message =
      std::string(operation) + ": " + cudaGetErrorString(result);
  return result == cudaErrorMemoryAllocation
      ? Status::ResourceExhausted(message)
      : Status::Internal(message);
}

Status CudaBlasStatus(const char* operation, cublasStatus_t result) {
  return result == CUBLAS_STATUS_SUCCESS
      ? Status::Ok()
      : Status::Internal(std::string(operation) + ": cuBLAS status " +
                         std::to_string(static_cast<int>(result)));
}

}  // namespace stlqpp
