#include "stlqpp/application/compute_backend.h"

#if STLQPP_HAS_CUDA_RUNTIME
#include <cuda_runtime_api.h>
#include <string>
#endif

namespace stlqpp::app {

Status ResolveRuntimeCudaExecutionPolicy(bool& use_cuda,
                                         bool& gpu_norm_enable) {
  if (!use_cuda) {
    gpu_norm_enable = false;
    return Status::Ok();
  }
  bool available = false;
#if STLQPP_HAS_CUDA_RUNTIME
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess && status != cudaErrorNoDevice &&
      status != cudaErrorInsufficientDriver) {
    return Status::FailedPrecondition(
        std::string("cannot query CUDA devices: ") + cudaGetErrorString(status));
  }
  available = status == cudaSuccess && device_count > 0;
#endif
  ResolveCudaExecutionPolicy(available, use_cuda, gpu_norm_enable);
  return Status::Ok();
}

}  // namespace stlqpp::app
