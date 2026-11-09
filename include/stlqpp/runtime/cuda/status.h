#pragma once

#include "stlqpp/core/status.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace stlqpp {

[[nodiscard]] Status CudaRuntimeStatus(const char* operation,
                                       cudaError_t result);
[[nodiscard]] Status CudaBlasStatus(const char* operation,
                                    cublasStatus_t result);

}  // namespace stlqpp
