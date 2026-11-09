#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/runtime/cuda/context.h"

namespace stlqpp {

enum class CudaGemmOperation {
  kNone,
  kTranspose,
};

[[nodiscard]] Status CudaGemmF32(
    CudaContext& context,
    CudaGemmOperation operation_a,
    CudaGemmOperation operation_b,
    int rows,
    int columns,
    int inner,
    float alpha,
    const float* a,
    int leading_a,
    const float* b,
    int leading_b,
    float beta,
    float* output,
    int leading_output);

}  // namespace stlqpp
