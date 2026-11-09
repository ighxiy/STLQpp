#include "stlqpp/runtime/cuda/gemm_f32.h"

#include "stlqpp/runtime/cuda/status.h"

#include <cublas_v2.h>

namespace stlqpp {

Status CudaGemmF32(CudaContext& context,
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
                   int leading_output) {
  const cublasOperation_t op_a = operation_a == CudaGemmOperation::kTranspose
      ? CUBLAS_OP_T
      : CUBLAS_OP_N;
  const cublasOperation_t op_b = operation_b == CudaGemmOperation::kTranspose
      ? CUBLAS_OP_T
      : CUBLAS_OP_N;
  cublasHandle_t handle =
      static_cast<cublasHandle_t>(context.native_blas());
  if (context.allow_tf32()) {
    return CudaBlasStatus(
        "cublasGemmEx TF32",
        cublasGemmEx(
            handle, op_a, op_b, rows, columns, inner, &alpha,
            a, CUDA_R_32F, leading_a, b, CUDA_R_32F, leading_b,
            &beta, output, CUDA_R_32F, leading_output,
            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  return CudaBlasStatus(
      "cublasSgemm",
      cublasSgemm(
          handle, op_a, op_b, rows, columns, inner, &alpha,
          a, leading_a, b, leading_b, &beta, output, leading_output));
}

}  // namespace stlqpp
