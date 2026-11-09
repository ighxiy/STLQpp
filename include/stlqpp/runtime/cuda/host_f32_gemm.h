#pragma once

#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/runtime/cuda/device_buffer.h"

namespace stlqpp {

// Host-facing GEMM adapter backed by the registered CUDA math mode. It is
// used for round-level model algebra that must share the same TF32 semantics
// as the CUDA basic encoder while remaining resident on the host afterward.
class CudaHostF32Gemm final : public F32GemmBackend {
 public:
  explicit CudaHostF32Gemm(CudaContext& context) noexcept
      : context_(&context) {}

  [[nodiscard]] Status MultiplyLeft(
      const LeftGemmF32& operation) const override;

  [[nodiscard]] Status MultiplyTransposedLeft(
      const TransposedLeftGemmF32& operation) const override;

 private:
  CudaContext* context_ = nullptr;
  mutable CudaDeviceBuffer<float> left_;
  mutable CudaDeviceBuffer<float> right_;
  mutable CudaDeviceBuffer<float> output_;
};

}  // namespace stlqpp
