#pragma once

#include <cstdint>
#include <memory>

#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/training/basic/basic_round_runtime.h"
#include "stlqpp/training/least_squares_cost_mode.h"

namespace stlqpp {

template <int M, typename Scalar>
class CpuRsdBasicRoundRuntimeFactory final
    : public BasicRoundRuntimeFactory<M, Scalar> {
 public:
  CpuRsdBasicRoundRuntimeFactory(const F32GemmBackend& gemm,
                                 const ParallelForExecutor& parallel,
                                 std::uint32_t maximum_rank) noexcept
      : gemm_(&gemm), parallel_(&parallel), maximum_rank_(maximum_rank) {}

  [[nodiscard]] StatusOr<std::unique_ptr<BasicRoundRuntime<Scalar>>>
  Create(const RootCodebooks<M>& codebooks,
         ColumnMajorMatrixF32View rotation) override;

 private:
  const F32GemmBackend* gemm_ = nullptr;
  const ParallelForExecutor* parallel_ = nullptr;
  std::uint32_t maximum_rank_ = 2U;
};

extern template class CpuRsdBasicRoundRuntimeFactory<5, std::uint8_t>;
extern template class CpuRsdBasicRoundRuntimeFactory<10, std::uint8_t>;
extern template class CpuRsdBasicRoundRuntimeFactory<5, float>;
extern template class CpuRsdBasicRoundRuntimeFactory<10, float>;

template <int M, typename Scalar>
class CudaRsdBasicRoundRuntimeFactory final
    : public BasicRoundRuntimeFactory<M, Scalar> {
 public:
  explicit CudaRsdBasicRoundRuntimeFactory(
      CudaContext& cuda, std::uint32_t maximum_rank = 2U,
      LeastSquaresCostMode cost_mode =
          LeastSquaresCostMode::kNormalEquation) noexcept
      : cuda_(&cuda), maximum_rank_(maximum_rank), cost_mode_(cost_mode) {}

  [[nodiscard]] StatusOr<std::unique_ptr<BasicRoundRuntime<Scalar>>>
  Create(const RootCodebooks<M>& codebooks,
         ColumnMajorMatrixF32View rotation) override;

 private:
  CudaContext* cuda_ = nullptr;
  std::uint32_t maximum_rank_ = 2U;
  LeastSquaresCostMode cost_mode_ = LeastSquaresCostMode::kNormalEquation;
};

extern template class CudaRsdBasicRoundRuntimeFactory<5, std::uint8_t>;
extern template class CudaRsdBasicRoundRuntimeFactory<10, std::uint8_t>;
extern template class CudaRsdBasicRoundRuntimeFactory<5, float>;
extern template class CudaRsdBasicRoundRuntimeFactory<10, float>;

}  // namespace stlqpp
