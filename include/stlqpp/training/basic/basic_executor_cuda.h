#pragma once

#include <cstdint>
#include <memory>

#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/training/basic/basic_encoder_cuda.h"
#include "stlqpp/training/basic/basic_executor.h"
#include "stlqpp/training/basic/basic_rotator_cuda.h"
#include "stlqpp/training/basic/large_root_algebra.h"

namespace stlqpp {

template <int M, typename Scalar>
class CudaBasicBlockExecutor final : public BasicBlockExecutor<Scalar> {
 public:
  [[nodiscard]] static StatusOr<CudaBasicBlockExecutor<M, Scalar>> Create(
      ColumnMajorMatrixF32View rotation, const BasicCodebookAlgebra<M>& algebra,
      CudaContext& context);

  ~CudaBasicBlockExecutor();
  CudaBasicBlockExecutor(CudaBasicBlockExecutor&&) noexcept;
  CudaBasicBlockExecutor& operator=(CudaBasicBlockExecutor&&) noexcept;
  CudaBasicBlockExecutor(const CudaBasicBlockExecutor&) = delete;
  CudaBasicBlockExecutor& operator=(const CudaBasicBlockExecutor&) = delete;

  [[nodiscard]] StatusOr<BasicEncodingBatch> Encode(
      ColumnMajorMatrixView<Scalar> samples, BasicSchedule schedule,
      std::uint32_t seed, std::uint64_t global_sample_offset) override;

  [[nodiscard]] StatusOr<RetainedModelSpaceBasicEncoding>
  EncodeRetainingModelSpace(ColumnMajorMatrixView<Scalar> samples,
                            BasicSchedule schedule, std::uint32_t seed,
                            std::uint64_t global_sample_offset) override;

  [[nodiscard]] StatusOr<RetainedModelSpaceBasicEncoding>
  EncodeFixedRootsRetainingModelSpace(
      ColumnMajorMatrixView<Scalar> samples,
      ArrayView<const std::uint32_t> fixed_roots, BasicSchedule schedule,
      std::uint32_t seed, std::uint64_t global_sample_offset) override;

 private:
  struct State;
  explicit CudaBasicBlockExecutor(std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

extern template class CudaBasicBlockExecutor<5, std::uint8_t>;
extern template class CudaBasicBlockExecutor<10, std::uint8_t>;
extern template class CudaBasicBlockExecutor<5, float>;
extern template class CudaBasicBlockExecutor<10, float>;

template <int M>
using CudaU8BasicBlockExecutor = CudaBasicBlockExecutor<M, std::uint8_t>;
template <int M>
using CudaF32BasicBlockExecutor = CudaBasicBlockExecutor<M, float>;

}  // namespace stlqpp
