#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/runtime/cuda/device_matrix.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace stlqpp {

template <typename Scalar>
class CudaBasicRotator final {
 public:
  [[nodiscard]] static StatusOr<CudaBasicRotator<Scalar>> Create(
      ColumnMajorMatrixF32View rotation,
      CudaContext& context);

  ~CudaBasicRotator();
  CudaBasicRotator(CudaBasicRotator&& other) noexcept;
  CudaBasicRotator& operator=(CudaBasicRotator&& other) noexcept;
  CudaBasicRotator(const CudaBasicRotator&) = delete;
  CudaBasicRotator& operator=(const CudaBasicRotator&) = delete;

  [[nodiscard]] std::uint32_t dimension() const noexcept;
  [[nodiscard]] StatusOr<CudaMatrixF32View> Rotate(
      ColumnMajorMatrixView<Scalar> samples);
  [[nodiscard]] StatusOr<std::vector<float>> CopyToHost(
      CudaMatrixF32View rotated);

 private:
  struct State;
  explicit CudaBasicRotator(std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

extern template class CudaBasicRotator<std::uint8_t>;
extern template class CudaBasicRotator<float>;

}  // namespace stlqpp
