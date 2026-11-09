#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/basic/large_root_algebra.h"
#include "stlqpp/training/basic/basic_projection.h"
#include "stlqpp/training/linkage/root_only/root_only_encoding.h"

#include <cstdint>

namespace stlqpp {

template <int M>
class CpuRootOnlyLinkageWorkerBackend;

// Semantic reference for registered init-linkage encoding:
// exact large-root scan, one native fixed-root global-greedy path, exact joint
// FP32 LS, then caller-ordered absolute-score ICM sweeps including layer 0.
// This algorithm never runs ILS and never inherits edge RSD.
template <int M>
class CpuRootOnlyLinkageEncoder final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  CpuRootOnlyLinkageEncoder(const BasicCodebookAlgebra<M>& algebra,
                            const F32GemmBackend& gemm,
                            const ParallelForExecutor& parallel,
                            std::uint32_t icm_rounds = 1U)
      : algebra_(algebra), gemm_(gemm), parallel_(parallel),
        icm_rounds_(icm_rounds) {}

  [[nodiscard]] StatusOr<RootOnlyLinkageEncoding<M>> Encode(
      ColumnMajorMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_orders) const;

 private:
  friend class CpuRootOnlyLinkageWorkerBackend<M>;
  [[nodiscard]] StatusOr<RootOnlyLinkageEncoding<M>> EncodeValidatedSharedOrder(
      ColumnMajorMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_order) const;
  [[nodiscard]] StatusOr<RootOnlyLinkageEncoding<M>> EncodeProjected(
      const BasicProjection<M>& projection,
      ArrayView<const std::uint8_t> icm_layer_orders,
      std::uint32_t order_stride) const;
  struct EncodeContext;
  template <std::uint32_t Capacity>
  static void EncodeSample(void* opaque,
                           std::uint32_t worker,
                           std::uint32_t sample) noexcept;

  const BasicCodebookAlgebra<M>& algebra_;
  const F32GemmBackend& gemm_;
  const ParallelForExecutor& parallel_;
  std::uint32_t icm_rounds_ = 1U;
};

extern template class CpuRootOnlyLinkageEncoder<5>;
extern template class CpuRootOnlyLinkageEncoder<10>;

}  // namespace stlqpp
