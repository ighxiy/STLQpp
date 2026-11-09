#pragma once

#include <cstdint>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/least_squares_cost_mode.h"
#include "stlqpp/training/linkage/replay/fixed_edge_replay_encoder.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/rsd/linkage_encoding.h"
#include "stlqpp/training/linkage/rsd/linkage_projection.h"

namespace stlqpp {

// Fixed-parent replay endpoint. Parent selection is frozen, but each residual
// still uses the configured RSD candidate competition and ICM sweep count.
template <int M>
class CpuFixedEdgeReplayEncoder final : public FixedEdgeReplayEncoder<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  CpuFixedEdgeReplayEncoder(
      const LinkageCodebookAlgebra<M>& algebra, const F32GemmBackend& gemm,
      const ParallelForExecutor& parallel, std::uint32_t maximum_rank = 2U,
      LeastSquaresCostMode cost_mode = LeastSquaresCostMode::kNormalEquation,
      std::uint32_t icm_rounds = 1U) noexcept
      : algebra_(algebra),
        gemm_(gemm),
        parallel_(parallel),
        maximum_rank_(maximum_rank),
        cost_mode_(cost_mode),
        icm_rounds_(icm_rounds) {}

  [[nodiscard]] StatusOr<LinkageEncoding<M>> Encode(
      ColumnMajorMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_orders) const override;

 private:
  [[nodiscard]] StatusOr<LinkageEncoding<M>> EncodeResidualsValidated(
      ColumnMajorMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_orders) const override;

  [[nodiscard]] StatusOr<LinkageEncoding<M>> EncodeImpl(
      ColumnMajorMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_orders,
      bool orders_validated) const;

  const LinkageCodebookAlgebra<M>& algebra_;
  const F32GemmBackend& gemm_;
  const ParallelForExecutor& parallel_;
  std::uint32_t maximum_rank_ = 2U;
  LeastSquaresCostMode cost_mode_ = LeastSquaresCostMode::kNormalEquation;
  std::uint32_t icm_rounds_ = 1U;
};

extern template class CpuFixedEdgeReplayEncoder<5>;
extern template class CpuFixedEdgeReplayEncoder<10>;

}  // namespace stlqpp
