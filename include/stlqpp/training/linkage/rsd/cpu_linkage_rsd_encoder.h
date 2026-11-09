#pragma once

#include <cstdint>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/least_squares_cost_mode.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/rsd/linkage_encoding.h"
#include "stlqpp/training/linkage/rsd/linkage_projection.h"

namespace stlqpp {

template <int M>
class CpuRsdLinkageWorkerBackend;
template <int M>
class CpuFixedEdgeReplayEncoder;

// Semantic CPU implementation of the production linkage endpoint: cumulative
// global-greedy RSD through the selected rank, exact joint-LS candidate
// ranking, then the requested ICM sweeps in the caller-supplied deterministic
// layer order.
template <int M>
class CpuLinkageRsdEncoder final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  CpuLinkageRsdEncoder(
      const LinkageCodebookAlgebra<M>& algebra,
      const ParallelForExecutor& parallel, std::uint32_t maximum_rank = 2U,
      LeastSquaresCostMode cost_mode = LeastSquaresCostMode::kNormalEquation,
      std::uint32_t icm_rounds = 1U) noexcept
      : algebra_(algebra),
        parallel_(parallel),
        maximum_rank_(maximum_rank),
        cost_mode_(cost_mode),
        icm_rounds_(icm_rounds) {}

  [[nodiscard]] StatusOr<LinkageEncoding<M>> Encode(
      const LinkageProjection<M>& projection,
      ArrayView<const std::uint8_t> icm_layer_orders) const;

 private:
  friend class CpuRsdLinkageWorkerBackend<M>;
  friend class CpuFixedEdgeReplayEncoder<M>;
  [[nodiscard]] StatusOr<LinkageEncoding<M>> EncodeValidatedOrders(
      const LinkageProjection<M>& projection,
      ArrayView<const std::uint8_t> icm_layer_orders) const;
  struct EncodeContext;

  template <std::uint32_t Capacity>
  static void EncodeSample(void* opaque, std::uint32_t worker,
                           std::uint32_t sample) noexcept;

  const LinkageCodebookAlgebra<M>& algebra_;
  const ParallelForExecutor& parallel_;
  std::uint32_t maximum_rank_ = 2U;
  LeastSquaresCostMode cost_mode_ = LeastSquaresCostMode::kNormalEquation;
  std::uint32_t icm_rounds_ = 1U;
};

extern template class CpuLinkageRsdEncoder<5>;
extern template class CpuLinkageRsdEncoder<10>;

}  // namespace stlqpp
