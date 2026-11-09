#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/rsd/linkage_rsd_cuda.h"
#include "stlqpp/training/linkage/replay/fixed_edge_replay_encoder.h"
#include "stlqpp/training/least_squares_cost_mode.h"

#include <memory>
#include <utility>

namespace stlqpp {

template <int M>
class CudaFixedEdgeReplayEncoder final : public FixedEdgeReplayEncoder<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<CudaFixedEdgeReplayEncoder<M>> Create(
      const LinkageCodebookAlgebra<M>& algebra,
      CudaContext& context,
      std::uint32_t rsd_rank,
      LeastSquaresCostMode cost_mode =
          LeastSquaresCostMode::kNormalEquation,
      std::uint32_t icm_rounds = 1U);

  [[nodiscard]] std::uint32_t rsd_rank() const noexcept;

  ~CudaFixedEdgeReplayEncoder() override;
  CudaFixedEdgeReplayEncoder(CudaFixedEdgeReplayEncoder&&) noexcept;
  CudaFixedEdgeReplayEncoder& operator=(
      CudaFixedEdgeReplayEncoder&&) noexcept;
  CudaFixedEdgeReplayEncoder(const CudaFixedEdgeReplayEncoder&) = delete;
  CudaFixedEdgeReplayEncoder& operator=(
      const CudaFixedEdgeReplayEncoder&) = delete;

  [[nodiscard]] StatusOr<LinkageEncoding<M>> Encode(
      ColumnMajorMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_orders) const override;

  [[nodiscard]] Status BeginWindow(
      ColumnMajorMatrixF32View roots, ArrayView<const std::uint32_t> root_columns,
      std::uint32_t window_columns) const override;

  void CancelWindow() const noexcept override;

  [[nodiscard]] bool RequiresWindowReconstructions() const noexcept override {
    return true;
  }

  [[nodiscard]] bool UsesHostResiduals()
      const noexcept override {
    return false;
  }

  [[nodiscard]] StatusOr<LinkageEncoding<M>> EncodeWindowBatch(
      ColumnMajorMatrixF32View exact_nodes,
      ColumnMajorMatrixF32View parent_reconstructions,
      ArrayView<const std::uint32_t> parent_columns,
      ArrayView<const std::uint32_t> destination_columns,
      ArrayView<const std::uint8_t> icm_layer_orders) const override;

 private:
  struct State;
  explicit CudaFixedEdgeReplayEncoder(std::unique_ptr<State> state) noexcept;

  [[nodiscard]] StatusOr<LinkageEncoding<M>> EncodeWindowBatchValidated(
      ColumnMajorMatrixF32View exact_nodes,
      ColumnMajorMatrixF32View parent_reconstructions,
      ArrayView<const std::uint32_t> parent_columns,
      ArrayView<const std::uint32_t> destination_columns,
      ArrayView<const std::uint8_t> icm_layer_orders) const override;

  [[nodiscard]] StatusOr<LinkageEncoding<M>> EncodeWindowBatchImpl(
      ColumnMajorMatrixF32View exact_nodes,
      ColumnMajorMatrixF32View parent_reconstructions,
      ArrayView<const std::uint32_t> parent_columns,
      ArrayView<const std::uint32_t> destination_columns,
      ArrayView<const std::uint8_t> icm_layer_orders,
      bool prevalidated) const;

  std::unique_ptr<State> state_;
};

extern template class CudaFixedEdgeReplayEncoder<5>;
extern template class CudaFixedEdgeReplayEncoder<10>;

}  // namespace stlqpp
