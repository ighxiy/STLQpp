#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/runtime/cuda/device_matrix.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/rsd/linkage_encoding.h"
#include "stlqpp/training/least_squares_cost_mode.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace stlqpp {

template <int M>
class CudaFixedEdgeReplayEncoder;
template <int M>
class CudaLinkageRsdDeviceModel;

// Non-owning view of row-major M-code RSD supports retained by the encoder.
// The view remains valid until the next call on that encoder.
struct CudaLinkageRsdSupportView final {
  const std::uint8_t* codes = nullptr;
  // Full request extent, not the workspace of its last raw-RSD chunk.
  const float* squared_norms = nullptr;
  std::uint32_t sample_count = 0U;
};

template <int M>
class CudaLinkageRsdEncoder final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<CudaLinkageRsdEncoder<M>> Create(
      const LinkageCodebookAlgebra<M>& algebra,
      CudaContext& context,
      std::uint32_t maximum_rank = 2U,
      LeastSquaresCostMode cost_mode =
          LeastSquaresCostMode::kNormalEquation,
      std::uint32_t icm_rounds = 1U);

  [[nodiscard]] static StatusOr<CudaLinkageRsdEncoder<M>> Create(
      const ProductCodebooks<M>& codebooks,
      CudaContext& context,
      std::uint32_t maximum_rank = 2U,
      LeastSquaresCostMode cost_mode =
          LeastSquaresCostMode::kNormalEquation,
      std::uint32_t icm_rounds = 1U);

  ~CudaLinkageRsdEncoder();

  [[nodiscard]] static StatusOr<CudaLinkageRsdEncoder<M>> Create(
      std::shared_ptr<const CudaLinkageRsdDeviceModel<M>> model,
      CudaContext& context,
      std::uint32_t maximum_rank = 2U,
      LeastSquaresCostMode cost_mode =
          LeastSquaresCostMode::kNormalEquation,
      std::uint32_t icm_rounds = 1U);
  CudaLinkageRsdEncoder(CudaLinkageRsdEncoder&&) noexcept;
  CudaLinkageRsdEncoder& operator=(
      CudaLinkageRsdEncoder&&) noexcept;
  CudaLinkageRsdEncoder(const CudaLinkageRsdEncoder&) = delete;
  CudaLinkageRsdEncoder& operator=(
      const CudaLinkageRsdEncoder&) = delete;

  [[nodiscard]] StatusOr<LinkageEncoding<M>> Encode(
      ColumnMajorMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_orders);

  // Variable-layer cumulative RSD selection through the configured rank and
  // joint LS only. No layer is fixed and no linkage or basic ICM is applied
  // at this boundary.
  [[nodiscard]] StatusOr<LinkageEncoding<M>> EncodeRawRsd(
      ColumnMajorMatrixF32View residuals);

  [[nodiscard]] StatusOr<LinkageEncoding<M>> EncodeNative(
      ColumnMajorMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_orders);

  // Refits and applies one caller-ordered linkage ICM sweep to an externally
  // selected C_one support. All M edge layers remain mutable.
  [[nodiscard]] StatusOr<LinkageEncoding<M>> RefineSupports(
      ColumnMajorMatrixF32View residuals,
      ArrayView<const std::uint8_t> initial_codes,
      ArrayView<const std::uint8_t> icm_layer_orders);

  [[nodiscard]] StatusOr<LinkageEncoding<M>> EncodeDevice(
      CudaMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_orders);

  [[nodiscard]] StatusOr<LinkageEncoding<M>> EncodeRawRsdDevice(
      CudaMatrixF32View residuals);

  // Basic-encoding fast path. It retains raw RSD supports on the producer's
  // CUDA stream instead of materializing unused coefficients and errors on
  // the host before immediately uploading the supports again.
  [[nodiscard]] StatusOr<CudaLinkageRsdSupportView>
  EncodeRawRsdSupportDeviceResident(CudaMatrixF32View residuals);

  [[nodiscard]] StatusOr<CudaLinkageRsdSupportView>
  EncodeRawRsdFixedLayerZeroSupportDeviceResident(
      CudaMatrixF32View residuals,
      ArrayView<const std::uint32_t> fixed_layer_zero_codes);

  // Select layer-zero's top-one root before searching the remaining layers,
  // retaining only the support consumed by basic refinement on the device.
  [[nodiscard]] StatusOr<CudaLinkageRsdSupportView>
  EncodeRawRsdTopOneRootSupportDeviceResident(
      CudaMatrixF32View residuals);

  [[nodiscard]] StatusOr<LinkageEncoding<M>>
  EncodeRawRsdFixedLayerZeroDevice(
      CudaMatrixF32View residuals,
      ArrayView<const std::uint32_t> fixed_layer_zero_codes);

  // Basic initializer with its layer-zero root fixed before RSD search.
  [[nodiscard]] StatusOr<LinkageEncoding<M>>
  EncodeRawRsdTopOneRootDevice(CudaMatrixF32View residuals);

  // Replay depth batches are semantic GEMM shapes in frozen DEV. Unlike the
  // general bounded standalone endpoint, this call must not split one depth
  // at the ordinary 8,192-row workspace boundary.
  [[nodiscard]] StatusOr<LinkageEncoding<M>> EncodeDeviceCompleteBatch(
      CudaMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_orders);

 private:
  friend class CudaFixedEdgeReplayEncoder<M>;

  struct CompleteBatchDeviceResult final {
    const std::uint8_t* codes;
    const float* coefficients;
  };
  // Only replay consumes this view, immediately after a successful complete
  // batch. The whole depth is resident (not just the final bounded chunk).
  // Enqueue consumers on the same context stream before the next encode.
  [[nodiscard]] CompleteBatchDeviceResult complete_batch_device_result() const;

  [[nodiscard]] StatusOr<LinkageEncoding<M>>
  EncodeDeviceCompleteBatchValidated(
      CudaMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_orders);

  enum class CandidateMode : std::uint8_t {
    kRsd = 0U,
    kNative = 1U,
    kInitialSupport = 2U,
    kRawRsd = 3U,
    kRawRsdFixedLayerZero = 4U,
    kRawRsdRootTopOne = 5U,
  };

  struct State;
  explicit CudaLinkageRsdEncoder(
      std::unique_ptr<State> state) noexcept;

  [[nodiscard]] StatusOr<LinkageEncoding<M>> EncodeImpl(
      const float* residuals,
      std::uint32_t sample_count,
      bool source_is_device,
      ArrayView<const std::uint8_t> icm_layer_orders,
      CandidateMode mode,
      std::uint32_t maximum_block_samples,
      ArrayView<const std::uint8_t> initial_codes,
      ArrayView<const std::uint32_t> fixed_layer_zero_codes = {},
      bool orders_prevalidated = false);

  [[nodiscard]] StatusOr<CudaLinkageRsdSupportView>
  EncodeRawRsdSupportDeviceResidentImpl(
      CudaMatrixF32View residuals,
      ArrayView<const std::uint32_t> fixed_layer_zero_codes,
      bool select_top_one_root = false);

  std::unique_ptr<State> state_;
};

extern template class CudaLinkageRsdEncoder<5>;
extern template class CudaLinkageRsdEncoder<10>;

}  // namespace stlqpp
