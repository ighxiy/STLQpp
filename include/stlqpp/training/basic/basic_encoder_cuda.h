#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/runtime/cuda/device_matrix.h"
#include "stlqpp/training/basic/basic_encoding.h"
#include "stlqpp/training/basic/basic_schedule.h"
#include "stlqpp/training/basic/large_root_algebra.h"

#include <cstdint>
#include <memory>

namespace stlqpp {

template <int M>
class CudaLinkageRsdDeviceModel;

namespace detail {
struct CudaBasicEncoderStorage;
}  // namespace detail

template <int M>
class CudaBasicEncoder final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  static constexpr std::uint32_t kVariableRootTileRows = 8192U;

  [[nodiscard]] static StatusOr<CudaBasicEncoder<M>> Create(
      const BasicCodebookAlgebra<M>& algebra,
      CudaContext& context);

  // Builds the ordinary large-root model directly on the device. The Gram
  // products never need a host materialization at this CUDA-only boundary.
  [[nodiscard]] static StatusOr<CudaBasicEncoder<M>> Create(
      const RootCodebooks<M>& codebooks,
      CudaContext& context);

  // IVF256 basic RSD stores its root/small books as the RSD model's edge
  // atoms. Share those immutable atoms, but retain the basic norm reductions
  // and Gram products. The model must contain these same root codebooks as
  // its edge atoms; ownership survives the caller.
  [[nodiscard]] static StatusOr<CudaBasicEncoder<M>> Create(
      const RootCodebooks<M>& codebooks,
      std::shared_ptr<const CudaLinkageRsdDeviceModel<M>> basic_rsd_model,
      CudaContext& context);

  ~CudaBasicEncoder();
  CudaBasicEncoder(CudaBasicEncoder&& other) noexcept;
  CudaBasicEncoder& operator=(CudaBasicEncoder&& other) noexcept;
  CudaBasicEncoder(const CudaBasicEncoder&) = delete;
  CudaBasicEncoder& operator=(const CudaBasicEncoder&) = delete;

  [[nodiscard]] StatusOr<BasicEncoding<M>> Encode(
      ColumnMajorMatrixF32View samples,
      BasicSchedule schedule,
      std::uint32_t seed,
      std::uint64_t global_sample_offset);

  // Consumes a contiguous model-space tile already resident on the context's
  // stream. The producer retains ownership until this call returns.
  [[nodiscard]] StatusOr<BasicEncoding<M>> EncodeDevice(
      CudaMatrixF32View samples,
      BasicSchedule schedule,
      std::uint32_t seed,
      std::uint64_t global_sample_offset);

  // Replay batch entry point. Layer zero is supplied by the frozen topology;
  // explicit stable IDs make ILS independent of gather and block order.
  [[nodiscard]] StatusOr<BasicEncoding<M>> EncodeFixedRoots(
      ColumnMajorMatrixF32View samples,
      ArrayView<const std::uint32_t> fixed_roots,
      ArrayView<const std::uint64_t> global_sample_ids,
      BasicSchedule schedule,
      std::uint32_t seed);

  [[nodiscard]] StatusOr<BasicEncoding<M>> EncodeFixedRootsDevice(
      CudaMatrixF32View samples,
      ArrayView<const std::uint32_t> fixed_roots,
      std::uint64_t global_sample_offset,
      BasicSchedule schedule,
      std::uint32_t seed);

  // Refines an already selected fixed-root support.  Unlike replay, ILS uses
  // the supplied contiguous sample offset; the caller owns support selection.
  [[nodiscard]] StatusOr<BasicEncoding<M>> RefineFixedRootSupports(
      ColumnMajorMatrixF32View samples,
      ArrayView<const std::uint32_t> fixed_roots,
      ArrayView<const std::uint8_t> initial_small_codes,
      BasicSchedule schedule,
      std::uint32_t seed,
      std::uint64_t global_sample_offset);

  [[nodiscard]] StatusOr<BasicEncoding<M>> RefineFixedRootSupportsDevice(
      CudaMatrixF32View samples,
      ArrayView<const std::uint32_t> fixed_roots,
      ArrayView<const std::uint8_t> initial_small_codes,
      BasicSchedule schedule,
      std::uint32_t seed,
      std::uint64_t global_sample_offset);

  // Consumes row-major M-code supports already resident on this context's
  // stream. The first code is the root and the remaining M-1 codes are the
  // fixed-root support. The producer retains ownership until this call
  // returns.
  [[nodiscard]] StatusOr<BasicEncoding<M>>
  RefineDeviceResidentSupports(
      CudaMatrixF32View samples,
      const std::uint8_t* device_support_codes,
      BasicSchedule schedule,
      std::uint32_t seed,
      std::uint64_t global_sample_offset,
      const float* device_sample_norms = nullptr);

  // Ordinary large-root RSD: uint32 root IDs and byte-sized small codes.
  [[nodiscard]] StatusOr<BasicEncoding<M>> EncodeRsdDevice(
      CudaMatrixF32View samples,
      ArrayView<const std::uint32_t> fixed_roots,
      std::uint32_t rsd_rank,
      LeastSquaresCostMode rsd_cost_mode,
      BasicSchedule schedule,
      std::uint32_t seed,
      std::uint64_t global_sample_offset);

  // DEV virtual-root path: CPU MKL produces C_all^T X and CPU norm2 once;
  // CUDA consumes those exact projections instead of recomputing them with a
  // different GEMM/reduction path.
  [[nodiscard]] StatusOr<BasicEncoding<M>> RefineFixedRootProjectedSupports(
      std::uint32_t sample_count,
      ArrayView<const std::uint32_t> fixed_roots,
      ArrayView<const std::uint8_t> initial_small_codes,
      ArrayView<const float> root_dots,
      ArrayView<const float> small_dots,
      ArrayView<const float> sample_norms2,
      BasicSchedule schedule,
      std::uint32_t seed,
      std::uint64_t global_sample_offset);

  // Exclusively owned immutable device storage; shared RSD model storage and
  // high-water per-call workspaces are excluded.
  [[nodiscard]] std::uint64_t model_resident_bytes() const noexcept;

 private:
  explicit CudaBasicEncoder(
      std::unique_ptr<detail::CudaBasicEncoderStorage> state) noexcept;

  [[nodiscard]] StatusOr<BasicEncoding<M>> EncodeImpl(
      const float* samples,
      std::uint32_t sample_count,
      bool source_is_device,
      BasicSchedule schedule,
      std::uint32_t seed,
      std::uint64_t global_sample_offset,
      ArrayView<const std::uint32_t> fixed_roots,
      ArrayView<const std::uint64_t> global_sample_ids,
      ArrayView<const std::uint8_t> initial_small_codes,
      ArrayView<const float> projected_root_dots,
      ArrayView<const float> projected_small_dots,
      ArrayView<const float> projected_sample_norms2,
      bool refine,
      const std::uint8_t* device_initial_support_codes = nullptr,
      const float* device_sample_norms = nullptr,
      std::uint32_t rsd_rank = 0U,
      LeastSquaresCostMode rsd_cost_mode =
          LeastSquaresCostMode::kNormalEquation);

  template <std::uint32_t Capacity>
  [[nodiscard]] StatusOr<BasicEncoding<M>> EncodeImplWithCapacity(
      const float* samples,
      std::uint32_t sample_count,
      bool source_is_device,
      BasicSchedule schedule,
      std::uint32_t seed,
      std::uint64_t global_sample_offset,
      ArrayView<const std::uint32_t> fixed_roots,
      ArrayView<const std::uint64_t> global_sample_ids,
      ArrayView<const std::uint8_t> initial_small_codes,
      ArrayView<const float> projected_root_dots,
      ArrayView<const float> projected_small_dots,
      ArrayView<const float> projected_sample_norms2,
      bool refine,
      const std::uint8_t* device_initial_support_codes = nullptr,
      const float* device_sample_norms = nullptr,
      std::uint32_t rsd_rank = 0U,
      LeastSquaresCostMode rsd_cost_mode =
          LeastSquaresCostMode::kNormalEquation);

  std::unique_ptr<detail::CudaBasicEncoderStorage> state_;
};

extern template class CudaBasicEncoder<5>;
extern template class CudaBasicEncoder<10>;

}  // namespace stlqpp
