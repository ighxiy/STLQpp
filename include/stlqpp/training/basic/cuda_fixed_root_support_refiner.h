#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/training/basic/basic_encoding.h"
#include "stlqpp/training/basic/basic_schedule.h"
#include "stlqpp/training/basic/large_root_algebra.h"

#include <cstdint>
#include <memory>

namespace stlqpp {

struct CudaFixedRootModelPreparation final {
  double allocation_seconds = 0.0;
  double host_preparation_seconds = 0.0;
  double upload_submission_seconds = 0.0;
  double upload_wait_seconds = 0.0;
};

// Immutable virtual-root algebra for one model version on one CUDA device.
// The uploading context is synchronized before this object is published.
template <int M>
class CudaFixedRootDeviceModel final {
 public:
  [[nodiscard]] static StatusOr<
      std::shared_ptr<const CudaFixedRootDeviceModel<M>>> Create(
      const BasicCodebookAlgebra<M>& algebra,
      CudaContext& context,
      CudaFixedRootModelPreparation* preparation = nullptr);

  // Production virtual-root preparation: no host root/small cross table.
  [[nodiscard]] static StatusOr<
      std::shared_ptr<const CudaFixedRootDeviceModel<M>>> Create(
      const RootCodebooks<M>& codebooks, const F32GemmBackend& gemm,
      CudaContext& context);

  ~CudaFixedRootDeviceModel();
  CudaFixedRootDeviceModel(const CudaFixedRootDeviceModel&) = delete;
  CudaFixedRootDeviceModel& operator=(
      const CudaFixedRootDeviceModel&) = delete;

  [[nodiscard]] int device() const noexcept;
  [[nodiscard]] ModelShape shape() const noexcept;
  [[nodiscard]] std::uint64_t logical_bytes() const noexcept;
  [[nodiscard]] const float* root_atoms() const noexcept;
  [[nodiscard]] const float* small_atoms() const noexcept;
  [[nodiscard]] const float* root_norm2() const noexcept;
  [[nodiscard]] const float* small_inverse_norm() const noexcept;
  [[nodiscard]] const float* small_gram() const noexcept;
  [[nodiscard]] const std::uint32_t* small_offsets() const noexcept;
  [[nodiscard]] const std::uint32_t* layer_sizes() const noexcept;
  [[nodiscard]] float host_root_norm2(std::uint32_t root) const noexcept;

 private:
  [[nodiscard]] static StatusOr<
      std::shared_ptr<const CudaFixedRootDeviceModel<M>>> CreateImpl(
      RuntimeRootCodebookView codebooks, ArrayView<const float> small_gram,
      CudaContext& context, CudaFixedRootModelPreparation* preparation);
  struct State;
  explicit CudaFixedRootDeviceModel(std::unique_ptr<State> state) noexcept;
  std::unique_ptr<State> state_;
};

// Fixed-root ICM/ILS backend used by virtual-root construction.  Its kernel
// boundaries and packed-upper solve order are part of the registered DEV
// numerical contract; ordinary basic encoding remains a separate backend.
template <int M>
class CudaFixedRootSupportRefiner final {
 public:
  [[nodiscard]] static StatusOr<CudaFixedRootSupportRefiner<M>> Create(
      const BasicCodebookAlgebra<M>& algebra,
      CudaContext& context,
      CudaFixedRootModelPreparation* preparation = nullptr);

  [[nodiscard]] static StatusOr<CudaFixedRootSupportRefiner<M>> Create(
      std::shared_ptr<const CudaFixedRootDeviceModel<M>> model,
      CudaContext& context);

  CudaFixedRootSupportRefiner(
      CudaFixedRootSupportRefiner&& other) noexcept;
  CudaFixedRootSupportRefiner& operator=(
      CudaFixedRootSupportRefiner&& other) noexcept;
  CudaFixedRootSupportRefiner(const CudaFixedRootSupportRefiner&) = delete;
  CudaFixedRootSupportRefiner& operator=(
      const CudaFixedRootSupportRefiner&) = delete;
  ~CudaFixedRootSupportRefiner();

  [[nodiscard]] StatusOr<BasicEncoding<M>> Refine(
      ColumnMajorMatrixF32View samples,
      std::uint32_t fixed_root,
      ArrayView<const std::uint64_t> sample_ids,
      BasicSchedule schedule,
      std::uint32_t seed) const;

  template <std::uint32_t Capacity>
  [[nodiscard]] StatusOr<BasicEncoding<M>> RefineWithCapacity(
      ColumnMajorMatrixF32View samples,
      std::uint32_t fixed_root,
      ArrayView<const std::uint64_t> sample_ids,
      BasicSchedule schedule,
      std::uint32_t seed) const;

  [[nodiscard]] Status ReserveWorkspace(
      std::uint32_t maximum_sample_count);

  [[nodiscard]] std::uint64_t model_resident_bytes() const noexcept;
  [[nodiscard]] std::uint64_t workspace_resident_bytes() const noexcept;

 private:
  struct State;
  explicit CudaFixedRootSupportRefiner(
      std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

extern template class CudaFixedRootSupportRefiner<5>;
extern template class CudaFixedRootSupportRefiner<10>;
extern template class CudaFixedRootDeviceModel<5>;
extern template class CudaFixedRootDeviceModel<10>;

}  // namespace stlqpp
