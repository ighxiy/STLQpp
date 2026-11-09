#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/training/basic/large_root_algebra.h"
#include "stlqpp/training/basic/cuda_fixed_root_support_refiner.h"
#include "stlqpp/training/basic/virtual_root_support_encoder.h"

#include <cstdint>
#include <memory>
#include <utility>

namespace stlqpp {

template <int M>
class CudaVirtualRootSupportEncoder final
    : public VirtualRootSupportEncoder<M> {
 public:
  [[nodiscard]] static StatusOr<CudaVirtualRootSupportEncoder<M>> Create(
      const BasicCodebookAlgebra<M>& algebra,
      CudaContext& context,
      BasicSchedule schedule = BasicSchedule::TrainingVirtualRoots(),
      CudaFixedRootModelPreparation* preparation = nullptr);

  [[nodiscard]] static StatusOr<CudaVirtualRootSupportEncoder<M>> Create(
      std::shared_ptr<const CudaFixedRootDeviceModel<M>> model,
      CudaContext& context,
      BasicSchedule schedule = BasicSchedule::TrainingVirtualRoots());

  CudaVirtualRootSupportEncoder(CudaVirtualRootSupportEncoder&&) noexcept =
      default;
  CudaVirtualRootSupportEncoder& operator=(
      CudaVirtualRootSupportEncoder&&) noexcept = default;
  CudaVirtualRootSupportEncoder(const CudaVirtualRootSupportEncoder&) = delete;
  CudaVirtualRootSupportEncoder& operator=(
      const CudaVirtualRootSupportEncoder&) = delete;

  [[nodiscard]] StatusOr<BasicEncoding<M>> Encode(
      ColumnMajorMatrixF32View samples,
      ArrayView<const std::uint64_t> sample_ids,
      std::uint32_t fixed_root,
      std::uint32_t seed) const override;

  [[nodiscard]] std::uint64_t model_resident_bytes() const noexcept {
    return refiner_.model_resident_bytes();
  }

  [[nodiscard]] Status ReserveWorkspace(
      std::uint32_t maximum_sample_count) {
    return refiner_.ReserveWorkspace(maximum_sample_count);
  }

  [[nodiscard]] std::uint64_t workspace_resident_bytes() const noexcept {
    return refiner_.workspace_resident_bytes();
  }

 private:
  CudaVirtualRootSupportEncoder(
      CudaFixedRootSupportRefiner<M> refiner,
      BasicSchedule schedule) noexcept
      : refiner_(std::move(refiner)), schedule_(schedule) {}

  mutable CudaFixedRootSupportRefiner<M> refiner_;
  BasicSchedule schedule_ = BasicSchedule::TrainingVirtualRoots();
};

extern template class CudaVirtualRootSupportEncoder<5>;
extern template class CudaVirtualRootSupportEncoder<10>;

}  // namespace stlqpp
