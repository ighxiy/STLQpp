#include "stlqpp/training/basic/cuda_virtual_root_support_encoder.h"

#include "stlqpp/training/basic/basic_schedule.h"

#include <cstdint>
#include <utility>

namespace stlqpp {

template <int M>
StatusOr<CudaVirtualRootSupportEncoder<M>>
CudaVirtualRootSupportEncoder<M>::Create(
    const BasicCodebookAlgebra<M>& algebra,
    CudaContext& context,
    BasicSchedule schedule,
    CudaFixedRootModelPreparation* preparation) {
  auto refiner = CudaFixedRootSupportRefiner<M>::Create(
      algebra, context, preparation);
  if (!refiner.ok()) {
    return refiner.status();
  }
  return CudaVirtualRootSupportEncoder<M>(
      std::move(refiner).value(), schedule);
}

template <int M>
StatusOr<CudaVirtualRootSupportEncoder<M>>
CudaVirtualRootSupportEncoder<M>::Create(
    std::shared_ptr<const CudaFixedRootDeviceModel<M>> model,
    CudaContext& context,
    BasicSchedule schedule) {
  auto refiner = CudaFixedRootSupportRefiner<M>::Create(
      std::move(model), context);
  if (!refiner.ok()) {
    return refiner.status();
  }
  return CudaVirtualRootSupportEncoder<M>(
      std::move(refiner).value(), schedule);
}

template <int M>
StatusOr<BasicEncoding<M>> CudaVirtualRootSupportEncoder<M>::Encode(
    ColumnMajorMatrixF32View samples,
    ArrayView<const std::uint64_t> sample_ids,
    std::uint32_t fixed_root,
    std::uint32_t seed) const {
  return refiner_.Refine(
      samples, fixed_root, sample_ids, schedule_, seed);
}

template class CudaVirtualRootSupportEncoder<0>;
template class CudaVirtualRootSupportEncoder<5>;
template class CudaVirtualRootSupportEncoder<12>;
template class CudaVirtualRootSupportEncoder<6>;
template class CudaVirtualRootSupportEncoder<10>;

}  // namespace stlqpp
