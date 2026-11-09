#pragma once

#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/training/basic/basic_schedule.h"
#include "stlqpp/training/basic/large_root_algebra.h"
#include "stlqpp/training/basic/virtual_root_support_encoder.h"

#include <cstdint>

namespace stlqpp {

// DEV's registered ordinary-training backend for UMAP virtual roots.  The
// owning CID is fixed, global-greedy selects the remaining supports, and the
// CPU absolute-score ICM/ILS contract refines them.
template <int M>
class CpuVirtualRootSupportEncoder final
    : public VirtualRootSupportEncoder<M> {
 public:
  CpuVirtualRootSupportEncoder(
      const BasicCodebookAlgebra<M>& algebra,
      const F32GemmBackend& gemm,
      BasicSchedule schedule = BasicSchedule::TrainingVirtualRoots()) noexcept
      : algebra_(&algebra), gemm_(&gemm), schedule_(schedule) {}

  [[nodiscard]] StatusOr<BasicEncoding<M>> Encode(
      ColumnMajorMatrixF32View samples,
      ArrayView<const std::uint64_t> sample_ids,
      std::uint32_t fixed_root,
      std::uint32_t seed) const override;

 private:
  const BasicCodebookAlgebra<M>* algebra_ = nullptr;
  const F32GemmBackend* gemm_ = nullptr;
  BasicSchedule schedule_ = BasicSchedule::TrainingVirtualRoots();
};

extern template class CpuVirtualRootSupportEncoder<5>;
extern template class CpuVirtualRootSupportEncoder<10>;

}  // namespace stlqpp
