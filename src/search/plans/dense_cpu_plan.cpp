#include "stlqpp/search/plans/dense_cpu_plan.h"

namespace stlqpp {

StatusOr<DenseCpuPlan> DenseCpuPlan::Create(const CpuFeatures& features) {
  if (!features.avx_register_state || !features.avx2 || !features.fma) {
    return Status::FailedPrecondition(
        "the STLQpp exact dense plan requires AVX2, FMA, and OS AVX state");
  }
  return DenseCpuPlan();
}

StatusOr<DenseCpuPlan> DenseCpuPlan::Detect() {
  return Create(DetectCpuFeatures());
}

}  // namespace stlqpp
