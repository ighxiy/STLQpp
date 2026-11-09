#include "stlqpp/training/basic/cpu_virtual_root_support_encoder.h"

#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/basic/basic_projection.h"
#include "stlqpp/training/basic/basic_schedule.h"
#include "stlqpp/training/basic/detail/basic_encoding_access.h"
#include "stlqpp/training/basic/icm_ils_cpu.h"
#include "stlqpp/training/basic/forced_root_global_greedy.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace stlqpp {

template <int M>
StatusOr<BasicEncoding<M>> CpuVirtualRootSupportEncoder<M>::Encode(
    ColumnMajorMatrixF32View samples,
    ArrayView<const std::uint64_t> sample_ids,
    std::uint32_t fixed_root,
    std::uint32_t seed) const {
  if (sample_ids.size() != samples.columns) {
    return Status::InvalidArgument(
        "virtual-root sample IDs must match the sample count");
  }
  auto supports = InitializeForcedRootGlobalGreedy<M>(
      samples, fixed_root, *algebra_, *gemm_);
  if (!supports.ok()) {
    return supports.status();
  }

  auto encoding =
      detail::BasicEncodingAccess::Create<M>(samples.columns, algebra_->shape().m);
  std::fill_n(detail::BasicEncodingAccess::Roots(&encoding),
              samples.columns, fixed_root);
  std::copy(supports.value().small_codes.begin(),
            supports.value().small_codes.end(),
            detail::BasicEncodingAccess::Codes(&encoding));

  InlineParallelFor parallel;
  // Keep the original BasicProjection norm accumulation order. The dots
  // already belong to the greedy initializer; no second GEMM is needed.
  std::vector<float> squared_norms(samples.columns, 0.0F);
  for (std::uint32_t sample = 0U; sample < samples.columns; ++sample) {
    float norm = 0.0F;
    for (std::uint32_t axis = 0U; axis < samples.rows; ++axis) {
      const float value = samples.column(sample)[axis];
      norm += value * value;
    }
    squared_norms[sample] = norm;
  }
  BasicProjection<M> projection(algebra_->shape().m);
  const Status bound = projection.BindFullAtomProjection(
      samples, squared_norms, supports.value().small_dots,
      algebra_->small_count(), 0U);
  if (!bound.ok()) return bound;
  CpuIcmIlsOptimizer<M> optimizer(*algebra_, parallel);
  const Status status = optimizer.OptimizeWithRootDotsAndSampleIds(
      projection, supports.value().root_dots,
      sample_ids, schedule_, seed, &encoding);
  if (!status.ok()) {
    return status;
  }
  return encoding;
}

template class CpuVirtualRootSupportEncoder<0>;
template class CpuVirtualRootSupportEncoder<5>;
template class CpuVirtualRootSupportEncoder<12>;
template class CpuVirtualRootSupportEncoder<6>;
template class CpuVirtualRootSupportEncoder<10>;

}  // namespace stlqpp
