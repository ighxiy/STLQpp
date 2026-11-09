#pragma once

#include <cstdint>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/basic/basic_encoding.h"
#include "stlqpp/training/basic/basic_projection.h"
#include "stlqpp/training/basic/basic_schedule.h"
#include "stlqpp/training/basic/large_root_algebra.h"
#include "stlqpp/training/basic/runtime_basic_encoding.h"

namespace stlqpp {

// Same sample-local ICM/ILS implementation and random stream as the fixed-M
// path. Empty IDs use offset+sample; root dots may be supplied by a GEMM owner.
[[nodiscard]] Status OptimizeRuntimeBasic(
    const RuntimeBasicCodebookAlgebra& algebra,
    const RuntimeBasicProjection& projection,
    const ParallelForExecutor& parallel, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t global_sample_offset,
    ArrayView<const std::uint64_t> sample_ids, ArrayView<const float> root_dots,
    RuntimeBasicEncoding* encoding);

namespace detail {
// For buffers produced by the encoder in this call; no release-time root scan.
[[nodiscard]] Status OptimizeRuntimeBasicTrusted(
    const RuntimeBasicCodebookAlgebra& algebra,
    const RuntimeBasicProjection& projection,
    const ParallelForExecutor& parallel, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t global_sample_offset,
    ArrayView<const std::uint64_t> sample_ids, ArrayView<const float> root_dots,
    RuntimeBasicEncoding* encoding);
}  // namespace detail

template <int M>
class CpuIcmIlsOptimizer final {
 public:
  static_assert(M == 0 ||
                (M >= kMinimumModelLayers && M <= kMaximumModelLayers));

  CpuIcmIlsOptimizer(const BasicCodebookAlgebra<M>& algebra,
                     const ParallelForExecutor& parallel) noexcept
      : algebra_(algebra), parallel_(parallel) {}

  [[nodiscard]] Status Optimize(const BasicProjection<M>& projection,
                                BasicSchedule schedule, std::uint32_t seed,
                                std::uint64_t global_sample_offset,
                                BasicEncoding<M>* encoding) const;

  // Gathered replay batches are not contiguous in dataset order. Explicit
  // stable IDs preserve ILS randomness across batching and worker order.
  [[nodiscard]] Status OptimizeWithSampleIds(
      const BasicProjection<M>& projection, BasicSchedule schedule,
      std::uint32_t seed, ArrayView<const std::uint64_t> global_sample_ids,
      BasicEncoding<M>* encoding) const;

  // Fixed-root virtual-center encoding already owns DEV's GEMM root
  // projection. Reusing it preserves the registered floating-point path.
  [[nodiscard]] Status OptimizeWithRootDots(
      const BasicProjection<M>& projection, ArrayView<const float> root_dots,
      BasicSchedule schedule, std::uint32_t seed,
      BasicEncoding<M>* encoding) const;

  [[nodiscard]] Status OptimizeWithRootDotsAndSampleIds(
      const BasicProjection<M>& projection, ArrayView<const float> root_dots,
      ArrayView<const std::uint64_t> global_sample_ids, BasicSchedule schedule,
      std::uint32_t seed, BasicEncoding<M>* encoding) const;

 private:
  const BasicCodebookAlgebra<M>& algebra_;
  const ParallelForExecutor& parallel_;
};

extern template class CpuIcmIlsOptimizer<5>;
extern template class CpuIcmIlsOptimizer<10>;

}  // namespace stlqpp
