#pragma once

#include <optional>

#include "stlqpp/training/basic/icm_ils_cpu.h"
#include "stlqpp/training/linkage/rsd/detail/linkage_rsd_kernel.h"

namespace stlqpp {

// CPU model-space encoder. Rotation and source conversion belong to the
// streaming caller, as for the existing fixed-M CPU basic encoder.
class RuntimeCpuRsdBasicEncoder final {
 public:
  [[nodiscard]] static StatusOr<RuntimeCpuRsdBasicEncoder> Create(
      RuntimeRootCodebookView codebooks, const F32GemmBackend& gemm,
      const ParallelForExecutor& parallel);
  [[nodiscard]] Status Encode(ColumnMajorMatrixF32View samples,
                              std::uint32_t rank, BasicSchedule schedule,
                              std::uint32_t seed,
                              std::uint64_t global_sample_offset,
                              ArrayView<const std::uint64_t> sample_ids,
                              ArrayView<const std::uint32_t> fixed_roots,
                              RuntimeBasicEncoding* output);

 private:
  RuntimeCpuRsdBasicEncoder(RuntimeBasicCodebookAlgebra algebra,
                            std::optional<LinkageCodebookAlgebra<0>> rsd,
                            const F32GemmBackend& gemm,
                            const ParallelForExecutor& parallel)
      : algebra_(std::move(algebra)),
        rsd_(std::move(rsd)),
        gemm_(&gemm),
        parallel_(&parallel),
        projection_(algebra_.shape().m) {}
  RuntimeBasicCodebookAlgebra algebra_;
  std::optional<LinkageCodebookAlgebra<0>> rsd_;
  const F32GemmBackend* gemm_;
  const ParallelForExecutor* parallel_;
  RuntimeBasicProjection projection_;
  std::vector<float> dots_, norms_, root_scores_, root_errors_, root_dots_;
  std::vector<std::uint32_t> roots_;
};

}  // namespace stlqpp
