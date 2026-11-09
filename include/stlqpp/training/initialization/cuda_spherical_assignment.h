#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/runtime/cuda/device_matrix.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/basic/basic_pipeline.h"
#include "stlqpp/training/initialization/rvq_assignment_store.h"
#include "stlqpp/training/initialization/spherical_hierarchical_layout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace stlqpp {

struct CudaSphericalAssignmentOptions final {
  std::size_t maximum_score_bytes = 256U * 1024U * 1024U;
  bool use_registered_hierarchy = false;
  std::uint32_t hierarchical_top_coarse = 1U;
  SphericalHierarchicalSplit hierarchical_split;
  const ParallelForExecutor* parallel = nullptr;
};

struct CudaSphericalAssignmentResult final {
  std::vector<std::uint32_t> codes;
  std::vector<float> best_dots;
};

struct CudaSphericalDeviceAssignment final {
  CudaMatrixF32View residuals;
  const std::uint32_t* codes = nullptr;
  const float* best_dots = nullptr;
  std::uint32_t count = 0U;
};

struct CudaSphericalResidualBlock final {
  CudaMatrixF32View residuals;
  const float* squared_norms = nullptr;
};

// One model-bound exact assignment engine. It retains prior codebooks and one
// current center table on device, while score tiles remain bounded by the
// registered runtime budget. This is a building block of the CUDA initializer,
// not a second IVF or a query path.
template <typename Scalar = std::uint8_t>
class CudaSphericalAssignment final {
 public:
  [[nodiscard]] static StatusOr<CudaSphericalAssignment<Scalar>> Create(
      CudaContext& context,
      const ModelShape& shape,
      ArrayView<const ColumnMajorMatrixF32View> prior_codebooks,
      std::uint32_t prior_layer_count,
      const RvqAssignmentStore* assignments,
      CudaSphericalAssignmentOptions options);

  ~CudaSphericalAssignment();
  CudaSphericalAssignment(CudaSphericalAssignment&&) noexcept;
  CudaSphericalAssignment& operator=(CudaSphericalAssignment&&) noexcept;
  CudaSphericalAssignment(const CudaSphericalAssignment&) = delete;
  CudaSphericalAssignment& operator=(const CudaSphericalAssignment&) =
      delete;

  [[nodiscard]] Status SetCenters(ColumnMajorMatrixF32View centers);
  // Builds the exact normalized stagewise residual and keeps it device
  // resident. Seeding and Lloyd assignment share this operation so their
  // spherical semantics cannot drift.
  [[nodiscard]] StatusOr<CudaMatrixF32View> PrepareResidualsDevice(
      const BasicInputBlock<Scalar>& input);
  [[nodiscard]] StatusOr<CudaSphericalDeviceAssignment> AssignDevice(
      const BasicInputBlock<Scalar>& input);
  // DEV cached-source entry point. The normalized input is copied into the
  // layer work buffer before prior-code projection, exactly as the frozen
  // partial-Xnorm slot path does.
  [[nodiscard]] StatusOr<CudaSphericalDeviceAssignment>
  AssignNormalizedDevice(CudaMatrixF32View normalized,
                         std::uint64_t first_id);
  // Streaming Lloyd path. Prior codes and metadata are appended to the
  // normalized-source slot's upload stream before that slot is published.
  [[nodiscard]] Status StagePriorCodes(std::uint64_t first_id,
                                       std::uint32_t count,
                                       std::uint8_t slot,
                                       void* native_upload_stream);
  // The caller supplies mutable slot storage for layers after the root layer.
  // Projection and renormalization are enqueued in place on the compute stream.
  [[nodiscard]] StatusOr<CudaSphericalResidualBlock> ProjectPriorInPlace(
      CudaMatrixF32View normalized, std::uint8_t slot);
  [[nodiscard]] StatusOr<CudaSphericalDeviceAssignment>
  AssignPreparedDevice(CudaMatrixF32View residuals);
  [[nodiscard]] StatusOr<std::vector<std::uint32_t>> CopyCodes(
      const CudaSphericalDeviceAssignment& assigned);
  [[nodiscard]] StatusOr<CudaSphericalAssignmentResult> Assign(
      const BasicInputBlock<Scalar>& input);

 private:
  struct State;
  explicit CudaSphericalAssignment(std::unique_ptr<State> state) noexcept;

  [[nodiscard]] StatusOr<CudaMatrixF32View>
  PrepareNormalizedResidualsDevice(CudaMatrixF32View normalized,
                                   std::uint64_t first_id);
  [[nodiscard]] StatusOr<CudaMatrixF32View> ApplyPriorProjection(
      std::uint64_t first_id,
      std::uint32_t count);
  std::unique_ptr<State> state_;
};

extern template class CudaSphericalAssignment<std::uint8_t>;
extern template class CudaSphericalAssignment<float>;

}  // namespace stlqpp
