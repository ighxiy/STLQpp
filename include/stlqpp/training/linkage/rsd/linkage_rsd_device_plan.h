#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/runtime/cuda/device_matrix.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/least_squares_cost_mode.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace stlqpp {

// Immutable edge algebra resident on one CUDA device. Scheduler workers may
// share this object while retaining independent streams, BLAS handles and
// writable workspaces. Create completes its uploads and algebra construction
// before publishing the model, so consumers need no cross-stream upload fence.
template <int M>
class CudaLinkageRsdDeviceModel final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<
      std::shared_ptr<const CudaLinkageRsdDeviceModel<M>>> Create(
      const LinkageCodebookAlgebra<M>& algebra,
      int device);

  [[nodiscard]] static StatusOr<
      std::shared_ptr<const CudaLinkageRsdDeviceModel<M>>> Create(
      const ProductCodebooks<M>& codebooks,
      CudaContext& context);

  ~CudaLinkageRsdDeviceModel();
  CudaLinkageRsdDeviceModel(
      const CudaLinkageRsdDeviceModel&) = delete;
  CudaLinkageRsdDeviceModel& operator=(
      const CudaLinkageRsdDeviceModel&) = delete;

  [[nodiscard]] int device() const noexcept;
  [[nodiscard]] ModelShape shape() const noexcept;
  [[nodiscard]] std::uint64_t logical_bytes() const noexcept;
  [[nodiscard]] const float* atoms() const noexcept;
  [[nodiscard]] const float* squared_norms() const noexcept;
  [[nodiscard]] const float* inverse_norms() const noexcept;
  [[nodiscard]] const float* gram() const noexcept;

 private:
  struct State;
  explicit CudaLinkageRsdDeviceModel(
      std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

template <int M>
struct CudaLinkageRsdDeviceOutput final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  std::uint8_t* codes = nullptr;
  float* coefficients = nullptr;
  float* squared_errors = nullptr;
};

// Stream-ordered device primitive shared by the standalone RSD encoder
// and the cross-cluster scheduler backend. Inputs and outputs stay on the
// caller's CUDA stream; synchronization and host result ownership remain with
// the caller.
template <int M>
class CudaLinkageRsdDevicePlan final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  static constexpr std::uint32_t kDefaultBlockSamples = 8192U;

  [[nodiscard]] static StatusOr<CudaLinkageRsdDevicePlan<M>> Create(
      const LinkageCodebookAlgebra<M>& algebra,
      CudaContext& context,
      std::uint32_t maximum_rank = 2U,
      LeastSquaresCostMode cost_mode =
          LeastSquaresCostMode::kNormalEquation,
      std::uint32_t icm_rounds = 1U);

  [[nodiscard]] static StatusOr<CudaLinkageRsdDevicePlan<M>> Create(
      std::shared_ptr<const CudaLinkageRsdDeviceModel<M>> model,
      CudaContext& context,
      std::uint32_t maximum_rank = 2U,
      LeastSquaresCostMode cost_mode =
          LeastSquaresCostMode::kNormalEquation,
      std::uint32_t icm_rounds = 1U);

  ~CudaLinkageRsdDevicePlan();
  CudaLinkageRsdDevicePlan(
      CudaLinkageRsdDevicePlan&&) noexcept;
  CudaLinkageRsdDevicePlan& operator=(
      CudaLinkageRsdDevicePlan&&) noexcept;
  CudaLinkageRsdDevicePlan(
      const CudaLinkageRsdDevicePlan&) = delete;
  CudaLinkageRsdDevicePlan& operator=(
      const CudaLinkageRsdDevicePlan&) = delete;

  [[nodiscard]] Status BeginBatch();

  // Materializes the high-water workspace before scheduler threads start.
  // This is a construction-time resource operation, not a launch-time guard.
  [[nodiscard]] Status ReserveWorkspace(std::uint32_t sample_count);

  [[nodiscard]] Status Enqueue(
      CudaMatrixF32View residuals,
      const std::uint8_t* device_icm_layer_orders,
      CudaLinkageRsdDeviceOutput<M> output,
      const float* device_residual_squared_norms = nullptr);

  // Selects and jointly refits cumulative RSD candidates without applying
  // linkage ICM. The plan's configured rank controls proposal generation;
  // basic and ordinary linkage may choose any admitted rank.
  [[nodiscard]] Status EnqueueRawRsd(
      CudaMatrixF32View residuals,
      CudaLinkageRsdDeviceOutput<M> output,
      const float* device_residual_squared_norms = nullptr);

  // Period-two basic replay keeps the historical root assignment. RSD still
  // constructs the native path and configured-rank deviations at remaining
  // layers, then jointly refits all M coefficients.
  [[nodiscard]] Status EnqueueRawRsdFixedLayerZero(
      CudaMatrixF32View residuals,
      const std::uint8_t* device_fixed_layer_zero_codes,
      CudaLinkageRsdDeviceOutput<M> output,
      const float* device_residual_squared_norms = nullptr);

  // Select the best one-term LS root before ordinary fixed-root RSD.
  [[nodiscard]] Status EnqueueRawRsdTopOneRoot(
      CudaMatrixF32View residuals,
      CudaLinkageRsdDeviceOutput<M> output,
      const float* device_residual_squared_norms = nullptr);

  // Stable scheduler primitive: reserve one transport-wide RSD
  // workspace, retain each source task's exact GEMM shape, then execute the
  // sample-independent RSD/ICM kernels once over all projected rows.
  [[nodiscard]] Status BeginProjectedBatch(std::uint32_t sample_count);
  [[nodiscard]] Status EnqueueProjectionSequence(
      CudaMatrixF32View residuals,
      ArrayView<const std::uint32_t> source_sample_offsets);
  [[nodiscard]] Status EnqueueProjectedRsd(
      const std::uint8_t* device_icm_layer_orders,
      CudaLinkageRsdDeviceOutput<M> output,
      const float* device_residual_squared_norms);

  // Native rank-1 global-greedy compatibility entry used by direct callers
  // and tests. Configured Replay uses Enqueue and evaluates RSD branches.
  [[nodiscard]] Status EnqueueNative(
      CudaMatrixF32View residuals,
      const std::uint8_t* device_icm_layer_orders,
      CudaLinkageRsdDeviceOutput<M> output);

  // Refits caller-selected layer-major support codes, then runs one complete
  // linkage ICM sweep in the caller-provided order. Unlike basic refinement,
  // every C_one layer, including layer zero, remains eligible for replacement.
  [[nodiscard]] Status EnqueueRefinedSupports(
      CudaMatrixF32View residuals,
      const std::uint8_t* device_icm_layer_orders,
      CudaLinkageRsdDeviceOutput<M> output);

  [[nodiscard]] Status EnqueueFailureReadback(
      std::uint32_t* host_failures);

 private:
  struct State;
  explicit CudaLinkageRsdDevicePlan(
      std::unique_ptr<State> state) noexcept;

  [[nodiscard]] Status EnqueueProjectedRsdImpl(
      std::uint32_t sample_count,
      const std::uint8_t* device_icm_layer_orders,
      CudaLinkageRsdDeviceOutput<M> output,
      const float* device_residual_squared_norms,
      bool run_icm,
      const std::uint8_t* device_fixed_layer_zero_codes = nullptr,
      bool record_unsolved_supports = true);

  template <std::uint32_t Capacity>
  [[nodiscard]] Status EnqueueProjectedRsdImplWithCapacity(
      std::uint32_t sample_count,
      const std::uint8_t* device_icm_layer_orders,
      CudaLinkageRsdDeviceOutput<M> output,
      const float* device_residual_squared_norms,
      bool run_icm,
      const std::uint8_t* device_fixed_layer_zero_codes = nullptr,
      bool record_unsolved_supports = true);

  [[nodiscard]] Status EnqueueRefinementImpl(
      CudaMatrixF32View residuals,
      const std::uint8_t* device_icm_layer_orders,
      CudaLinkageRsdDeviceOutput<M> output,
      bool initialize_native_support);

  template <std::uint32_t Capacity>
  [[nodiscard]] Status EnqueueRefinementImplWithCapacity(
      CudaMatrixF32View residuals,
      const std::uint8_t* device_icm_layer_orders,
      CudaLinkageRsdDeviceOutput<M> output,
      bool initialize_native_support);

  std::unique_ptr<State> state_;
};

extern template class CudaLinkageRsdDevicePlan<5>;
extern template class CudaLinkageRsdDevicePlan<10>;
extern template class CudaLinkageRsdDeviceModel<5>;
extern template class CudaLinkageRsdDeviceModel<10>;

}  // namespace stlqpp
