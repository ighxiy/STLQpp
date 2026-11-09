#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/runtime/cuda/device_matrix.h"
#include "stlqpp/training/basic/large_root_algebra.h"

#include <cstdint>
#include <memory>

namespace stlqpp {

struct CudaRootOnlyLinkageModelOptions final {
  // DEV's large-root single-call kernel caps the root-dot tile to the smaller
  // of these two budgets.  The outer wrapper additionally chunks samples so
  // a complete root-dot matrix fits in full_root_dot_workspace_bytes.
  std::uint64_t root_dot_workspace_bytes =
      32ULL * 1024ULL * 1024ULL;
  std::uint64_t full_root_dot_workspace_bytes =
      256ULL * 1024ULL * 1024ULL;
  std::uint64_t encoder_workspace_bytes =
      512ULL * 1024ULL * 1024ULL;
};

template <int M>
class CudaRootOnlyLinkageDevicePlan;

// One immutable device copy of C_root and its algebra is shared by all
// scheduler workers on a device.  In particular, the nlist x (M-1)*256
// root/small cross table is never replicated per CUDA stream.
template <int M>
class CudaRootOnlyLinkageModel final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<
      std::shared_ptr<CudaRootOnlyLinkageModel<M>>> Create(
      const BasicCodebookAlgebra<M>& algebra,
      int device,
      bool allow_tf32,
      CudaRootOnlyLinkageModelOptions options = {});

  [[nodiscard]] static StatusOr<
      std::shared_ptr<CudaRootOnlyLinkageModel<M>>> Create(
      const RootCodebooks<M>& codebooks, const F32GemmBackend& gemm,
      int device, bool allow_tf32,
      CudaRootOnlyLinkageModelOptions options = {});

  ~CudaRootOnlyLinkageModel();
  CudaRootOnlyLinkageModel(const CudaRootOnlyLinkageModel&) = delete;
  CudaRootOnlyLinkageModel& operator=(
      const CudaRootOnlyLinkageModel&) = delete;

  [[nodiscard]] int device() const noexcept;
  [[nodiscard]] std::uint32_t dimension() const noexcept;
  [[nodiscard]] std::uint32_t m() const noexcept;
  [[nodiscard]] std::uint64_t device_bytes() const noexcept;

 private:
  friend class CudaRootOnlyLinkageDevicePlan<M>;
  struct State;
  [[nodiscard]] static StatusOr<
      std::shared_ptr<CudaRootOnlyLinkageModel<M>>> CreateImpl(
      RuntimeRootCodebookView codebooks, ArrayView<const float> small_gram,
      int device, bool allow_tf32, CudaRootOnlyLinkageModelOptions options);
  explicit CudaRootOnlyLinkageModel(std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

template <int M>
struct CudaRootOnlyLinkageDeviceOutput final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  std::uint32_t* first_codes = nullptr;
  std::uint8_t* trailing_codes = nullptr;
  float* coefficients = nullptr;
  float* squared_errors = nullptr;
};

// Stream-ordered root-only init primitive. One Enqueue call performs exactly
// one native global-greedy path, exact joint LS and configured shared-order ICM sweeps
// including layer zero. No synchronization, host copies, ILS or RSD
// are hidden inside this object.
template <int M>
class CudaRootOnlyLinkageDevicePlan final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  static constexpr std::uint32_t kDefaultBlockSamples = 8192U;

  [[nodiscard]] static StatusOr<CudaRootOnlyLinkageDevicePlan<M>> Create(
      std::shared_ptr<const CudaRootOnlyLinkageModel<M>> model,
      CudaContext& context,
      std::uint32_t icm_rounds = 1U);

  ~CudaRootOnlyLinkageDevicePlan();
  CudaRootOnlyLinkageDevicePlan(
      CudaRootOnlyLinkageDevicePlan&&) noexcept;
  CudaRootOnlyLinkageDevicePlan& operator=(
      CudaRootOnlyLinkageDevicePlan&&) noexcept;
  CudaRootOnlyLinkageDevicePlan(
      const CudaRootOnlyLinkageDevicePlan&) = delete;
  CudaRootOnlyLinkageDevicePlan& operator=(
      const CudaRootOnlyLinkageDevicePlan&) = delete;

  [[nodiscard]] Status BeginBatch();

  // Sample extent passed to one DEV-compatible large-root encoder call.  DEV
  // rounds the requested extent down according to its workspace policy, so
  // this is deliberately a function of the complete batch size.
  [[nodiscard]] std::uint32_t sample_chunk_size(
      std::uint32_t requested_sample_count) const noexcept;

  // Materializes the high-water workspace before scheduler threads start.
  // The immutable root algebra remains shared by every worker on the device.
  [[nodiscard]] Status ReserveWorkspace(std::uint32_t sample_count);

  [[nodiscard]] Status Enqueue(
      CudaMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_order,
      CudaRootOnlyLinkageDeviceOutput<M> output);

  template <std::uint32_t Capacity>
  [[nodiscard]] Status EnqueueWithCapacity(
      CudaMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_order,
      CudaRootOnlyLinkageDeviceOutput<M> output);

  [[nodiscard]] Status EnqueueFailureReadback(
      std::uint32_t* host_failures);

 private:
  struct State;
  explicit CudaRootOnlyLinkageDevicePlan(
      std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

extern template class CudaRootOnlyLinkageModel<5>;
extern template class CudaRootOnlyLinkageModel<10>;
extern template class CudaRootOnlyLinkageDevicePlan<5>;
extern template class CudaRootOnlyLinkageDevicePlan<10>;

}  // namespace stlqpp
