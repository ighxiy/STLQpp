#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "stlqpp/core/status_or.h"
#include "stlqpp/training/least_squares_cost_mode.h"
#include "stlqpp/training/linkage/root_only/root_only_linkage_device_plan.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/scheduler/bounded_best_fit_cache.h"
#include "stlqpp/training/linkage/scheduler/linkage_worker_backend.h"

namespace stlqpp {

template <int M>
class CudaLinkageRsdDeviceModel;

struct CudaLinkageWorkerWorkspaceLimits final {
  std::size_t maximum_batch_tasks = 0U;
  std::uint32_t maximum_batch_nodes = 0U;
  std::uint32_t maximum_batch_pairs = 0U;
  std::uint32_t maximum_update_columns = 0U;
  std::uint32_t internal_chunk_max_pairs = 0U;
};

template <int M, LinkageEvaluationKind Kind>
class CudaTypedLinkageWorkerBackend final : public LinkageWorkerBackend {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  static_assert(Kind == LinkageEvaluationKind::kEdgeRsd ||
                Kind == LinkageEvaluationKind::kRootOnlyInitialization);

  [[nodiscard]] static StatusOr<
      std::unique_ptr<CudaTypedLinkageWorkerBackend<M, Kind>>>
  CreateRsd(
      const LinkageCodebookAlgebra<M>& edge_algebra, int device,
      bool allow_tf32, BestFitCacheLimits rfull_cache_limits = {},
      std::uint32_t maximum_rank = 2U,
      LeastSquaresCostMode cost_mode = LeastSquaresCostMode::kNormalEquation,
      std::uint32_t icm_rounds = 1U);

  [[nodiscard]] static StatusOr<
      std::unique_ptr<CudaTypedLinkageWorkerBackend<M, Kind>>>
  CreateRsd(
      std::shared_ptr<const CudaLinkageRsdDeviceModel<M>> edge_model,
      int device, bool allow_tf32, BestFitCacheLimits rfull_cache_limits = {},
      std::uint32_t maximum_rank = 2U,
      LeastSquaresCostMode cost_mode = LeastSquaresCostMode::kNormalEquation,
      std::uint32_t icm_rounds = 1U);

  // Builds the rsd backend on an existing direct-pool lane. The
  // shared context lifetime lets virtual-root encoding and direct rsd
  // evaluation use the same DEV-equivalent CUDA lane under one lease table.
  [[nodiscard]] static StatusOr<
      std::unique_ptr<CudaTypedLinkageWorkerBackend<M, Kind>>>
  CreateRsdOnContext(
      const LinkageCodebookAlgebra<M>& edge_algebra,
      std::shared_ptr<CudaContext> context,
      BestFitCacheLimits rfull_cache_limits = {},
      std::uint32_t maximum_rank = 2U,
      LeastSquaresCostMode cost_mode = LeastSquaresCostMode::kNormalEquation,
      std::uint32_t icm_rounds = 1U);

  [[nodiscard]] static StatusOr<
      std::unique_ptr<CudaTypedLinkageWorkerBackend<M, Kind>>>
  CreateRsdOnContext(
      std::shared_ptr<const CudaLinkageRsdDeviceModel<M>> edge_model,
      std::shared_ptr<CudaContext> context,
      BestFitCacheLimits rfull_cache_limits = {},
      std::uint32_t maximum_rank = 2U,
      LeastSquaresCostMode cost_mode = LeastSquaresCostMode::kNormalEquation,
      std::uint32_t icm_rounds = 1U);

  [[nodiscard]] static StatusOr<
      std::unique_ptr<CudaTypedLinkageWorkerBackend<M, Kind>>>
  CreateRootOnly(std::shared_ptr<const CudaRootOnlyLinkageModel<M>> root_model,
                 int device, bool allow_tf32,
                 BestFitCacheLimits rfull_cache_limits = {},
                 std::uint32_t icm_rounds = 1U);

  ~CudaTypedLinkageWorkerBackend() override;
  CudaTypedLinkageWorkerBackend(const CudaTypedLinkageWorkerBackend&) = delete;
  CudaTypedLinkageWorkerBackend& operator=(
      const CudaTypedLinkageWorkerBackend&) = delete;

  // Allocates every scheduler-owned device and pinned update buffer at its
  // admitted high-water extent. The CUDA scheduler factory calls this before
  // worker threads start.
  [[nodiscard]] Status ReserveWorkspace(
      CudaLinkageWorkerWorkspaceLimits limits);

  [[nodiscard]] StatusOr<LinkageBackendCluster> Register(
      LinkageBackendRegistration registration) override;

  [[nodiscard]] Status ApplyUpdates(
      ArrayView<const LinkageBackendUpdate> updates) override;

  [[nodiscard]] StatusOr<std::vector<LinkageEvaluationResult>> Evaluate(
      ArrayView<const LinkageBackendEvaluation> tasks) override;
  [[nodiscard]] StatusOr<std::vector<LinkageEvaluationResult>> EvaluateStable(
      ArrayView<const LinkageBackendEvaluation> tasks) override;

  [[nodiscard]] bool SupportsAsynchronousEvaluation() const noexcept override {
    return true;
  }

  [[nodiscard]] Status BeginEvaluate(
      ArrayView<const LinkageBackendEvaluation> tasks,
      LinkageBackendEvaluationCompletion completion) override;

  [[nodiscard]] Status Unregister(LinkageBackendCluster cluster) override;

  [[nodiscard]] LinkageBackendResourceStats ResourceStats()
      const noexcept override;

 private:
  struct State;
  [[nodiscard]] Status EnqueueEvaluation(
      ArrayView<const LinkageBackendEvaluation> tasks, bool stable);
  [[nodiscard]] StatusOr<std::vector<LinkageEvaluationResult>>
  DecodeEvaluation();
  explicit CudaTypedLinkageWorkerBackend(std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

template <int M>
using CudaLinkageWorkerBackend =
    CudaTypedLinkageWorkerBackend<M, LinkageEvaluationKind::kEdgeRsd>;

template <int M>
using CudaRootOnlyLinkageWorkerBackend = CudaTypedLinkageWorkerBackend<
    M, LinkageEvaluationKind::kRootOnlyInitialization>;

extern template class CudaTypedLinkageWorkerBackend<
    5, LinkageEvaluationKind::kEdgeRsd>;
extern template class CudaTypedLinkageWorkerBackend<
    10, LinkageEvaluationKind::kEdgeRsd>;
extern template class CudaTypedLinkageWorkerBackend<
    5, LinkageEvaluationKind::kRootOnlyInitialization>;
extern template class CudaTypedLinkageWorkerBackend<
    10, LinkageEvaluationKind::kRootOnlyInitialization>;

}  // namespace stlqpp
