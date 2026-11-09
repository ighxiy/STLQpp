#pragma once

#include <memory>

#include "stlqpp/model/model_shape.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/basic/large_root_algebra.h"
#include "stlqpp/training/least_squares_cost_mode.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/scheduler/linkage_worker_backend.h"

namespace stlqpp {

// Frozen DEV's CPU fallback keeps one candidate projection in thread-local
// scratch. PP retains batched GEMM, but caps that execution-only batching so
// 32 direct lanes cannot materialize full producer tasks at once. The task
// remains the numerical/causal unit; its immutable pair stream is reduced in
// original order.
inline constexpr std::uint32_t kCpuLinkageEvaluationTilePairs = 512U;

template <int M>
[[nodiscard]] constexpr std::uint64_t CpuEdgeLinkageMatrixWorkspaceBytes(
    std::uint32_t dimension, std::uint32_t maximum_task_pairs,
    std::uint32_t m = FixedModelLayers<M>()) noexcept {
  const std::uint64_t pairs =
      maximum_task_pairs < kCpuLinkageEvaluationTilePairs
          ? maximum_task_pairs
          : kCpuLinkageEvaluationTilePairs;
  return sizeof(float) * pairs *
         (static_cast<std::uint64_t>(dimension) +
          static_cast<std::uint64_t>(m) * 256U);
}

template <int M>
[[nodiscard]] constexpr std::uint64_t CpuRootLinkageMatrixWorkspaceBytes(
    std::uint32_t dimension, std::uint32_t maximum_task_pairs,
    std::uint32_t m = FixedModelLayers<M>()) noexcept {
  const std::uint64_t pairs =
      maximum_task_pairs < kCpuLinkageEvaluationTilePairs
          ? maximum_task_pairs
          : kCpuLinkageEvaluationTilePairs;
  return sizeof(float) * pairs *
         (static_cast<std::uint64_t>(dimension) +
          static_cast<std::uint64_t>(m - 1U) * 256U);
}

// CPU-only production backends for the two existing linkage semantics.  The
// kind is fixed by the concrete type, so an evaluation hot path never checks
// a runtime capability flag or mixes root-only and edge-RSD algorithms.
template <int M>
class CpuRsdLinkageWorkerBackend final : public LinkageWorkerBackend {
 public:
  CpuRsdLinkageWorkerBackend(
      const LinkageCodebookAlgebra<M>& algebra, const F32GemmBackend& gemm,
      const ParallelForExecutor& parallel, std::uint32_t maximum_rank = 2U,
      LeastSquaresCostMode cost_mode = LeastSquaresCostMode::kNormalEquation,
      std::uint32_t icm_rounds = 1U) noexcept;
  ~CpuRsdLinkageWorkerBackend() override;

  [[nodiscard]] StatusOr<LinkageBackendCluster> Register(
      LinkageBackendRegistration registration) override;
  [[nodiscard]] Status ApplyUpdates(
      ArrayView<const LinkageBackendUpdate> updates) override;
  [[nodiscard]] StatusOr<std::vector<LinkageEvaluationResult>> Evaluate(
      ArrayView<const LinkageBackendEvaluation> tasks) override;
  [[nodiscard]] Status Unregister(LinkageBackendCluster cluster) override;
  [[nodiscard]] LinkageBackendResourceStats ResourceStats()
      const noexcept override;

 private:
  [[nodiscard]] StatusOr<std::vector<LinkageEvaluationResult>>
  EvaluatePrevalidatedDirect(
      ArrayView<const LinkageBackendEvaluation> tasks) override;
  [[nodiscard]] StatusOr<std::vector<LinkageEvaluationResult>> EvaluateImpl(
      ArrayView<const LinkageBackendEvaluation> tasks, bool prevalidated);
  struct State;
  std::unique_ptr<State> state_;
};

template <int M>
class CpuRootOnlyLinkageWorkerBackend final : public LinkageWorkerBackend {
 public:
  CpuRootOnlyLinkageWorkerBackend(const BasicCodebookAlgebra<M>& algebra,
                                  const F32GemmBackend& gemm,
                                  const ParallelForExecutor& parallel,
                                  std::uint32_t icm_rounds = 1U) noexcept;
  ~CpuRootOnlyLinkageWorkerBackend() override;

  [[nodiscard]] StatusOr<LinkageBackendCluster> Register(
      LinkageBackendRegistration registration) override;
  [[nodiscard]] Status ApplyUpdates(
      ArrayView<const LinkageBackendUpdate> updates) override;
  [[nodiscard]] StatusOr<std::vector<LinkageEvaluationResult>> Evaluate(
      ArrayView<const LinkageBackendEvaluation> tasks) override;
  [[nodiscard]] Status Unregister(LinkageBackendCluster cluster) override;
  [[nodiscard]] LinkageBackendResourceStats ResourceStats()
      const noexcept override;

 private:
  [[nodiscard]] StatusOr<std::vector<LinkageEvaluationResult>>
  EvaluatePrevalidatedDirect(
      ArrayView<const LinkageBackendEvaluation> tasks) override;
  [[nodiscard]] StatusOr<std::vector<LinkageEvaluationResult>> EvaluateImpl(
      ArrayView<const LinkageBackendEvaluation> tasks, bool prevalidated);
  struct State;
  std::unique_ptr<State> state_;
};

extern template class CpuRsdLinkageWorkerBackend<5>;
extern template class CpuRsdLinkageWorkerBackend<10>;
extern template class CpuRootOnlyLinkageWorkerBackend<5>;
extern template class CpuRootOnlyLinkageWorkerBackend<10>;

}  // namespace stlqpp
