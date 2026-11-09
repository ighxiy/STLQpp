#pragma once

#include <cstdint>
#include <memory>

#include "stlqpp/core/status_or.h"
#include "stlqpp/model/root_codebooks.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/spd_solver_f32.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_tail_evaluator.h"

namespace stlqpp {

template <int M>
class RootOnlyInitializationCpuTailModel final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<
      std::unique_ptr<RootOnlyInitializationCpuTailModel<M>>>
  Create(const RootCodebooks<M>& codebooks, const F32GemmBackend& gemm);

  ~RootOnlyInitializationCpuTailModel();
  RootOnlyInitializationCpuTailModel(
      RootOnlyInitializationCpuTailModel&&) noexcept;
  RootOnlyInitializationCpuTailModel& operator=(
      RootOnlyInitializationCpuTailModel&&) noexcept;
  RootOnlyInitializationCpuTailModel(
      const RootOnlyInitializationCpuTailModel&) = delete;
  RootOnlyInitializationCpuTailModel& operator=(
      const RootOnlyInitializationCpuTailModel&) = delete;

 private:
  template <int>
  friend class RootOnlyInitializationCpuTailEvaluator;
  struct State;
  explicit RootOnlyInitializationCpuTailModel(
      std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

// One instance belongs to one outer cluster worker. Small atoms/Gram belong
// to the immutable model. Large-IVF workers cache only the current root and
// its cross products; no worker copies or rebuilds the shared small Gram.
template <int M>
class RootOnlyInitializationCpuTailEvaluator final
    : public RootOnlyInitializationTailEvaluator<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  RootOnlyInitializationCpuTailEvaluator(
      const RootOnlyInitializationCpuTailModel<M>& model,
      const F32GemmBackend& gemm, const SpdSolverF32& solver,
      std::uint32_t icm_rounds = 1U) noexcept;
  ~RootOnlyInitializationCpuTailEvaluator() override;
  RootOnlyInitializationCpuTailEvaluator(
      const RootOnlyInitializationCpuTailEvaluator&) = delete;
  RootOnlyInitializationCpuTailEvaluator& operator=(
      const RootOnlyInitializationCpuTailEvaluator&) = delete;

  [[nodiscard]] StatusOr<LinkageEvaluationResult> Evaluate(
      const ClusterTopologyState<M>& state, const LinkageEvaluationTask& task,
      std::uint32_t cluster_seed, std::uint32_t real_local_id) override;

 private:
  template <std::uint32_t Capacity>
  [[nodiscard]] StatusOr<LinkageEvaluationResult> EvaluateWithCapacity(
      const ClusterTopologyState<M>&, const LinkageEvaluationTask&,
      std::uint32_t cluster_seed, std::uint32_t real_local_id);
  struct State;
  std::unique_ptr<State> state_;
};

extern template class RootOnlyInitializationCpuTailModel<5>;
extern template class RootOnlyInitializationCpuTailModel<10>;
extern template class RootOnlyInitializationCpuTailEvaluator<5>;
extern template class RootOnlyInitializationCpuTailEvaluator<10>;

}  // namespace stlqpp
