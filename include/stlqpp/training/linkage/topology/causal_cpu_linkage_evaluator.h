#pragma once

#include <cstdint>
#include <vector>

#include "stlqpp/core/status_or.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/training/least_squares_cost_mode.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/scheduler/linkage_evaluation.h"
#include "stlqpp/training/linkage/topology/topology_random_stream.h"

namespace stlqpp {

template <int M>
class ClusterTopologyState;

// CPU endpoint for tiny forced batches and resource fallback. Both use the
// configured RSD/refinement; tiny batches retain per-pair cluster ICM draws,
// resource fallback uses the supplied task order. Each worker owns its scratch.
template <int M>
class CausalCpuLinkageEvaluator final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  struct Scratch final {
    std::vector<float> residual;
    std::vector<float> residual_atom_dots;
  };

  [[nodiscard]] static StatusOr<CausalCpuLinkageEvaluator<M>> Create(
      const ProductCodebooks<M>& codebooks,
      const LinkageCodebookAlgebra<M>& edge_algebra,
      std::uint32_t maximum_rank = 2U,
      LeastSquaresCostMode cost_mode = LeastSquaresCostMode::kNormalEquation,
      std::uint32_t icm_rounds = 1U);

  CausalCpuLinkageEvaluator(CausalCpuLinkageEvaluator&&) noexcept = default;
  CausalCpuLinkageEvaluator& operator=(CausalCpuLinkageEvaluator&&) noexcept =
      default;
  CausalCpuLinkageEvaluator(const CausalCpuLinkageEvaluator&) = delete;
  CausalCpuLinkageEvaluator& operator=(const CausalCpuLinkageEvaluator&) =
      delete;

  [[nodiscard]] Scratch MakeScratch() const;

  // Visibility, depth and candidate order are established by the caller.
  // The resource endpoint never synchronizes or consumes topology RNG.
  [[nodiscard]] StatusOr<LinkageEvaluationResult> Evaluate(
      const ClusterTopologyState<M>& state, const LinkageEvaluationTask& task,
      Scratch* scratch) const;

  // The tiny endpoint draws one order for each successfully fitted pair when
  // refinement is enabled, preserving the cluster stream for later topology.
  [[nodiscard]] StatusOr<LinkageEvaluationResult> EvaluateTinyTail(
      const ClusterTopologyState<M>& state, const LinkageEvaluationTask& task,
      TopologyRandomStream& stream, Scratch* scratch) const;

 private:
  CausalCpuLinkageEvaluator() = default;
  [[nodiscard]] StatusOr<LinkageEvaluationResult> EvaluateImpl(
      const ClusterTopologyState<M>&, const LinkageEvaluationTask&,
      TopologyRandomStream*, Scratch*) const;

  template <std::uint32_t Capacity>
  [[nodiscard]] StatusOr<LinkageEvaluationResult> EvaluateWithCapacity(
      const ClusterTopologyState<M>&, const LinkageEvaluationTask&,
      TopologyRandomStream*, Scratch*) const;

  const ProductCodebooks<M>* codebooks_ = nullptr;
  const LinkageCodebookAlgebra<M>* edge_algebra_ = nullptr;
  std::uint32_t maximum_rank_ = 2U;
  LeastSquaresCostMode cost_mode_ = LeastSquaresCostMode::kNormalEquation;
  std::uint32_t icm_rounds_ = 1U;
};

extern template class CausalCpuLinkageEvaluator<5>;
extern template class CausalCpuLinkageEvaluator<10>;

}  // namespace stlqpp
