#include "stlqpp/training/linkage/topology/causal_cpu_linkage_evaluator.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "stlqpp/training/linkage/rsd/detail/linkage_rsd_kernel.h"
#include "stlqpp/training/linkage/rsd/linkage_support.h"
#include "stlqpp/training/linkage/rsd/ranked_single_deviation.h"
#include "stlqpp/training/linkage/topology/cluster_topology_state.h"

namespace stlqpp {
namespace {

template <int M, std::uint32_t Capacity>
LinkageCodeTuple<M> ToCodeTuple(const LinkageSupport<M, Capacity>& support,
                                std::uint32_t m) noexcept {
  LinkageCodeTuple<M> output;
  output.first = support.codes[0];
  for (std::uint32_t layer = 1U; layer < m; ++layer) {
    output.trailing[layer - 1U] = support.codes[layer];
  }
  return output;
}

}  // namespace

template <int M>
StatusOr<CausalCpuLinkageEvaluator<M>> CausalCpuLinkageEvaluator<M>::Create(
    const ProductCodebooks<M>& codebooks,
    const LinkageCodebookAlgebra<M>& edge_algebra,
    std::uint32_t maximum_rank, LeastSquaresCostMode cost_mode,
    std::uint32_t icm_rounds) {
  if (codebooks.shape().dimension != edge_algebra.shape().dimension ||
      codebooks.shape().nlist != edge_algebra.shape().nlist ||
      codebooks.shape().m != edge_algebra.shape().m ||
      !IsSupportedRsdRank(maximum_rank)) {
    return Status::InvalidArgument(
        "causal CPU evaluator model components disagree");
  }

  CausalCpuLinkageEvaluator<M> output;
  output.codebooks_ = &codebooks;
  output.edge_algebra_ = &edge_algebra;
  output.maximum_rank_ = maximum_rank;
  output.cost_mode_ = cost_mode;
  output.icm_rounds_ = icm_rounds;
  return output;
}

template <int M>
typename CausalCpuLinkageEvaluator<M>::Scratch
CausalCpuLinkageEvaluator<M>::MakeScratch() const {
  Scratch output;
  const std::size_t atoms = edge_algebra_->atom_count();
  output.residual.resize(codebooks_->shape().dimension);
  output.residual_atom_dots.resize(atoms);
  return output;
}

template <int M>
StatusOr<LinkageEvaluationResult> CausalCpuLinkageEvaluator<M>::Evaluate(
    const ClusterTopologyState<M>& state, const LinkageEvaluationTask& task,
    Scratch* scratch) const {
  return EvaluateImpl(state, task, nullptr, scratch);
}

template <int M>
StatusOr<LinkageEvaluationResult> CausalCpuLinkageEvaluator<M>::EvaluateTinyTail(
    const ClusterTopologyState<M>& state, const LinkageEvaluationTask& task,
    TopologyRandomStream& stream, Scratch* scratch) const {
  return EvaluateImpl(state, task,
                      &stream, scratch);
}

template <int M>
StatusOr<LinkageEvaluationResult> CausalCpuLinkageEvaluator<M>::EvaluateImpl(
    const ClusterTopologyState<M>& state, const LinkageEvaluationTask& task,
    TopologyRandomStream* stream, Scratch* scratch) const {
  if constexpr (M == 0) {
    if (codebooks_->shape().m > kLegacyRuntimeScratchLayers)
      return EvaluateWithCapacity<kWideRuntimeScratchLayers>(state, task, stream, scratch);
  }
  return EvaluateWithCapacity<kModelScratchLayers<M>>(state, task, stream, scratch);
}

template <int M>
template <std::uint32_t Capacity>
StatusOr<LinkageEvaluationResult> CausalCpuLinkageEvaluator<M>::EvaluateWithCapacity(
    const ClusterTopologyState<M>& state, const LinkageEvaluationTask& task,
    TopologyRandomStream* stream, Scratch* scratch) const {
  assert(task.kind == LinkageEvaluationKind::kEdgeRsd);
  assert(task.node_count() != 0U);
  assert(scratch != nullptr);
  const std::uint32_t m = codebooks_->shape().m;
  const std::uint32_t kAtoms = m * 256U;
  const std::uint32_t dimension = codebooks_->shape().dimension;
  const ColumnMajorMatrixF32View atoms = edge_algebra_->atoms();

  LinkageEvaluationResult output;
  const std::uint32_t nodes = task.node_count();
  output.parent_columns.reserve(nodes);
  output.supports.resize(nodes, m);
  output.coefficients.reserve(static_cast<std::size_t>(nodes) * m);
  output.squared_errors.reserve(nodes);
  for (std::uint32_t row = 0U; row < nodes; ++row) {
    const float* exact =
        task.node_vectors.data() + static_cast<std::size_t>(row) * dimension;
    LinkageSolution<M, Capacity> best;
    best.squared_error = std::numeric_limits<float>::infinity();
    std::uint32_t best_column = 0U;
    bool solved = false;
    const std::uint32_t begin = task.row_offsets[row];
    const std::uint32_t end = task.row_offsets[row + 1U];
    for (std::uint32_t pair = begin; pair < end; ++pair) {
      const std::uint32_t column = task.candidate_columns[pair];
      const LinkageParentReference parent_reference =
          column < state.virtual_count()
              ? LinkageParentReference{LinkageParentKind::kVirtual, column}
              : LinkageParentReference{LinkageParentKind::kReal,
                                       column - state.virtual_count()};
      const ArrayView<const float> parent =
          state.parent_reconstruction(parent_reference);
#pragma omp simd
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        scratch->residual[axis] = exact[axis] - parent[axis];
      }
      float residual_squared_norm = 0.0F;
      const float* residual = scratch->residual.data();
#pragma omp simd reduction(+ : residual_squared_norm)
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        residual_squared_norm += residual[axis] * residual[axis];
      }
      for (std::uint32_t atom = 0U; atom < kAtoms; ++atom) {
        const float* codeword = atoms.column(atom);
        float dot = 0.0F;
#pragma omp simd reduction(+ : dot)
        for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
          dot += codeword[axis] * residual[axis];
        }
        scratch->residual_atom_dots[atom] = dot;
      }
      LinkageSolution<M, Capacity> candidate;
      // Preserve the configured RSD search; tiny-tail refinement consumes the
      // historical per-pair cluster stream after a successful fit.
      const bool encoded = detail::FitRsdOne(
          *edge_algebra_, scratch->residual_atom_dots.data(),
          residual_squared_norm, &candidate, maximum_rank_, cost_mode_);
      if (encoded && icm_rounds_ != 0U) {
        std::array<std::uint8_t, Capacity> sampled_order{};
        const std::uint8_t* order = task.icm_layer_orders.data() +
            static_cast<std::size_t>(row) * m;
        if (stream != nullptr) {
          stream->FillIcmLayerOrder(m, sampled_order.data());
          order = sampled_order.data();
        }
        for (std::uint32_t round = 0U; round < icm_rounds_; ++round) {
          detail::RefineRsdOne(*edge_algebra_, scratch->residual_atom_dots.data(),
                              residual_squared_norm, order, &candidate, cost_mode_);
        }
      }
      if (!encoded) {
        continue;
      }
      if (!solved || candidate.squared_error < best.squared_error) {
        best = candidate;
        best_column = column;
        solved = true;
      }
    }
    if (!solved) {
      return Status::FailedPrecondition(
          "causal CPU tail contains no nonsingular support");
    }
    output.parent_columns.push_back(best_column);
    output.supports.set_tuple(row, ToCodeTuple(best.support, m));
    output.coefficients.insert(output.coefficients.end(),
                               best.coefficients.begin(),
                               best.coefficients.begin() + m);
    output.squared_errors.push_back(best.squared_error);
  }
  return output;
}

template class CausalCpuLinkageEvaluator<0>;
template class CausalCpuLinkageEvaluator<5>;
template class CausalCpuLinkageEvaluator<12>;
template class CausalCpuLinkageEvaluator<6>;
template class CausalCpuLinkageEvaluator<10>;

}  // namespace stlqpp
