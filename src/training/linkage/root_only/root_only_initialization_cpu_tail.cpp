#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/linkage/root_only/root_only_initialization_cpu_tail.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "stlqpp/training/linkage/topology/cluster_topology_state.h"

namespace stlqpp {
namespace {

constexpr float kAcceptanceEpsilon = 1.0e-6F;
constexpr std::uint32_t kWideRootThreshold = 256U;

template <int M>
struct RootTailPrecompute final {
  std::uint32_t m = M;
  std::uint32_t dimension = 0U;
  std::uint32_t root_domain = 0U;
  std::uint32_t atom_count = 0U;
  std::array<std::uint32_t, (M == 0 ? kMaximumModelLayers : M)> offsets{};
  std::array<std::uint32_t, (M == 0 ? kMaximumModelLayers : M)> layer_sizes{};
  std::vector<float> atoms;
  std::vector<float> gram;
  std::vector<float> inverse_norms;
  std::vector<std::uint8_t> flat_layers;

  const RootTailPrecompute<M>* small = nullptr;

  [[nodiscard]] const float* atom(std::uint32_t flat) const noexcept {
    if (small != nullptr && flat != 0U) return small->atom(flat - 1U);
    return atoms.data() + static_cast<std::size_t>(flat) * dimension;
  }
  [[nodiscard]] float inverse_norm(std::uint32_t flat) const noexcept {
    return small != nullptr && flat != 0U
        ? small->inverse_norms[flat - 1U] : inverse_norms[flat];
  }
  [[nodiscard]] const float* gram_segment(std::uint32_t flat,
                                         std::uint32_t begin) const noexcept {
    if (small == nullptr) return gram.data() + static_cast<std::size_t>(flat) * atom_count + begin;
    if (flat == 0U) return gram.data() + begin;
    if (begin == 0U) return gram.data() + flat;
    return small->gram.data() + static_cast<std::size_t>(flat - 1U) * small->atom_count + begin - 1U;
  }
  [[nodiscard]] float dot(std::uint32_t row,
                          std::uint32_t column) const noexcept {
    return *gram_segment(column, row);
  }

};

template <int M>
StatusOr<RootTailPrecompute<M>> BuildPrecompute(
    const RootCodebooks<M>& codebooks, const F32GemmBackend& gemm,
    std::uint32_t root_begin, std::uint32_t root_count) {
  RootTailPrecompute<M> output;
  const auto m = codebooks.m();
  output.m = m;
  output.dimension = codebooks.shape().dimension;
  output.root_domain = root_count;
  output.layer_sizes[0] = root_count;
  output.offsets[0] = 0U;
  output.atom_count = root_count;
  for (std::uint32_t layer = 1U; layer < m; ++layer) {
    output.offsets[layer] = output.atom_count;
    output.layer_sizes[layer] = 256U;
    output.atom_count += 256U;
  }

  output.atoms.resize(static_cast<std::size_t>(output.dimension) *
                      output.atom_count);
  output.inverse_norms.resize(output.atom_count);
  output.flat_layers.resize(output.atom_count);
  for (std::uint32_t root = 0U; root < root_count; ++root) {
    std::copy_n(codebooks.root0().column(root_begin + root), output.dimension,
                output.atoms.data() +
                    static_cast<std::size_t>(root) * output.dimension);
  }
  for (std::uint32_t layer = 1U; layer < m; ++layer) {
    const ColumnMajorMatrixF32View source = codebooks.root_small(layer);
    std::copy_n(
        source.values, static_cast<std::size_t>(output.dimension) * 256U,
        output.atoms.data() +
            static_cast<std::size_t>(output.offsets[layer]) * output.dimension);
  }
  for (std::uint32_t layer = 0U; layer < m; ++layer) {
    const std::uint32_t end = output.offsets[layer] + output.layer_sizes[layer];
    for (std::uint32_t flat = output.offsets[layer]; flat < end; ++flat) {
      float squared_norm = 0.0F;
      const float* values = output.atom(flat);
      for (std::uint32_t axis = 0U; axis < output.dimension; ++axis) {
        squared_norm += values[axis] * values[axis];
      }
      output.inverse_norms[flat] =
          squared_norm > 0.0F ? 1.0F / std::sqrt(squared_norm) : 0.0F;
      output.flat_layers[flat] = static_cast<std::uint8_t>(layer);
    }
  }

  output.gram.resize(static_cast<std::size_t>(output.atom_count) *
                     output.atom_count);
  const Status status = gemm.MultiplyTransposedLeft({
      .left = {output.atoms.data(), output.dimension, output.atom_count},
      .right = output.atoms.data(),
      .query_count = output.atom_count,
      .right_leading_dimension = output.dimension,
      .output = output.gram.data(),
      .output_leading_dimension = output.atom_count,
  });
  if (!status.ok()) {
    return status;
  }
  return output;
}

template <int M>
StatusOr<RootTailPrecompute<M>> BuildFixedPrecompute(
    const RootCodebooks<M>& codebooks, const F32GemmBackend& gemm,
    std::uint32_t cid, const RootTailPrecompute<M>& small) {
  RootTailPrecompute<M> output;
  output.m = small.m;
  output.dimension = small.dimension;
  output.root_domain = 1U;
  output.atom_count = small.atom_count + 1U;
  output.small = &small;
  output.layer_sizes[0] = 1U;
  for (std::uint32_t layer = 1U; layer < output.m; ++layer) {
    output.offsets[layer] = small.offsets[layer] + 1U;
    output.layer_sizes[layer] = 256U;
  }
  const float* root = codebooks.root0().column(cid);
  output.atoms.assign(root, root + output.dimension);
  output.gram.resize(output.atom_count);
  output.inverse_norms.resize(1U);
  output.flat_layers.resize(output.atom_count);
  float norm = 0.0F;
  for (std::uint32_t axis = 0U; axis < output.dimension; ++axis) norm += root[axis] * root[axis];
  output.gram[0] = norm;
  output.inverse_norms[0] = norm > 0.0F ? 1.0F / std::sqrt(norm) : 0.0F;
  for (std::uint32_t flat = 1U; flat < output.atom_count; ++flat)
    output.flat_layers[flat] = small.flat_layers[flat - 1U];
  const Status status = gemm.MultiplyTransposedLeft({
      .left = {small.atoms.data(), small.dimension, small.atom_count},
      .right = root, .query_count = 1U,
      .right_leading_dimension = output.dimension,
      .output = output.gram.data() + 1U,
      .output_leading_dimension = small.atom_count});
  if (!status.ok()) return status;
  return output;
}

template <int M, std::uint32_t Capacity>
struct RootTailSolution final {
  std::array<std::uint32_t, Capacity> codes{};
  std::array<float, Capacity> coefficients{};
  float squared_error = std::numeric_limits<float>::infinity();
};

template <int M>
void Project(const RootTailPrecompute<M>& precompute, const float* vector,
             float* output) noexcept {
  for (std::uint32_t flat = 0U; flat < precompute.atom_count; ++flat) {
    const float* atom = precompute.atom(flat);
    float dot = 0.0F;
#pragma omp simd reduction(+ : dot)
    for (std::uint32_t axis = 0U; axis < precompute.dimension; ++axis) {
      dot += atom[axis] * vector[axis];
    }
    output[flat] = dot;
  }
}

template <int M>
std::uint32_t InternalCode(const RootTailPrecompute<M>& precompute,
                           const LinkageCodeTuple<M>& support,
                           std::uint32_t layer) noexcept {
  if (layer != 0U) {
    return support.trailing[layer - 1U];
  }
  return precompute.root_domain == 1U ? 0U : support.first;
}

template <int M, std::uint32_t Capacity>
void GreedySupport(const RootTailPrecompute<M>& precompute,
                   const float* residual,
                   const std::vector<float>* initial_projection,
                   RootTailSolution<M, Capacity>* solution,
                   std::vector<float>* remaining) {
  remaining->resize(precompute.atom_count);
  if (initial_projection != nullptr) {
    std::copy(initial_projection->begin(), initial_projection->end(),
              remaining->begin());
  } else {
    Project(precompute, residual, remaining->data());
  }
  std::uint32_t available = (1U << precompute.m) - 1U;
  for (std::uint32_t step = 0U; step < precompute.m; ++step) {
    std::uint32_t best_flat = 0U;
    float best_absolute = -std::numeric_limits<float>::infinity();
    float best_adjusted = 0.0F;
    for (std::uint32_t flat = 0U; flat < precompute.atom_count; ++flat) {
      const std::uint32_t layer = precompute.flat_layers[flat];
      if ((available & (1U << layer)) == 0U) {
        continue;
      }
      const float adjusted =
          (*remaining)[flat] * precompute.inverse_norm(flat);
      const float absolute = std::fabs(adjusted);
      if (absolute > best_absolute) {
        best_absolute = absolute;
        best_adjusted = adjusted;
        best_flat = flat;
      }
    }
    const std::uint32_t layer = precompute.flat_layers[best_flat];
    solution->codes[layer] = best_flat - precompute.offsets[layer];
    solution->coefficients[layer] =
        best_adjusted * precompute.inverse_norm(best_flat);
    available &= ~(1U << layer);
    if (step + 1U == precompute.m) break;
    for (std::uint32_t next_layer = 0U; next_layer < precompute.m;
         ++next_layer) {
      if ((available & (1U << next_layer)) == 0U) continue;
      const std::uint32_t begin = precompute.offsets[next_layer];
      const std::uint32_t end = begin + precompute.layer_sizes[next_layer];
      const float* gram = precompute.gram_segment(best_flat, begin);
#pragma omp simd
      for (std::uint32_t flat = begin; flat < end; ++flat) {
        (*remaining)[flat] -= solution->coefficients[layer] * gram[flat - begin];
      }
    }
  }
}

template <int M, std::uint32_t Capacity>
StatusOr<bool> SolveSupport(const RootTailPrecompute<M>& precompute,
                            const SpdSolverF32& solver,
                            const std::vector<float>& adjusted_projection,
                            RootTailSolution<M, Capacity>* solution) {
  std::array<float, Capacity * Capacity> matrix{};
  std::array<float, Capacity> right_hand_side{};
  float bump = 0.0F;
  for (std::uint32_t attempt = 0U; attempt < 4U; ++attempt) {
    matrix.fill(0.0F);
    for (std::uint32_t column = 0U; column < precompute.m; ++column) {
      const std::uint32_t column_flat =
          precompute.offsets[column] + solution->codes[column];
      right_hand_side[column] = adjusted_projection[column_flat];
      for (std::uint32_t row = 0U; row <= column; ++row) {
        const std::uint32_t row_flat =
            precompute.offsets[row] + solution->codes[row];
        matrix[static_cast<std::size_t>(column) * precompute.m + row] =
            precompute.dot(row_flat, column_flat);
      }
      matrix[static_cast<std::size_t>(column) * (precompute.m + 1U)] += bump;
    }
    auto solved = solver.FactorAndSolve({
        .matrix = matrix.data(),
        .order = precompute.m,
        .matrix_leading_dimension = precompute.m,
        .right_hand_sides = right_hand_side.data(),
        .right_hand_side_count = 1U,
        .rhs_leading_dimension = precompute.m,
    });
    if (!solved.ok()) {
      return solved.status();
    }
    if (solved.value() == SpdSolveOutcome::kSolved) {
      std::copy_n(right_hand_side.data(), precompute.m,
                  solution->coefficients.data());
      return true;
    }
    bump = bump == 0.0F ? kAcceptanceEpsilon : bump * 10.0F;
  }
  return false;
}

template <int M, std::uint32_t Capacity>
float SquaredCost(const RootTailPrecompute<M>& precompute,
                  const std::vector<float>& adjusted_projection,
                  const RootTailSolution<M, Capacity>& solution,
                  float residual_squared_norm) noexcept {
  float data_dot = 0.0F;
  float reconstruction_norm = 0.0F;
  for (std::uint32_t left = 0U; left < precompute.m; ++left) {
    const std::uint32_t left_flat =
        precompute.offsets[left] + solution.codes[left];
    data_dot += solution.coefficients[left] * adjusted_projection[left_flat];
    for (std::uint32_t right = 0U; right < precompute.m; ++right) {
      const std::uint32_t right_flat =
          precompute.offsets[right] + solution.codes[right];
      reconstruction_norm += solution.coefficients[left] *
                             solution.coefficients[right] *
                             precompute.dot(left_flat, right_flat);
    }
  }
  return residual_squared_norm - 2.0F * data_dot + reconstruction_norm;
}

template <int M, std::uint32_t Capacity>
void BuildRemaining(const RootTailPrecompute<M>& precompute,
                    const std::vector<float>& adjusted_projection,
                    const RootTailSolution<M, Capacity>& solution,
                    std::vector<float>* remaining) {
  *remaining = adjusted_projection;
  for (std::uint32_t layer = 0U; layer < precompute.m; ++layer) {
    const std::uint32_t flat =
        precompute.offsets[layer] + solution.codes[layer];
    for (std::uint32_t target = 0U; target < precompute.m; ++target) {
      const auto begin = precompute.offsets[target];
      const auto end = begin + precompute.layer_sizes[target];
      const float* gram = precompute.gram_segment(flat, begin);
#pragma omp simd
      for (std::uint32_t atom = begin; atom < end; ++atom)
        (*remaining)[atom] -= solution.coefficients[layer] * gram[atom - begin];
    }
  }
}

template <int M, std::uint32_t Capacity>
StatusOr<bool> RefineOneIcmRound(const RootTailPrecompute<M>& precompute,
                                 const SpdSolverF32& solver,
                                 const std::vector<float>& adjusted_projection,
                                 float residual_squared_norm,
                                 std::mt19937* random,
                                 RootTailSolution<M, Capacity>* solution,
                                 std::vector<float>* remaining) {
  solution->squared_error = SquaredCost(precompute, adjusted_projection,
                                        *solution, residual_squared_norm);
  BuildRemaining(precompute, adjusted_projection, *solution, remaining);
  std::array<std::uint32_t, Capacity> order{};
  std::iota(order.begin(), order.begin() + precompute.m, 0U);
  std::shuffle(order.begin(), order.begin() + precompute.m, *random);
  for (std::uint32_t position = 0U; position < precompute.m; ++position) {
    const std::uint32_t layer = order[position];
    const std::uint32_t old_code = solution->codes[layer];
    const std::uint32_t old_flat = precompute.offsets[layer] + old_code;
    std::uint32_t best_code = old_code;
    float best_absolute = -std::numeric_limits<float>::infinity();
    const std::uint32_t end =
        precompute.offsets[layer] + precompute.layer_sizes[layer];
    for (std::uint32_t flat = precompute.offsets[layer]; flat < end; ++flat) {
      const float adjusted =
          ((*remaining)[flat] +
           solution->coefficients[layer] * precompute.dot(flat, old_flat)) *
          precompute.inverse_norm(flat);
      const float absolute = std::fabs(adjusted);
      if (absolute > best_absolute) {
        best_absolute = absolute;
        best_code = flat - precompute.offsets[layer];
      }
    }
    if (best_code == old_code) {
      continue;
    }

    RootTailSolution<M, Capacity> trial = *solution;
    trial.codes[layer] = best_code;
    auto solved = SolveSupport(precompute, solver, adjusted_projection, &trial);
    if (!solved.ok()) {
      return solved.status();
    }
    if (solved.value()) {
      trial.squared_error = SquaredCost(precompute, adjusted_projection, trial,
                                        residual_squared_norm);
      if (trial.squared_error + kAcceptanceEpsilon < solution->squared_error) {
        *solution = trial;
        if (position + 1U < precompute.m) {
          BuildRemaining(precompute, adjusted_projection, *solution, remaining);
        }
      }
    }
  }
  return std::isfinite(solution->squared_error);
}

template <int M, std::uint32_t Capacity>
LinkageCodeTuple<M> ToPersistentSupport(const RootTailPrecompute<M>& precompute,
                                        const RootTailSolution<M, Capacity>& solution,
                                        std::uint32_t cid) noexcept {
  LinkageCodeTuple<M> output;
  output.first = precompute.root_domain == 1U ? cid : solution.codes[0];
  for (std::uint32_t layer = 1U; layer < precompute.m; ++layer) {
    output.trailing[layer - 1U] =
        static_cast<std::uint8_t>(solution.codes[layer]);
  }
  return output;
}

}  // namespace

template <int M>
struct RootOnlyInitializationCpuTailModel<M>::State final {
  const RootCodebooks<M>* codebooks = nullptr;
  std::unique_ptr<RootTailPrecompute<M>> dense;
  std::unique_ptr<RootTailPrecompute<M>> small;
};

template <int M>
struct RootOnlyInitializationCpuTailEvaluator<M>::State final {
  const RootOnlyInitializationCpuTailModel<M>* model = nullptr;
  const F32GemmBackend* gemm = nullptr;
  const SpdSolverF32* solver = nullptr;
  std::uint32_t icm_rounds = 1U;
  std::uint32_t cached_cid = std::numeric_limits<std::uint32_t>::max();
  std::unique_ptr<RootTailPrecompute<M>> fixed;
  std::vector<float> residual;
  std::vector<float> base_projection;
  std::vector<float> adjusted_projection;
  std::vector<float> remaining;
};

template <int M>
StatusOr<std::unique_ptr<RootOnlyInitializationCpuTailModel<M>>>
RootOnlyInitializationCpuTailModel<M>::Create(const RootCodebooks<M>& codebooks,
                                              const F32GemmBackend& gemm) {
  auto state = std::make_unique<State>();
  state->codebooks = &codebooks;
  if (codebooks.shape().nlist <= kWideRootThreshold) {
    auto dense = BuildPrecompute(codebooks, gemm, 0U, codebooks.shape().nlist);
    if (!dense.ok()) {
      return dense.status();
    }
    state->dense =
        std::make_unique<RootTailPrecompute<M>>(std::move(dense).value());
  } else {
    auto small = BuildPrecompute(codebooks, gemm, 0U, 0U);
    if (!small.ok()) return small.status();
    state->small = std::make_unique<RootTailPrecompute<M>>(std::move(small).value());
  }
  return std::unique_ptr<RootOnlyInitializationCpuTailModel<M>>(
      new RootOnlyInitializationCpuTailModel<M>(std::move(state)));
}

template <int M>
RootOnlyInitializationCpuTailModel<M>::RootOnlyInitializationCpuTailModel(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
RootOnlyInitializationCpuTailModel<M>::~RootOnlyInitializationCpuTailModel() =
    default;

template <int M>
RootOnlyInitializationCpuTailModel<M>::RootOnlyInitializationCpuTailModel(
    RootOnlyInitializationCpuTailModel&&) noexcept = default;

template <int M>
RootOnlyInitializationCpuTailModel<M>&
RootOnlyInitializationCpuTailModel<M>::operator=(
    RootOnlyInitializationCpuTailModel&&) noexcept = default;

template <int M>
RootOnlyInitializationCpuTailEvaluator<M>::
    RootOnlyInitializationCpuTailEvaluator(
        const RootOnlyInitializationCpuTailModel<M>& model,
        const F32GemmBackend& gemm, const SpdSolverF32& solver,
        std::uint32_t icm_rounds) noexcept
    : state_(std::make_unique<State>()) {
  state_->model = &model;
  state_->gemm = &gemm;
  state_->solver = &solver;
  state_->icm_rounds = icm_rounds;
}

template <int M>
RootOnlyInitializationCpuTailEvaluator<
    M>::~RootOnlyInitializationCpuTailEvaluator() = default;

template <int M>
StatusOr<LinkageEvaluationResult>
RootOnlyInitializationCpuTailEvaluator<M>::Evaluate(
    const ClusterTopologyState<M>& topology, const LinkageEvaluationTask& task,
    std::uint32_t cluster_seed, std::uint32_t real_local_id) {
  if constexpr (M == 0) {
    if (state_->model->state_->codebooks->m() > kLegacyRuntimeScratchLayers)
      return EvaluateWithCapacity<kWideRuntimeScratchLayers>(topology, task, cluster_seed, real_local_id);
  }
  return EvaluateWithCapacity<kModelScratchLayers<M>>(topology, task, cluster_seed, real_local_id);
}

template <int M>
template <std::uint32_t Capacity>
StatusOr<LinkageEvaluationResult>
RootOnlyInitializationCpuTailEvaluator<M>::EvaluateWithCapacity(
    const ClusterTopologyState<M>& topology, const LinkageEvaluationTask& task,
    std::uint32_t cluster_seed, std::uint32_t real_local_id) {
  assert(task.kind == LinkageEvaluationKind::kRootOnlyInitialization);
  assert(task.node_count() == 1U);
  assert(task.pair_count() != 0U);
  assert(task.dimension == topology.dimension());
  assert(real_local_id < topology.real_count());

  const RootTailPrecompute<M>* precompute = state_->model->state_->dense.get();
  if (precompute == nullptr) {
    if (state_->cached_cid != topology.cid()) {
      auto fixed = BuildFixedPrecompute(*state_->model->state_->codebooks,
          *state_->gemm, topology.cid(), *state_->model->state_->small);
      if (!fixed.ok()) {
        return fixed.status();
      }
      state_->fixed =
          std::make_unique<RootTailPrecompute<M>>(std::move(fixed).value());
      state_->cached_cid = topology.cid();
    }
    precompute = state_->fixed.get();
  }

  const float* exact = task.node_vectors.data();
  if (state_->model->state_->dense != nullptr) {
    state_->base_projection.resize(precompute->atom_count);
    Project(*precompute, exact, state_->base_projection.data());
  }
  state_->residual.resize(precompute->dimension);
  state_->adjusted_projection.resize(precompute->atom_count);
  std::mt19937 random(cluster_seed ^ (real_local_id * 0x9E3779B9U) ^
                      0xA7F1C3D5U);

  RootTailSolution<M, Capacity> best;
  std::uint32_t best_column = 0U;
  std::uint32_t best_pair = std::numeric_limits<std::uint32_t>::max();
  for (std::uint32_t pair = task.row_offsets[0]; pair < task.row_offsets[1];
       ++pair) {
    const std::uint32_t column = task.candidate_columns[pair];
    assert(column >= topology.virtual_count());
    const std::uint32_t parent_local = column - topology.virtual_count();
    const ArrayView<const float> parent = topology.parent_reconstruction(
        {LinkageParentKind::kReal, parent_local});
    float residual_squared_norm = 0.0F;
#pragma omp simd reduction(+ : residual_squared_norm)
    for (std::uint32_t axis = 0U; axis < precompute->dimension; ++axis) {
      const float value = exact[axis] - parent[axis];
      state_->residual[axis] = value;
      residual_squared_norm += value * value;
    }

    if (state_->model->state_->dense != nullptr) {
      std::copy(state_->base_projection.begin(), state_->base_projection.end(),
                state_->adjusted_projection.begin());
      std::uint32_t local = parent_local;
      for (;;) {
        const LinkageCodeTuple<M>& support = topology.support(local);
        const ArrayView<const float> coefficients =
            topology.coefficients(local);
        for (std::uint32_t layer = 0U; layer < precompute->m; ++layer) {
          const float coefficient = coefficients[layer];
          if (coefficient == 0.0F) {
            continue;
          }
          const std::uint32_t flat = precompute->offsets[layer] +
                                     InternalCode(*precompute, support, layer);
          const float* gram = precompute->gram_segment(flat, 0U);
#pragma omp simd
          for (std::uint32_t atom = 0U; atom < precompute->atom_count; ++atom) {
            state_->adjusted_projection[atom] -= coefficient * gram[atom];
          }
        }
        const auto& ancestor = topology.parent(local);
        if (!ancestor.has_value()) {
          break;
        }
        assert(ancestor->kind == LinkageParentKind::kReal);
        local = ancestor->local_id;
      }
    } else {
      Project(*precompute, state_->residual.data(),
              state_->adjusted_projection.data());
    }

    RootTailSolution<M, Capacity> candidate;
    GreedySupport(*precompute, state_->residual.data(),
                  state_->model->state_->dense == nullptr
                      ? &state_->adjusted_projection
                      : nullptr,
                  &candidate, &state_->remaining);
    auto solved = SolveSupport(*precompute, *state_->solver,
                               state_->adjusted_projection, &candidate);
    if (!solved.ok()) {
      return solved.status();
    }
    if (!solved.value()) {
      continue;
    }
    if (state_->icm_rounds == 0U) {
      candidate.squared_error =
          SquaredCost(*precompute, state_->adjusted_projection, candidate,
                      residual_squared_norm);
      if (!std::isfinite(candidate.squared_error)) {
        continue;
      }
    } else {
      bool valid = true;
      for (std::uint32_t round = 0U; round < state_->icm_rounds; ++round) {
        auto refined = RefineOneIcmRound(
            *precompute, *state_->solver, state_->adjusted_projection,
            residual_squared_norm, &random, &candidate, &state_->remaining);
        if (!refined.ok()) {
          return refined.status();
        }
        if (!refined.value()) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        continue;
      }
    }
    if (best_pair == std::numeric_limits<std::uint32_t>::max() ||
        candidate.squared_error < best.squared_error) {
      best = candidate;
      best_column = column;
      best_pair = pair;
    }
  }
  if (best_pair == std::numeric_limits<std::uint32_t>::max()) {
    return Status::FailedPrecondition(
        "root-only tiny CPU tail contains no nonsingular support");
  }

  LinkageEvaluationResult output;
  output.parent_columns.push_back(best_column);
  output.supports.resize(1U, precompute->m);
  output.supports.set_tuple(
      0U, ToPersistentSupport(*precompute, best, topology.cid()));
  output.coefficients.insert(output.coefficients.end(),
                             best.coefficients.begin(),
                             best.coefficients.begin() + precompute->m);
  output.squared_errors.push_back(best.squared_error);
  return output;
}

template class RootOnlyInitializationCpuTailModel<0>;
template class RootOnlyInitializationCpuTailEvaluator<0>;
template class RootOnlyInitializationCpuTailModel<5>;
template class RootOnlyInitializationCpuTailModel<12>;
template class RootOnlyInitializationCpuTailModel<6>;
template class RootOnlyInitializationCpuTailModel<10>;
template class RootOnlyInitializationCpuTailEvaluator<5>;
template class RootOnlyInitializationCpuTailEvaluator<12>;
template class RootOnlyInitializationCpuTailEvaluator<6>;
template class RootOnlyInitializationCpuTailEvaluator<10>;

}  // namespace stlqpp
