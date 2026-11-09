#include "stlqpp/training/linkage/scheduler/cpu_linkage_worker_backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stlqpp/training/linkage/root_only/cpu_root_only_linkage_encoder.h"
#include "stlqpp/training/linkage/rsd/cpu_linkage_rsd_encoder.h"
#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"
#include "stlqpp/training/linkage/rsd/linkage_projection.h"

namespace stlqpp {
namespace {

struct CpuRegisteredCluster final {
  std::uint32_t dimension = 0U;
  std::uint32_t columns = 0U;
  std::uint64_t generation = 0U;
  std::vector<float> reconstructions;
};

Status ValidateRegistration(const LinkageBackendRegistration& registration) {
  if (registration.dimension == 0U || registration.column_count == 0U ||
      registration.reconstructions.size() !=
          static_cast<std::size_t>(registration.dimension) *
              registration.column_count) {
    return Status::InvalidArgument(
        "CPU linkage registration has an invalid matrix");
  }
  return Status::Ok();
}

Status ApplyCpuUpdates(
    std::unordered_map<LinkageBackendCluster, CpuRegisteredCluster>* clusters,
    ArrayView<const LinkageBackendUpdate> updates) {
  for (const LinkageBackendUpdate& update : updates) {
    const auto found = clusters->find(update.cluster);
    if (found == clusters->end()) {
      return Status::FailedPrecondition(
          "CPU linkage update references an unavailable cluster");
    }
    CpuRegisteredCluster& cluster = found->second;
    if (update.generation != cluster.generation + 1U ||
        update.dimension != cluster.dimension || update.columns.empty() ||
        update.reconstructions.size() !=
            static_cast<std::size_t>(cluster.dimension) *
                update.columns.size()) {
      return Status::InvalidArgument(
          "CPU linkage update has an invalid generation or matrix");
    }
    for (std::uint32_t index = 0U; index < update.columns.size(); ++index) {
      const std::uint32_t column = update.columns[index];
      if (column >= cluster.columns) {
        return Status::InvalidArgument(
            "CPU linkage update column is outside registration");
      }
      std::copy_n(update.reconstructions.data() +
                      static_cast<std::size_t>(index) * cluster.dimension,
                  cluster.dimension,
                  cluster.reconstructions.data() +
                      static_cast<std::size_t>(column) * cluster.dimension);
    }
    cluster.generation = update.generation;
  }
  return Status::Ok();
}

Status ValidateEvaluationState(const LinkageBackendEvaluation& evaluation,
                               LinkageEvaluationKind expected,
                               const CpuRegisteredCluster& cluster) {
  const LinkageEvaluationTask& task = evaluation.task;
  if (task.kind != expected || task.dimension != cluster.dimension ||
      cluster.generation < task.required_generation) {
    return Status::FailedPrecondition(
        "CPU linkage evaluation references stale or incompatible state");
  }
  return Status::Ok();
}

Status ValidateEvaluation(const LinkageBackendEvaluation& evaluation,
                          LinkageEvaluationKind expected,
                          const CpuRegisteredCluster& cluster) {
  const LinkageEvaluationTask& task = evaluation.task;
  if (const Status status =
          ValidateEvaluationState(evaluation, expected, cluster);
      !status.ok()) {
    return status;
  }
  if (task.node_count() == 0U || task.row_offsets.front() != 0U ||
      task.row_offsets.back() != task.pair_count() ||
      task.node_vectors.size() !=
          static_cast<std::size_t>(task.dimension) * task.node_count()) {
    return Status::FailedPrecondition(
        "CPU linkage evaluation references stale or incompatible state");
  }
  for (std::uint32_t pair = 0U; pair < task.pair_count(); ++pair) {
    if (task.candidate_columns[pair] >= cluster.columns) {
      return Status::InvalidArgument(
          "CPU linkage candidate is outside registration");
    }
  }
  for (std::uint32_t node = 0U; node < task.node_count(); ++node) {
    if (task.row_offsets[node] > task.row_offsets[node + 1U]) {
      return Status::InvalidArgument(
          "CPU linkage evaluation row offsets are not ordered");
    }
    if (expected == LinkageEvaluationKind::kEdgeRsd &&
        task.row_offsets[node] == task.row_offsets[node + 1U]) {
      return Status::InvalidArgument(
          "CPU edge-RSD evaluation contains an empty row");
    }
  }
  const std::uint32_t order_count =
      expected == LinkageEvaluationKind::kEdgeRsd ? task.node_count() : 1U;
  return ValidateIcmLayerOrders<0>(order_count, task.icm_layer_orders, task.m);
}

void GatherResidualTile(const LinkageEvaluationTask& task,
                        const CpuRegisteredCluster& cluster,
                        std::uint32_t first_pair, std::uint32_t pair_count,
                        std::vector<float>* residuals,
                        std::vector<std::uint32_t>* pair_nodes) {
  residuals->resize(static_cast<std::size_t>(task.dimension) * pair_count);
  pair_nodes->resize(pair_count);
  std::uint32_t node = 0U;
  while (node + 1U < task.row_offsets.size() &&
         task.row_offsets[node + 1U] <= first_pair) {
    ++node;
  }
  for (std::uint32_t local_pair = 0U; local_pair < pair_count; ++local_pair) {
    const std::uint32_t pair = first_pair + local_pair;
    while (node + 1U < task.row_offsets.size() &&
           task.row_offsets[node + 1U] <= pair) {
      ++node;
    }
    (*pair_nodes)[local_pair] = node;
    const float* source = task.node_vectors.data() +
                          static_cast<std::size_t>(node) * task.dimension;
    const float* parent =
        cluster.reconstructions.data() +
        static_cast<std::size_t>(task.candidate_columns[pair]) * task.dimension;
    float* residual = residuals->data() +
                      static_cast<std::size_t>(local_pair) * task.dimension;
    for (std::uint32_t axis = 0U; axis < task.dimension; ++axis) {
      residual[axis] = source[axis] - parent[axis];
    }
  }
}

LinkageEvaluationResult EmptyResult(std::uint32_t node_count, std::uint32_t m) {
  LinkageEvaluationResult result;
  result.parent_columns.resize(node_count,
                               std::numeric_limits<std::uint32_t>::max());
  result.supports.resize(node_count, m);
  result.coefficients.resize(static_cast<std::size_t>(node_count) * m);
  result.squared_errors.resize(node_count,
                               std::numeric_limits<float>::infinity());
  return result;
}

template <int M>
void CopyEdgeWinner(const LinkageEvaluationTask& task,
                    const LinkageEncoding<M>& encoded, std::uint32_t task_pair,
                    std::uint32_t tile_pair, std::uint32_t node,
                    LinkageEvaluationResult* result) {
  result->parent_columns[node] = task.candidate_columns[task_pair];
  auto support = result->supports[node];
  support.first = encoded.code(tile_pair, 0U);
  for (std::uint32_t layer = 1U; layer < task.m; ++layer) {
    support.trailing[layer - 1U] = encoded.code(tile_pair, layer);
  }
  for (std::uint32_t layer = 0U; layer < task.m; ++layer) {
    result->coefficients[static_cast<std::size_t>(node) * task.m + layer] =
        encoded.coefficient(tile_pair, layer);
  }
  result->squared_errors[node] = encoded.squared_error(tile_pair);
}

template <int M>
void CopyRootWinner(const LinkageEvaluationTask& task,
                    const RootOnlyLinkageEncoding<M>& encoded,
                    std::uint32_t task_pair, std::uint32_t tile_pair,
                    std::uint32_t node, LinkageEvaluationResult* result) {
  result->parent_columns[node] = task.candidate_columns[task_pair];
  result->supports.set_tuple(node, encoded.support(tile_pair));
  for (std::uint32_t layer = 0U; layer < task.m; ++layer) {
    result->coefficients[static_cast<std::size_t>(node) * task.m + layer] =
        encoded.coefficient(tile_pair, layer);
  }
  result->squared_errors[node] = encoded.squared_error(tile_pair);
}

template <int M, typename EncodeTile>
StatusOr<LinkageEvaluationResult> EvaluateEdgeTask(
    const LinkageBackendEvaluation& evaluation,
    const CpuRegisteredCluster& cluster,
    const LinkageCodebookAlgebra<M>& algebra, const F32GemmBackend& gemm,
    const ParallelForExecutor& parallel, std::uint32_t maximum_rank,
    LeastSquaresCostMode cost_mode, std::uint32_t icm_rounds,
    EncodeTile encode_tile) {
  const LinkageEvaluationTask& task = evaluation.task;
  LinkageEvaluationResult result = EmptyResult(task.node_count(), task.m);
  std::vector<float> residuals;
  std::vector<std::uint32_t> pair_nodes;
  std::vector<std::uint8_t> orders;
  CpuLinkageRsdEncoder<M> encoder(algebra, parallel, maximum_rank, cost_mode,
                                  icm_rounds);
  LinkageProjection<M> projection;
  for (std::uint32_t first_pair = 0U; first_pair < task.pair_count();) {
    const std::uint32_t pair_count = std::min(kCpuLinkageEvaluationTilePairs,
                                              task.pair_count() - first_pair);
    GatherResidualTile(task, cluster, first_pair, pair_count, &residuals,
                       &pair_nodes);
    const Status projection_status =
        projection.Recompute({residuals.data(), task.dimension, pair_count},
                             algebra, gemm, parallel);
    if (!projection_status.ok()) return projection_status;
    orders.resize(static_cast<std::size_t>(pair_count) * task.m);
    for (std::uint32_t pair = 0U; pair < pair_count; ++pair) {
      std::copy_n(task.icm_layer_orders.data() +
                      static_cast<std::size_t>(pair_nodes[pair]) * task.m,
                  task.m,
                  orders.data() + static_cast<std::size_t>(pair) * task.m);
    }
    auto encoded = encode_tile(encoder, projection, orders);
    if (!encoded.ok()) return encoded.status();
    for (std::uint32_t pair = 0U; pair < pair_count; ++pair) {
      const std::uint32_t node = pair_nodes[pair];
      if (encoded.value().squared_error(pair) < result.squared_errors[node]) {
        CopyEdgeWinner(task, encoded.value(), first_pair + pair, pair, node,
                       &result);
      }
    }
    first_pair += pair_count;
  }
  return result;
}

template <int M, typename EncodeTile>
StatusOr<LinkageEvaluationResult> EvaluateRootTask(
    const LinkageBackendEvaluation& evaluation,
    const CpuRegisteredCluster& cluster, const BasicCodebookAlgebra<M>& algebra,
    const F32GemmBackend& gemm, const ParallelForExecutor& parallel,
    std::uint32_t icm_rounds, EncodeTile encode_tile) {
  const LinkageEvaluationTask& task = evaluation.task;
  LinkageEvaluationResult result = EmptyResult(task.node_count(), task.m);
  std::vector<float> residuals;
  std::vector<std::uint32_t> pair_nodes;
  CpuRootOnlyLinkageEncoder<M> encoder(algebra, gemm, parallel, icm_rounds);
  for (std::uint32_t first_pair = 0U; first_pair < task.pair_count();) {
    const std::uint32_t pair_count = std::min(kCpuLinkageEvaluationTilePairs,
                                              task.pair_count() - first_pair);
    GatherResidualTile(task, cluster, first_pair, pair_count, &residuals,
                       &pair_nodes);
    auto encoded =
        encode_tile(encoder, {residuals.data(), task.dimension, pair_count},
                    task.icm_layer_orders);
    if (!encoded.ok()) return encoded.status();
    for (std::uint32_t pair = 0U; pair < pair_count; ++pair) {
      const std::uint32_t node = pair_nodes[pair];
      if (encoded.value().squared_error(pair) < result.squared_errors[node]) {
        CopyRootWinner(task, encoded.value(), first_pair + pair, pair, node,
                       &result);
      }
    }
    first_pair += pair_count;
  }
  return result;
}

}  // namespace

template <int M>
struct CpuRsdLinkageWorkerBackend<M>::State final {
  State(const LinkageCodebookAlgebra<M>& input_algebra,
        const F32GemmBackend& input_gemm,
        const ParallelForExecutor& input_parallel,
        std::uint32_t input_maximum_rank, LeastSquaresCostMode input_cost_mode,
        std::uint32_t input_icm_rounds) noexcept
      : algebra(&input_algebra),
        gemm(&input_gemm),
        parallel(&input_parallel),
        maximum_rank(input_maximum_rank),
        cost_mode(input_cost_mode),
        icm_rounds(input_icm_rounds) {}
  const LinkageCodebookAlgebra<M>* algebra;
  const F32GemmBackend* gemm;
  const ParallelForExecutor* parallel;
  std::uint32_t maximum_rank = 2U;
  LeastSquaresCostMode cost_mode = LeastSquaresCostMode::kNormalEquation;
  std::uint32_t icm_rounds = 1U;
  LinkageBackendCluster next_cluster = 1U;
  std::unordered_map<LinkageBackendCluster, CpuRegisteredCluster> clusters;
  LinkageBackendResourceStats resources;
};

template <int M>
CpuRsdLinkageWorkerBackend<M>::CpuRsdLinkageWorkerBackend(
    const LinkageCodebookAlgebra<M>& algebra, const F32GemmBackend& gemm,
    const ParallelForExecutor& parallel, std::uint32_t maximum_rank,
    LeastSquaresCostMode cost_mode, std::uint32_t icm_rounds) noexcept
    : state_(std::make_unique<State>(algebra, gemm, parallel, maximum_rank,
                                     cost_mode, icm_rounds)) {}

template <int M>
CpuRsdLinkageWorkerBackend<M>::~CpuRsdLinkageWorkerBackend() = default;

template <int M>
StatusOr<LinkageBackendCluster> CpuRsdLinkageWorkerBackend<M>::Register(
    LinkageBackendRegistration registration) {
  if (const Status status = ValidateRegistration(registration); !status.ok()) {
    return status;
  }
  const LinkageBackendCluster handle = state_->next_cluster++;
  state_->resources.live_registration_bytes +=
      registration.reconstructions.size() * sizeof(float);
  state_->clusters.emplace(
      handle, CpuRegisteredCluster{
                  .dimension = registration.dimension,
                  .columns = registration.column_count,
                  .generation = 0U,
                  .reconstructions = std::move(registration.reconstructions),
              });
  return handle;
}

template <int M>
Status CpuRsdLinkageWorkerBackend<M>::ApplyUpdates(
    ArrayView<const LinkageBackendUpdate> updates) {
  return ApplyCpuUpdates(&state_->clusters, updates);
}

template <int M>
StatusOr<std::vector<LinkageEvaluationResult>>
CpuRsdLinkageWorkerBackend<M>::Evaluate(
    ArrayView<const LinkageBackendEvaluation> tasks) {
  return EvaluateImpl(tasks, false);
}

template <int M>
StatusOr<std::vector<LinkageEvaluationResult>>
CpuRsdLinkageWorkerBackend<M>::EvaluatePrevalidatedDirect(
    ArrayView<const LinkageBackendEvaluation> tasks) {
  return EvaluateImpl(tasks, true);
}

template <int M>
StatusOr<std::vector<LinkageEvaluationResult>>
CpuRsdLinkageWorkerBackend<M>::EvaluateImpl(
    ArrayView<const LinkageBackendEvaluation> tasks, bool prevalidated) {
  if (tasks.empty()) {
    return Status::InvalidArgument("CPU edge-RSD batch must not be empty");
  }
  std::vector<LinkageEvaluationResult> results;
  results.reserve(tasks.size());
  for (const LinkageBackendEvaluation& task : tasks) {
    if (task.task.m != state_->algebra->shape().m)
      return Status::InvalidArgument("CPU linkage task M does not match model");
    const auto found = state_->clusters.find(task.cluster);
    if (found == state_->clusters.end()) {
      return Status::FailedPrecondition(
          "CPU edge-RSD evaluation references an unavailable cluster");
    }
    const Status status =
        prevalidated ? ValidateEvaluationState(
                           task, LinkageEvaluationKind::kEdgeRsd, found->second)
                     : ValidateEvaluation(task, LinkageEvaluationKind::kEdgeRsd,
                                          found->second);
    if (!status.ok()) {
      return status;
    }
    assert(
        !prevalidated ||
        ValidateEvaluation(task, LinkageEvaluationKind::kEdgeRsd, found->second)
            .ok());
    auto result = EvaluateEdgeTask(
        task, found->second, *state_->algebra, *state_->gemm, *state_->parallel,
        state_->maximum_rank, state_->cost_mode, state_->icm_rounds,
        [](const CpuLinkageRsdEncoder<M>& encoder,
           const LinkageProjection<M>& projection,
           ArrayView<const std::uint8_t> orders) {
          return encoder.EncodeValidatedOrders(projection, orders);
        });
    if (!result.ok()) return result.status();
    results.push_back(std::move(result).value());
  }
  return results;
}

template <int M>
Status CpuRsdLinkageWorkerBackend<M>::Unregister(
    LinkageBackendCluster cluster) {
  const auto found = state_->clusters.find(cluster);
  if (found == state_->clusters.end()) {
    return Status::FailedPrecondition(
        "CPU edge-RSD unregister references an unavailable cluster");
  }
  state_->resources.live_registration_bytes -=
      found->second.reconstructions.size() * sizeof(float);
  state_->clusters.erase(found);
  return Status::Ok();
}

template <int M>
LinkageBackendResourceStats CpuRsdLinkageWorkerBackend<M>::ResourceStats()
    const noexcept {
  return state_->resources;
}

template <int M>
struct CpuRootOnlyLinkageWorkerBackend<M>::State final {
  State(const BasicCodebookAlgebra<M>& input_algebra,
        const F32GemmBackend& input_gemm,
        const ParallelForExecutor& input_parallel,
        std::uint32_t input_icm_rounds) noexcept
      : algebra(&input_algebra),
        gemm(&input_gemm),
        parallel(&input_parallel),
        icm_rounds(input_icm_rounds) {}
  const BasicCodebookAlgebra<M>* algebra;
  const F32GemmBackend* gemm;
  const ParallelForExecutor* parallel;
  std::uint32_t icm_rounds = 1U;
  LinkageBackendCluster next_cluster = 1U;
  std::unordered_map<LinkageBackendCluster, CpuRegisteredCluster> clusters;
  LinkageBackendResourceStats resources;
};

template <int M>
CpuRootOnlyLinkageWorkerBackend<M>::CpuRootOnlyLinkageWorkerBackend(
    const BasicCodebookAlgebra<M>& algebra, const F32GemmBackend& gemm,
    const ParallelForExecutor& parallel, std::uint32_t icm_rounds) noexcept
    : state_(std::make_unique<State>(algebra, gemm, parallel, icm_rounds)) {}

template <int M>
CpuRootOnlyLinkageWorkerBackend<M>::~CpuRootOnlyLinkageWorkerBackend() =
    default;

template <int M>
StatusOr<LinkageBackendCluster> CpuRootOnlyLinkageWorkerBackend<M>::Register(
    LinkageBackendRegistration registration) {
  if (const Status status = ValidateRegistration(registration); !status.ok()) {
    return status;
  }
  const LinkageBackendCluster handle = state_->next_cluster++;
  state_->resources.live_registration_bytes +=
      registration.reconstructions.size() * sizeof(float);
  state_->clusters.emplace(
      handle, CpuRegisteredCluster{
                  .dimension = registration.dimension,
                  .columns = registration.column_count,
                  .generation = 0U,
                  .reconstructions = std::move(registration.reconstructions),
              });
  return handle;
}

template <int M>
Status CpuRootOnlyLinkageWorkerBackend<M>::ApplyUpdates(
    ArrayView<const LinkageBackendUpdate> updates) {
  return ApplyCpuUpdates(&state_->clusters, updates);
}

template <int M>
StatusOr<std::vector<LinkageEvaluationResult>>
CpuRootOnlyLinkageWorkerBackend<M>::Evaluate(
    ArrayView<const LinkageBackendEvaluation> tasks) {
  return EvaluateImpl(tasks, false);
}

template <int M>
StatusOr<std::vector<LinkageEvaluationResult>>
CpuRootOnlyLinkageWorkerBackend<M>::EvaluatePrevalidatedDirect(
    ArrayView<const LinkageBackendEvaluation> tasks) {
  return EvaluateImpl(tasks, true);
}

template <int M>
StatusOr<std::vector<LinkageEvaluationResult>>
CpuRootOnlyLinkageWorkerBackend<M>::EvaluateImpl(
    ArrayView<const LinkageBackendEvaluation> tasks, bool prevalidated) {
  if (tasks.empty()) {
    return Status::InvalidArgument(
        "CPU root-only linkage batch must not be empty");
  }
  std::vector<LinkageEvaluationResult> results;
  results.reserve(tasks.size());
  for (const LinkageBackendEvaluation& task : tasks) {
    if (task.task.m != state_->algebra->shape().m)
      return Status::InvalidArgument("CPU linkage task M does not match model");
    const auto found = state_->clusters.find(task.cluster);
    if (found == state_->clusters.end()) {
      return Status::FailedPrecondition(
          "CPU root-only evaluation references an unavailable cluster");
    }
    const Status status =
        prevalidated ? ValidateEvaluationState(
                           task, LinkageEvaluationKind::kRootOnlyInitialization,
                           found->second)
                     : ValidateEvaluation(
                           task, LinkageEvaluationKind::kRootOnlyInitialization,
                           found->second);
    if (!status.ok()) {
      return status;
    }
    assert(!prevalidated ||
           ValidateEvaluation(task,
                              LinkageEvaluationKind::kRootOnlyInitialization,
                              found->second)
               .ok());
    auto result = EvaluateRootTask(
        task, found->second, *state_->algebra, *state_->gemm, *state_->parallel,
        state_->icm_rounds,
        [](const CpuRootOnlyLinkageEncoder<M>& encoder,
           ColumnMajorMatrixF32View residuals,
           ArrayView<const std::uint8_t> order) {
          return encoder.EncodeValidatedSharedOrder(residuals, order);
        });
    if (!result.ok()) return result.status();
    results.push_back(std::move(result).value());
  }
  return results;
}

template <int M>
Status CpuRootOnlyLinkageWorkerBackend<M>::Unregister(
    LinkageBackendCluster cluster) {
  const auto found = state_->clusters.find(cluster);
  if (found == state_->clusters.end()) {
    return Status::FailedPrecondition(
        "CPU root-only unregister references an unavailable cluster");
  }
  state_->resources.live_registration_bytes -=
      found->second.reconstructions.size() * sizeof(float);
  state_->clusters.erase(found);
  return Status::Ok();
}

template <int M>
LinkageBackendResourceStats CpuRootOnlyLinkageWorkerBackend<M>::ResourceStats()
    const noexcept {
  return state_->resources;
}

template class CpuRsdLinkageWorkerBackend<0>;
template class CpuRootOnlyLinkageWorkerBackend<0>;
template class CpuRsdLinkageWorkerBackend<5>;
template class CpuRsdLinkageWorkerBackend<12>;
template class CpuRsdLinkageWorkerBackend<6>;
template class CpuRsdLinkageWorkerBackend<10>;
template class CpuRootOnlyLinkageWorkerBackend<5>;
template class CpuRootOnlyLinkageWorkerBackend<12>;
template class CpuRootOnlyLinkageWorkerBackend<6>;
template class CpuRootOnlyLinkageWorkerBackend<10>;

}  // namespace stlqpp
