#include "stlqpp/training/linkage/initialization/root_only_initialization_cluster_builder.h"

#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"
#include "stlqpp/training/linkage/topology/cluster_topology_session.h"
#include "stlqpp/training/linkage/topology/radial_layer_plan.h"
#include "stlqpp/training/linkage/topology/topology_random_stream.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

Status ValidatePolicy(
    const RootOnlyInitializationTopologyPolicy& policy) {
  if (policy.radial_layer_count == 0U ||
      !std::isfinite(policy.root_fraction) ||
      policy.root_fraction <= 0.0 || policy.root_fraction > 1.0 ||
      policy.depth_k == 0U || policy.knn_k == 0U ||
      policy.maximum_depth == 0U ||
      policy.batching.inner_batch_rows == 0U ||
      policy.batching.maximum_batch_pairs == 0U ||
      policy.batching.same_layer_block_rows == 0U ||
      policy.batching.same_layer_window_maximum_pairs == 0U ||
      policy.batching.same_layer_maximum_pending_nodes == 0U ||
      (policy.batching.tiny_forced_cpu_enabled &&
       policy.batching.tiny_forced_pair_limit == 0U)) {
    return Status::InvalidArgument(
        "root-only initialization topology policy is invalid");
  }
  return Status::Ok();
}

template <int M>
Status ValidateRequest(
    const RootCodebooks<M>& codebooks,
    const RootOnlyInitializationClusterRequest<M>& request) {
  const ModelShape& shape = codebooks.shape();
  const std::uint32_t count = request.exact_real.columns;
  if (request.cid >= shape.nlist || count == 0U ||
      request.exact_real.values == nullptr ||
      request.exact_real.rows != shape.dimension ||
      request.basic_real_reconstructions.values == nullptr ||
      request.basic_real_reconstructions.rows != shape.dimension ||
      request.basic_real_reconstructions.columns != count ||
      request.basic_supports.size() != count ||
      request.basic_coefficients.size() !=
          static_cast<std::size_t>(count) * codebooks.m() ||
      request.basic_squared_errors.size() != count) {
    return Status::InvalidArgument(
        "root-only initialization cluster input has an invalid shape");
  }
  return ValidateIcmLayerOrders<M>(1U, request.icm_layer_order, codebooks.m());
}

}  // namespace

template <int M>
StatusOr<ClusterTopologyResult<M>>
RootOnlyInitializationClusterBuilder<M>::Build(
    const RootOnlyInitializationClusterRequest<M>& request,
    const RootOnlyInitializationTopologyPolicy& policy) const {
  if (const Status status = ValidatePolicy(policy); !status.ok()) {
    return status;
  }
  if (const Status status = ValidateRequest(*codebooks_, request);
      !status.ok()) {
    return status;
  }

  const std::uint32_t cluster_seed = policy.shared_cluster_seed
      ? policy.seed : policy.seed + 997U * request.cid;
  TopologyRandomStream cluster_random(cluster_seed);
  auto radial_plan = RadialLayerPlan::Build(
      request.exact_real,
      {codebooks_->root0().column(request.cid),
       codebooks_->shape().dimension},
      {
          .layer_count = policy.radial_layer_count,
          .root_fraction = policy.root_fraction,
      });
  if (!radial_plan.ok()) {
    return radial_plan.status();
  }

  auto session_result =
      ClusterTopologySession<M>::CreateRootOnlyInitialization(
          *codebooks_,
          {
              .cid = request.cid,
              .exact_real = request.exact_real,
              .basic_real_reconstructions =
                  request.basic_real_reconstructions,
              .virtual_reconstructions = {},
              .basic_supports = request.basic_supports,
              .basic_coefficients = request.basic_coefficients,
              .basic_squared_errors = request.basic_squared_errors,
              .validated_basic = request.validated_basic,
          },
          *scheduler_, *tail_evaluator_, cluster_seed, policy.batching);
  if (!session_result.ok()) {
    return session_result.status();
  }
  auto session = std::move(session_result).value();

  const std::vector<RadialLayerView> ordered =
      radial_plan.value().Views();
  for (const std::uint32_t root : ordered.front().local_ids) {
    if (const Status status = session->PublishInitialRoot(root);
        !status.ok()) {
      return status;
    }
  }

  std::vector<RadialLayerView> candidate_layers;
  candidate_layers.reserve(ordered.size());
  std::vector<std::uint32_t> layer_of_real(
      request.exact_real.columns, 0U);
  for (std::uint32_t layer = 0U;
       layer < static_cast<std::uint32_t>(ordered.size()); ++layer) {
    candidate_layers.push_back({ordered[layer].local_ids});
    for (const std::uint32_t local_id : ordered[layer].local_ids) {
      layer_of_real[local_id] = layer;
    }
  }

  std::vector<std::uint32_t> layer_counts(ordered.size(), 0U);
  std::vector<LinkageParentReference> selected;
  for (std::uint32_t layer = 1U;
       layer < static_cast<std::uint32_t>(ordered.size()); ++layer) {
    if (ordered[layer].local_ids.empty()) {
      continue;
    }
    const std::uint32_t inner_begin =
        layer > policy.depth_k ? layer - policy.depth_k : 0U;
    const std::vector<std::uint32_t> processing_order =
        cluster_random.Permutation(
            static_cast<std::uint32_t>(ordered[layer].local_ids.size()));
    auto prepared = candidates_->BuildValidatedLayers({
        .cluster_vectors = request.exact_real,
        .layers = candidate_layers,
        .current_layer = layer,
        .inner_layer_begin = inner_begin,
        .inner_layer_end_exclusive = layer,
        .current_layer_order = processing_order,
        .knn_k = policy.knn_k,
    });
    if (!prepared.ok()) {
      return prepared.status();
    }
    std::vector<CausalTopologyRow<M>> causal_rows;
    causal_rows.reserve(prepared.value().row_count());
    for (std::uint32_t row_index = 0U;
         row_index < prepared.value().row_count(); ++row_index) {
      const PreparedParentCandidateRow row =
          prepared.value().row(row_index);
      std::fill(layer_counts.begin(), layer_counts.end(), 0U);
      selected.clear();
      selected.reserve(row.parent_local_ids.size());
      for (const std::uint32_t parent : row.parent_local_ids) {
        const std::uint32_t parent_layer = layer_of_real[parent];
        if (layer_counts[parent_layer] < policy.knn_k) {
          ++layer_counts[parent_layer];
          selected.push_back({LinkageParentKind::kReal, parent});
        }
      }
      CausalTopologyRow<M> causal;
      causal.real_local_id = row.query_local_id;
      causal.ordered_candidates = selected;
      std::copy_n(request.icm_layer_order.data(), codebooks_->m(),
                  causal.icm_layer_order.begin());
      causal_rows.push_back(std::move(causal));
    }
    if (!causal_rows.empty()) {
      if (const Status status =
              session->ProcessValidatedRootOnlyInitializationLayer(
              causal_rows, ordered[layer].local_ids, policy.maximum_depth);
          !status.ok()) {
        return status;
      }
    }
  }
  return session->Finish();
}

template class RootOnlyInitializationClusterBuilder<0>;
template class RootOnlyInitializationClusterBuilder<5>;
template class RootOnlyInitializationClusterBuilder<12>;
template class RootOnlyInitializationClusterBuilder<6>;
template class RootOnlyInitializationClusterBuilder<10>;

}  // namespace stlqpp
