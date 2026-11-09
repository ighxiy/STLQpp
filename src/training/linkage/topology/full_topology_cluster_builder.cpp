#include "stlqpp/training/linkage/topology/full_topology_cluster_builder.h"

#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"
#include "stlqpp/training/linkage/rsd/ranked_single_deviation.h"
#include "stlqpp/training/linkage/topology/bad_cluster_topology_builder.h"
#include "stlqpp/training/linkage/topology/good_cluster_topology_builder.h"
#include "stlqpp/training/linkage/topology/radial_layer_plan.h"
#include "stlqpp/training/linkage/topology/topology_random_stream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

Status ValidatePolicy(const FullTopologyPolicy& policy) {
  if (policy.radial_layer_count == 0U ||
      !std::isfinite(policy.root_fraction) ||
      policy.root_fraction <= 0.0 || policy.root_fraction > 1.0 ||
      policy.depth_k == 0U || policy.knn_k == 0U ||
      policy.maximum_depth == 0U ||
      !IsSupportedRsdRank(policy.rsd_rank) ||
      policy.batching.inner_batch_rows == 0U ||
      policy.batching.maximum_batch_pairs == 0U ||
      policy.batching.same_layer_block_rows == 0U ||
      policy.batching.same_layer_window_maximum_pairs == 0U ||
      policy.batching.same_layer_maximum_pending_nodes == 0U ||
      (policy.batching.tiny_forced_cpu_enabled &&
       policy.batching.tiny_forced_pair_limit == 0U) ||
      policy.batching.bad_frozen_window_rows == 0U ||
      policy.batching.bad_frozen_maximum_pairs == 0U ||
      !std::isfinite(policy.virtual_ratio) ||
      policy.virtual_ratio < 0.0 || policy.virtual_ratio > 1.0 ||
      policy.maximum_virtual_roots < policy.minimum_virtual_roots) {
    return Status::InvalidArgument("full-topology policy is invalid");
  }
  return Status::Ok();
}

std::uint32_t ResolveVirtualRootCount(std::uint32_t real_count,
                                      const FullTopologyPolicy& policy) {
  const auto proportional = policy.use_fixed_virtual_roots
      ? static_cast<std::uint64_t>(policy.fixed_virtual_roots)
      : static_cast<std::uint64_t>(std::llround(
            policy.virtual_ratio * static_cast<double>(real_count)));
  const std::uint64_t bounded = std::min<std::uint64_t>(
      real_count,
      std::min<std::uint64_t>(
          policy.maximum_virtual_roots,
          std::max<std::uint64_t>(
              policy.minimum_virtual_roots, proportional)));
  return static_cast<std::uint32_t>(bounded);
}

}  // namespace

template <int M>
StatusOr<FullTopologyClusterProduct<M>>
FullTopologyClusterBuilder<M>::Build(
    const FullTopologyClusterBuildRequest<M>& request,
    const FullTopologyPolicy& policy) const {
  if (const Status status = ValidatePolicy(policy); !status.ok()) {
    return status;
  }
  const ModelShape& shape = codebooks_->shape();
  const std::uint32_t real_count = request.rotated_exact_real.columns;
  const bool bad_cluster = policy.virtual_enabled && request.bad_cluster;
  if (request.cid >= shape.nlist || real_count == 0U ||
      request.rotated_exact_real.values == nullptr ||
      request.rotated_exact_real.rows != shape.dimension ||
      request.unrotated_exact_real.rows != shape.dimension ||
      request.unrotated_exact_real.columns != real_count ||
      (bad_cluster &&
       request.unrotated_exact_real.values == nullptr) ||
      request.real_sample_ids.size() != real_count) {
    return Status::InvalidArgument(
        "full-topology cluster input does not match its model");
  }
  const std::uint32_t cluster_seed =
      policy.seed + 997U * request.cid;
  auto icm_orders = BuildSharedTaskIcmLayerOrders<M>(
      cluster_seed, real_count, shape.m);
  if (!icm_orders.ok()) {
    return icm_orders.status();
  }

  FullTopologyClusterProduct<M> output;
  if (!bad_cluster || real_count == 1U) {
    TopologyRandomStream cluster_random(cluster_seed);
    auto radial_plan = RadialLayerPlan::Build(
        request.rotated_exact_real,
        {codebooks_->root0().column(request.cid), shape.dimension},
        {.layer_count = policy.radial_layer_count,
         .root_fraction = policy.root_fraction});
    if (!radial_plan.ok()) {
      return radial_plan.status();
    }
    const std::vector<RadialLayerView> radial_views =
        radial_plan.value().Views();
    GoodClusterTopologyBuilder<M> builder(
        *codebooks_, *good_candidates_, *scheduler_, policy.batching,
        cpu_evaluator_);
    auto topology = builder.BuildProduced({
        .cid = request.cid,
        .exact_real = request.rotated_exact_real,
        .borrow_exact_real = request.borrow_rotated_exact_real,
        .basic_real_reconstructions = request.basic_real_reconstructions,
        .basic_supports = request.basic_supports,
        .basic_coefficients = request.basic_coefficients,
        .basic_squared_errors = request.basic_squared_errors,
        .validated_basic = request.validated_basic,
        .basic_ownership = request.basic_ownership,
        .radial_layers = radial_views,
        .icm_layer_orders = icm_orders.value(),
        .random_stream = &cluster_random,
        .depth_k = policy.depth_k,
        .knn_k = policy.knn_k,
        .maximum_depth = policy.maximum_depth,
    });
    if (!topology.ok()) {
      return topology.status();
    }
    output.topology = std::move(topology).value();
    return output;
  }

  auto graph = BuildBadClusterKnn(*exact_knn_,
      request.rotated_exact_real,
      std::max(policy.knn_k, policy.virtual_policy.neighbor_count), policy.cluster_knn,
      request.exact_knn_cpu_budget);
  if (!graph.ok()) {
    return graph.status();
  }
  const PreparedKnnRows& real_graph = graph.value();
  const std::uint32_t virtual_count =
      ResolveVirtualRootCount(real_count, policy);
  if (virtual_count == 0U) {
    return Status::InvalidArgument(
        "bad full-topology cluster resolved zero virtual roots");
  }
  auto virtual_plan = UmapVirtualRootPlan::Build(
      real_graph, virtual_count, policy.virtual_policy);
  if (!virtual_plan.ok()) {
    return virtual_plan.status();
  }
  auto virtual_roots = EncodeVirtualRoots<M>(
      request.cid, request.unrotated_exact_real,
      request.rotated_exact_real, virtual_plan.value(), *codebooks_,
      *root_encoder_, policy.virtual_root_seed +
          1337U * request.cid + 17U);
  if (!virtual_roots.ok()) {
    return virtual_roots.status();
  }
  auto virtual_neighbors = exact_knn_->BuildBipartite(
      virtual_roots.value().reconstructions(),
      request.rotated_exact_real, policy.knn_k, request.exact_knn_cpu_budget);
  if (!virtual_neighbors.ok()) {
    return virtual_neighbors.status();
  }
  std::vector<float> owned_virtual_reconstructions =
      virtual_roots.value().TakeReconstructions();
  virtual_roots.value().ReleaseConstructionOnly();
  TopologyRandomStream cluster_random(cluster_seed);
  const std::vector<std::uint32_t> processing =
      cluster_random.Permutation(real_count);
  BadClusterTopologyBuilder<M> builder(
      *codebooks_, *scheduler_, policy.batching, cpu_evaluator_);
  const BadClusterTopologyBuildRequest<M> bad_request = {
      .cid = request.cid,
      .exact_real = request.rotated_exact_real,
      .borrow_exact_real = request.borrow_rotated_exact_real,
      .basic_real_reconstructions = request.basic_real_reconstructions,
      .virtual_reconstructions = {
          owned_virtual_reconstructions.data(), shape.dimension,
          virtual_count},
      .basic_supports = request.basic_supports,
      .basic_coefficients = request.basic_coefficients,
      .basic_squared_errors = request.basic_squared_errors,
      .validated_basic = request.validated_basic,
      .basic_ownership = request.basic_ownership,
      .owned_virtual_reconstructions = &owned_virtual_reconstructions,
      .processing_order = processing,
      .icm_layer_orders = icm_orders.value(),
      .random_stream = &cluster_random,
      .knn_k = policy.knn_k,
      .maximum_depth = policy.maximum_depth,
  };
  auto topology = real_graph.producer_guaranteed_unique() &&
          virtual_neighbors.value().producer_guaranteed_unique()
      ? builder.BuildProduced(
            bad_request, real_graph, virtual_neighbors.value())
      : builder.Build(
            bad_request, real_graph, virtual_neighbors.value());
  if (!topology.ok()) {
    return topology.status();
  }
  output.topology = std::move(topology).value();
  output.virtual_roots =
      std::move(virtual_roots).value().TakePublication();
  return output;
}

template class FullTopologyClusterBuilder<0>;
template class FullTopologyClusterBuilder<5>;
template class FullTopologyClusterBuilder<12>;
template class FullTopologyClusterBuilder<6>;
template class FullTopologyClusterBuilder<10>;

}  // namespace stlqpp
