#include "stlqpp/training/linkage/topology/good_cluster_topology_builder.h"

#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"
#include "stlqpp/training/linkage/topology/cluster_topology_session.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

template <int M>
Status ValidateRequest(
    const ProductCodebooks<M>& codebooks,
    const GoodClusterTopologyBuildRequest<M>& request,
    std::vector<std::uint32_t>* layer_of_real,
    std::vector<RadialLayerView>* candidate_layers,
    bool produced_partition) {
  if (request.cid >= codebooks.shape().nlist ||
      request.exact_real.values == nullptr ||
      request.exact_real.rows != codebooks.shape().dimension ||
      request.exact_real.columns == 0U ||
      request.basic_real_reconstructions.values == nullptr ||
      request.basic_real_reconstructions.rows != request.exact_real.rows ||
      request.basic_real_reconstructions.columns !=
          request.exact_real.columns ||
      request.radial_layers.empty() || request.depth_k == 0U ||
      request.knn_k == 0U || request.maximum_depth == 0U ||
      request.random_stream == nullptr) {
    return Status::InvalidArgument(
        "good-cluster topology request has an invalid shape or policy");
  }
  if (!produced_partition) {
    if (const Status status = ValidateIcmLayerOrders<M>(
            request.exact_real.columns, request.icm_layer_orders, codebooks.shape().m);
        !status.ok()) {
      return status;
    }
  } else if (request.icm_layer_orders.size() !=
             static_cast<std::size_t>(request.exact_real.columns) * codebooks.shape().m) {
    return Status::InvalidArgument(
        "generated ICM layer orders have an invalid extent");
  }
#ifndef NDEBUG
  if (produced_partition) {
    std::vector<std::uint32_t> checked_layers;
    std::vector<RadialLayerView> checked_views;
    assert(ValidateRequest(codebooks, request, &checked_layers,
                           &checked_views, false).ok());
  }
#endif
  const std::uint32_t real_count = request.exact_real.columns;
  layer_of_real->assign(real_count, 0U);
  std::vector<std::uint8_t> seen_real;
  if (!produced_partition) seen_real.assign(real_count, 0U);
  candidate_layers->clear();
  candidate_layers->reserve(request.radial_layers.size());
  for (std::uint32_t layer = 0U;
       layer < static_cast<std::uint32_t>(request.radial_layers.size());
       ++layer) {
    const RadialLayerView source = request.radial_layers[layer];
    if (source.local_ids.data() == nullptr && !source.local_ids.empty()) {
      return Status::InvalidArgument(
          "radial layer requires a complete local-ID view");
    }
    for (std::size_t position = 0U; position < source.local_ids.size();
         ++position) {
      const std::uint32_t local_id = source.local_ids[position];
      if (!produced_partition) {
        if (local_id >= real_count || seen_real[local_id] != 0U) {
          return Status::InvalidArgument(
              "radial layers must partition the real local IDs");
        }
        seen_real[local_id] = 1U;
      }
      (*layer_of_real)[local_id] = layer;
    }
    candidate_layers->push_back(source);
  }
  if (!produced_partition &&
      std::find(seen_real.begin(), seen_real.end(), 0U) != seen_real.end()) {
    return Status::InvalidArgument(
        "radial layers do not cover every real local ID");
  }
  return Status::Ok();
}

}  // namespace

template <int M>
StatusOr<ClusterTopologyResult<M>> GoodClusterTopologyBuilder<M>::Build(
    const GoodClusterTopologyBuildRequest<M>& request) const {
  return BuildImpl(request, false);
}

template <int M>
StatusOr<ClusterTopologyResult<M>>
GoodClusterTopologyBuilder<M>::BuildProduced(
    const GoodClusterTopologyBuildRequest<M>& request) const {
  return BuildImpl(request, true);
}

template <int M>
StatusOr<ClusterTopologyResult<M>> GoodClusterTopologyBuilder<M>::BuildImpl(
    const GoodClusterTopologyBuildRequest<M>& request,
    bool produced_partition) const {
  std::vector<std::uint32_t> layer_of_real;
  std::vector<RadialLayerView> candidate_layers;
  if (const Status status = ValidateRequest(
          *codebooks_, request, &layer_of_real, &candidate_layers,
          produced_partition);
      !status.ok()) {
    return status;
  }
  auto session_result = ClusterTopologySession<M>::Create(
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
      *scheduler_, batching_, cpu_evaluator_, request.random_stream,
      request.basic_ownership, nullptr,
      request.borrow_exact_real);
  if (!session_result.ok()) {
    return session_result.status();
  }
  auto session = std::move(session_result).value();
  for (const std::uint32_t root : request.radial_layers[0].local_ids) {
    if (const Status status = session->PublishInitialRoot(root); !status.ok()) {
      return status;
    }
  }
  std::vector<std::uint32_t> layer_counts(request.radial_layers.size(), 0U);
  std::vector<LinkageParentReference> selected;

  for (std::uint32_t layer = 1U;
       layer < static_cast<std::uint32_t>(request.radial_layers.size());
       ++layer) {
    if (request.radial_layers[layer].local_ids.empty()) {
      continue;
    }
    const std::uint32_t inner_begin =
        layer > request.depth_k ? layer - request.depth_k : 0U;
    const std::vector<std::uint32_t> processing_order =
        request.random_stream->Permutation(
            static_cast<std::uint32_t>(
                request.radial_layers[layer].local_ids.size()));
    auto prepared = candidates_->BuildValidatedLayers({
        .cluster_vectors = request.exact_real,
        .layers = candidate_layers,
        .current_layer = layer,
        .inner_layer_begin = inner_begin,
        .inner_layer_end_exclusive = layer,
        .current_layer_order = processing_order,
        .knn_k = request.knn_k,
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
        if (layer_counts[parent_layer] < request.knn_k) {
          ++layer_counts[parent_layer];
          selected.push_back({LinkageParentKind::kReal, parent});
        }
      }
      CausalTopologyRow<M> causal;
      causal.real_local_id = row.query_local_id;
      causal.ordered_candidates = selected;
      const std::uint8_t* order = request.icm_layer_orders.data() +
          static_cast<std::size_t>(row.query_local_id) * codebooks_->shape().m;
      std::copy_n(order, codebooks_->shape().m, causal.icm_layer_order.begin());
      causal_rows.push_back(std::move(causal));
    }
    if (!causal_rows.empty()) {
      if (const Status status = session->ProcessRadialLayerRows(
              causal_rows, request.maximum_depth);
          !status.ok()) {
        return status;
      }
    }
  }
  return session->Finish();
}

template class GoodClusterTopologyBuilder<0>;
template class GoodClusterTopologyBuilder<5>;
template class GoodClusterTopologyBuilder<12>;
template class GoodClusterTopologyBuilder<6>;
template class GoodClusterTopologyBuilder<10>;

}  // namespace stlqpp
