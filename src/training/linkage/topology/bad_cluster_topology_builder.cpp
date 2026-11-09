#include "stlqpp/training/linkage/topology/bad_cluster_topology_builder.h"

#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"
#include "stlqpp/training/linkage/topology/cluster_topology_session.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

std::uint32_t RealCandidateSearchWidth(
    const PreparedKnnRows& graph,
    std::uint32_t knn_k) noexcept {
  const std::uint64_t expanded = std::max(
      4ULL * knn_k, static_cast<std::uint64_t>(knn_k) + 16ULL);
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(
      expanded, graph.neighbors_per_row()));
}

Status ValidateCandidateRows(const PreparedKnnRows& rows,
                             std::uint32_t row_count,
                             std::uint32_t row_width,
                             std::uint32_t label_limit,
                             bool exclude_self) {
  if (rows.row_count() != row_count ||
      rows.neighbors_per_row() != row_width) {
    return Status::InvalidArgument(
        "bad-cluster candidate rows have an unexpected shape");
  }
  std::vector<std::uint32_t> labels;
  labels.reserve(row_width);
  for (std::uint32_t row = 0U; row < row_count; ++row) {
    labels.clear();
    const PreparedKnnRow candidates = rows.row(row);
    for (const std::uint32_t label : candidates.local_ids) {
      if (label >= label_limit || (exclude_self && label == row) ||
          std::find(labels.begin(), labels.end(), label) != labels.end()) {
        return Status::InvalidArgument(
            "bad-cluster candidate row contains an invalid label");
      }
      labels.push_back(label);
    }
  }
  return Status::Ok();
}

template <int M>
Status ValidateRequest(const ProductCodebooks<M>& codebooks,
                       const BadClusterTopologyBuildRequest<M>& request,
                       const PreparedKnnRows& real_graph,
                       const PreparedKnnRows& virtual_neighbors,
                       bool produced_rows_and_order) {
  const std::uint32_t real_count = request.exact_real.columns;
  const std::uint32_t virtual_count =
      request.virtual_reconstructions.columns;
  if (request.cid >= codebooks.shape().nlist || real_count == 0U ||
      virtual_count == 0U || request.exact_real.values == nullptr ||
      request.exact_real.rows != codebooks.shape().dimension ||
      request.virtual_reconstructions.values == nullptr ||
      request.virtual_reconstructions.rows != request.exact_real.rows ||
      request.processing_order.size() != real_count || request.knn_k == 0U ||
      request.maximum_depth == 0U || request.random_stream == nullptr) {
    return Status::InvalidArgument(
        "bad-cluster topology request has an invalid shape or policy");
  }
  if (const Status status = ValidateIcmLayerOrders<M>(
          real_count, request.icm_layer_orders, codebooks.shape().m);
      !status.ok()) {
    return status;
  }
  const std::uint32_t real_graph_width = real_graph.neighbors_per_row();
  if ((real_count > 1U && real_graph_width == 0U) ||
      real_graph_width > real_count - 1U) {
    return Status::InvalidArgument(
        "bad-cluster real graph has an invalid row width");
  }
  if (real_graph.row_count() != real_count ||
      virtual_neighbors.row_count() != real_count ||
      virtual_neighbors.neighbors_per_row() !=
          std::min(request.knn_k, virtual_count)) {
    return Status::InvalidArgument(
        "bad-cluster candidate rows have an unexpected shape");
  }
  if (!produced_rows_and_order) {
    if (const Status status = ValidateCandidateRows(
            real_graph, real_count, real_graph_width, real_count, true);
        !status.ok()) {
      return status;
    }
    if (const Status status = ValidateCandidateRows(
            virtual_neighbors, real_count,
            std::min(request.knn_k, virtual_count), virtual_count, false);
        !status.ok()) {
      return status;
    }

    std::vector<std::uint8_t> seen(real_count, 0U);
    for (const std::uint32_t local_id : request.processing_order) {
      if (local_id >= real_count || seen[local_id] != 0U) {
        return Status::InvalidArgument(
            "bad-cluster processing order is not a real-ID permutation");
      }
      seen[local_id] = 1U;
    }
  }
#ifndef NDEBUG
  if (produced_rows_and_order) {
    assert(ValidateCandidateRows(
        real_graph, real_count, real_graph_width, real_count, true).ok());
    assert(ValidateCandidateRows(
        virtual_neighbors, real_count,
        std::min(request.knn_k, virtual_count), virtual_count, false).ok());
    std::vector<std::uint8_t> seen(real_count, 0U);
    for (const std::uint32_t local_id : request.processing_order) {
      assert(local_id < real_count && seen[local_id] == 0U);
      seen[local_id] = 1U;
    }
  }
#endif
  return Status::Ok();
}

}  // namespace

template <int M>
StatusOr<ClusterTopologyResult<M>> BadClusterTopologyBuilder<M>::Build(
    const BadClusterTopologyBuildRequest<M>& request,
    const PreparedKnnRows& real_graph,
    const PreparedKnnRows& virtual_neighbors) const {
  return BuildImpl(request, real_graph, virtual_neighbors, false);
}

template <int M>
StatusOr<ClusterTopologyResult<M>> BadClusterTopologyBuilder<M>::BuildProduced(
    const BadClusterTopologyBuildRequest<M>& request,
    const PreparedKnnRows& real_graph,
    const PreparedKnnRows& virtual_neighbors) const {
  return BuildImpl(request, real_graph, virtual_neighbors, true);
}

template <int M>
StatusOr<ClusterTopologyResult<M>> BadClusterTopologyBuilder<M>::BuildImpl(
    const BadClusterTopologyBuildRequest<M>& request,
    const PreparedKnnRows& real_graph,
    const PreparedKnnRows& virtual_neighbors,
    bool produced_rows_and_order) const {
  if (const Status status = ValidateRequest(
          *codebooks_, request, real_graph, virtual_neighbors,
          produced_rows_and_order);
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
          .virtual_reconstructions = request.virtual_reconstructions,
          .basic_supports = request.basic_supports,
          .basic_coefficients = request.basic_coefficients,
          .basic_squared_errors = request.basic_squared_errors,
          .validated_basic = request.validated_basic,
      },
      *scheduler_, batching_, cpu_evaluator_, request.random_stream,
      request.basic_ownership, request.owned_virtual_reconstructions,
      request.borrow_exact_real);
  if (!session_result.ok()) {
    return session_result.status();
  }
  auto session = std::move(session_result).value();

  const std::uint32_t real_count = request.exact_real.columns;
  const std::uint32_t real_search_width =
      RealCandidateSearchWidth(real_graph, request.knn_k);
  std::vector<std::uint8_t> processed(real_count, 0U);
  std::vector<LinkageParentReference> selected;
  selected.reserve(
      static_cast<std::size_t>(request.knn_k) +
      virtual_neighbors.neighbors_per_row());
  std::vector<CausalTopologyRow<M>> causal_rows;
  causal_rows.reserve(real_count);
  for (const std::uint32_t node : request.processing_order) {
    selected.clear();
    for (const std::uint32_t root :
         virtual_neighbors.row(node).local_ids) {
      selected.push_back({LinkageParentKind::kVirtual, root});
    }
    std::uint32_t selected_real = 0U;
    const PreparedKnnRow real_row = real_graph.row(node);
    for (std::uint32_t candidate = 0U;
         candidate < real_search_width; ++candidate) {
      const std::uint32_t parent = real_row.local_ids[candidate];
      if (processed[parent] != 0U) {
        selected.push_back({LinkageParentKind::kReal, parent});
        if (++selected_real == request.knn_k) {
          break;
        }
      }
    }
    CausalTopologyRow<M> causal;
    causal.real_local_id = node;
    causal.ordered_candidates = selected;
    const std::uint8_t* order = request.icm_layer_orders.data() +
        static_cast<std::size_t>(node) * codebooks_->shape().m;
    std::copy_n(order, codebooks_->shape().m, causal.icm_layer_order.begin());
    causal_rows.push_back(std::move(causal));
    processed[node] = 1U;
  }
  if (const Status status = session->ProcessGraphRows(
          causal_rows, request.maximum_depth);
      !status.ok()) {
    return status;
  }
  return session->Finish();
}

template class BadClusterTopologyBuilder<0>;
template class BadClusterTopologyBuilder<5>;
template class BadClusterTopologyBuilder<12>;
template class BadClusterTopologyBuilder<6>;
template class BadClusterTopologyBuilder<10>;

}  // namespace stlqpp
