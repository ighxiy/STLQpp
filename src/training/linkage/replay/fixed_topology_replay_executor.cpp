#include "stlqpp/training/linkage/replay/fixed_topology_replay_executor.h"

#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"
#include "stlqpp/training/linkage/replay/fixed_topology_replay_plan.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

Status ValidateExactMatrix(ColumnMajorMatrixF32View matrix,
                           std::uint32_t dimension) {
  if (matrix.rows != dimension ||
      (matrix.columns != 0U && matrix.values == nullptr)) {
    return Status::InvalidArgument(
        "fixed-topology replay matrix has an invalid shape");
  }
  const std::size_t count =
      static_cast<std::size_t>(matrix.rows) * matrix.columns;
  for (std::size_t index = 0U; index < count; ++index) {
    if (!std::isfinite(matrix.values[index])) {
      return Status::InvalidArgument(
          "fixed-topology replay matrix contains non-finite data");
    }
  }
  return Status::Ok();
}

template <int M>
Status ValidateRequest(const ProductCodebooks<M>& codebooks,
                       const FixedTopologyReplayRequest& request,
                       bool validated_checkpoint_payload) {
  const std::uint32_t dimension = codebooks.shape().dimension;
  const auto m = codebooks.shape().m;
  const std::uint32_t real_count = request.exact_real.columns;
  const std::size_t virtual_count = request.virtual_sample_ids.size();
  if (request.m != m || request.cid >= codebooks.shape().nlist ||
      virtual_count > std::numeric_limits<std::uint32_t>::max() ||
      request.real_sample_ids.size() != real_count ||
      request.basic_small_codes.size() !=
          static_cast<std::size_t>(real_count) * (m - 1U) ||
      request.basic_coefficients.size() !=
          static_cast<std::size_t>(real_count) * m ||
      request.frozen_virtual_small_codes.size() !=
          static_cast<std::size_t>(virtual_count) * (m - 1U) ||
      request.frozen_virtual_coefficients.size() !=
          static_cast<std::size_t>(virtual_count) * m ||
      request.frozen_roles.size() != real_count ||
      request.frozen_parents.size() != real_count ||
      request.frozen_depths.size() != real_count) {
    return Status::InvalidArgument(
        "fixed-topology replay request has inconsistent row counts");
  }
  if (const Status status =
          ValidateExactMatrix(request.exact_real, dimension);
      !status.ok()) {
    return status;
  }
  if (request.unrotated_exact_real.rows != dimension ||
      request.unrotated_exact_real.columns != real_count ||
      (real_count != 0U && request.unrotated_exact_real.values == nullptr)) {
    return Status::InvalidArgument(
        "fixed-topology replay unrotated rows have an invalid shape");
  }
  if (!validated_checkpoint_payload) {
    for (const float coefficient : request.basic_coefficients) {
      if (!std::isfinite(coefficient)) {
        return Status::InvalidArgument(
            "fixed-topology replay basic coefficient is non-finite");
      }
    }
    for (const float coefficient : request.frozen_virtual_coefficients) {
      if (!std::isfinite(coefficient)) {
        return Status::InvalidArgument(
            "fixed-topology replay virtual coefficient is non-finite");
      }
    }
    return ValidateIcmLayerOrders<M>(
        real_count, request.edge_icm_layer_orders, m);
  }
#ifndef NDEBUG
  for (const float coefficient : request.basic_coefficients) {
    assert(std::isfinite(coefficient));
  }
  for (const float coefficient : request.frozen_virtual_coefficients) {
    assert(std::isfinite(coefficient));
  }
  assert(ValidateIcmLayerOrders<M>(
      real_count, request.edge_icm_layer_orders, m).ok());
#endif
  return Status::Ok();
}

template <int M>
float ReconstructRoot(const ProductCodebooks<M>& codebooks,
                      std::uint32_t cid,
                      const std::uint8_t* small_codes,
                      const float* coefficients,
                      const float* exact,
                      float* reconstruction) noexcept {
  const std::uint32_t dimension = codebooks.shape().dimension;
  const auto m = codebooks.shape().m;
  const float* first = codebooks.root0().column(cid);
  for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
    reconstruction[axis] = coefficients[0] * first[axis];
  }
  for (std::uint32_t layer = 1U;
       layer < m; ++layer) {
    const float* atom = codebooks.root_small(layer).column(
        small_codes[layer - 1U]);
    const float coefficient = coefficients[layer];
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      reconstruction[axis] += coefficient * atom[axis];
    }
  }
  float squared_error = 0.0F;
  if (exact == nullptr) return squared_error;
  for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
    const float difference = exact[axis] - reconstruction[axis];
    squared_error += difference * difference;
  }
  return squared_error;
}

template <int M>
struct ReplayClusterState final {
  const FixedTopologyReplayRequest* request = nullptr;
  FixedTopologyReplayPlan plan;
  FixedTopologyReplayResult<M> output;
  // Scratch belongs to execution, not to the queued publication result.
  // Direct callers requesting reconstructions receive it by move at the end.
  std::vector<float> real_reconstructions;
  std::uint32_t window_column_base = 0U;
};

template <int M>
ReplayClusterState<M> InitializeCluster(
    const ProductCodebooks<M>& codebooks,
    const FixedTopologyReplayRequest& request,
    FixedTopologyReplayPlan plan) {
  const std::uint32_t dimension = codebooks.shape().dimension;
  const auto m = codebooks.shape().m;
  const std::uint32_t real_count = request.exact_real.columns;
  const std::uint32_t virtual_count = static_cast<std::uint32_t>(
      request.virtual_sample_ids.size());
  ReplayClusterState<M> state;
  state.request = &request;
  state.plan = std::move(plan);
  ClusterTopologyResult<M>& topology = state.output.topology;
  topology.m = m;
  topology.cid = request.cid;
  topology.dimension = dimension;
  topology.virtual_count = virtual_count;
  topology.roles.assign(request.frozen_roles.begin(),
                        request.frozen_roles.end());
  topology.parents.assign(request.frozen_parents.begin(),
                          request.frozen_parents.end());
  topology.depths.assign(request.frozen_depths.begin(),
                         request.frozen_depths.end());
  topology.supports.resize(real_count);
  topology.coefficients.assign(
      static_cast<std::size_t>(real_count) * m, 0.0F);
  topology.squared_errors.assign(real_count, 0.0F);
  state.real_reconstructions.assign(
      static_cast<std::size_t>(dimension) * real_count, 0.0F);
  state.output.virtual_small_codes.resize(
      static_cast<std::size_t>(virtual_count) * (m - 1));
  state.output.virtual_coefficients.resize(
      static_cast<std::size_t>(virtual_count) * m);
  state.output.virtual_reconstructions.resize(
      static_cast<std::size_t>(dimension) * virtual_count);
  return state;
}

struct EdgeDestination final {
  std::uint32_t cluster = 0U;
  std::uint32_t real_local_id = 0U;
};

}  // namespace

template <int M>
StatusOr<FixedTopologyReplayResult<M>>
FixedTopologyReplayExecutor<M>::Execute(
    const FixedTopologyReplayRequest& request) const {
  const std::array<FixedTopologyReplayRequest, 1U> requests{request};
  auto result = ExecuteWindow(requests);
  if (!result.ok()) {
    return result.status();
  }
  auto outputs = std::move(result).value();
  return std::move(outputs.front());
}

template <int M>
StatusOr<std::vector<FixedTopologyReplayResult<M>>>
FixedTopologyReplayExecutor<M>::ExecuteWindow(
    ArrayView<const FixedTopologyReplayRequest> requests,
    bool retain_real_reconstructions) const {
  return ExecuteWindowImpl(requests, retain_real_reconstructions, false);
}

template <int M>
StatusOr<std::vector<FixedTopologyReplayResult<M>>>
FixedTopologyReplayExecutor<M>::ExecuteWindowFromCheckpoint(
    ArrayView<const FixedTopologyReplayRequest> requests,
    bool retain_real_reconstructions) const {
  return ExecuteWindowImpl(requests, retain_real_reconstructions, true);
}

template <int M>
StatusOr<std::vector<FixedTopologyReplayResult<M>>>
FixedTopologyReplayExecutor<M>::ExecuteWindowImpl(
    ArrayView<const FixedTopologyReplayRequest> requests,
    bool retain_real_reconstructions,
    bool validated_checkpoint_payload) const {
  if (requests.empty() || requests.size() > kMaximumWindowClusters) {
    return Status::InvalidArgument(
        "fixed-topology replay window must contain 1..64 clusters");
  }
  const std::uint32_t seed = requests[0].seed;
  const auto m = codebooks_->shape().m;
  std::vector<ReplayClusterState<M>> clusters;
  clusters.reserve(requests.size());
  std::uint32_t maximum_depth = 0U;
  for (const auto& request : requests) {
    if (request.seed != seed) {
      return Status::InvalidArgument(
          "one replay window requires one deterministic round seed");
    }
    if (const Status status = ValidateRequest(
            *codebooks_, request, validated_checkpoint_payload);
        !status.ok()) {
      return status;
    }
    const TypedForestView forest = {
        .virtual_count = static_cast<std::uint32_t>(
            request.virtual_sample_ids.size()),
        .real_roles = request.frozen_roles,
        .real_parents = request.frozen_parents,
    };
    auto plan = validated_checkpoint_payload
        ? FixedTopologyReplayPlanFromCheckpoint::Build(
              forest, request.frozen_depths)
        : BuildFixedTopologyReplayPlan(
              forest, request.frozen_depths);
    if (!plan.ok()) {
      return plan.status();
    }
    auto state = InitializeCluster(
        *codebooks_, request, std::move(plan).value());
    if (!state.plan.linked_by_depth.empty()) {
      maximum_depth = std::max(
          maximum_depth,
          state.plan.linked_by_depth.back().depth);
    }
    clusters.push_back(std::move(state));
  }
  const std::uint32_t dimension = codebooks_->shape().dimension;
  std::uint64_t window_columns64 = 0U;
  for (ReplayClusterState<M>& cluster : clusters) {
    const std::uint64_t cluster_columns =
        static_cast<std::uint64_t>(
            cluster.request->virtual_sample_ids.size()) +
        cluster.request->exact_real.columns;
    if (window_columns64 + cluster_columns >
        std::numeric_limits<std::uint32_t>::max()) {
      return Status::InvalidArgument(
          "fixed-topology replay window columns exceed uint32");
    }
    cluster.window_column_base =
        static_cast<std::uint32_t>(window_columns64);
    window_columns64 += cluster_columns;
  }
  const std::uint32_t window_columns =
      static_cast<std::uint32_t>(window_columns64);
  for (ReplayClusterState<M>& cluster : clusters) {
    const auto& request = *cluster.request;
    for (std::uint32_t root = 0U;
         root < request.virtual_sample_ids.size(); ++root) {
      const std::uint8_t* small_codes =
          request.frozen_virtual_small_codes.data() +
          static_cast<std::size_t>(root) * (m - 1U);
      const float* coefficients =
          request.frozen_virtual_coefficients.data() +
          static_cast<std::size_t>(root) * m;
      std::copy_n(
          small_codes, m - 1U,
          cluster.output.virtual_small_codes.data() +
              static_cast<std::size_t>(root) * (m - 1U));
      std::copy_n(
          coefficients, m,
          cluster.output.virtual_coefficients.data() +
              static_cast<std::size_t>(root) * m);
      static_cast<void>(ReconstructRoot(
          *codebooks_, request.cid, small_codes, coefficients,
          nullptr,
          cluster.output.virtual_reconstructions.data() +
              static_cast<std::size_t>(root) * dimension));
    }
    for (const std::uint32_t real : cluster.plan.real_roots) {
      const std::uint8_t* small_codes = request.basic_small_codes.data() +
          static_cast<std::size_t>(real) * (m - 1U);
      const float* coefficients = request.basic_coefficients.data() +
          static_cast<std::size_t>(real) * m;
      auto& topology = cluster.output.topology;
      LinkageCodeTuple<M>& support = topology.supports[real];
      support.first = request.cid;
      std::copy_n(small_codes, m - 1U, support.trailing.begin());
      std::copy_n(
          coefficients, m,
          topology.coefficients.data() +
              static_cast<std::size_t>(real) * m);
      topology.squared_errors[real] = ReconstructRoot(
          *codebooks_, request.cid, small_codes, coefficients,
          request.exact_real.column(real),
          cluster.real_reconstructions.data() +
              static_cast<std::size_t>(real) * dimension);
    }
  }

  std::vector<float> window_roots;
  std::vector<std::uint32_t> window_root_columns;
  std::vector<std::size_t> depth_cursors(clusters.size(), 0U);
  // CPU: residuals; CUDA: exact rows for its resident-parent gather.
  std::vector<float> depth_vectors;
  std::vector<std::uint8_t> batch_orders;
  std::vector<std::uint32_t> parent_columns;
  std::vector<std::uint32_t> destination_columns;
  std::vector<EdgeDestination> edge_destinations;
  struct WindowLifetime {
    const FixedEdgeReplayEncoder<M>* encoder = nullptr;
    ~WindowLifetime() { if (encoder != nullptr) encoder->CancelWindow(); }
  } window_lifetime;
  // Root-only windows never invoke the edge backend. In mixed windows,
  // pack virtual/real roots once; linked columns have no initial consumer.
  if (maximum_depth != 0U && edge_encoder_->RequiresWindowReconstructions()) {
    std::size_t root_count = 0U;
    for (const auto& cluster : clusters)
      root_count += cluster.request->virtual_sample_ids.size() + cluster.plan.real_roots.size();
    window_roots.reserve(root_count * dimension);
    window_root_columns.reserve(root_count);
    for (const auto& cluster : clusters) {
      const auto virtual_count = static_cast<std::uint32_t>(cluster.request->virtual_sample_ids.size());
      window_roots.insert(window_roots.end(), cluster.output.virtual_reconstructions.begin(),
                          cluster.output.virtual_reconstructions.end());
      for (std::uint32_t root = 0U; root < virtual_count; ++root)
        window_root_columns.push_back(cluster.window_column_base + root);
      for (const auto real : cluster.plan.real_roots) {
        const float* vector = cluster.real_reconstructions.data() + static_cast<std::size_t>(real) * dimension;
        window_roots.insert(window_roots.end(), vector, vector + dimension);
        window_root_columns.push_back(cluster.window_column_base + virtual_count + real);
      }
    }
    window_lifetime.encoder = edge_encoder_;
    if (const auto status = edge_encoder_->BeginWindow(
            {window_roots.data(), dimension, static_cast<std::uint32_t>(root_count)},
            window_root_columns, window_columns); !status.ok()) return status;
  }

  const bool host_residuals = edge_encoder_->UsesHostResiduals();
  const bool reconstruct_linked_on_host =
      host_residuals || retain_real_reconstructions;
  for (std::uint32_t depth = 1U; depth <= maximum_depth; ++depth) {
    std::uint64_t batch_size64 = 0U;
    for (std::uint32_t cluster_id = 0U;
         cluster_id < clusters.size(); ++cluster_id) {
      const auto& batches = clusters[cluster_id].plan.linked_by_depth;
      const std::size_t cursor = depth_cursors[cluster_id];
      if (cursor < batches.size() && batches[cursor].depth == depth) {
        batch_size64 += batches[cursor].real_local_ids.size();
      }
    }
    if (batch_size64 == 0U) {
      continue;
    }
    if (batch_size64 > std::numeric_limits<std::uint32_t>::max()) {
      return Status::InvalidArgument(
          "fixed-topology replay depth batch exceeds uint32");
    }
    const std::uint32_t batch_size =
        static_cast<std::uint32_t>(batch_size64);
    depth_vectors.resize(static_cast<std::size_t>(dimension) * batch_size);
    batch_orders.resize(static_cast<std::size_t>(batch_size) * m);
    parent_columns.clear();
    destination_columns.clear();
    if (!host_residuals) {
      parent_columns.reserve(batch_size);
      destination_columns.reserve(batch_size);
    }
    edge_destinations.clear();
    edge_destinations.reserve(batch_size);

    std::uint32_t gathered = 0U;
    for (std::uint32_t cluster_id = 0U;
         cluster_id < clusters.size(); ++cluster_id) {
      ReplayClusterState<M>& cluster = clusters[cluster_id];
      const auto& batches = cluster.plan.linked_by_depth;
      const std::size_t cursor = depth_cursors[cluster_id];
      if (cursor >= batches.size() || batches[cursor].depth != depth) {
        continue;
      }
      const auto& request = *cluster.request;
      auto& topology = cluster.output.topology;
      for (const std::uint32_t real : batches[cursor].real_local_ids) {
        const LinkageParentReference parent = *topology.parents[real];
        const float* exact = request.exact_real.column(real);
        float* destination = depth_vectors.data() +
            static_cast<std::size_t>(gathered) * dimension;
        if (host_residuals) {
          const float* parent_reconstruction =
              parent.kind == LinkageParentKind::kVirtual
              ? cluster.output.virtual_reconstructions.data() +
                    static_cast<std::size_t>(parent.local_id) * dimension
              : cluster.real_reconstructions.data() +
                    static_cast<std::size_t>(parent.local_id) * dimension;
          for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
            destination[axis] = exact[axis] - parent_reconstruction[axis];
          }
        } else {
          std::copy_n(exact, dimension, destination);
          const std::uint32_t parent_column = cluster.window_column_base +
              (parent.kind == LinkageParentKind::kVirtual
                   ? parent.local_id
                   : static_cast<std::uint32_t>(
                         request.virtual_sample_ids.size()) + parent.local_id);
          parent_columns.push_back(parent_column);
          destination_columns.push_back(
              cluster.window_column_base + static_cast<std::uint32_t>(
                  request.virtual_sample_ids.size()) + real);
        }
        std::copy_n(
            request.edge_icm_layer_orders.data() +
                static_cast<std::size_t>(real) * m,
            m, batch_orders.data() +
                   static_cast<std::size_t>(gathered) * m);
        edge_destinations.push_back({cluster_id, real});
        ++gathered;
      }
      ++depth_cursors[cluster_id];
    }

    const ColumnMajorMatrixF32View batch{
        depth_vectors.data(), dimension, batch_size};
    auto encoded = host_residuals
        ? edge_encoder_->EncodeResidualsValidated(batch, batch_orders)
        : edge_encoder_->EncodeWindowBatchValidated(
              batch, {nullptr, dimension, batch_size},
              parent_columns, destination_columns, batch_orders);
    if (!encoded.ok()) {
      return encoded.status();
    }
    assert(encoded.value().sample_count() == batch_size);
    for (std::uint32_t row = 0U; row < batch_size; ++row) {
      const EdgeDestination destination = edge_destinations[row];
      ReplayClusterState<M>& cluster = clusters[destination.cluster];
      auto& topology = cluster.output.topology;
      const std::uint32_t real = destination.real_local_id;
      LinkageCodeTuple<M>& support = topology.supports[real];
      support.first = encoded.value().code(row, 0U);
      for (std::uint32_t layer = 1U;
           layer < m; ++layer) {
        support.trailing[layer - 1U] = encoded.value().code(row, layer);
      }
      float* coefficients = topology.coefficients.data() +
          static_cast<std::size_t>(real) * m;
      for (std::uint32_t layer = 0U;
           layer < m; ++layer) {
        coefficients[layer] = encoded.value().coefficient(row, layer);
      }
      topology.squared_errors[real] =
          encoded.value().squared_error(row);

      // CUDA already scattered this child into the window-resident device
      // matrix. Production sinks consume its support/error, not a second host
      // reconstruction; direct callers still request the historical result.
      if (!reconstruct_linked_on_host) {
        continue;
      }

      const LinkageParentReference parent = *topology.parents[real];
      const float* parent_reconstruction =
          parent.kind == LinkageParentKind::kVirtual
          ? cluster.output.virtual_reconstructions.data() +
                static_cast<std::size_t>(parent.local_id) * dimension
          : cluster.real_reconstructions.data() +
                static_cast<std::size_t>(parent.local_id) * dimension;
      float* reconstruction = cluster.real_reconstructions.data() +
          static_cast<std::size_t>(real) * dimension;
      std::copy_n(parent_reconstruction, dimension, reconstruction);
      for (std::uint32_t layer = 0U;
           layer < m; ++layer) {
        const float* atom = codebooks_->edge(layer).column(
            support.code(layer));
        const float coefficient = coefficients[layer];
        for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
          reconstruction[axis] += coefficient * atom[axis];
        }
      }
    }
  }

  std::vector<FixedTopologyReplayResult<M>> results;
  results.reserve(clusters.size());
  for (auto& cluster : clusters) {
    if (retain_real_reconstructions) {
      cluster.output.topology.real_reconstructions =
          std::move(cluster.real_reconstructions);
    }
    results.push_back(std::move(cluster.output));
  }
  window_lifetime.encoder = nullptr;
  return results;
}

template class FixedTopologyReplayExecutor<0>;
template class FixedTopologyReplayExecutor<5>;
template class FixedTopologyReplayExecutor<12>;
template class FixedTopologyReplayExecutor<6>;
template class FixedTopologyReplayExecutor<10>;

}  // namespace stlqpp
