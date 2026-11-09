#include "stlqpp/training/linkage/candidates/cpu_exact_parent_candidates.h"

#include "exact_parent_prefilter.h"
#include "ranked_neighbor_heap.h"

#include "stlqpp/platform/cpu_features.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace stlqpp {
namespace {


constexpr std::uint32_t kParentTile = 1024U;
constexpr std::uint32_t kQueryTile = 256U;
constexpr std::uint32_t kSameLayerTile = 256U;

using linkage_candidates::OfferRankedNeighbor;
using linkage_candidates::PrefilterEightParentCandidatesAvx2;
using linkage_candidates::RankedNeighbor;
using linkage_candidates::RankedNeighborLess;

void OfferParentRange(float query_norm,
                      const float* parent_norms,
                      const float* dots,
                      const std::uint32_t* local_ids,
                      std::uint32_t candidate_count,
                      RankedNeighbor* heap,
                      std::uint32_t* heap_count,
                      std::uint32_t capacity) {
  std::uint32_t candidate = 0U;
  while (candidate < candidate_count && *heap_count < capacity) {
    const float distance = query_norm + parent_norms[candidate] -
        2.0F * dots[candidate];
    OfferRankedNeighbor(
        {distance, local_ids[candidate]}, heap, heap_count, capacity);
    ++candidate;
  }

  static const bool use_avx2 = [] {
    const CpuFeatures features = DetectCpuFeatures();
    return features.avx_register_state && features.avx2;
  }();
  if (use_avx2) {
    alignas(32) float distances[8];
    for (; candidate + 8U <= candidate_count; candidate += 8U) {
      const RankedNeighbor threshold = heap[0];
      const std::uint32_t mask = PrefilterEightParentCandidatesAvx2(
          query_norm, parent_norms + candidate, dots + candidate,
          local_ids + candidate, threshold.distance, threshold.local_id,
          distances);
      for (std::uint32_t lane = 0U; lane < 8U; ++lane) {
        if ((mask & (1U << lane)) == 0U) continue;
        OfferRankedNeighbor(
            {distances[lane], local_ids[candidate + lane]}, heap,
            heap_count, capacity);
      }
    }
  }
  for (; candidate < candidate_count; ++candidate) {
    const float distance = query_norm + parent_norms[candidate] -
        2.0F * dots[candidate];
    OfferRankedNeighbor(
        {distance, local_ids[candidate]}, heap, heap_count, capacity);
  }
}

[[nodiscard]] float SquaredNorm(const float* vector,
                                std::uint32_t dimension) noexcept {
  float result = 0.0F;
#pragma omp simd reduction(+:result)
  for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
    result += vector[axis] * vector[axis];
  }
  return result;
}

Status ValidateRequest(const ExactParentCandidateRequest& request,
                       std::vector<std::uint32_t>* ordered_queries,
                       std::vector<std::uint32_t>* inner_parents) {
  if (request.cluster_vectors.values == nullptr ||
      request.cluster_vectors.rows == 0U ||
      request.cluster_vectors.columns == 0U ||
      request.layers.data() == nullptr || request.layers.empty() ||
      request.current_layer >= request.layers.size() ||
      request.inner_layer_begin > request.inner_layer_end_exclusive ||
      request.inner_layer_end_exclusive > request.current_layer ||
      request.knn_k == 0U) {
    return Status::InvalidArgument(
        "exact parent candidate request has invalid matrix or layer extents");
  }

  const RadialLayerView current = request.layers[request.current_layer];
  if ((current.local_ids.data() == nullptr && !current.local_ids.empty()) ||
      request.current_layer_order.size() != current.local_ids.size() ||
      (request.current_layer_order.data() == nullptr &&
       !request.current_layer_order.empty())) {
    return Status::InvalidArgument(
        "current radial layer order must be a complete permutation");
  }

  const std::uint32_t cluster_count = request.cluster_vectors.columns;
  std::vector<std::uint8_t> membership(cluster_count, 0U);
  inner_parents->clear();
  for (std::uint32_t layer = request.inner_layer_begin;
       layer < request.inner_layer_end_exclusive; ++layer) {
    const ArrayView<const std::uint32_t> ids = request.layers[layer].local_ids;
    if (ids.data() == nullptr && !ids.empty()) {
      return Status::InvalidArgument("inner radial layer has a null extent");
    }
    for (const std::uint32_t local_id : ids) {
      if (local_id >= cluster_count || membership[local_id] != 0U) {
        return Status::InvalidArgument(
            "inner radial layers must contain unique cluster-local IDs");
      }
      membership[local_id] = 1U;
      inner_parents->push_back(local_id);
    }
  }

  std::vector<std::uint8_t> seen_order(current.local_ids.size(), 0U);
  ordered_queries->resize(current.local_ids.size());
  for (std::size_t rank = 0U; rank < current.local_ids.size(); ++rank) {
    const std::uint32_t position = request.current_layer_order[rank];
    if (position >= current.local_ids.size() || seen_order[position] != 0U) {
      return Status::InvalidArgument(
          "current radial layer order is not a permutation");
    }
    seen_order[position] = 1U;
    const std::uint32_t local_id = current.local_ids[position];
    if (local_id >= cluster_count || membership[local_id] != 0U) {
      return Status::InvalidArgument(
          "current radial layer must be unique and disjoint from inner layers");
    }
    membership[local_id] = 2U;
    (*ordered_queries)[rank] = local_id;
  }
  return Status::Ok();
}

// GOOD and root-only initialization build their complete radial partition
// through RadialLayerPlan; their order comes from Permutation().
Status PrepareValidatedLayers(const ExactParentCandidateRequest& request,
                              std::vector<std::uint32_t>* ordered_queries,
                              std::vector<std::uint32_t>* inner_parents) {
#ifndef NDEBUG
  std::vector<std::uint32_t> checked_queries;
  std::vector<std::uint32_t> checked_parents;
  assert(ValidateRequest(request, &checked_queries, &checked_parents).ok());
#endif
  if (request.cluster_vectors.values == nullptr ||
      request.cluster_vectors.rows == 0U ||
      request.cluster_vectors.columns == 0U ||
      request.layers.empty() ||
      request.current_layer >= request.layers.size() ||
      request.inner_layer_begin > request.inner_layer_end_exclusive ||
      request.inner_layer_end_exclusive > request.current_layer ||
      request.knn_k == 0U ||
      request.current_layer_order.size() !=
          request.layers[request.current_layer].local_ids.size()) {
    return Status::InvalidArgument(
        "validated exact parent candidate request has invalid extents");
  }
  inner_parents->clear();
  for (std::uint32_t layer = request.inner_layer_begin;
       layer < request.inner_layer_end_exclusive; ++layer) {
    const auto ids = request.layers[layer].local_ids;
    if (!ids.empty()) {
      inner_parents->insert(inner_parents->end(), ids.begin(), ids.end());
    }
  }
  const auto current = request.layers[request.current_layer].local_ids;
  ordered_queries->resize(current.size());
  for (std::size_t rank = 0U; rank < current.size(); ++rank) {
    const std::uint32_t position = request.current_layer_order[rank];
    assert(position < current.size());
    (*ordered_queries)[rank] = current[position];
  }
#ifndef NDEBUG
  assert(*ordered_queries == checked_queries);
  assert(*inner_parents == checked_parents);
#endif
  return Status::Ok();
}

void Gather(const ColumnMajorMatrixF32View& source,
            ArrayView<const std::uint32_t> local_ids,
            std::vector<float>* matrix,
            std::vector<float>* norms) {
  const std::size_t dimension = source.rows;
  matrix->resize(dimension * local_ids.size());
  norms->resize(local_ids.size());
  for (std::size_t column = 0U; column < local_ids.size(); ++column) {
    const float* input = source.column(local_ids[column]);
    float* output = matrix->data() + column * dimension;
    std::copy_n(input, dimension, output);
    (*norms)[column] = SquaredNorm(input, source.rows);
  }
}

Status OfferInnerParents(
    const F32GemmBackend& gemm,
    const ExactParentCandidateRequest& request,
    const std::vector<std::uint32_t>& inner_parents,
    const std::vector<float>& queries,
    const std::vector<float>& query_norms,
    std::uint32_t capacity,
    std::vector<RankedNeighbor>* heaps,
    std::vector<std::uint32_t>* counts) {
  if (inner_parents.empty() || query_norms.empty()) {
    return Status::Ok();
  }
  const std::uint32_t dimension = request.cluster_vectors.rows;
  const std::uint32_t query_count =
      static_cast<std::uint32_t>(query_norms.size());
  std::vector<float> parents;
  std::vector<float> parent_norms;
  std::vector<float> dots;

  for (std::uint32_t parent_begin = 0U;
       parent_begin < inner_parents.size(); parent_begin += kParentTile) {
    const std::uint32_t parent_count = std::min(
        kParentTile,
        static_cast<std::uint32_t>(inner_parents.size()) - parent_begin);
    const ArrayView<const std::uint32_t> parent_ids(
        inner_parents.data() + parent_begin, parent_count);
    Gather(request.cluster_vectors, parent_ids, &parents, &parent_norms);

    for (std::uint32_t query_begin = 0U; query_begin < query_count;
         query_begin += kQueryTile) {
      const std::uint32_t tile_queries =
          std::min(kQueryTile, query_count - query_begin);
      dots.resize(static_cast<std::size_t>(parent_count) * tile_queries);
      const Status status = gemm.MultiplyTransposedLeft({
          .left = {parents.data(), dimension, parent_count},
          .right = queries.data() +
              static_cast<std::size_t>(query_begin) * dimension,
          .query_count = tile_queries,
          .right_leading_dimension = dimension,
          .output = dots.data(),
          .output_leading_dimension = parent_count,
      });
      if (!status.ok()) {
        return status;
      }
      for (std::uint32_t query = 0U; query < tile_queries; ++query) {
        const std::uint32_t query_index = query_begin + query;
        RankedNeighbor* heap = heaps->data() +
            static_cast<std::size_t>(query_index) * capacity;
        std::uint32_t* count = counts->data() + query_index;
        const float* query_dots = dots.data() +
            static_cast<std::size_t>(query) * parent_count;
        OfferParentRange(
            query_norms[query_index], parent_norms.data(), query_dots,
            parent_ids.data(), parent_count, heap, count, capacity);
      }
    }
  }
  return Status::Ok();
}

Status OfferCommittedPrefix(
    const F32GemmBackend& gemm,
    const std::vector<std::uint32_t>& ordered_queries,
    const std::vector<float>& queries,
    const std::vector<float>& query_norms,
    std::uint32_t dimension,
    std::uint32_t capacity,
    std::vector<RankedNeighbor>* heaps,
    std::vector<std::uint32_t>* counts) {
  const std::uint32_t query_count =
      static_cast<std::uint32_t>(ordered_queries.size());
  std::vector<float> dots;
  for (std::uint32_t query_begin = 0U; query_begin < query_count;
       query_begin += kSameLayerTile) {
    const std::uint32_t tile_queries =
        std::min(kSameLayerTile, query_count - query_begin);
    for (std::uint32_t parent_begin = 0U; parent_begin <= query_begin;
         parent_begin += kSameLayerTile) {
      const std::uint32_t parent_count =
          std::min(kSameLayerTile, query_count - parent_begin);
      dots.resize(static_cast<std::size_t>(parent_count) * tile_queries);
      const float* parent_values = queries.data() +
          static_cast<std::size_t>(parent_begin) * dimension;
      const Status status = parent_begin == query_begin
          ? gemm.MultiplySelfTransposedUpper({
                .input = {parent_values, dimension, parent_count},
                .output = dots.data(),
                .output_leading_dimension = parent_count,
            })
          : gemm.MultiplyTransposedLeft({
                .left = {parent_values, dimension, parent_count},
                .right = queries.data() +
                    static_cast<std::size_t>(query_begin) * dimension,
                .query_count = tile_queries,
                .right_leading_dimension = dimension,
                .output = dots.data(),
                .output_leading_dimension = parent_count,
            });
      if (!status.ok()) {
        return status;
      }
      for (std::uint32_t query = 0U; query < tile_queries; ++query) {
        const std::uint32_t query_rank = query_begin + query;
        RankedNeighbor* heap = heaps->data() +
            static_cast<std::size_t>(query_rank) * capacity;
        std::uint32_t* count = counts->data() + query_rank;
        const std::uint32_t valid_parent_count =
            parent_begin == query_begin ? query : parent_count;
        const float* query_dots = dots.data() +
            static_cast<std::size_t>(query) * parent_count;
        OfferParentRange(
            query_norms[query_rank], query_norms.data() + parent_begin,
            query_dots, ordered_queries.data() + parent_begin,
            valid_parent_count, heap, count, capacity);
      }
    }
  }
  return Status::Ok();
}

}  // namespace

StatusOr<PreparedParentCandidates> CpuExactParentCandidateProvider::Build(
    const ExactParentCandidateRequest& request) const {
  return BuildImpl(request, false);
}

StatusOr<PreparedParentCandidates>
CpuExactParentCandidateProvider::BuildValidatedLayers(
    const ExactParentCandidateRequest& request) const {
  return BuildImpl(request, true);
}

StatusOr<PreparedParentCandidates>
CpuExactParentCandidateProvider::BuildImpl(
    const ExactParentCandidateRequest& request, bool validated_layers) const {
  std::vector<std::uint32_t> ordered_queries;
  std::vector<std::uint32_t> inner_parents;
  const Status validation = validated_layers
      ? PrepareValidatedLayers(request, &ordered_queries, &inner_parents)
      : ValidateRequest(request, &ordered_queries, &inner_parents);
  if (!validation.ok()) {
    return validation;
  }

  const std::uint64_t depth_span =
      request.inner_layer_end_exclusive - request.inner_layer_begin;
  const std::uint64_t raw_capacity =
      (depth_span + 1U) * request.knn_k / 2U;
  const std::uint32_t capacity = static_cast<std::uint32_t>(std::min(
      std::max<std::uint64_t>(1U, raw_capacity),
      static_cast<std::uint64_t>(request.cluster_vectors.columns)));
  const std::uint64_t heap_count =
      static_cast<std::uint64_t>(ordered_queries.size()) * capacity;
  if (heap_count > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "prepared parent candidate CSR exceeds the u32 cluster extent");
  }

  std::vector<float> queries;
  std::vector<float> query_norms;
  Gather(request.cluster_vectors, ordered_queries, &queries, &query_norms);
  std::vector<RankedNeighbor> heaps(static_cast<std::size_t>(heap_count));
  std::vector<std::uint32_t> counts(ordered_queries.size(), 0U);

  Status status = OfferInnerParents(
      *gemm_, request, inner_parents, queries, query_norms, capacity,
      &heaps, &counts);
  if (!status.ok()) {
    return status;
  }
  status = OfferCommittedPrefix(
      *gemm_, ordered_queries, queries, query_norms,
      request.cluster_vectors.rows, capacity, &heaps, &counts);
  if (!status.ok()) {
    return status;
  }

  PreparedParentCandidates result;
  result.query_local_ids_ = std::move(ordered_queries);
  result.offsets_.resize(result.query_local_ids_.size() + 1U, 0U);
  result.parent_local_ids_.reserve(static_cast<std::size_t>(heap_count));
  for (std::size_t row = 0U; row < counts.size(); ++row) {
    RankedNeighbor* begin = heaps.data() + row * capacity;
    std::sort(begin, begin + counts[row], RankedNeighborLess);
    for (std::uint32_t index = 0U; index < counts[row]; ++index) {
      result.parent_local_ids_.push_back(begin[index].local_id);
    }
    result.offsets_[row + 1U] =
        static_cast<std::uint32_t>(result.parent_local_ids_.size());
  }
  return result;
}

}  // namespace stlqpp
