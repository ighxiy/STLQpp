#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/training/linkage/candidates/prepared_parent_candidates.h"

#include <cstdint>

namespace stlqpp {

template <int M>
class GoodClusterTopologyBuilder;
template <int M>
class RootOnlyInitializationClusterBuilder;

struct RadialLayerView final {
  ArrayView<const std::uint32_t> local_ids;
};

struct ExactParentCandidateRequest final {
  ColumnMajorMatrixF32View cluster_vectors;
  ArrayView<const RadialLayerView> layers;
  std::uint32_t current_layer = 0U;
  std::uint32_t inner_layer_begin = 0U;
  std::uint32_t inner_layer_end_exclusive = 0U;
  ArrayView<const std::uint32_t> current_layer_order;
  std::uint32_t knn_k = 0U;
};

class CpuExactParentCandidateProvider final {
 public:
  explicit CpuExactParentCandidateProvider(
      const F32GemmBackend& gemm) noexcept : gemm_(&gemm) {}

  [[nodiscard]] StatusOr<PreparedParentCandidates> Build(
      const ExactParentCandidateRequest& request) const;

 private:
  template <int M>
  friend class GoodClusterTopologyBuilder;
  template <int M>
  friend class RootOnlyInitializationClusterBuilder;

  [[nodiscard]] StatusOr<PreparedParentCandidates> BuildValidatedLayers(
      const ExactParentCandidateRequest& request) const;
  [[nodiscard]] StatusOr<PreparedParentCandidates> BuildImpl(
      const ExactParentCandidateRequest& request, bool validated_layers) const;
  const F32GemmBackend* gemm_ = nullptr;
};

}  // namespace stlqpp
