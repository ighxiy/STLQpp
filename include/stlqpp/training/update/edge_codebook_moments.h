#pragma once

#include <cstdint>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/linkage_code_tuple.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/model/topology/canonical_bfs_forest_order.h"
#include "stlqpp/training/linkage/initialization/root_to_edge_code_map.h"
#include "stlqpp/training/linkage/topology/cluster_topology_state.h"

namespace stlqpp {

template <int M>
class RoundModelUpdater;

template <int M>
class InitializationModelUpdater;

template <int M>
struct LinkedTrainingClusterView final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  ColumnMajorMatrixF32View exact_real;
  const ClusterTopologyResult<M>* topology = nullptr;
  ColumnMajorMatrixF32View real_reconstructions;
  ColumnMajorMatrixF32View virtual_reconstructions;
};

// Frozen topology plus the minimal streams required to reconstruct only nodes
// that can be consumed as parents while accumulating exact C_one moments.
// This avoids materializing leaf reconstructions without changing the target
// or moment accumulation order.
template <int M>
struct FrozenLinkedTrainingClusterView final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  ColumnMajorMatrixF32View exact_real;
  const ClusterTopologyResult<M>* topology = nullptr;
  ArrayView<const std::uint8_t> virtual_small_codes;
  ArrayView<const float> virtual_coefficients;
};

// FP64 streaming sufficient statistics for the exact joint C_one solve.
// The packed Gram stores its upper triangle by column; the RHS stores one
// contiguous dimension-vector per flattened (layer, code) atom.
template <int M>
class EdgeCodebookMoments final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  static constexpr std::uint32_t kAtoms = M * 256U;

  [[nodiscard]] static StatusOr<EdgeCodebookMoments<M>> Create(
      std::uint32_t dimension, std::uint32_t m = FixedModelLayers<M>());

  [[nodiscard]] Status AccumulateTargets(
      ColumnMajorMatrixF32View targets,
      ArrayView<const LinkageCodeTuple<M>> supports,
      ArrayView<const float> coefficients);

  [[nodiscard]] Status AccumulateLinkedCluster(
      const LinkedTrainingClusterView<M>& cluster, double sample_weight);

  // Equivalent to ReconstructFrozenForest() followed by
  // AccumulateLinkedCluster(), but reconstructs only real/virtual nodes that
  // have at least one child. Canonical BFS order and per-sample arithmetic
  // remain unchanged, so leaf reconstruction work is removed rather than
  // approximated. Scratch capacity is retained per moment shard.
  [[nodiscard]] Status AccumulateFrozenLinkedCluster(
      const ProductCodebooks<M>& codebooks,
      const FrozenLinkedTrainingClusterView<M>& cluster, double sample_weight,
      const CanonicalBfsForestOrder* order = nullptr);

  // Initial linkage encodes residuals with C_root. Layer zero is mapped into
  // the byte C_one domain before accumulating the same exact-parent targets;
  // the remaining support and all FP32 coefficients are retained unchanged.
  [[nodiscard]] Status AccumulateRootOnlyInitializationCluster(
      const LinkedTrainingClusterView<M>& cluster,
      const RootToEdgeCodeMap& root_to_edge, double sample_weight);

  // Reduces an independently accumulated CID shard. Callers merge shards in
  // a fixed order so the result is deterministic regardless of worker timing.
  [[nodiscard]] Status MergeFrom(const EdgeCodebookMoments<M>& shard);

  [[nodiscard]] std::uint32_t m() const noexcept {
    if constexpr (M != 0) return M;
    return m_;
  }
  [[nodiscard]] std::uint32_t atom_count() const noexcept { return m() * 256U; }
  [[nodiscard]] std::uint32_t dimension() const noexcept { return dimension_; }
  [[nodiscard]] std::uint64_t sample_count() const noexcept {
    return sample_count_;
  }
  [[nodiscard]] ArrayView<const double> packed_upper_gram() const noexcept {
    return packed_upper_gram_;
  }
  [[nodiscard]] ArrayView<const double> atom_major_rhs() const noexcept {
    return atom_major_rhs_;
  }

  [[nodiscard]] static constexpr std::size_t PackedUpperIndex(
      std::uint32_t row, std::uint32_t column) noexcept {
    return static_cast<std::size_t>(column) * (column + 1U) / 2U + row;
  }

 private:
  friend class RoundModelUpdater<M>;
  friend class InitializationModelUpdater<M>;
  explicit EdgeCodebookMoments(std::uint32_t dimension, std::uint32_t m);

  [[nodiscard]] Status AccumulateFrozenLinkedClusterAfterValidation(
      const ProductCodebooks<M>& codebooks,
      const FrozenLinkedTrainingClusterView<M>& cluster, double sample_weight,
      const CanonicalBfsForestOrder& order);
  [[nodiscard]] Status AccumulateFrozenLinkedClusterAfterReplayValidation(
      const ProductCodebooks<M>& codebooks,
      const FrozenLinkedTrainingClusterView<M>& cluster, double sample_weight);

  [[nodiscard]] Status AccumulateRootOnlyInitializationClusterAfterValidation(
      const LinkedTrainingClusterView<M>& cluster,
      const RootToEdgeCodeMap& root_to_edge, double sample_weight);

  template <bool Prevalidated>
  [[nodiscard]] Status AccumulateRootOnlyInitializationClusterImpl(
      const LinkedTrainingClusterView<M>& cluster,
      const RootToEdgeCodeMap& root_to_edge, double sample_weight);

  template <bool Prevalidated, bool ExactValidated>
  [[nodiscard]] Status AccumulateFrozenLinkedClusterImpl(
      const ProductCodebooks<M>& codebooks,
      const FrozenLinkedTrainingClusterView<M>& cluster, double sample_weight,
      const CanonicalBfsForestOrder* order);

  void AccumulateOne(const float* target, const LinkageCodeTuple<M>& support,
                     const float* coefficients, double sample_weight) noexcept;

  std::uint32_t m_ = 0U;
  std::uint32_t dimension_ = 0U;
  std::uint64_t sample_count_ = 0U;
  std::vector<double> packed_upper_gram_;
  std::vector<double> atom_major_rhs_;

  // Reused only by AccumulateFrozenLinkedCluster(). Full-size backing keeps
  // parent lookup by original local id cheap; leaf columns are never filled.
  std::vector<float> frozen_real_reconstructions_;
  std::vector<float> frozen_virtual_reconstructions_;
  std::vector<std::uint8_t> frozen_real_has_child_;
  std::vector<std::uint8_t> frozen_virtual_has_child_;
  std::vector<float> frozen_target_;
};

using RuntimeEdgeCodebookMoments = EdgeCodebookMoments<0>;
extern template class EdgeCodebookMoments<0>;
extern template class EdgeCodebookMoments<5>;
extern template class EdgeCodebookMoments<10>;

}  // namespace stlqpp
