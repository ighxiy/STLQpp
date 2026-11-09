#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/training/update/round_update_transcript.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

struct FrozenForestReconstruction final {
  std::uint32_t dimension = 0U;
  std::uint32_t real_count = 0U;
  std::uint32_t virtual_count = 0U;
  std::vector<float> real;
  std::vector<float> virtual_roots;

  [[nodiscard]] ColumnMajorMatrixF32View real_view() const noexcept {
    return {real.data(), dimension, real_count};
  }
  [[nodiscard]] ColumnMajorMatrixF32View virtual_view() const noexcept {
    return {virtual_roots.data(), dimension, virtual_count};
  }
};

// Reconstructs one frozen forest in DEV's exact arithmetic order. Roots use
// layer-zero assignment followed by layer-major additions. Linked rows first
// form the complete edge residual and only then add their parent once.
template <int M>
[[nodiscard]] StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest(
    const ProductCodebooks<M>& codebooks,
    const RoundUpdateTranscriptCluster<M>& cluster);
template <int M>
[[nodiscard]] Status ReconstructFrozenForest(
    const ProductCodebooks<M>& codebooks,
    const RoundUpdateTranscriptCluster<M>& cluster,
    FrozenForestReconstruction* output);

template <int M>
[[nodiscard]] StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest(
    const ProductCodebooks<M>& codebooks,
    const ClusterTopologyResult<M>& topology,
    ArrayView<const std::uint8_t> virtual_small_codes,
    ArrayView<const float> virtual_coefficients);
template <int M>
[[nodiscard]] Status ReconstructFrozenForest(
    const ProductCodebooks<M>& codebooks,
    const ClusterTopologyResult<M>& topology,
    ArrayView<const std::uint8_t> virtual_small_codes,
    ArrayView<const float> virtual_coefficients,
    FrozenForestReconstruction* output);

extern template StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest<5>(
    const ProductCodebooks<5>&,
    const RoundUpdateTranscriptCluster<5>&);
extern template StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest<10>(
    const ProductCodebooks<10>&,
    const RoundUpdateTranscriptCluster<10>&);
extern template Status ReconstructFrozenForest<5>(
    const ProductCodebooks<5>&,
    const RoundUpdateTranscriptCluster<5>&,
    FrozenForestReconstruction*);
extern template Status ReconstructFrozenForest<10>(
    const ProductCodebooks<10>&,
    const RoundUpdateTranscriptCluster<10>&,
    FrozenForestReconstruction*);
extern template StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest<5>(
    const ProductCodebooks<5>&,
    const ClusterTopologyResult<5>&,
    ArrayView<const std::uint8_t>,
    ArrayView<const float>);
extern template StatusOr<FrozenForestReconstruction>
ReconstructFrozenForest<10>(
    const ProductCodebooks<10>&,
    const ClusterTopologyResult<10>&,
    ArrayView<const std::uint8_t>,
    ArrayView<const float>);
extern template Status ReconstructFrozenForest<5>(
    const ProductCodebooks<5>&,
    const ClusterTopologyResult<5>&,
    ArrayView<const std::uint8_t>,
    ArrayView<const float>,
    FrozenForestReconstruction*);
extern template Status ReconstructFrozenForest<10>(
    const ProductCodebooks<10>&,
    const ClusterTopologyResult<10>&,
    ArrayView<const std::uint8_t>,
    ArrayView<const float>,
    FrozenForestReconstruction*);

}  // namespace stlqpp
