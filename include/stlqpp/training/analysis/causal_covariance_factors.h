#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/training/update/round_update_transcript.h"

namespace stlqpp::detail {

// Compare dense BLAS work with a conservative allowance for sparse gathers.
// Small/low-dimensional CIDs retain the ordinary sample covariance.
[[nodiscard]] constexpr bool UseCausalCovarianceFactors(
    std::uint32_t dimension, std::uint32_t m, std::uint32_t real_count,
    std::uint32_t virtual_count, bool has_linkage) noexcept {
  const std::uint64_t atoms = 1U + (m - 1U) * 256U + (has_linkage ? m * 256U : 0U);
  const std::uint64_t nodes = static_cast<std::uint64_t>(real_count) + virtual_count;
  return static_cast<std::uint64_t>(real_count) * dimension >
      2U * atoms * dimension + 64U * nodes * m;
}

// Trusted frozen-forest numerical boundary. Original samples become subtree
// sums in place; the consumed reconstruction buffers become virtual sums and
// dictionary factors. No vector, coefficient or support packing is needed.
// Factors: one root0 column, (M-1)*256 root-small columns, then M*256 edge
// columns when linkage exists. The caller validates input values before the
// destructive reuse and owns all buffers through the following GEMMs.
template <int M>
void BuildCausalCovarianceFactors(
    const RoundUpdateTranscriptCluster<M>& forest,
    ArrayView<const std::uint32_t> causal_order, std::uint32_t dimension,
    ArrayView<float> real_subtrees, ArrayView<float> virtual_subtrees,
    ArrayView<float> factors) noexcept;

}  // namespace stlqpp::detail
