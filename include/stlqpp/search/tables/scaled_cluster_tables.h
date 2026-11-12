#pragma once

#include "stlqpp/search/layout/resident_cluster.h"
#include "stlqpp/search/query_tables.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace stlqpp {

#if defined(_MSC_VER)
#define STLQPP_TABLE_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define STLQPP_TABLE_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define STLQPP_TABLE_ALWAYS_INLINE inline
#endif

#if defined(_MSC_VER)
#pragma warning(push)
// The SIMD tables intentionally contain padding to align each array to 32 bytes.
#pragma warning(disable : 4324)
#endif

template <int M>
class ScaledClusterQueryTables final {
 public:
  static_assert(M >= 2 && M <= 20);

  [[nodiscard]] float root0() const noexcept { return root0_; }
  [[nodiscard]] const float* root_small(std::uint32_t layer) const noexcept {
    return root_small_.data() +
           static_cast<std::size_t>(layer - 1U) * 256U;
  }
  [[nodiscard]] const float* edge(std::uint32_t layer) const noexcept {
    return edge_.data() + static_cast<std::size_t>(layer) * 256U;
  }

 private:
  template <int N, bool NegativeTwo>
  friend void PrepareScaledClusterQueryTablesImpl(
      const TrustedQueryTables<N>&,
      const ResidentClusterView<N>&,
      ScaledClusterQueryTables<N>* __restrict) noexcept;
  template <int N>
  friend void PrepareScaledClusterQueryTables(
      const TrustedQueryTables<N>&,
      const ResidentClusterView<N>&,
      ScaledClusterQueryTables<N>* __restrict) noexcept;
  template <int N>
  friend void PrepareNegativeTwoScaledClusterQueryTables(
      const TrustedQueryTables<N>&,
      const ResidentClusterView<N>&,
      ScaledClusterQueryTables<N>* __restrict) noexcept;

  float root0_ = 0.0F;
  alignas(32) std::array<float, (M - 1) * 256> root_small_{};
  alignas(32) std::array<float, M * 256> edge_{};
};

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

template <int M, bool NegativeTwo>
STLQPP_TABLE_ALWAYS_INLINE void PrepareScaledClusterQueryTablesImpl(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict output) noexcept {
  constexpr float score_scale = NegativeTwo ? -2.0F : 1.0F;
  output->root0_ = query.root0(cluster.cid()) *
      (score_scale * cluster.root_scale(0U));
  for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(M);
       ++layer) {
    float* __restrict destination = output->root_small_.data() +
        static_cast<std::size_t>(layer - 1U) * 256U;
    const float scale = score_scale * cluster.root_scale(layer);
    for (std::uint32_t code = 0U; code < 256U; ++code) {
      destination[code] = query.root_small(layer, static_cast<std::uint8_t>(code)) *
                          scale;
    }
  }
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(M);
       ++layer) {
    float* __restrict destination = output->edge_.data() +
        static_cast<std::size_t>(layer) * 256U;
    const float scale = score_scale * cluster.linkage_scale(layer);
    for (std::uint32_t code = 0U; code < 256U; ++code) {
      destination[code] = query.edge(layer, static_cast<std::uint8_t>(code)) *
                          scale;
    }
  }
}

template <int M>
STLQPP_TABLE_ALWAYS_INLINE void PrepareScaledClusterQueryTables(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict output) noexcept {
  PrepareScaledClusterQueryTablesImpl<M, false>(query, cluster, output);
}

template <int M>
STLQPP_TABLE_ALWAYS_INLINE void PrepareNegativeTwoScaledClusterQueryTables(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict output) noexcept {
  // Reconstruction-norm8 has one global dot multiplier. Folding that exact
  // power-of-two scale into the already-required per-CID coefficient scaling
  // removes the final per-row FMA without adding a table pass. Scaled-score8
  // cannot use this convention because its multiplier is selected per row.
  PrepareScaledClusterQueryTablesImpl<M, true>(query, cluster, output);
}

#undef STLQPP_TABLE_ALWAYS_INLINE

}  // namespace stlqpp
