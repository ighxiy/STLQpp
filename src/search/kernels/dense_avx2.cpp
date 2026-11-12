#include "stlqpp/search/kernels/dense_avx2.h"

#include "stlqpp/model/topology/parent_louds.h"

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace stlqpp {
namespace {

#if defined(_MSC_VER)
#define STLQPP_DENSE_ALWAYS_INLINE __forceinline
#define STLQPP_DENSE_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define STLQPP_DENSE_ALWAYS_INLINE inline __attribute__((always_inline))
#define STLQPP_DENSE_NOINLINE __attribute__((noinline))
#else
#define STLQPP_DENSE_ALWAYS_INLINE inline
#define STLQPP_DENSE_NOINLINE
#endif

[[nodiscard]] STLQPP_DENSE_ALWAYS_INLINE __m256 LoadI8x8AsFloat(
    const std::uint8_t* __restrict source) noexcept {
  const __m128i bytes = _mm_loadl_epi64(
      reinterpret_cast<const __m128i*>(source));
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
}

[[nodiscard]] STLQPP_DENSE_ALWAYS_INLINE __m256 Gather8(
    const float* __restrict table,
    const std::uint8_t* __restrict codes) noexcept {
  return _mm256_setr_ps(table[codes[0]], table[codes[1]],
                        table[codes[2]], table[codes[3]],
                        table[codes[4]], table[codes[5]],
                        table[codes[6]], table[codes[7]]);
}

[[nodiscard]] STLQPP_DENSE_ALWAYS_INLINE float SignedCoefficient(
    std::uint8_t byte) noexcept {
  return static_cast<float>(static_cast<std::int8_t>(byte));
}

template <int M>
STLQPP_DENSE_ALWAYS_INLINE void ComputeRootDots(
    const ResidentClusterView<M>& cluster,
    const ScaledClusterQueryTables<M>& query,
    float* __restrict dots) noexcept {
  using Layout = DenseAoSoA16Layout<M>;
  const std::uint32_t root_count = cluster.root_count();
  const std::uint32_t tiled =
      root_count / Layout::kTileRows * Layout::kTileRows;
  const __m256 root0 = _mm256_set1_ps(query.root0());
  const std::uint8_t* __restrict root_payload = cluster.root_payload();

  for (std::uint32_t row = 0U; row < tiled; row += Layout::kTileRows) {
    const std::uint8_t* tile = root_payload +
        static_cast<std::size_t>(row / Layout::kTileRows) *
            Layout::kRootTileBytes;
    __m256 low = _mm256_mul_ps(LoadI8x8AsFloat(tile), root0);
    __m256 high = _mm256_mul_ps(
        LoadI8x8AsFloat(tile + 8U), root0);
#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
    for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(M);
         ++layer) {
      const std::uint8_t* pair = tile + Layout::kTileRows +
          static_cast<std::size_t>(layer - 1U) * 2U * Layout::kTileRows;
      const std::uint8_t* codes = pair + Layout::kTileRows;
      const float* table = query.root_small(layer);
      low = _mm256_fmadd_ps(LoadI8x8AsFloat(pair),
                            Gather8(table, codes), low);
      high = _mm256_fmadd_ps(LoadI8x8AsFloat(pair + 8U),
                             Gather8(table, codes + 8U), high);
    }
    _mm256_storeu_ps(dots + row, low);
    _mm256_storeu_ps(dots + row + 8U, high);
  }

  const std::uint8_t* trailer = root_payload +
      static_cast<std::size_t>(tiled / Layout::kTileRows) *
          Layout::kRootTileBytes;
  for (std::uint32_t root = tiled; root < root_count; ++root) {
    const std::uint8_t* record = trailer +
        static_cast<std::size_t>(root - tiled) * Layout::kRootBytesPerRow;
    float dot = SignedCoefficient(record[0]) * query.root0();
#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
    for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(M);
         ++layer) {
      dot += SignedCoefficient(record[1U + 2U * (layer - 1U)]) *
             query.root_small(layer)[record[2U + 2U * (layer - 1U)]];
    }
    dots[root] = dot;
  }
}

template <int M>
STLQPP_DENSE_ALWAYS_INLINE void ComputeLinkedDotsNativeLouds(
    const ResidentClusterView<M>& cluster,
    const ScaledClusterQueryTables<M>& query,
    float* __restrict dots) noexcept {
  using Layout = DenseAoSoA16Layout<M>;
  const std::uint32_t root_count = cluster.root_count();
  const std::uint32_t linked_count = cluster.linked_count();
  const std::uint32_t tiled =
      linked_count / Layout::kTileRows * Layout::kTileRows;
  const std::uint8_t* __restrict linked_payload = cluster.linked_payload();
  ParentLoudsSequentialDecoder parent_decoder(
      cluster.parent_louds_words());
  parent_decoder.InitializeAfterValidatedRootPrefix(root_count);
  alignas(32) float edge[Layout::kTileRows];

  for (std::uint32_t row = 0U; row < tiled; row += Layout::kTileRows) {
    const std::uint8_t* tile = linked_payload +
        static_cast<std::size_t>(row / Layout::kTileRows) *
            Layout::kLinkedTileBytes;
    const std::uint8_t* pair = tile;
    const std::uint8_t* codes = pair + Layout::kTileRows;
    __m256 low = _mm256_mul_ps(LoadI8x8AsFloat(pair),
                               Gather8(query.edge(0U), codes));
    __m256 high = _mm256_mul_ps(LoadI8x8AsFloat(pair + 8U),
                                Gather8(query.edge(0U), codes + 8U));
#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
    for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(M);
         ++layer) {
      pair = tile +
          static_cast<std::size_t>(layer) * 2U * Layout::kTileRows;
      codes = pair + Layout::kTileRows;
      low = _mm256_fmadd_ps(LoadI8x8AsFloat(pair),
                            Gather8(query.edge(layer), codes), low);
      high = _mm256_fmadd_ps(LoadI8x8AsFloat(pair + 8U),
                             Gather8(query.edge(layer), codes + 8U), high);
    }
    _mm256_store_ps(edge, low);
    _mm256_store_ps(edge + 8U, high);
#if defined(__GNUC__)
#pragma GCC unroll 16
#endif
    for (std::uint32_t lane = 0U; lane < Layout::kTileRows; ++lane) {
      const std::uint32_t local = root_count + row + lane;
      const std::uint32_t parent =
          parent_decoder.NextParent1Based() - 1U;
      dots[local] = dots[parent] + edge[lane];
    }
  }

  const std::uint8_t* trailer = linked_payload +
      static_cast<std::size_t>(tiled / Layout::kTileRows) *
          Layout::kLinkedTileBytes;
  for (std::uint32_t row = tiled; row < linked_count; ++row) {
    const std::uint8_t* record = trailer +
        static_cast<std::size_t>(row - tiled) * Layout::kLinkedBytesPerRow;
    float edge_dot = 0.0F;
#if defined(__GNUC__)
#pragma GCC unroll 8
#endif
    for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(M);
         ++layer) {
      edge_dot += SignedCoefficient(record[2U * layer]) *
                  query.edge(layer)[record[2U * layer + 1U]];
    }
    const std::uint32_t local = root_count + row;
    const std::uint32_t parent =
        parent_decoder.NextParent1Based() - 1U;
    dots[local] = dots[parent] + edge_dot;
  }
}

template <int M, typename Parent>
STLQPP_DENSE_ALWAYS_INLINE void ComputeLinkedDotsMaterialized(
    const ResidentClusterView<M>& cluster,
    const ScaledClusterQueryTables<M>& query,
    const Parent* __restrict parents,
    float* __restrict dots) noexcept {
  using Layout = DenseAoSoA16Layout<M>;
  const std::uint32_t root_count = cluster.root_count();
  const std::uint32_t linked_count = cluster.linked_count();
  const std::uint32_t tiled =
      linked_count / Layout::kTileRows * Layout::kTileRows;
  const std::uint8_t* __restrict linked_payload = cluster.linked_payload();
  const Parent* __restrict linked_parents = parents + root_count;
  alignas(32) float edge[Layout::kTileRows];

  for (std::uint32_t row = 0U; row < tiled; row += Layout::kTileRows) {
    const std::uint8_t* tile = linked_payload +
        static_cast<std::size_t>(row / Layout::kTileRows) *
            Layout::kLinkedTileBytes;
    const std::uint8_t* pair = tile;
    const std::uint8_t* codes = pair + Layout::kTileRows;
    __m256 low = _mm256_mul_ps(LoadI8x8AsFloat(pair),
                               Gather8(query.edge(0U), codes));
    __m256 high = _mm256_mul_ps(LoadI8x8AsFloat(pair + 8U),
                                Gather8(query.edge(0U), codes + 8U));
#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
    for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(M);
         ++layer) {
      pair = tile +
          static_cast<std::size_t>(layer) * 2U * Layout::kTileRows;
      codes = pair + Layout::kTileRows;
      low = _mm256_fmadd_ps(LoadI8x8AsFloat(pair),
                            Gather8(query.edge(layer), codes), low);
      high = _mm256_fmadd_ps(LoadI8x8AsFloat(pair + 8U),
                             Gather8(query.edge(layer), codes + 8U), high);
    }
    _mm256_store_ps(edge, low);
    _mm256_store_ps(edge + 8U, high);
#if defined(__GNUC__)
#pragma GCC unroll 16
#endif
    for (std::uint32_t lane = 0U; lane < Layout::kTileRows; ++lane) {
      const std::uint32_t local = root_count + row + lane;
      const std::uint32_t parent =
          static_cast<std::uint32_t>(linked_parents[row + lane]) - 1U;
      dots[local] = dots[parent] + edge[lane];
    }
  }

  const std::uint8_t* trailer = linked_payload +
      static_cast<std::size_t>(tiled / Layout::kTileRows) *
          Layout::kLinkedTileBytes;
  for (std::uint32_t row = tiled; row < linked_count; ++row) {
    const std::uint8_t* record = trailer +
        static_cast<std::size_t>(row - tiled) * Layout::kLinkedBytesPerRow;
    float edge_dot = 0.0F;
#if defined(__GNUC__)
#pragma GCC unroll 8
#endif
    for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(M);
         ++layer) {
      edge_dot += SignedCoefficient(record[2U * layer]) *
                  query.edge(layer)[record[2U * layer + 1U]];
    }
    const std::uint32_t local = root_count + row;
    const std::uint32_t parent =
        static_cast<std::uint32_t>(linked_parents[row]) - 1U;
    dots[local] = dots[parent] + edge_dot;
  }
}

template <bool NegativeTwoPrefolded>
STLQPP_DENSE_NOINLINE void ScoreF32NormRows(
    std::uint32_t count, const std::uint32_t* __restrict ids,
    const float* __restrict real_dots, const float* __restrict rows,
    QueryTopK* __restrict top_k) noexcept {
  std::uint32_t pos = 0U;

  while (pos < count && !top_k->full()) {
    const float distance = NegativeTwoPrefolded
        ? rows[pos] + real_dots[pos]
        : rows[pos] - 2.0F * real_dots[pos];
    top_k->Consider(ids[pos], distance);
    ++pos;
  }

  alignas(32) float distances[8];
  const __m256 negative_two = _mm256_set1_ps(-2.0F);
  float worst_distance = top_k->cutoff_distance();
  std::uint32_t worst_id = top_k->cutoff_id();
  for (; pos + 8U <= count; pos += 8U) {
    const __m256 norms = _mm256_loadu_ps(rows + pos);
    const __m256 score = NegativeTwoPrefolded
        ? _mm256_add_ps(_mm256_loadu_ps(real_dots + pos), norms)
        : _mm256_fmadd_ps(
              negative_two, _mm256_loadu_ps(real_dots + pos), norms);
    const __m256 cutoff = _mm256_set1_ps(worst_distance);
    unsigned int mask = static_cast<unsigned int>(_mm256_movemask_ps(
        _mm256_cmp_ps(score, cutoff, _CMP_LE_OQ)));
    if (mask == 0U) {
      [[likely]] continue;
    }
    _mm256_store_ps(distances, score);
    while (mask != 0U) {
      const std::uint32_t lane = std::countr_zero(mask);
      const float distance = distances[lane];
      const std::uint32_t id = ids[pos + lane];
      if (distance < worst_distance ||
          (distance == worst_distance && id < worst_id)) {
        top_k->ConsiderAssumeBetter(id, distance);
        worst_distance = top_k->cutoff_distance();
        worst_id = top_k->cutoff_id();
      }
      mask &= mask - 1U;
    }
  }
  for (; pos < count; ++pos) {
    const float distance = NegativeTwoPrefolded
        ? rows[pos] + real_dots[pos]
        : rows[pos] - 2.0F * real_dots[pos];
    const std::uint32_t id = ids[pos];
    if (distance < worst_distance ||
        (distance == worst_distance && id < worst_id)) {
      top_k->ConsiderAssumeBetter(id, distance);
      worst_distance = top_k->cutoff_distance();
      worst_id = top_k->cutoff_id();
    }
  }
}
STLQPP_DENSE_NOINLINE void ScoreF32ScaledRows(
    std::uint32_t count, const std::uint32_t* __restrict ids,
    const float* __restrict real_dots, const float* __restrict biases,
    const float* __restrict multipliers, QueryTopK* __restrict top_k) noexcept {
  std::uint32_t pos = 0U;

  while (pos < count && !top_k->full()) {
    const std::uint32_t label = pos;
    const float distance =
        std::fma(multipliers[label], real_dots[pos], biases[label]);
    top_k->Consider(ids[pos], distance);
    ++pos;
  }

  alignas(32) float distances[8];
  float worst_distance = top_k->cutoff_distance();
  std::uint32_t worst_id = top_k->cutoff_id();
  for (; pos + 8U <= count; pos += 8U) {
    const __m256 bias = _mm256_loadu_ps(biases + pos);
    const __m256 multiplier = _mm256_loadu_ps(multipliers + pos);
    const __m256 score = _mm256_fmadd_ps(
        multiplier, _mm256_loadu_ps(real_dots + pos), bias);
    const __m256 cutoff = _mm256_set1_ps(worst_distance);
    unsigned int mask = static_cast<unsigned int>(_mm256_movemask_ps(
        _mm256_cmp_ps(score, cutoff, _CMP_LE_OQ)));
    if (mask == 0U) {
      [[likely]] continue;
    }
    _mm256_store_ps(distances, score);
    while (mask != 0U) {
      const std::uint32_t lane = std::countr_zero(mask);
      const float distance = distances[lane];
      const std::uint32_t id = ids[pos + lane];
      if (distance < worst_distance ||
          (distance == worst_distance && id < worst_id)) {
        top_k->ConsiderAssumeBetter(id, distance);
        worst_distance = top_k->cutoff_distance();
        worst_id = top_k->cutoff_id();
      }
      mask &= mask - 1U;
    }
  }
  for (; pos < count; ++pos) {
    const std::uint32_t label = pos;
    const float distance =
        std::fma(multipliers[label], real_dots[pos], biases[label]);
    const std::uint32_t id = ids[pos];
    if (distance < worst_distance ||
        (distance == worst_distance && id < worst_id)) {
      top_k->ConsiderAssumeBetter(id, distance);
      worst_distance = top_k->cutoff_distance();
      worst_id = top_k->cutoff_id();
    }
  }
}
template <bool NegativeTwoPrefolded, int M>
STLQPP_DENSE_ALWAYS_INLINE void ScoreRealRowsNorm8(
    const ResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    const float* __restrict dots) noexcept {
  if (cluster.row_norm2() != nullptr) [[unlikely]] {
    ScoreF32NormRows<NegativeTwoPrefolded>(cluster.real_count(), cluster.real_ids(),
        dots + cluster.virtual_count(), cluster.row_norm2(), top_k);
    return;
  }
  const std::uint32_t count = cluster.real_count();
  const float* __restrict real_dots = dots + cluster.virtual_count();
  const std::uint32_t* __restrict ids = cluster.real_ids();
  const std::uint8_t* __restrict norm8 = cluster.real_norm8();
  const float* __restrict centers = cluster.norm8_centers();
  std::uint32_t pos = 0U;

  while (pos < count && !top_k->full()) {
    const float distance = NegativeTwoPrefolded
        ? centers[norm8[pos]] + real_dots[pos]
        : centers[norm8[pos]] - 2.0F * real_dots[pos];
    top_k->Consider(ids[pos], distance);
    ++pos;
  }

  alignas(32) float distances[8];
  const __m256 negative_two = _mm256_set1_ps(-2.0F);
  float worst_distance = top_k->cutoff_distance();
  std::uint32_t worst_id = top_k->cutoff_id();
  for (; pos + 8U <= count; pos += 8U) {
    const __m256 norms = _mm256_setr_ps(
        centers[norm8[pos]], centers[norm8[pos + 1U]],
        centers[norm8[pos + 2U]], centers[norm8[pos + 3U]],
        centers[norm8[pos + 4U]], centers[norm8[pos + 5U]],
        centers[norm8[pos + 6U]], centers[norm8[pos + 7U]]);
    const __m256 score = NegativeTwoPrefolded
        ? _mm256_add_ps(_mm256_loadu_ps(real_dots + pos), norms)
        : _mm256_fmadd_ps(
              negative_two, _mm256_loadu_ps(real_dots + pos), norms);
    const __m256 cutoff = _mm256_set1_ps(worst_distance);
    unsigned int mask = static_cast<unsigned int>(_mm256_movemask_ps(
        _mm256_cmp_ps(score, cutoff, _CMP_LE_OQ)));
    if (mask == 0U) {
      [[likely]] continue;
    }
    _mm256_store_ps(distances, score);
    while (mask != 0U) {
      const std::uint32_t lane = std::countr_zero(mask);
      const float distance = distances[lane];
      const std::uint32_t id = ids[pos + lane];
      if (distance < worst_distance ||
          (distance == worst_distance && id < worst_id)) {
        top_k->ConsiderAssumeBetter(id, distance);
        worst_distance = top_k->cutoff_distance();
        worst_id = top_k->cutoff_id();
      }
      mask &= mask - 1U;
    }
  }
  for (; pos < count; ++pos) {
    const float distance = NegativeTwoPrefolded
        ? centers[norm8[pos]] + real_dots[pos]
        : centers[norm8[pos]] - 2.0F * real_dots[pos];
    const std::uint32_t id = ids[pos];
    if (distance < worst_distance ||
        (distance == worst_distance && id < worst_id)) {
      top_k->ConsiderAssumeBetter(id, distance);
      worst_distance = top_k->cutoff_distance();
      worst_id = top_k->cutoff_id();
    }
  }
}

template <int M>
STLQPP_DENSE_ALWAYS_INLINE void ScoreRealRowsScaled(
    const ResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    const float* __restrict dots) noexcept {
  if (cluster.row_dot_multipliers() != nullptr) [[unlikely]] {
    ScoreF32ScaledRows(cluster.real_count(), cluster.real_ids(), dots + cluster.virtual_count(),
        cluster.row_biases(), cluster.row_dot_multipliers(), top_k);
    return;
  }
  const std::uint32_t count = cluster.real_count();
  const float* __restrict real_dots = dots + cluster.virtual_count();
  const std::uint32_t* __restrict ids = cluster.real_ids();
  const std::uint8_t* __restrict labels = cluster.real_norm8();
  const float* __restrict biases = cluster.norm8_centers();
  const float* __restrict multipliers =
      cluster.scaled_score_dot_multipliers();
  std::uint32_t pos = 0U;

  while (pos < count && !top_k->full()) {
    const std::uint8_t label = labels[pos];
    const float distance =
        std::fma(multipliers[label], real_dots[pos], biases[label]);
    top_k->Consider(ids[pos], distance);
    ++pos;
  }

  alignas(32) float distances[8];
  float worst_distance = top_k->cutoff_distance();
  std::uint32_t worst_id = top_k->cutoff_id();
  for (; pos + 8U <= count; pos += 8U) {
    const __m256 bias = _mm256_setr_ps(
        biases[labels[pos]], biases[labels[pos + 1U]],
        biases[labels[pos + 2U]], biases[labels[pos + 3U]],
        biases[labels[pos + 4U]], biases[labels[pos + 5U]],
        biases[labels[pos + 6U]], biases[labels[pos + 7U]]);
    const __m256 multiplier = _mm256_setr_ps(
        multipliers[labels[pos]], multipliers[labels[pos + 1U]],
        multipliers[labels[pos + 2U]], multipliers[labels[pos + 3U]],
        multipliers[labels[pos + 4U]], multipliers[labels[pos + 5U]],
        multipliers[labels[pos + 6U]], multipliers[labels[pos + 7U]]);
    const __m256 score = _mm256_fmadd_ps(
        multiplier, _mm256_loadu_ps(real_dots + pos), bias);
    const __m256 cutoff = _mm256_set1_ps(worst_distance);
    unsigned int mask = static_cast<unsigned int>(_mm256_movemask_ps(
        _mm256_cmp_ps(score, cutoff, _CMP_LE_OQ)));
    if (mask == 0U) {
      [[likely]] continue;
    }
    _mm256_store_ps(distances, score);
    while (mask != 0U) {
      const std::uint32_t lane = std::countr_zero(mask);
      const float distance = distances[lane];
      const std::uint32_t id = ids[pos + lane];
      if (distance < worst_distance ||
          (distance == worst_distance && id < worst_id)) {
        top_k->ConsiderAssumeBetter(id, distance);
        worst_distance = top_k->cutoff_distance();
        worst_id = top_k->cutoff_id();
      }
      mask &= mask - 1U;
    }
  }
  for (; pos < count; ++pos) {
    const std::uint8_t label = labels[pos];
    const float distance =
        std::fma(multipliers[label], real_dots[pos], biases[label]);
    const std::uint32_t id = ids[pos];
    if (distance < worst_distance ||
        (distance == worst_distance && id < worst_id)) {
      top_k->ConsiderAssumeBetter(id, distance);
      worst_distance = top_k->cutoff_distance();
      worst_id = top_k->cutoff_id();
    }
  }
}

template <int M>
STLQPP_DENSE_ALWAYS_INLINE void ScoreRealRows(
    const ResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    const float* __restrict dots) noexcept {
  if (cluster.scaled_score_dot_multipliers() == nullptr) {
    ScoreRealRowsNorm8<false>(cluster, top_k, dots);
  } else {
    ScoreRealRowsScaled(cluster, top_k, dots);
  }
}

template <int M>
STLQPP_DENSE_ALWAYS_INLINE void PrepareScoreTables(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict output) noexcept {
  if (cluster.scaled_score_dot_multipliers() == nullptr) {
    // Scratch stores -2*qdot for the legacy score codec.
    PrepareNegativeTwoScaledClusterQueryTables(query, cluster, output);
  } else {
    // Scaled-score8 needs the unscaled qdot for its per-label FMA.
    PrepareScaledClusterQueryTables(query, cluster, output);
  }
}

template <int M>
STLQPP_DENSE_ALWAYS_INLINE void ScorePreparedRealRows(
    const ResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    const float* __restrict dots) noexcept {
  if (cluster.scaled_score_dot_multipliers() == nullptr) {
    ScoreRealRowsNorm8<true>(cluster, top_k, dots);
  } else {
    ScoreRealRowsScaled(cluster, top_k, dots);
  }
}

template <int M>
STLQPP_DENSE_ALWAYS_INLINE void ComputeProfileScaledDistances(
    const ResidentClusterView<M>& cluster,
    const float* __restrict dots,
    float* __restrict distances) noexcept {
  const std::uint32_t count = cluster.real_count();
  const float* __restrict real_dots = dots + cluster.virtual_count();
  const std::uint8_t* __restrict labels = cluster.real_norm8();
  const bool rows = cluster.row_dot_multipliers() != nullptr;
  const float* __restrict biases = rows ? cluster.row_biases() : cluster.norm8_centers();
  const float* __restrict multipliers =
      rows ? cluster.row_dot_multipliers() : cluster.scaled_score_dot_multipliers();
  std::uint32_t pos = 0U;
  for (; pos + 8U <= count; pos += 8U) {
    const __m256 bias = _mm256_setr_ps(
        biases[rows ? pos : labels[pos]], biases[rows ? pos + 1U : labels[pos + 1U]],
        biases[rows ? pos + 2U : labels[pos + 2U]], biases[rows ? pos + 3U : labels[pos + 3U]],
        biases[rows ? pos + 4U : labels[pos + 4U]], biases[rows ? pos + 5U : labels[pos + 5U]],
        biases[rows ? pos + 6U : labels[pos + 6U]], biases[rows ? pos + 7U : labels[pos + 7U]]);
    const __m256 multiplier = _mm256_setr_ps(
        multipliers[rows ? pos : labels[pos]], multipliers[rows ? pos + 1U : labels[pos + 1U]],
        multipliers[rows ? pos + 2U : labels[pos + 2U]], multipliers[rows ? pos + 3U : labels[pos + 3U]],
        multipliers[rows ? pos + 4U : labels[pos + 4U]], multipliers[rows ? pos + 5U : labels[pos + 5U]],
        multipliers[rows ? pos + 6U : labels[pos + 6U]], multipliers[rows ? pos + 7U : labels[pos + 7U]]);
    _mm256_storeu_ps(
        distances + pos,
        _mm256_fmadd_ps(
            multiplier, _mm256_loadu_ps(real_dots + pos), bias));
  }
  for (; pos < count; ++pos) {
    const std::uint32_t label = rows ? pos : labels[pos];
    distances[pos] =
        std::fma(multipliers[label], real_dots[pos], biases[label]);
  }
}

template <int M>
STLQPP_DENSE_ALWAYS_INLINE void ConsumeProfileDistances(
    const ResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    const float* __restrict distances) noexcept {
  const std::uint32_t count = cluster.real_count();
  const std::uint32_t* __restrict ids = cluster.real_ids();
  std::uint32_t pos = 0U;
  while (pos < count && !top_k->full()) {
    top_k->Consider(ids[pos], distances[pos]);
    ++pos;
  }
  if (!top_k->full()) return;

  float worst_distance = top_k->cutoff_distance();
  std::uint32_t worst_id = top_k->cutoff_id();
  for (; pos + 8U <= count; pos += 8U) {
    const __m256 score = _mm256_loadu_ps(distances + pos);
    const __m256 cutoff = _mm256_set1_ps(worst_distance);
    unsigned int mask = static_cast<unsigned int>(_mm256_movemask_ps(
        _mm256_cmp_ps(score, cutoff, _CMP_LE_OQ)));
    if (mask == 0U) {
      [[likely]] continue;
    }
    while (mask != 0U) {
      const std::uint32_t lane = std::countr_zero(mask);
      const float distance = distances[pos + lane];
      const std::uint32_t id = ids[pos + lane];
      if (distance < worst_distance ||
          (distance == worst_distance && id < worst_id)) {
        top_k->ConsiderAssumeBetter(id, distance);
        worst_distance = top_k->cutoff_distance();
        worst_id = top_k->cutoff_id();
      }
      mask &= mask - 1U;
    }
  }
  for (; pos < count; ++pos) {
    const float distance = distances[pos];
    const std::uint32_t id = ids[pos];
    if (distance < worst_distance ||
        (distance == worst_distance && id < worst_id)) {
      top_k->ConsiderAssumeBetter(id, distance);
      worst_distance = top_k->cutoff_distance();
      worst_id = top_k->cutoff_id();
    }
  }
}

template <bool ProfileTiming, int M>
STLQPP_DENSE_ALWAYS_INLINE void ScorePreparedRealRowsDispatch(
    const ResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    const float* __restrict dots,
    float* __restrict distance_scratch,
    DenseIntScanTiming* __restrict timing) noexcept {
  if constexpr (!ProfileTiming) {
    (void)distance_scratch;
    (void)timing;
    ScorePreparedRealRows(cluster, top_k, dots);
  } else {
    using Clock = std::chrono::steady_clock;
    auto begin = Clock::now();
    assert(cluster.scaled_score_dot_multipliers() != nullptr || cluster.row_dot_multipliers() != nullptr);
    ComputeProfileScaledDistances(cluster, dots, distance_scratch);
    timing->scan_seconds +=
        std::chrono::duration<double>(Clock::now() - begin).count();

    begin = Clock::now();
    ConsumeProfileDistances(cluster, top_k, distance_scratch);
    timing->online_topk_seconds +=
        std::chrono::duration<double>(Clock::now() - begin).count();
  }
}

template <int M>
STLQPP_DENSE_ALWAYS_INLINE void ComputeFloatRootDots(
    const FloatResidentClusterView<M>& cluster,
    const TrustedQueryTables<M>& query,
    float* __restrict dots) noexcept {
  const std::uint32_t count = cluster.root_count();
  const __m256 root0 = _mm256_set1_ps(query.root0(cluster.cid()));
  std::uint32_t row = 0U;
  for (; row + 8U <= count; row += 8U) {
    __m256 values = _mm256_mul_ps(
        _mm256_loadu_ps(cluster.root_coefficients(0U) + row), root0);
    for (std::uint32_t layer = 1U;
         layer < static_cast<std::uint32_t>(M); ++layer) {
      const std::uint8_t* codes = cluster.root_small_codes(layer) + row;
      const __m256 table = _mm256_setr_ps(
          query.root_small(layer, codes[0]),
          query.root_small(layer, codes[1]),
          query.root_small(layer, codes[2]),
          query.root_small(layer, codes[3]),
          query.root_small(layer, codes[4]),
          query.root_small(layer, codes[5]),
          query.root_small(layer, codes[6]),
          query.root_small(layer, codes[7]));
      values = _mm256_fmadd_ps(
          _mm256_loadu_ps(cluster.root_coefficients(layer) + row),
          table, values);
    }
    _mm256_storeu_ps(dots + row, values);
  }
  for (; row < count; ++row) {
    float dot = cluster.root_coefficients(0U)[row] *
        query.root0(cluster.cid());
    for (std::uint32_t layer = 1U;
         layer < static_cast<std::uint32_t>(M); ++layer) {
      dot += cluster.root_coefficients(layer)[row] *
          query.root_small(layer, cluster.root_small_codes(layer)[row]);
    }
    dots[row] = dot;
  }
}

template <int M>
STLQPP_DENSE_ALWAYS_INLINE void ComputeFloatLinkedDotsNativeLouds(
    const FloatResidentClusterView<M>& cluster,
    const TrustedQueryTables<M>& query,
    float* __restrict dots) noexcept {
  const std::uint32_t root_count = cluster.root_count();
  const std::uint32_t count = cluster.linked_count();
  ParentLoudsSequentialDecoder parent_decoder(
      cluster.parent_louds_words());
  parent_decoder.InitializeAfterValidatedRootPrefix(root_count);
  alignas(32) float edge[8];
  std::uint32_t row = 0U;
  for (; row + 8U <= count; row += 8U) {
    __m256 values = _mm256_setzero_ps();
    for (std::uint32_t layer = 0U;
         layer < static_cast<std::uint32_t>(M); ++layer) {
      const std::uint8_t* codes = cluster.linked_codes(layer) + row;
      const __m256 table = _mm256_setr_ps(
          query.edge(layer, codes[0]), query.edge(layer, codes[1]),
          query.edge(layer, codes[2]), query.edge(layer, codes[3]),
          query.edge(layer, codes[4]), query.edge(layer, codes[5]),
          query.edge(layer, codes[6]), query.edge(layer, codes[7]));
      values = _mm256_fmadd_ps(
          _mm256_loadu_ps(cluster.linked_coefficients(layer) + row),
          table, values);
    }
    _mm256_store_ps(edge, values);
    for (std::uint32_t lane = 0U; lane < 8U; ++lane) {
      const std::uint32_t parent =
          parent_decoder.NextParent1Based() - 1U;
      dots[root_count + row + lane] = dots[parent] + edge[lane];
    }
  }
  for (; row < count; ++row) {
    float edge_dot = 0.0F;
    for (std::uint32_t layer = 0U;
         layer < static_cast<std::uint32_t>(M); ++layer) {
      edge_dot += cluster.linked_coefficients(layer)[row] *
          query.edge(layer, cluster.linked_codes(layer)[row]);
    }
    const std::uint32_t parent =
        parent_decoder.NextParent1Based() - 1U;
    dots[root_count + row] = dots[parent] + edge_dot;
  }
}

template <int M, typename Parent>
STLQPP_DENSE_ALWAYS_INLINE void ComputeFloatLinkedDotsMaterialized(
    const FloatResidentClusterView<M>& cluster,
    const TrustedQueryTables<M>& query,
    const Parent* __restrict parents,
    float* __restrict dots) noexcept {
  const std::uint32_t root_count = cluster.root_count();
  const std::uint32_t count = cluster.linked_count();
  const Parent* __restrict linked_parents = parents + root_count;
  alignas(32) float edge[8];
  std::uint32_t row = 0U;
  for (; row + 8U <= count; row += 8U) {
    __m256 values = _mm256_setzero_ps();
    for (std::uint32_t layer = 0U;
         layer < static_cast<std::uint32_t>(M); ++layer) {
      const std::uint8_t* codes = cluster.linked_codes(layer) + row;
      const __m256 table = _mm256_setr_ps(
          query.edge(layer, codes[0]), query.edge(layer, codes[1]),
          query.edge(layer, codes[2]), query.edge(layer, codes[3]),
          query.edge(layer, codes[4]), query.edge(layer, codes[5]),
          query.edge(layer, codes[6]), query.edge(layer, codes[7]));
      values = _mm256_fmadd_ps(
          _mm256_loadu_ps(cluster.linked_coefficients(layer) + row),
          table, values);
    }
    _mm256_store_ps(edge, values);
    for (std::uint32_t lane = 0U; lane < 8U; ++lane) {
      const std::uint32_t parent =
          static_cast<std::uint32_t>(linked_parents[row + lane]) - 1U;
      dots[root_count + row + lane] = dots[parent] + edge[lane];
    }
  }
  for (; row < count; ++row) {
    float edge_dot = 0.0F;
    for (std::uint32_t layer = 0U;
         layer < static_cast<std::uint32_t>(M); ++layer) {
      edge_dot += cluster.linked_coefficients(layer)[row] *
          query.edge(layer, cluster.linked_codes(layer)[row]);
    }
    const std::uint32_t parent =
        static_cast<std::uint32_t>(linked_parents[row]) - 1U;
    dots[root_count + row] = dots[parent] + edge_dot;
  }
}

template <int M>
STLQPP_DENSE_ALWAYS_INLINE void ScoreFloatRealRowsClassic(
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    const float* __restrict dots) noexcept {
  const std::uint32_t count = cluster.real_count();
  const float* __restrict real_dots = dots + cluster.virtual_count();
  const float* __restrict norms = cluster.real_norm2();
  const std::uint32_t* __restrict ids = cluster.real_ids();
  std::uint32_t pos = 0U;
  while (pos < count && !top_k->full()) {
    top_k->Consider(ids[pos], norms[pos] - 2.0F * real_dots[pos]);
    ++pos;
  }
  alignas(32) float distances[8];
  const __m256 negative_two = _mm256_set1_ps(-2.0F);
  float worst_distance = top_k->cutoff_distance();
  std::uint32_t worst_id = top_k->cutoff_id();
  for (; pos + 8U <= count; pos += 8U) {
    const __m256 score = _mm256_fmadd_ps(
        negative_two, _mm256_loadu_ps(real_dots + pos),
        _mm256_loadu_ps(norms + pos));
    const __m256 cutoff = _mm256_set1_ps(worst_distance);
    unsigned int mask = static_cast<unsigned int>(_mm256_movemask_ps(
        _mm256_cmp_ps(score, cutoff, _CMP_LE_OQ)));
    if (mask == 0U) continue;
    _mm256_store_ps(distances, score);
    while (mask != 0U) {
      const std::uint32_t lane = std::countr_zero(mask);
      const float distance = distances[lane];
      const std::uint32_t id = ids[pos + lane];
      if (distance < worst_distance ||
          (distance == worst_distance && id < worst_id)) {
        top_k->ConsiderAssumeBetter(id, distance);
        worst_distance = top_k->cutoff_distance();
        worst_id = top_k->cutoff_id();
      }
      mask &= mask - 1U;
    }
  }
  for (; pos < count; ++pos) {
    const float distance = norms[pos] - 2.0F * real_dots[pos];
    const std::uint32_t id = ids[pos];
    if (distance < worst_distance ||
        (distance == worst_distance && id < worst_id)) {
      top_k->ConsiderAssumeBetter(id, distance);
      worst_distance = top_k->cutoff_distance();
      worst_id = top_k->cutoff_id();
    }
  }
}

template <int M>
STLQPP_DENSE_ALWAYS_INLINE void ScoreFloatRealRowsScaled(
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    const float* __restrict dots) noexcept {
  const std::uint32_t count = cluster.real_count();
  const float* __restrict real_dots = dots + cluster.virtual_count();
  const std::uint32_t* __restrict ids = cluster.real_ids();
  const float* __restrict biases = cluster.scaled_score_biases();
  const float* __restrict multipliers =
      cluster.scaled_score_dot_multipliers();
  std::uint32_t pos = 0U;
  while (pos < count && !top_k->full()) {
    top_k->Consider(
        ids[pos], std::fma(multipliers[pos], real_dots[pos], biases[pos]));
    ++pos;
  }
  alignas(32) float distances[8];
  float worst_distance = top_k->cutoff_distance();
  std::uint32_t worst_id = top_k->cutoff_id();
  for (; pos + 8U <= count; pos += 8U) {
    const __m256 bias = _mm256_loadu_ps(biases + pos);
    const __m256 multiplier = _mm256_loadu_ps(multipliers + pos);
    const __m256 score = _mm256_fmadd_ps(
        multiplier, _mm256_loadu_ps(real_dots + pos), bias);
    const __m256 cutoff = _mm256_set1_ps(worst_distance);
    unsigned int mask = static_cast<unsigned int>(_mm256_movemask_ps(
        _mm256_cmp_ps(score, cutoff, _CMP_LE_OQ)));
    if (mask == 0U) continue;
    _mm256_store_ps(distances, score);
    while (mask != 0U) {
      const std::uint32_t lane = std::countr_zero(mask);
      const float distance = distances[lane];
      const std::uint32_t id = ids[pos + lane];
      if (distance < worst_distance ||
          (distance == worst_distance && id < worst_id)) {
        top_k->ConsiderAssumeBetter(id, distance);
        worst_distance = top_k->cutoff_distance();
        worst_id = top_k->cutoff_id();
      }
      mask &= mask - 1U;
    }
  }
  for (; pos < count; ++pos) {
    const float distance =
        std::fma(multipliers[pos], real_dots[pos], biases[pos]);
    const std::uint32_t id = ids[pos];
    if (distance < worst_distance ||
        (distance == worst_distance && id < worst_id)) {
      top_k->ConsiderAssumeBetter(id, distance);
      worst_distance = top_k->cutoff_distance();
      worst_id = top_k->cutoff_id();
    }
  }
}

// Cosine uses the reconstructed norm; MIPS uses the same computed dots but
// does not normalize. Neither path changes the L2/scaled scan loops.
template <bool Normalize, bool InverseNorm = false, typename NormAt>
STLQPP_DENSE_ALWAYS_INLINE void ScoreAngularRealRows(
    const std::uint32_t* __restrict ids,
    const float* __restrict dots, std::uint32_t count, NormAt norm_at,
    QueryTopK* __restrict top_k) noexcept {
  [[maybe_unused]] constexpr float kMinNorm2 = 1e-20F;
  if constexpr (!Normalize) (void)norm_at;
  std::uint32_t pos = 0U;
  while (pos < count && !top_k->full()) {
    float score = -dots[pos];
    if constexpr (Normalize) {
      if constexpr (InverseNorm) score *= norm_at(pos);
      else score /= std::sqrt(std::max(norm_at(pos), kMinNorm2));
    }
    top_k->Consider(ids[pos], score);
    ++pos;
  }
  if (!top_k->full()) return;

  alignas(32) float scores[8];
  float worst_score = top_k->cutoff_distance();
  std::uint32_t worst_id = top_k->cutoff_id();
  const __m256 zero = _mm256_setzero_ps();
  for (; pos + 8U <= count; pos += 8U) {
    __m256 score = _mm256_sub_ps(zero, _mm256_loadu_ps(dots + pos));
    if constexpr (Normalize) {
      [[maybe_unused]] const __m256 minimum = _mm256_set1_ps(kMinNorm2);
      const __m256 norm2 = _mm256_setr_ps(
          norm_at(pos), norm_at(pos + 1U), norm_at(pos + 2U),
          norm_at(pos + 3U), norm_at(pos + 4U), norm_at(pos + 5U),
          norm_at(pos + 6U), norm_at(pos + 7U));
      if constexpr (InverseNorm) score = _mm256_mul_ps(score, norm2);
      else score = _mm256_div_ps(
          score, _mm256_sqrt_ps(_mm256_max_ps(norm2, minimum)));
    }
    const __m256 cutoff = _mm256_set1_ps(worst_score);
    unsigned int mask = static_cast<unsigned int>(_mm256_movemask_ps(
        _mm256_cmp_ps(score, cutoff, _CMP_LE_OQ)));
    if (mask == 0U) continue;
    _mm256_store_ps(scores, score);
    while (mask != 0U) {
      const std::uint32_t lane = std::countr_zero(mask);
      const float candidate = scores[lane];
      const std::uint32_t id = ids[pos + lane];
      if (candidate < worst_score ||
          (candidate == worst_score && id < worst_id)) {
        top_k->ConsiderAssumeBetter(id, candidate);
        worst_score = top_k->cutoff_distance();
        worst_id = top_k->cutoff_id();
      }
      mask &= mask - 1U;
    }
  }
  for (; pos < count; ++pos) {
    float candidate = -dots[pos];
    if constexpr (Normalize) {
      if constexpr (InverseNorm) candidate *= norm_at(pos);
      else candidate /= std::sqrt(std::max(norm_at(pos), kMinNorm2));
    }
    const std::uint32_t id = ids[pos];
    if (candidate < worst_score ||
        (candidate == worst_score && id < worst_id)) {
      top_k->ConsiderAssumeBetter(id, candidate);
      worst_score = top_k->cutoff_distance();
      worst_id = top_k->cutoff_id();
    }
  }
}

template <bool ProfileTiming, int M>
STLQPP_DENSE_ALWAYS_INLINE void PrepareAndScanMaterializedImpl(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch,
    float* __restrict distance_scratch,
    DenseIntScanTiming* __restrict timing) noexcept {
  std::chrono::steady_clock::time_point begin;
  if constexpr (ProfileTiming) begin = std::chrono::steady_clock::now();
  PrepareScoreTables(query, cluster, scaled_query);
  ComputeRootDots(cluster, *scaled_query, dot_scratch);
  if (const std::uint16_t* parents = cluster.parent_1based_u16();
      parents != nullptr) {
    ComputeLinkedDotsMaterialized(cluster, *scaled_query, parents,
                                  dot_scratch);
  } else {
    ComputeLinkedDotsMaterialized(cluster, *scaled_query,
                                  cluster.parent_1based_u32(), dot_scratch);
  }
  if constexpr (ProfileTiming) {
    timing->scan_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
  }
  ScorePreparedRealRowsDispatch<ProfileTiming>(
      cluster, top_k, dot_scratch, distance_scratch, timing);
}

template <bool ProfileTiming, int M>
STLQPP_DENSE_ALWAYS_INLINE void PrepareAndScanMaterializedU16Impl(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch,
    float* __restrict distance_scratch,
    DenseIntScanTiming* __restrict timing) noexcept {
  std::chrono::steady_clock::time_point begin;
  if constexpr (ProfileTiming) begin = std::chrono::steady_clock::now();
  PrepareScoreTables(query, cluster, scaled_query);
  ComputeRootDots(cluster, *scaled_query, dot_scratch);
  ComputeLinkedDotsMaterialized(
      cluster, *scaled_query, cluster.parent_1based_u16(), dot_scratch);
  if constexpr (ProfileTiming) {
    timing->scan_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
  }
  ScorePreparedRealRowsDispatch<ProfileTiming>(
      cluster, top_k, dot_scratch, distance_scratch, timing);
}

template <bool ProfileTiming, int M>
STLQPP_DENSE_ALWAYS_INLINE void PrepareAndScanMaterializedU32Impl(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch,
    float* __restrict distance_scratch,
    DenseIntScanTiming* __restrict timing) noexcept {
  std::chrono::steady_clock::time_point begin;
  if constexpr (ProfileTiming) begin = std::chrono::steady_clock::now();
  PrepareScoreTables(query, cluster, scaled_query);
  ComputeRootDots(cluster, *scaled_query, dot_scratch);
  ComputeLinkedDotsMaterialized(
      cluster, *scaled_query, cluster.parent_1based_u32(), dot_scratch);
  if constexpr (ProfileTiming) {
    timing->scan_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
  }
  ScorePreparedRealRowsDispatch<ProfileTiming>(
      cluster, top_k, dot_scratch, distance_scratch, timing);
}

template <bool ProfileTiming, int M>
STLQPP_DENSE_ALWAYS_INLINE void PrepareAndScanNativeLoudsImpl(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch,
    float* __restrict distance_scratch,
    DenseIntScanTiming* __restrict timing) noexcept {
  std::chrono::steady_clock::time_point begin;
  if constexpr (ProfileTiming) begin = std::chrono::steady_clock::now();
  PrepareScoreTables(query, cluster, scaled_query);
  ComputeRootDots(cluster, *scaled_query, dot_scratch);
  ComputeLinkedDotsNativeLouds(cluster, *scaled_query, dot_scratch);
  if constexpr (ProfileTiming) {
    timing->scan_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
  }
  ScorePreparedRealRowsDispatch<ProfileTiming>(
      cluster, top_k, dot_scratch, distance_scratch, timing);
}

#undef STLQPP_DENSE_ALWAYS_INLINE

}  // namespace

template <int M>
void ScanDenseAvx2(const ResidentClusterView<M>& cluster,
                   const ScaledClusterQueryTables<M>& query,
                   QueryTopK* __restrict top_k,
                   float* __restrict dot_scratch) noexcept {
  ComputeRootDots(cluster, query, dot_scratch);
  ComputeLinkedDotsNativeLouds(cluster, query, dot_scratch);
  ScoreRealRows(cluster, top_k, dot_scratch);
}

template <int M>
void PrepareAndScanDenseMaterializedAvx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept {
  PrepareAndScanMaterializedImpl<false>(
      query, cluster, scaled_query, top_k, dot_scratch, nullptr, nullptr);
}

template <int M>
void PrepareAndScanDenseMaterializedU16Avx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept {
  PrepareAndScanMaterializedU16Impl<false>(
      query, cluster, scaled_query, top_k, dot_scratch, nullptr, nullptr);
}

template <int M>
void PrepareAndScanDenseMaterializedU32Avx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept {
  PrepareAndScanMaterializedU32Impl<false>(
      query, cluster, scaled_query, top_k, dot_scratch, nullptr, nullptr);
}

template <int M>
void PrepareAndScanDenseNativeLoudsAvx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept {
  PrepareAndScanNativeLoudsImpl<false>(
      query, cluster, scaled_query, top_k, dot_scratch, nullptr, nullptr);
}

template <int M>
void PrepareAndScanDenseMaterializedProfiledAvx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch,
    float* __restrict distance_scratch,
    DenseIntScanTiming* __restrict timing) noexcept {
  PrepareAndScanMaterializedImpl<true>(
      query, cluster, scaled_query, top_k, dot_scratch, distance_scratch,
      timing);
}

template <int M>
void PrepareAndScanDenseMaterializedU16ProfiledAvx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch,
    float* __restrict distance_scratch,
    DenseIntScanTiming* __restrict timing) noexcept {
  PrepareAndScanMaterializedU16Impl<true>(
      query, cluster, scaled_query, top_k, dot_scratch, distance_scratch,
      timing);
}

template <int M>
void PrepareAndScanDenseMaterializedU32ProfiledAvx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch,
    float* __restrict distance_scratch,
    DenseIntScanTiming* __restrict timing) noexcept {
  PrepareAndScanMaterializedU32Impl<true>(
      query, cluster, scaled_query, top_k, dot_scratch, distance_scratch,
      timing);
}

template <int M>
void ScanFloatDenseMaterializedAvx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept {
  ComputeFloatRootDots(cluster, query, dot_scratch);
  if (const std::uint16_t* parents = cluster.parent_1based_u16();
      parents != nullptr) {
    ComputeFloatLinkedDotsMaterialized(cluster, query, parents, dot_scratch);
  } else {
    ComputeFloatLinkedDotsMaterialized(
        cluster, query, cluster.parent_1based_u32(), dot_scratch);
  }
  ScoreFloatRealRowsClassic(cluster, top_k, dot_scratch);
}

template <int M>
void ScanFloatDenseMaterializedU16Avx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept {
  ComputeFloatRootDots(cluster, query, dot_scratch);
  ComputeFloatLinkedDotsMaterialized(
      cluster, query, cluster.parent_1based_u16(), dot_scratch);
  ScoreFloatRealRowsClassic(cluster, top_k, dot_scratch);
}

template <int M>
void ScanFloatDenseMaterializedU32Avx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept {
  ComputeFloatRootDots(cluster, query, dot_scratch);
  ComputeFloatLinkedDotsMaterialized(
      cluster, query, cluster.parent_1based_u32(), dot_scratch);
  ScoreFloatRealRowsClassic(cluster, top_k, dot_scratch);
}

template <int M>
void ScanFloatDenseNativeLoudsAvx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept {
  ComputeFloatRootDots(cluster, query, dot_scratch);
  ComputeFloatLinkedDotsNativeLouds(cluster, query, dot_scratch);
  ScoreFloatRealRowsClassic(cluster, top_k, dot_scratch);
}

template <int M>
void ScanFloatScaledDenseMaterializedAvx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept {
  ComputeFloatRootDots(cluster, query, dot_scratch);
  if (const std::uint16_t* parents = cluster.parent_1based_u16();
      parents != nullptr) {
    ComputeFloatLinkedDotsMaterialized(cluster, query, parents, dot_scratch);
  } else {
    ComputeFloatLinkedDotsMaterialized(
        cluster, query, cluster.parent_1based_u32(), dot_scratch);
  }
  ScoreFloatRealRowsScaled(cluster, top_k, dot_scratch);
}

template <int M>
void ScanFloatScaledDenseMaterializedU16Avx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept {
  ComputeFloatRootDots(cluster, query, dot_scratch);
  ComputeFloatLinkedDotsMaterialized(
      cluster, query, cluster.parent_1based_u16(), dot_scratch);
  ScoreFloatRealRowsScaled(cluster, top_k, dot_scratch);
}

template <int M>
void ScanFloatScaledDenseMaterializedU32Avx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept {
  ComputeFloatRootDots(cluster, query, dot_scratch);
  ComputeFloatLinkedDotsMaterialized(
      cluster, query, cluster.parent_1based_u32(), dot_scratch);
  ScoreFloatRealRowsScaled(cluster, top_k, dot_scratch);
}

template <int M>
void ScanFloatScaledDenseNativeLoudsAvx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept {
  ComputeFloatRootDots(cluster, query, dot_scratch);
  ComputeFloatLinkedDotsNativeLouds(cluster, query, dot_scratch);
  ScoreFloatRealRowsScaled(cluster, top_k, dot_scratch);
}

template <int M>
void PrepareAndScanAngularDenseAvx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch,
    bool normalize, const float* cosine_inverse_norms) noexcept {
  assert(cluster.scaled_score_dot_multipliers() == nullptr &&
      cluster.row_dot_multipliers() == nullptr);
  PrepareScaledClusterQueryTables(query, cluster, scaled_query);
  ComputeRootDots(cluster, *scaled_query, dot_scratch);
  if (cluster.parent_1based_u16() != nullptr) {
    ComputeLinkedDotsMaterialized(
        cluster, *scaled_query, cluster.parent_1based_u16(), dot_scratch);
  } else if (cluster.parent_1based_u32() != nullptr) {
    ComputeLinkedDotsMaterialized(
        cluster, *scaled_query, cluster.parent_1based_u32(), dot_scratch);
  } else {
    ComputeLinkedDotsNativeLouds(cluster, *scaled_query, dot_scratch);
  }
  if (normalize) {
    const std::uint8_t* labels = cluster.real_norm8();
    const float* centers = cluster.norm8_centers();
    const float* rows = cluster.row_norm2();
    const float* inverses = cosine_inverse_norms;
    if (rows == nullptr && inverses != nullptr) {
      ScoreAngularRealRows<true, true>(
          cluster.real_ids(), dot_scratch + cluster.virtual_count(),
          cluster.real_count(), [labels, inverses](std::uint32_t row) noexcept {
            return inverses[labels[row]];
          }, top_k);
      return;
    }
    ScoreAngularRealRows<true>(
        cluster.real_ids(), dot_scratch + cluster.virtual_count(),
        cluster.real_count(),
        [labels, centers, rows](std::uint32_t row) noexcept {
          return rows != nullptr ? rows[row] : centers[labels[row]];
        }, top_k);
  } else {
    ScoreAngularRealRows<false>(
        cluster.real_ids(), dot_scratch + cluster.virtual_count(),
        cluster.real_count(),
        [](std::uint32_t) noexcept { return 0.0F; }, top_k);
  }
}

template <int M>
void ScanFloatAngularDenseAvx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch,
    bool normalize) noexcept {
  ComputeFloatRootDots(cluster, query, dot_scratch);
  if (cluster.parent_1based_u16() != nullptr) {
    ComputeFloatLinkedDotsMaterialized(
        cluster, query, cluster.parent_1based_u16(), dot_scratch);
  } else if (cluster.parent_1based_u32() != nullptr) {
    ComputeFloatLinkedDotsMaterialized(
        cluster, query, cluster.parent_1based_u32(), dot_scratch);
  } else {
    ComputeFloatLinkedDotsNativeLouds(cluster, query, dot_scratch);
  }
  if (normalize) {
    const float* norms = cluster.real_norm2();
    ScoreAngularRealRows<true>(
        cluster.real_ids(), dot_scratch + cluster.virtual_count(),
        cluster.real_count(),
        [norms](std::uint32_t row) noexcept { return norms[row]; }, top_k);
  } else {
    ScoreAngularRealRows<false>(
        cluster.real_ids(), dot_scratch + cluster.virtual_count(),
        cluster.real_count(),
        [](std::uint32_t) noexcept { return 0.0F; }, top_k);
  }
}

#define STLQPP_INSTANTIATE_ANGULAR_SCAN(m) \
  template void PrepareAndScanAngularDenseAvx2<m>( \
      const TrustedQueryTables<m>&, const ResidentClusterView<m>&, \
      ScaledClusterQueryTables<m>* __restrict, QueryTopK* __restrict, \
      float* __restrict, bool, const float*) noexcept; \
  template void ScanFloatAngularDenseAvx2<m>( \
      const TrustedQueryTables<m>&, const FloatResidentClusterView<m>&, \
      QueryTopK* __restrict, float* __restrict, bool) noexcept;
STLQPP_INSTANTIATE_ANGULAR_SCAN(5)
STLQPP_INSTANTIATE_ANGULAR_SCAN(6)
STLQPP_INSTANTIATE_ANGULAR_SCAN(10)
STLQPP_INSTANTIATE_ANGULAR_SCAN(12)
#undef STLQPP_INSTANTIATE_ANGULAR_SCAN

template void PrepareAndScanDenseMaterializedAvx2<5>(
    const TrustedQueryTables<5>&,
    const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict,
    QueryTopK* __restrict,
    float* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedAvx2<12>(
    const TrustedQueryTables<12>&,
    const ResidentClusterView<12>&,
    ScaledClusterQueryTables<12>* __restrict,
    QueryTopK* __restrict,
    float* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedAvx2<6>(
    const TrustedQueryTables<6>&,
    const ResidentClusterView<6>&,
    ScaledClusterQueryTables<6>* __restrict,
    QueryTopK* __restrict,
    float* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedAvx2<10>(
    const TrustedQueryTables<10>&,
    const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict,
    QueryTopK* __restrict,
    float* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU16Avx2<5>(
    const TrustedQueryTables<5>&, const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU16Avx2<12>(
    const TrustedQueryTables<12>&, const ResidentClusterView<12>&,
    ScaledClusterQueryTables<12>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU16Avx2<6>(
    const TrustedQueryTables<6>&, const ResidentClusterView<6>&,
    ScaledClusterQueryTables<6>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU16Avx2<10>(
    const TrustedQueryTables<10>&, const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU32Avx2<5>(
    const TrustedQueryTables<5>&, const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU32Avx2<12>(
    const TrustedQueryTables<12>&, const ResidentClusterView<12>&,
    ScaledClusterQueryTables<12>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU32Avx2<6>(
    const TrustedQueryTables<6>&, const ResidentClusterView<6>&,
    ScaledClusterQueryTables<6>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU32Avx2<10>(
    const TrustedQueryTables<10>&, const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
template void PrepareAndScanDenseNativeLoudsAvx2<5>(
    const TrustedQueryTables<5>&,
    const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
template void PrepareAndScanDenseNativeLoudsAvx2<12>(
    const TrustedQueryTables<12>&,
    const ResidentClusterView<12>&,
    ScaledClusterQueryTables<12>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
template void PrepareAndScanDenseNativeLoudsAvx2<6>(
    const TrustedQueryTables<6>&,
    const ResidentClusterView<6>&,
    ScaledClusterQueryTables<6>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
template void PrepareAndScanDenseNativeLoudsAvx2<10>(
    const TrustedQueryTables<10>&,
    const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedProfiledAvx2<5>(
    const TrustedQueryTables<5>&, const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedProfiledAvx2<12>(
    const TrustedQueryTables<12>&, const ResidentClusterView<12>&,
    ScaledClusterQueryTables<12>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedProfiledAvx2<6>(
    const TrustedQueryTables<6>&, const ResidentClusterView<6>&,
    ScaledClusterQueryTables<6>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedProfiledAvx2<10>(
    const TrustedQueryTables<10>&, const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU16ProfiledAvx2<5>(
    const TrustedQueryTables<5>&, const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU16ProfiledAvx2<12>(
    const TrustedQueryTables<12>&, const ResidentClusterView<12>&,
    ScaledClusterQueryTables<12>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU16ProfiledAvx2<6>(
    const TrustedQueryTables<6>&, const ResidentClusterView<6>&,
    ScaledClusterQueryTables<6>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU16ProfiledAvx2<10>(
    const TrustedQueryTables<10>&, const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU32ProfiledAvx2<5>(
    const TrustedQueryTables<5>&, const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU32ProfiledAvx2<12>(
    const TrustedQueryTables<12>&, const ResidentClusterView<12>&,
    ScaledClusterQueryTables<12>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU32ProfiledAvx2<6>(
    const TrustedQueryTables<6>&, const ResidentClusterView<6>&,
    ScaledClusterQueryTables<6>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
template void PrepareAndScanDenseMaterializedU32ProfiledAvx2<10>(
    const TrustedQueryTables<10>&, const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
template void ScanFloatDenseMaterializedAvx2<5>(
    const TrustedQueryTables<5>&,
    const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseMaterializedAvx2<12>(
    const TrustedQueryTables<12>&,
    const FloatResidentClusterView<12>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseMaterializedAvx2<6>(
    const TrustedQueryTables<6>&,
    const FloatResidentClusterView<6>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseMaterializedAvx2<10>(
    const TrustedQueryTables<10>&,
    const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseMaterializedU16Avx2<5>(
    const TrustedQueryTables<5>&, const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseMaterializedU16Avx2<12>(
    const TrustedQueryTables<12>&, const FloatResidentClusterView<12>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseMaterializedU16Avx2<6>(
    const TrustedQueryTables<6>&, const FloatResidentClusterView<6>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseMaterializedU16Avx2<10>(
    const TrustedQueryTables<10>&, const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseMaterializedU32Avx2<5>(
    const TrustedQueryTables<5>&, const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseMaterializedU32Avx2<12>(
    const TrustedQueryTables<12>&, const FloatResidentClusterView<12>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseMaterializedU32Avx2<6>(
    const TrustedQueryTables<6>&, const FloatResidentClusterView<6>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseMaterializedU32Avx2<10>(
    const TrustedQueryTables<10>&, const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseNativeLoudsAvx2<5>(
    const TrustedQueryTables<5>&,
    const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseNativeLoudsAvx2<12>(
    const TrustedQueryTables<12>&,
    const FloatResidentClusterView<12>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseNativeLoudsAvx2<6>(
    const TrustedQueryTables<6>&,
    const FloatResidentClusterView<6>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatDenseNativeLoudsAvx2<10>(
    const TrustedQueryTables<10>&,
    const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseMaterializedAvx2<5>(
    const TrustedQueryTables<5>&, const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseMaterializedAvx2<12>(
    const TrustedQueryTables<12>&, const FloatResidentClusterView<12>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseMaterializedAvx2<6>(
    const TrustedQueryTables<6>&, const FloatResidentClusterView<6>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseMaterializedAvx2<10>(
    const TrustedQueryTables<10>&, const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseMaterializedU16Avx2<5>(
    const TrustedQueryTables<5>&, const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseMaterializedU16Avx2<12>(
    const TrustedQueryTables<12>&, const FloatResidentClusterView<12>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseMaterializedU16Avx2<6>(
    const TrustedQueryTables<6>&, const FloatResidentClusterView<6>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseMaterializedU16Avx2<10>(
    const TrustedQueryTables<10>&, const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseMaterializedU32Avx2<5>(
    const TrustedQueryTables<5>&, const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseMaterializedU32Avx2<12>(
    const TrustedQueryTables<12>&, const FloatResidentClusterView<12>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseMaterializedU32Avx2<6>(
    const TrustedQueryTables<6>&, const FloatResidentClusterView<6>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseMaterializedU32Avx2<10>(
    const TrustedQueryTables<10>&, const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseNativeLoudsAvx2<5>(
    const TrustedQueryTables<5>&, const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseNativeLoudsAvx2<12>(
    const TrustedQueryTables<12>&, const FloatResidentClusterView<12>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseNativeLoudsAvx2<6>(
    const TrustedQueryTables<6>&, const FloatResidentClusterView<6>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanFloatScaledDenseNativeLoudsAvx2<10>(
    const TrustedQueryTables<10>&, const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanDenseAvx2<5>(
    const ResidentClusterView<5>&,
    const ScaledClusterQueryTables<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanDenseAvx2<12>(
    const ResidentClusterView<12>&,
    const ScaledClusterQueryTables<12>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanDenseAvx2<6>(
    const ResidentClusterView<6>&,
    const ScaledClusterQueryTables<6>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
template void ScanDenseAvx2<10>(
    const ResidentClusterView<10>&,
    const ScaledClusterQueryTables<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;

}  // namespace stlqpp
