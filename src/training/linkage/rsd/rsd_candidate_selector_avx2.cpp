#include "stlqpp/training/linkage/rsd/detail/rsd_candidate_selector.h"

#include <immintrin.h>

#include <cstdint>
#include <limits>

namespace stlqpp::detail {
namespace {

[[nodiscard]] bool Better(RsdRankedAtom candidate,
                          RsdRankedAtom incumbent) noexcept {
  return candidate.score > incumbent.score ||
      (candidate.score == incumbent.score &&
       candidate.atom < incumbent.atom);
}

void Insert(RsdRankedAtom candidate,
            std::uint32_t rank_count,
            RsdRankedAtom* output) noexcept {
  if (!Better(candidate, output[rank_count - 1U])) {
    return;
  }
  std::uint32_t position = rank_count - 1U;
  while (position != 0U && Better(candidate, output[position - 1U])) {
    output[position] = output[position - 1U];
    --position;
  }
  output[position] = candidate;
}

}  // namespace

void SelectRsdCandidatesAvx2(
    const float* residual_dots,
    const float* inverse_norms,
    std::uint32_t layer_count,
    std::uint32_t available_layers,
    std::uint32_t rank_count,
    RsdRankedAtom* output) noexcept {
  const float negative_infinity =
      -std::numeric_limits<float>::infinity();
  const __m256 negative_infinities = _mm256_set1_ps(negative_infinity);
  const __m256 absolute_mask = _mm256_castsi256_ps(
      _mm256_set1_epi32(0x7fffffff));
  const __m256i sentinel_atoms = _mm256_set1_epi32(
      std::numeric_limits<std::int32_t>::max());

  for (std::uint32_t rank = 0U; rank < rank_count; ++rank) {
    output[rank] = {
        negative_infinity,
        std::numeric_limits<std::uint32_t>::max(),
    };
  }

  for (std::uint32_t layer = 0U; layer < layer_count; ++layer) {
    if ((available_layers & (1U << layer)) == 0U) {
      continue;
    }
    __m256 lane_scores[kMaximumRsdSelectedCandidates] = {
        negative_infinities, negative_infinities, negative_infinities};
    __m256i lane_atoms[kMaximumRsdSelectedCandidates] = {
        sentinel_atoms, sentinel_atoms, sentinel_atoms};
    const std::uint32_t begin = layer * 256U;

    for (std::uint32_t code = 0U; code < 256U; code += 8U) {
      const std::uint32_t atom = begin + code;
      __m256 candidate_score = _mm256_and_ps(
          _mm256_mul_ps(
              _mm256_loadu_ps(residual_dots + atom),
              _mm256_loadu_ps(inverse_norms + atom)),
          absolute_mask);
      __m256i candidate_atom = _mm256_setr_epi32(
          static_cast<int>(atom), static_cast<int>(atom + 1U),
          static_cast<int>(atom + 2U), static_cast<int>(atom + 3U),
          static_cast<int>(atom + 4U), static_cast<int>(atom + 5U),
          static_cast<int>(atom + 6U), static_cast<int>(atom + 7U));

      for (std::uint32_t rank = 0U; rank < rank_count; ++rank) {
        const __m256 old_score = lane_scores[rank];
        const __m256i old_atom = lane_atoms[rank];
        const __m256 greater = _mm256_cmp_ps(
            candidate_score, old_score, _CMP_GT_OQ);
        const __m256 equal = _mm256_cmp_ps(
            candidate_score, old_score, _CMP_EQ_OQ);
        const __m256 atom_before = _mm256_castsi256_ps(
            _mm256_cmpgt_epi32(old_atom, candidate_atom));
        const __m256 better = _mm256_or_ps(
            greater, _mm256_and_ps(equal, atom_before));
        lane_scores[rank] = _mm256_blendv_ps(
            old_score, candidate_score, better);
        lane_atoms[rank] = _mm256_blendv_epi8(
            old_atom, candidate_atom, _mm256_castps_si256(better));
        candidate_score = _mm256_blendv_ps(
            candidate_score, old_score, better);
        candidate_atom = _mm256_blendv_epi8(
            candidate_atom, old_atom, _mm256_castps_si256(better));
      }
    }

    alignas(32) float local_scores[kMaximumRsdSelectedCandidates][8];
    alignas(32) std::int32_t local_atoms[kMaximumRsdSelectedCandidates][8];
    for (std::uint32_t rank = 0U; rank < rank_count; ++rank) {
      _mm256_store_ps(local_scores[rank], lane_scores[rank]);
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(local_atoms[rank]), lane_atoms[rank]);
    }
    for (std::uint32_t rank = 0U; rank < rank_count; ++rank) {
      for (std::uint32_t lane = 0U; lane < 8U; ++lane) {
        if (local_atoms[rank][lane] ==
            std::numeric_limits<std::int32_t>::max()) {
          continue;
        }
        Insert({
                   local_scores[rank][lane],
                   static_cast<std::uint32_t>(local_atoms[rank][lane]),
               },
               rank_count, output);
      }
    }
  }
}

}  // namespace stlqpp::detail
