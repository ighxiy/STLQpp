#include "stlqpp/training/basic/detail/icm_candidate_selector.h"

#include <immintrin.h>

#include <cstdint>
#include <limits>

namespace stlqpp::detail {

IcmCandidateSelection SelectIcmCandidateAvx2(
    const float* residual_dots,
    const float* inverse_norms,
    std::uint32_t old_code,
    bool use_absolute_score) noexcept {
  const __m256 negative_infinity = _mm256_set1_ps(
      -std::numeric_limits<float>::infinity());
  const __m256 absolute_mask = _mm256_castsi256_ps(
      _mm256_set1_epi32(0x7fffffff));
  __m256 lane_scores = negative_infinity;
  __m256i lane_codes = _mm256_set1_epi32(
      static_cast<int>(old_code));

  for (std::uint32_t code = 0U; code < 256U; code += 8U) {
    __m256 scores = _mm256_mul_ps(
        _mm256_loadu_ps(residual_dots + code),
        _mm256_loadu_ps(inverse_norms + code));
    if (use_absolute_score) {
      scores = _mm256_and_ps(scores, absolute_mask);
    }
    // Codes within each lane arrive in ascending order. Strict replacement
    // therefore preserves the scalar loop's first-code tie rule locally.
    const __m256 better = _mm256_cmp_ps(
        scores, lane_scores, _CMP_GT_OQ);
    lane_scores = _mm256_blendv_ps(lane_scores, scores, better);
    const __m256i codes = _mm256_setr_epi32(
        static_cast<int>(code), static_cast<int>(code + 1U),
        static_cast<int>(code + 2U), static_cast<int>(code + 3U),
        static_cast<int>(code + 4U), static_cast<int>(code + 5U),
        static_cast<int>(code + 6U), static_cast<int>(code + 7U));
    lane_codes = _mm256_blendv_epi8(
        lane_codes, codes, _mm256_castps_si256(better));
  }

  alignas(32) float scores[8];
  alignas(32) std::int32_t codes[8];
  _mm256_store_ps(scores, lane_scores);
  _mm256_store_si256(
      reinterpret_cast<__m256i*>(codes), lane_codes);

  IcmCandidateSelection best{
      .code = old_code,
      .score = -std::numeric_limits<float>::infinity(),
  };
  for (std::uint32_t lane = 0U; lane < 8U; ++lane) {
    const float score = scores[lane];
    const std::uint32_t code = static_cast<std::uint32_t>(codes[lane]);
    if (score > best.score ||
        (score == best.score &&
         score > -std::numeric_limits<float>::infinity() &&
         code < best.code)) {
      best.score = score;
      best.code = code;
    }
  }
  return best;
}

}  // namespace stlqpp::detail
