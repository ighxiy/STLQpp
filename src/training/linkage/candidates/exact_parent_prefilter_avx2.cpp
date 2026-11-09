#include "exact_parent_prefilter.h"

#include <immintrin.h>

#include <cstdint>

namespace stlqpp::linkage_candidates {

std::uint32_t PrefilterEightParentCandidatesAvx2(
    float query_norm,
    const float* parent_norms,
    const float* dots,
    const std::uint32_t* local_ids,
    float worst_distance,
    std::uint32_t worst_local_id,
    float* distances) noexcept {
  const __m256 query_norms = _mm256_set1_ps(query_norm);
  const __m256 doubled_dots = _mm256_mul_ps(
      _mm256_loadu_ps(dots), _mm256_set1_ps(2.0F));
  const __m256 candidate_distances = _mm256_sub_ps(
      _mm256_add_ps(query_norms, _mm256_loadu_ps(parent_norms)),
      doubled_dots);
  _mm256_storeu_ps(distances, candidate_distances);

  const __m256 threshold = _mm256_set1_ps(worst_distance);
  const __m256 less = _mm256_cmp_ps(
      candidate_distances, threshold, _CMP_LT_OQ);
  const __m256 equal = _mm256_cmp_ps(
      candidate_distances, threshold, _CMP_EQ_OQ);

  const __m256i sign = _mm256_set1_epi32(
      static_cast<std::int32_t>(0x80000000U));
  const __m256i candidate_ids = _mm256_xor_si256(
      _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(local_ids)),
      sign);
  const __m256i threshold_ids = _mm256_set1_epi32(
      static_cast<std::int32_t>(worst_local_id ^ 0x80000000U));
  const __m256 id_before = _mm256_castsi256_ps(
      _mm256_cmpgt_epi32(threshold_ids, candidate_ids));
  const __m256 better = _mm256_or_ps(
      less, _mm256_and_ps(equal, id_before));
  return static_cast<std::uint32_t>(_mm256_movemask_ps(better));
}

}  // namespace stlqpp::linkage_candidates
