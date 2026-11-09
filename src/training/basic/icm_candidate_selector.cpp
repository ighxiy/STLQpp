#include "stlqpp/training/basic/detail/icm_candidate_selector.h"

#include "stlqpp/platform/cpu_features.h"

#include <cmath>
#include <limits>

namespace stlqpp::detail {
namespace {

using Selector = IcmCandidateSelection (*)(
    const float*, const float*, std::uint32_t, bool) noexcept;

[[nodiscard]] Selector ResolveSelector() noexcept {
  const CpuFeatures features = DetectCpuFeatures();
  if (features.avx_register_state && features.avx2) {
    return SelectIcmCandidateAvx2;
  }
  return SelectIcmCandidateScalar;
}

}  // namespace

IcmCandidateSelection SelectIcmCandidateScalar(
    const float* residual_dots,
    const float* inverse_norms,
    std::uint32_t old_code,
    bool use_absolute_score) noexcept {
  IcmCandidateSelection best{
      .code = old_code,
      .score = -std::numeric_limits<float>::infinity(),
  };
  for (std::uint32_t code = 0U; code < 256U; ++code) {
    float score = residual_dots[code] * inverse_norms[code];
    if (use_absolute_score) {
      score = std::fabs(score);
    }
    if (score > best.score) {
      best.score = score;
      best.code = code;
    }
  }
  return best;
}

IcmCandidateSelection SelectIcmCandidate(
    const float* residual_dots,
    const float* inverse_norms,
    std::uint32_t old_code,
    bool use_absolute_score) noexcept {
  static const Selector selector = ResolveSelector();
  return selector(
      residual_dots, inverse_norms, old_code, use_absolute_score);
}

}  // namespace stlqpp::detail
