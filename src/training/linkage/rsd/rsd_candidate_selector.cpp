#include "stlqpp/training/linkage/rsd/detail/rsd_candidate_selector.h"

#include "stlqpp/platform/cpu_features.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace stlqpp::detail {
namespace {

using Selector = void (*)(const float*, const float*, std::uint32_t,
                          std::uint32_t, std::uint32_t,
                          RsdRankedAtom*) noexcept;

[[nodiscard]] bool Better(RsdRankedAtom candidate,
                          RsdRankedAtom incumbent) noexcept {
  return candidate.score > incumbent.score ||
      (candidate.score == incumbent.score &&
       candidate.atom < incumbent.atom);
}

void Initialize(std::uint32_t rank_count,
                RsdRankedAtom* output) noexcept {
  for (std::uint32_t rank = 0U; rank < rank_count; ++rank) {
    output[rank] = {
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<std::uint32_t>::max(),
    };
  }
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

[[nodiscard]] Selector ResolveSelector() noexcept {
  const CpuFeatures features = DetectCpuFeatures();
  if (features.avx_register_state && features.avx2) {
    return SelectRsdCandidatesAvx2;
  }
  return SelectRsdCandidatesScalar;
}

}  // namespace

void SelectRsdCandidatesScalar(
    const float* residual_dots,
    const float* inverse_norms,
    std::uint32_t layer_count,
    std::uint32_t available_layers,
    std::uint32_t rank_count,
    RsdRankedAtom* output) noexcept {
  Initialize(rank_count, output);
  for (std::uint32_t layer = 0U; layer < layer_count; ++layer) {
    if ((available_layers & (1U << layer)) == 0U) {
      continue;
    }
    const std::uint32_t begin = layer * 256U;
    for (std::uint32_t code = 0U; code < 256U; ++code) {
      const std::uint32_t atom = begin + code;
      Insert({
                 std::fabs(residual_dots[atom] * inverse_norms[atom]),
                 atom,
             },
             rank_count, output);
    }
  }
}

void SelectRsdCandidates(
    const float* residual_dots,
    const float* inverse_norms,
    std::uint32_t layer_count,
    std::uint32_t available_layers,
    std::uint32_t rank_count,
    RsdRankedAtom* output) noexcept {
  static const Selector selector = ResolveSelector();
  selector(residual_dots, inverse_norms, layer_count, available_layers,
           rank_count, output);
}

}  // namespace stlqpp::detail
