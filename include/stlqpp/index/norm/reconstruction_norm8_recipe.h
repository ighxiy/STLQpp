#pragma once

#include <cstdint>

namespace stlqpp {

inline constexpr std::uint32_t kRegisteredNorm8CenterCount = 256U;

// The fitting rule is producer provenance because it changes both persisted
// centers and per-row labels.  Identity Lloyd is the ordinary LUT.  The
// tail-weighted rule is retained only for source-compatible historical
// comparisons and uses DEV's fixed p99/p999 weights (1, 4 and 16).
enum class ReconstructionNorm8Mode : std::uint8_t {
  kIdentityLloyd = 0,
  kTailWeightedLloyd = 1,
  // Eval-only: global reconstruction mean, log-domain labels and linear
  // residual norm2 centers. Never written as a canonical index recipe.
  kReferenceLloyd = 2,
};

inline constexpr const char* ReconstructionNorm8ModeName(
    ReconstructionNorm8Mode mode) noexcept {
  return mode == ReconstructionNorm8Mode::kReferenceLloyd ? "reference"
       : mode == ReconstructionNorm8Mode::kTailWeightedLloyd ? "tail_weighted"
       : "identity";
}

inline constexpr ReconstructionNorm8Mode PublishedNorm8Mode(
    ReconstructionNorm8Mode mode) noexcept {
  return mode == ReconstructionNorm8Mode::kReferenceLloyd
      ? ReconstructionNorm8Mode::kIdentityLloyd : mode;
}

}  // namespace stlqpp
