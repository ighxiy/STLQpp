#pragma once

#include <cstdint>

namespace stlqpp {

struct SphericalRvqSeedStartLog final {
  bool gpu = false;
  std::uint32_t sample_count = 0U;
  std::uint32_t center_count = 0U;
  std::uint32_t requested_rounds = 0U;
  float oversample_factor = 0.0F;
  std::uint32_t candidate_cap = 0U;
};

struct SphericalRvqSeedFinishLog final {
  std::uint32_t center_count = 0U;
  std::uint32_t rounds_ran = 0U;
  std::uint32_t picked_raw = 0U;
  std::uint32_t picked_new = 0U;
  std::uint32_t unique_candidates = 0U;
  std::uint32_t padded_candidates = 0U;
  std::uint32_t capped_candidates = 0U;
  std::uint32_t reducer_input_candidates = 0U;
};

// DEV-compatible spherical kmeans|| transcript formatting. The algorithm
// owner supplies actual counters at the seed boundaries; formatting remains
// outside candidate and assignment hot loops.
void LogSphericalRvqSeedStart(const SphericalRvqSeedStartLog& event);
void LogSphericalRvqSeedFinish(const SphericalRvqSeedFinishLog& event);

}  // namespace stlqpp
