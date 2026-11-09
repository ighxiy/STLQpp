#include "stlqpp/training/initialization/spherical_rvq_log.h"

#include "stlqpp/runtime/telemetry/logger.h"

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace stlqpp {

void LogSphericalRvqSeedStart(const SphericalRvqSeedStartLog& event) {
  std::ostringstream line;
  line << "[kmeans_seed] method=kmeans|| gpu=" << (event.gpu ? 1 : 0)
       << " sample_n=" << event.sample_count
       << " K=" << event.center_count
       << " rounds=" << event.requested_rounds
       << " l_fac=" << std::fixed << std::setprecision(2)
       << event.oversample_factor
       << " l_abs=" << static_cast<std::uint64_t>(
              event.oversample_factor *
              static_cast<float>(event.center_count))
       << " cap=" << event.candidate_cap;
  LogInfo(line.str());
}

void LogSphericalRvqSeedFinish(const SphericalRvqSeedFinishLog& event) {
  LogInfo(
      "[kmeans_seed_pp] K=" + std::to_string(event.center_count) +
      " rounds_ran=" + std::to_string(event.rounds_ran) +
      " picked_raw=" + std::to_string(event.picked_raw) +
      " picked_new=" + std::to_string(event.picked_new) +
      " cand_unique=" + std::to_string(event.unique_candidates) +
      " padded=" + std::to_string(event.padded_candidates) +
      " capped=" + std::to_string(event.capped_candidates) +
      " pp_input=" + std::to_string(event.reducer_input_candidates));
}

}  // namespace stlqpp
