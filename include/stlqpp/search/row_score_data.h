#pragma once

#include <vector>
#include <cmath>
#include "stlqpp/core/status.h"

namespace stlqpp {

// Prepared independently of the coefficient representation. Owned by the
// caller during index opening; the resident index takes its own score arrays.
// No reconstruction, fitting or allocation happens in the query scan.
struct ClusterRowScores final {
  std::vector<float> norm2;
  std::vector<float> biases;
  std::vector<float> dot_multipliers;
  // Reference terms have been folded into norm2/bias at loading. These
  // approximate intercepts can be negative; the scan uses the same arrays.
  bool has_reference_correction = false;
};

inline Status ValidateClusterRowScores(const ClusterRowScores& scores,
                                      std::size_t count) {
  if (scores.norm2.size() != count ||
      ((!scores.biases.empty() || !scores.dot_multipliers.empty()) &&
       (scores.biases.size() != count || scores.dot_multipliers.size() != count)))
    return Status::DataLoss("evaluation row score extent disagrees with CID");
  for (float value : scores.norm2)
    if (!std::isfinite(value) || (!scores.has_reference_correction && value < 0.0F))
      return Status::DataLoss("invalid evaluation reconstruction norm");
  for (std::size_t row = 0; row < scores.biases.size(); ++row)
    if (!std::isfinite(scores.biases[row]) ||
        (!scores.has_reference_correction && scores.biases[row] < 0.0F) ||
        !std::isfinite(scores.dot_multipliers[row]) || scores.dot_multipliers[row] > 0.0F)
      return Status::DataLoss("invalid evaluation scaled row score");
  return Status::Ok();
}

struct IndexRowScores final {
  std::vector<ClusterRowScores> clusters;
};

}  // namespace stlqpp
