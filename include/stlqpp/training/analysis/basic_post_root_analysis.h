#pragma once

#include <cstdint>

namespace stlqpp {

struct BasicPostRootAnalysis final {
  std::uint64_t rows = 0U;
  double squared_error_sum = 0.0;
};

}  // namespace stlqpp
