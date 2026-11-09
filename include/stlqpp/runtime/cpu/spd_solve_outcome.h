#pragma once

#include <cstdint>

namespace stlqpp {

enum class SpdSolveOutcome : std::uint8_t {
  kSolved,
  kNotPositiveDefinite,
};

}  // namespace stlqpp
