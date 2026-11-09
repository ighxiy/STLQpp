#pragma once

#include <cstdint>

namespace stlqpp {

// Selects how the residual cost of an already solved least-squares support is
// evaluated. The normal-equation form exploits the stationary condition
// Gc=b (or (G+lambda I)c=b after a registered diagonal retry).
enum class LeastSquaresCostMode : std::uint8_t {
  kExplicitQuadratic = 0U,
  kNormalEquation = 1U,
};

[[nodiscard]] constexpr const char* LeastSquaresCostModeName(
    LeastSquaresCostMode mode) noexcept {
  return mode == LeastSquaresCostMode::kNormalEquation
      ? "normal_equation" : "explicit_quadratic";
}

}  // namespace stlqpp
