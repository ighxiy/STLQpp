#include "stlqpp/training/initialization/spherical_lloyd_code_emission.h"

namespace stlqpp {

bool SphericalLloydCodeEmission::BeginIteration(
    std::uint32_t iteration,
    std::uint32_t maximum_iterations) noexcept {
  return retain_codes_ && !emitted_done_ &&
      (emitted_armed_ || iteration == maximum_iterations);
}

bool SphericalLloydCodeEmission::EndIteration(bool emitted,
                                              bool converged) noexcept {
  if (converged) {
    if (retain_codes_ && !emitted_done_ && !emitted) {
      emitted_armed_ = true;
      return false;
    }
    if (emitted) {
      emitted_done_ = true;
      emitted_armed_ = false;
    }
    return true;
  }
  if (emitted) {
    emitted_done_ = true;
    emitted_armed_ = false;
  }
  return false;
}

}  // namespace stlqpp
