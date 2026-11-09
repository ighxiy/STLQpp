#pragma once

#include <cstdint>

namespace stlqpp {

// Frozen DEV retained-code transition state. A convergence before emission
// arms the next executed Lloyd pass. Emission happens once; a non-converged
// emitting pass does not force termination.
class SphericalLloydCodeEmission final {
 public:
  explicit SphericalLloydCodeEmission(bool retain_codes) noexcept
      : retain_codes_(retain_codes) {}

  [[nodiscard]] bool BeginIteration(std::uint32_t iteration,
                                    std::uint32_t maximum_iterations)
      noexcept;

  // Returns true only when the DEV convergence rule terminates the loop.
  [[nodiscard]] bool EndIteration(bool emitted,
                                  bool converged) noexcept;

  [[nodiscard]] bool complete() const noexcept {
    return !retain_codes_ || emitted_done_;
  }

 private:
  bool retain_codes_ = false;
  bool emitted_armed_ = false;
  bool emitted_done_ = false;
};

}  // namespace stlqpp
