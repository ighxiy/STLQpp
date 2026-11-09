#pragma once

#include "stlqpp/core/status.h"

#include <cstdint>

namespace stlqpp {

struct SquareSvdSystemF32 final {
  // Borrowed column-major input. Solvers preserve it; factors use FP32 at
  // this boundary even when the decomposition is computed in FP64.
  const float* matrix = nullptr;
  std::uint32_t order = 0U;
  float* singular_values = nullptr;
  float* left_vectors = nullptr;
  float* right_vectors_transposed = nullptr;
  // Thread-budget permission only; never selects numerical precision.
  bool allow_phase_parallelism = false;
};

class SquareSvdF32 {
 public:
  virtual ~SquareSvdF32() = default;
  SquareSvdF32(const SquareSvdF32&) = delete;
  SquareSvdF32& operator=(const SquareSvdF32&) = delete;

  [[nodiscard]] virtual Status Solve(
      const SquareSvdSystemF32& system) const = 0;

 protected:
  SquareSvdF32() = default;
};

}  // namespace stlqpp
