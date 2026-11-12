#pragma once

#include "stlqpp/core/status.h"

namespace stlqpp::app {

// Resolve requested acceleration against the build capability. The main
// CUDA switch governs every GPU operation, including norm preparation.
inline void ResolveCudaExecutionPolicy(bool cuda_available,
                                        bool& use_cuda,
                                        bool& gpu_norm_enable) noexcept {
  use_cuda = cuda_available && use_cuda;
  gpu_norm_enable = use_cuda && gpu_norm_enable;
}

// Resolve the compiled backend and visible devices before composing recipes.
[[nodiscard]] Status ResolveRuntimeCudaExecutionPolicy(
    bool& use_cuda, bool& gpu_norm_enable);

}  // namespace stlqpp::app
