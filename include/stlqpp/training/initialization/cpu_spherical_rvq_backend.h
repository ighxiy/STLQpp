#pragma once

#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/initialization/root_model_initializer.h"

#include <cstdint>

namespace stlqpp {

struct CpuSphericalRvqBackendOptions final {
  bool collect_metrics = true;
};

// CPU implementation of the registered spherical-RVQ initialization recipe.
// It preserves the CUDA backend's source window, kmeans||, reduction, Lloyd,
// annealing and retained-code boundaries with CPU-native reductions.
template <typename Scalar = std::uint8_t>
class CpuSphericalRvqInitializationBackend final
    : public SphericalRvqInitializationBackend<Scalar> {
 public:
  CpuSphericalRvqInitializationBackend(
      const ParallelForExecutor& parallel,
      CpuSphericalRvqBackendOptions options = {}) noexcept
      : parallel_(&parallel), options_(options) {}

  [[nodiscard]] StatusOr<SphericalRvqBackendResult> Initialize(
      InitializationBasicBlockSourceFactory<Scalar>& source,
      const SphericalRvqInitializationRequest& request) override;

 private:
  const ParallelForExecutor* parallel_ = nullptr;
  CpuSphericalRvqBackendOptions options_;
};

extern template class CpuSphericalRvqInitializationBackend<std::uint8_t>;
extern template class CpuSphericalRvqInitializationBackend<float>;

}  // namespace stlqpp
