#pragma once

#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/training/initialization/cuda_kmeans_parallel_seeder.h"
#include "stlqpp/training/initialization/cuda_streaming_spherical_lloyd.h"
#include "stlqpp/training/initialization/root_model_initializer.h"

namespace stlqpp {

struct CudaSphericalRvqBackendOptions final {
  CudaKmeansParallelSeedOptions seeding;
  CudaStreamingSphericalLloydOptions lloyd;
};

template <typename Scalar = std::uint8_t>
class CudaSphericalRvqInitializationBackend final
    : public SphericalRvqInitializationBackend<Scalar> {
 public:
  CudaSphericalRvqInitializationBackend(
      CudaContext& context,
      const ParallelForExecutor& parallel,
      CudaSphericalRvqBackendOptions options) noexcept
      : context_(context), parallel_(parallel), options_(options) {}

  [[nodiscard]] StatusOr<SphericalRvqBackendResult> Initialize(
      InitializationBasicBlockSourceFactory<Scalar>& source,
      const SphericalRvqInitializationRequest& request) override;

 private:
  CudaContext& context_;
  const ParallelForExecutor& parallel_;
  CudaSphericalRvqBackendOptions options_;
};

extern template class CudaSphericalRvqInitializationBackend<std::uint8_t>;
extern template class CudaSphericalRvqInitializationBackend<float>;

}  // namespace stlqpp
