#include "stlqpp/index/norm/reconstruction_norm8_cuda.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {

StatusOr<ReconstructionNorm8Codebook> FitIdentityNorm8Cuda(
    ArrayView<const float> values, std::uint32_t iterations, int device) {
  if (values.empty() || iterations == 0U) {
    return Status::InvalidArgument("CUDA norm8 fit requires values and passes");
  }
  for (const float value : values) {
    if (!std::isfinite(value) || value < 0.0F) {
      return Status::InvalidArgument("CUDA norm8 input is invalid");
    }
  }
  auto gpu = CudaNorm8Fitter::Create(values, device);
  if (!gpu.ok()) return gpu.status();
  auto centers = gpu.value()->Fit(iterations);
  if (!centers.ok()) return centers.status();
  return ReconstructionNorm8Codebook::FromIdentityCenters(centers.value());
}

}  // namespace stlqpp
