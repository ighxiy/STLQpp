#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/index/norm/reconstruction_norm8.h"

#include <array>
#include <memory>

namespace stlqpp {

// Bounded publication-time GPU fit. The complete norm population is already
// required by the global equal-quantile initializer; device values are kept
// resident for all Lloyd passes. No per-CID index data is copied to the GPU.
class CudaNorm8Fitter final {
 public:
  static StatusOr<std::unique_ptr<CudaNorm8Fitter>> Create(
      ArrayView<const float> values, int device = 0);
  ~CudaNorm8Fitter();
  CudaNorm8Fitter(const CudaNorm8Fitter&) = delete;
  CudaNorm8Fitter& operator=(const CudaNorm8Fitter&) = delete;

  StatusOr<std::array<float, 256>> InitialCenters();
  Status Accumulate(
      Norm8CodebookView codebook,
      std::array<std::uint64_t, 256>& counts,
      std::array<double, 256>& sums);
  StatusOr<std::array<float, 256>> Fit(std::uint32_t iterations);

 private:
  CudaNorm8Fitter() = default;
  struct State;
  std::unique_ptr<State> state_;
};

// Reusable per-publication GPU label encoder. Input/output buffers grow to the
// largest CID and are reused, so small clusters still execute on CUDA without
// per-CID allocation or a hidden CPU route.
class CudaNorm8LabelEncoder final {
 public:
  static StatusOr<std::unique_ptr<CudaNorm8LabelEncoder>> Create(
      Norm8CodebookView codebook, int device = 0);
  ~CudaNorm8LabelEncoder();
  CudaNorm8LabelEncoder(const CudaNorm8LabelEncoder&) = delete;
  CudaNorm8LabelEncoder& operator=(const CudaNorm8LabelEncoder&) = delete;

  Status Encode(ArrayView<const float> values,
                ArrayView<std::uint8_t> labels);

 private:
  CudaNorm8LabelEncoder() = default;
  struct State;
  std::unique_ptr<State> state_;
};

// The CUDA fitter intentionally keeps the norm population resident for the
// complete global fit. Failure is returned to the caller; there is no CPU
// fallback when GPU norm is requested.
StatusOr<ReconstructionNorm8Codebook> FitIdentityNorm8Cuda(
    ArrayView<const float> values, std::uint32_t iterations,
    int device = 0);

}  // namespace stlqpp
