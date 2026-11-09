#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/index/norm/reconstruction_norm8_recipe.h"
#include "stlqpp/model/norm8_codebook.h"
#include "stlqpp/runtime/cpu/parallel_for.h"

#include <array>
#include <cstdint>
#include <vector>

namespace stlqpp {

struct PreparedSpoolPublicationAccess;
struct NormOnlyDerivationAccess;

class Norm8LloydAccumulator final {
 public:
  Norm8LloydAccumulator() = default;

  void Clear() noexcept;
  [[nodiscard]] Status Accumulate(Norm8CodebookView codebook,
                                  ArrayView<const float> norm2,
                                  ReconstructionNorm8Mode mode =
                                      ReconstructionNorm8Mode::kIdentityLloyd,
                                  float tail_p99 = 0.0F,
                                  float tail_p999 = 0.0F);
  void Merge(const Norm8LloydAccumulator& other) noexcept;
  void MergeBins(const std::array<std::uint64_t, 256>& counts,
                 const std::array<double, 256>& sums) noexcept;

 private:
  friend class ReconstructionNorm8Codebook;
  friend struct PreparedSpoolPublicationAccess;
  friend struct NormOnlyDerivationAccess;

  [[nodiscard]] Status AccumulateValidated(
      Norm8CodebookView codebook, ArrayView<const float> norm2,
      ReconstructionNorm8Mode mode, float tail_p99, float tail_p999);
  [[nodiscard]] Status AccumulateImpl(
      Norm8CodebookView codebook, ArrayView<const float> norm2,
      ReconstructionNorm8Mode mode, float tail_p99, float tail_p999,
      bool validated_norms);

  std::array<std::uint64_t, 256> counts_{};
  std::array<double, 256> sums_{};
};

// Global 256-entry identity-domain Lloyd quantizer for squared norms. The
// initializer consumes the complete norm population, matching DEV's exact
// equal-quantile initialization. Sorted fitting retains that population only
// through the fit and computes exact nearest-center intervals.
class ReconstructionNorm8Codebook final {
 public:
  [[nodiscard]] static StatusOr<ReconstructionNorm8Codebook>
  InitializeFromValues(
      std::vector<float> values,
      ReconstructionNorm8Mode mode =
          ReconstructionNorm8Mode::kIdentityLloyd);

  [[nodiscard]] static StatusOr<ReconstructionNorm8Codebook> FitFromValues(
      std::vector<float> values, std::uint32_t iterations,
      const ParallelForExecutor& parallel,
      ReconstructionNorm8Mode mode = ReconstructionNorm8Mode::kIdentityLloyd);

  // Accepts a completed, sorted identity-domain fit (for example a CUDA
  // Lloyd fit). This validates and copies only the 256 final centers.
  [[nodiscard]] static StatusOr<ReconstructionNorm8Codebook>
  FromIdentityCenters(const std::array<float, 256>& centers);

  [[nodiscard]] Norm8CodebookView view() const noexcept {
    return {centers_.data(), static_cast<std::uint32_t>(centers_.size())};
  }
  [[nodiscard]] const std::array<float, 256>& centers() const noexcept {
    return centers_;
  }
  [[nodiscard]] ReconstructionNorm8Mode mode() const noexcept {
    return mode_;
  }
  [[nodiscard]] float tail_p99() const noexcept { return tail_p99_; }
  [[nodiscard]] float tail_p999() const noexcept { return tail_p999_; }

  // Updates occupied centers from one complete pass. Empty centers retain
  // their previous values. Returns true once every center moves by <=1e-6.
  [[nodiscard]] bool FinishPass(const Norm8LloydAccumulator& pass) noexcept;

  [[nodiscard]] Status Encode(ArrayView<const float> norm2,
                              ArrayView<std::uint8_t> labels) const;

 private:
  friend struct PreparedSpoolPublicationAccess;
  friend struct NormOnlyDerivationAccess;

  [[nodiscard]] static StatusOr<ReconstructionNorm8Codebook> FitFromValidatedValues(
      std::vector<float> values, std::uint32_t iterations,
      const ParallelForExecutor& parallel, ReconstructionNorm8Mode mode);
  [[nodiscard]] static StatusOr<ReconstructionNorm8Codebook>
  InitializeSorted(ArrayView<const float> values,
                   ReconstructionNorm8Mode mode);
  [[nodiscard]] Status EncodeValidated(ArrayView<const float> norm2,
                                       ArrayView<std::uint8_t> labels) const;
  [[nodiscard]] Status EncodeImpl(ArrayView<const float> norm2,
                                  ArrayView<std::uint8_t> labels,
                                  bool validated_norms) const;

  std::array<float, 256> centers_{};
  ReconstructionNorm8Mode mode_ = ReconstructionNorm8Mode::kIdentityLloyd;
  float tail_p99_ = 0.0F;
  float tail_p999_ = 0.0F;
};

}  // namespace stlqpp
