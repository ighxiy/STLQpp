#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/norm8_codebook.h"

#include <array>
#include <cstdint>

namespace stlqpp {

struct PreparedSpoolPublicationAccess;
class ParallelForExecutor;

// Preserves the historical FP32 result for ordinary ratios. Only a ratio that
// overflows or underflows in FP32 uses a double intermediate. The returned
// scale is guaranteed to remain finite after multiplication by -2 in the
// scoring path.
[[nodiscard]] StatusOr<float> ComputeRadialScoreScale(
    float source_norm2, float reconstruction_norm2);

struct ScaledScore8CodebookView final {
  const float* scales = nullptr;
  const float* biases = nullptr;
  std::uint32_t center_count = 0U;

  [[nodiscard]] bool enabled() const noexcept {
    return scales != nullptr && biases != nullptr;
  }
};

class ScaledScore8Codebook final {
 public:
  [[nodiscard]] static StatusOr<ScaledScore8Codebook> Create(
      ArrayView<const float> scales, ArrayView<const float> biases);
  [[nodiscard]] const std::array<float, 256>& scales() const noexcept {
    return scales_;
  }
  [[nodiscard]] const std::array<float, 256>& biases() const noexcept {
    return biases_;
  }
  [[nodiscard]] ScaledScore8CodebookView view() const noexcept {
    return enabled_
        ? ScaledScore8CodebookView{scales_.data(), biases_.data(), 256U}
        : ScaledScore8CodebookView{};
  }

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }

 private:
  friend class ScaledScore8Accumulator;
  std::array<float, 256> scales_{};
  std::array<float, 256> biases_{};
  bool enabled_ = false;
};

// Geometry-only fixed-label fit.  Labels come from classic reconstruction
// norm8.  No query, query moment or ground truth participates.
class ScaledScore8Accumulator final {
 public:
  [[nodiscard]] Status Accumulate(
      ArrayView<const std::uint8_t> labels,
      ArrayView<const float> reconstruction_norm2,
      ArrayView<const float> source_norm2);
  // Publication/eval preparation only. Small populations stay serial; larger
  // populations reduce bounded partial tables through the caller's executor.
  [[nodiscard]] Status Accumulate(
      ArrayView<const std::uint8_t> labels,
      ArrayView<const float> reconstruction_norm2,
      ArrayView<const float> source_norm2,
      const ParallelForExecutor& parallel);
  void Merge(const ScaledScore8Accumulator& other) noexcept;
  [[nodiscard]] StatusOr<ScaledScore8Codebook> Finish(
      Norm8CodebookView classic) const;

 private:
  friend struct PreparedSpoolPublicationAccess;

  [[nodiscard]] Status AccumulateValidated(
      ArrayView<const std::uint8_t> labels,
      ArrayView<const float> reconstruction_norm2,
      ArrayView<const float> source_norm2);
  [[nodiscard]] Status AccumulateValidated(
      ArrayView<const std::uint8_t> labels,
      ArrayView<const float> reconstruction_norm2,
      ArrayView<const float> source_norm2,
      const ParallelForExecutor& parallel);
  [[nodiscard]] Status AccumulateParallel(
      ArrayView<const std::uint8_t> labels,
      ArrayView<const float> reconstruction_norm2,
      ArrayView<const float> source_norm2,
      bool validated_norms,
      const ParallelForExecutor& parallel);
  [[nodiscard]] Status AccumulateImpl(
      ArrayView<const std::uint8_t> labels,
      ArrayView<const float> reconstruction_norm2,
      ArrayView<const float> source_norm2,
      bool validated_norms);

  std::array<double, 256> scale_numerators_{};
  std::array<double, 256> scale_denominators_{};
  std::array<double, 256> bias_sums_{};
  std::array<std::uint64_t, 256> counts_{};
};

}  // namespace stlqpp
