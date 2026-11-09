#pragma once

#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/runtime_codebooks.h"

namespace stlqpp {

struct RuntimeCanonicalClusterCodeView final {
  std::uint32_t cid = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  std::uint32_t linked_real_count = 0U;
  ArrayView<const std::uint8_t> virtual_codes_small;
  ArrayView<const std::uint8_t> real_codes_small;
  ArrayView<const std::uint8_t> edge_code0;
};

struct RuntimeReconstructionNormInput final {
  RuntimeCanonicalClusterCodeView codes;
  ArrayView<const std::uint32_t> parent_local_1based;
  ArrayView<const std::int8_t> coefficients_layer_major;
  ArrayView<const float> root_scales;
  ArrayView<const float> linkage_scales;
};

struct RuntimeFloatReconstructionNormInput final {
  RuntimeCanonicalClusterCodeView codes;
  ArrayView<const std::uint32_t> parent_local_1based;
  ArrayView<const float> coefficients_layer_major;
};

// External cluster boundary, shared by CPU and CUDA. It does not rescan an
// immutable model, and performs no checks inside reconstruction's hot loops.
[[nodiscard]] Status ValidateRuntimeNormInput(
    const ModelShape& shape, const RuntimeReconstructionNormInput& input);
[[nodiscard]] Status ValidateRuntimeNormInput(
    const ModelShape& shape, const RuntimeFloatReconstructionNormInput& input);

// Explicit-vector test oracle. Production norm preparation uses Gram tables.
// Borrows an already validated model. The model must outlive the computer.
class RuntimeReconstructionNormComputer final {
 public:
  explicit RuntimeReconstructionNormComputer(
      RuntimeProductCodebookView model) noexcept
      : model_(model) {}
  [[nodiscard]] StatusOr<std::vector<float>> Compute(
      const RuntimeReconstructionNormInput& input,
      ArrayView<const float> reference = {},
      ArrayView<double> vector_sum = {}) const;
  [[nodiscard]] StatusOr<std::vector<float>> ComputeFloat(
      const RuntimeFloatReconstructionNormInput& input,
      ArrayView<const float> reference = {},
      ArrayView<double> vector_sum = {}) const;

 private:
  RuntimeProductCodebookView model_;
};

// Fixed-reference projection at index loading, using scalar forest DP. No
// vector reconstruction or query-path work is required on a cache hit.
class RuntimeReconstructionProjectionComputer final {
 public:
  [[nodiscard]] static StatusOr<RuntimeReconstructionProjectionComputer> Create(
      RuntimeProductCodebookView model, ArrayView<const float> reference);
  [[nodiscard]] StatusOr<std::vector<double>> Compute(
      const RuntimeReconstructionNormInput& input) const;
  [[nodiscard]] StatusOr<std::vector<double>> ComputeFloat(
      const RuntimeFloatReconstructionNormInput& input) const;
 private:
  ModelShape shape_;
  std::vector<double> root0_, root_small_, edge_;
};

// Sums represented real vectors without materializing them. Each node's
// local atom coefficients are weighted by its number of real descendants;
// the final model-sized reduction is the only dimension-dependent work.
// Borrows an already validated model for the accumulator's lifetime.
class RuntimeReconstructionMeanAccumulator final {
 public:
  explicit RuntimeReconstructionMeanAccumulator(RuntimeProductCodebookView model);
  [[nodiscard]] Status Accumulate(const RuntimeReconstructionNormInput& input);
  [[nodiscard]] Status AccumulateFloat(const RuntimeFloatReconstructionNormInput& input);
  [[nodiscard]] StatusOr<std::vector<float>> Finish() const;

 private:
  RuntimeProductCodebookView model_;
  std::vector<double> root0_, root_small_, edge_;
  std::uint64_t real_count_ = 0U;
};

}  // namespace stlqpp
