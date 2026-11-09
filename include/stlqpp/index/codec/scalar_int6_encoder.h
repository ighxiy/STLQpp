#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/codec/canonical_huffman.h"
#include "stlqpp/model/codec/coefficient_bit_width.h"
#include "stlqpp/model/codec/scalar_int6_codec.h"
#include "stlqpp/index/codec/selected_atom_gram.h"
#include "stlqpp/runtime/cpu/spd_solver_f64.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {

template <int M>
class FinalClusterPostprocessor;

inline constexpr std::uint32_t kScalarInt6RefinementSweeps = 3U;
inline constexpr std::uint32_t kScalarInt6RefinementStepLimit = 1U;

struct ScalarCoefficientCodecOptions final {
  bool weighted_quantile = true;
  bool allow_clipping = true;
  bool fit_reconstruction_scales = true;
  std::uint32_t refinement_sweeps = kScalarInt6RefinementSweeps;
  // Zero refines every layer; a positive value refines only that prefix.
  std::uint32_t refinement_max_layer = 0U;
  // Zero uses the unconstrained rounded coordinate optimum.
  std::uint32_t refinement_step_limit = kScalarInt6RefinementStepLimit;
  // Internal reporting demand; does not change quantile selection or fitting.
  bool collect_sse_statistics = true;
};

// One canonical-BFS cluster at the final coefficient-codec boundary.
// Coefficients are layer-major. Each selected-atom Gram matrix is row-major
// and local to one node; callers may build and release these bounded buffers
// CID by CID.
template <int M>
struct ScalarInt6EncodingInput final {
  std::uint32_t m = M;
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t virtual_root_count = 0U;
  std::uint32_t real_root_count = 0U;
  std::uint32_t linked_real_count = 0U;
  ArrayView<const float> coefficients_layer_major;
  ArrayView<const float> selected_atom_grams_row_major;
};

template <int M>
struct SelectedScalarInt6EncodingInput final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t virtual_root_count = 0U;
  std::uint32_t real_root_count = 0U;
  std::uint32_t linked_real_count = 0U;
  ArrayView<const float> coefficients_layer_major;
  const SelectedAtomGramView<M>* selected_atom_grams = nullptr;
};

struct ScalarInt6EncodingStatistics final {
  double before_integer_refinement_sse = 0.0;
  double final_sse = 0.0;
  std::uint64_t root_payload_bits = 0U;
  std::uint64_t linkage_payload_bits = 0U;
};

template <int M>
class EncodedScalarInt6Cluster final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] ScalarInt6ClusterView<M> view() const noexcept {
    return {
        .m = static_cast<std::uint32_t>(root_scales_.size()),
        .virtual_root_count = virtual_root_count_,
        .real_root_count = real_root_count_,
        .linked_real_count = linked_real_count_,
        .root_scales = root_scales_,
        .linkage_scales = linkage_scales_,
        .root = {.code_lengths = root_.code_lengths,
                 .payload = root_.payload},
        .linkage = {.code_lengths = linkage_.code_lengths,
                    .payload = linkage_.payload},
    };
  }

  [[nodiscard]] const ModelLayers<float, M>& root_scales() const noexcept {
    return root_scales_;
  }
  [[nodiscard]] const ModelLayers<float, M>& linkage_scales() const noexcept {
    return linkage_scales_;
  }
  [[nodiscard]] const std::vector<std::int8_t>& quantized_coefficients()
      const noexcept {
    return coefficients_layer_major_;
  }
  [[nodiscard]] const ScalarInt6EncodingStatistics& statistics() const
      noexcept {
    return statistics_;
  }

  [[nodiscard]] std::uint64_t resident_bytes() const noexcept {
    return static_cast<std::uint64_t>(
        coefficients_layer_major_.capacity() * sizeof(std::int8_t) +
        root_.code_lengths.capacity() + root_.payload.capacity() +
        linkage_.code_lengths.capacity() + linkage_.payload.capacity());
  }

 private:
  template <int N>
  friend class ScalarInt6Encoder;

  std::uint32_t virtual_root_count_ = 0U;
  std::uint32_t real_root_count_ = 0U;
  std::uint32_t linked_real_count_ = 0U;
  ModelLayers<float, M> root_scales_{};
  ModelLayers<float, M> linkage_scales_{};
  std::vector<std::int8_t> coefficients_layer_major_;
  EncodedCanonicalHuffman root_;
  EncodedCanonicalHuffman linkage_;
  ScalarInt6EncodingStatistics statistics_;
};

// The registered final coefficient codec:
//   signed int6 + configured weighted-quantile initialization
//   reconstruction-aligned per-layer scale fit
//   three +/-1 integer coordinate sweeps
//   one canonical Huffman stream per CID and node role
template <int M>
class ScalarInt6Encoder final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  ScalarInt6Encoder(const SpdSolverF64& solver,
                    double magnitude_quantile) noexcept
      : solver_(&solver),
        base_bits_(M, kDefaultCoefficientBits),
        root_bits_(M, kDefaultCoefficientBits),
        linkage_bits_(M, kDefaultCoefficientBits),
        first_magnitude_quantiles_{magnitude_quantile},
        rest_magnitude_quantiles_{magnitude_quantile} {}

  ScalarInt6Encoder(
      const SpdSolverF64& solver,
      std::vector<double> first_magnitude_quantiles,
      std::vector<double> rest_magnitude_quantiles) noexcept
      : solver_(&solver),
        base_bits_(M, kDefaultCoefficientBits),
        root_bits_(M, kDefaultCoefficientBits),
        linkage_bits_(M, kDefaultCoefficientBits),
        first_magnitude_quantiles_(
            std::move(first_magnitude_quantiles)),
        rest_magnitude_quantiles_(
            std::move(rest_magnitude_quantiles)) {}

  ScalarInt6Encoder(
      const SpdSolverF64& solver,
      std::vector<std::uint32_t> base_bits,
      std::vector<std::uint32_t> root_bits,
      std::vector<std::uint32_t> linkage_bits,
      std::vector<double> first_magnitude_quantiles,
      std::vector<double> rest_magnitude_quantiles,
      ScalarCoefficientCodecOptions options = {}) noexcept
      : solver_(&solver),
        base_bits_(std::move(base_bits)),
        root_bits_(std::move(root_bits)),
        linkage_bits_(std::move(linkage_bits)),
        first_magnitude_quantiles_(std::move(first_magnitude_quantiles)),
        rest_magnitude_quantiles_(std::move(rest_magnitude_quantiles)),
        options_(options) {}

  [[nodiscard]] StatusOr<EncodedScalarInt6Cluster<M>> Encode(
      const ScalarInt6EncodingInput<M>& input) const;
  [[nodiscard]] StatusOr<EncodedScalarInt6Cluster<M>> EncodeSelected(
      const SelectedScalarInt6EncodingInput<M>& input) const;

 private:
  friend class FinalClusterPostprocessor<M>;

  // The postprocessor uses this only after canonical assembly validation.
  [[nodiscard]] StatusOr<EncodedScalarInt6Cluster<M>>
  EncodeSelectedValidated(
      const SelectedScalarInt6EncodingInput<M>& input) const;
  [[nodiscard]] StatusOr<EncodedScalarInt6Cluster<M>>
  EncodeSelectedImpl(
      const SelectedScalarInt6EncodingInput<M>& input,
      bool validated_coefficients) const;

  const SpdSolverF64* solver_ = nullptr;
  std::vector<std::uint32_t> base_bits_;
  std::vector<std::uint32_t> root_bits_;
  std::vector<std::uint32_t> linkage_bits_;
  std::vector<double> first_magnitude_quantiles_;
  std::vector<double> rest_magnitude_quantiles_;
  ScalarCoefficientCodecOptions options_;
};

extern template class EncodedScalarInt6Cluster<5>;
extern template class EncodedScalarInt6Cluster<10>;
extern template class ScalarInt6Encoder<5>;
extern template class ScalarInt6Encoder<10>;

}  // namespace stlqpp
