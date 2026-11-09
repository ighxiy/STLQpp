#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/index/codec/selected_atom_gram.h"
#include "stlqpp/index/norm/runtime_reconstruction_norms.h"
#include "stlqpp/model/product_codebooks.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

template <int M>
class FinalClusterPostprocessor;

template <int M>
struct ReconstructionNormInput final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  CanonicalClusterCodeView<M> codes;
  ArrayView<const std::uint32_t> parent_local_1based;
  ArrayView<const std::int8_t> coefficients_layer_major;
  ArrayView<const float> root_scales;
  ArrayView<const float> linkage_scales;
};

template <int M>
struct FloatReconstructionNormInput final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  CanonicalClusterCodeView<M> codes;
  ArrayView<const std::uint32_t> parent_local_1based;
  ArrayView<const float> coefficients_layer_major;
};

// Explicit-vector test oracle; production uses the Gram service below.
// Reconstructs one CID exactly under the resident signed-int8 coefficient
// semantics and emits norm squared for real rows only. Virtual rows remain
// transient causal prefixes and never receive persistent norm labels.
template <int M>
class ReconstructionNormComputer final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  explicit ReconstructionNormComputer(
      const ProductCodebooks<M>& codebooks) noexcept
      : codebooks_(&codebooks) {}

  [[nodiscard]] StatusOr<std::vector<float>> Compute(
      const ReconstructionNormInput<M>& input) const;

 private:
  const ProductCodebooks<M>* codebooks_ = nullptr;
};

// Explicit-vector test oracle for the native-float forest.
// This is deliberately a separate input and implementation from the int6
// reconstruction path: neither representation may borrow the other's norm.
template <int M>
class FloatReconstructionNormComputer final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  explicit FloatReconstructionNormComputer(
      const ProductCodebooks<M>& codebooks) noexcept
      : codebooks_(&codebooks) {}

  [[nodiscard]] StatusOr<std::vector<float>> Compute(
      const FloatReconstructionNormInput<M>& input) const;

 private:
  const ProductCodebooks<M>* codebooks_ = nullptr;
};

// Production norm path aligned with DEV's LinkageNormProviderLookup. It uses
// model-level Gram tables and a causal ancestor recurrence, so no d-by-node
// reconstruction matrix is materialized during final publication.
template <int M>
class SelectedAtomReconstructionNormComputer final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] StatusOr<std::vector<float>> Compute(
      const SelectedAtomGramView<M>& grams,
      const ReconstructionNormInput<M>& input) const;
  [[nodiscard]] StatusOr<std::vector<float>> ComputeFloat(
      const SelectedAtomGramView<M>& grams,
      const FloatReconstructionNormInput<M>& input) const;

 private:
  friend class FinalClusterPostprocessor<M>;

  // The postprocessor uses this only for assembled float coefficients.
  [[nodiscard]] StatusOr<std::vector<float>> ComputeFloatValidated(
      const SelectedAtomGramView<M>& grams,
      const FloatReconstructionNormInput<M>& input) const;
  [[nodiscard]] StatusOr<std::vector<float>> ComputeFloatImpl(
      const SelectedAtomGramView<M>& grams,
      const FloatReconstructionNormInput<M>& input,
      bool validated_coefficients) const;
};

extern template class ReconstructionNormComputer<5>;
extern template class ReconstructionNormComputer<10>;
extern template class FloatReconstructionNormComputer<5>;
extern template class FloatReconstructionNormComputer<10>;
extern template class SelectedAtomReconstructionNormComputer<5>;
extern template class SelectedAtomReconstructionNormComputer<10>;

// Model-level read-only tables; every Compute owns only CID cross tables and
// scalar causal state. The validated borrowed model outlives this computer.
class RuntimeGramReconstructionNormComputer final {
 public:
  [[nodiscard]] static StatusOr<RuntimeGramReconstructionNormComputer> Create(
      RuntimeProductCodebookView model);
  [[nodiscard]] StatusOr<std::vector<float>> Compute(
      const RuntimeReconstructionNormInput& input) const;
  [[nodiscard]] StatusOr<std::vector<float>> ComputeFloat(
      const RuntimeFloatReconstructionNormInput& input) const;
 private:
  explicit RuntimeGramReconstructionNormComputer(SelectedAtomGramTables<0> grams)
      : grams_(std::move(grams)) {}
  SelectedAtomGramTables<0> grams_;
};

}  // namespace stlqpp
