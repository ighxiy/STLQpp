#include "stlqpp/index/norm/reconstruction_norms.h"

#include <cassert>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#if STLQPP_HAS_CBLAS_GRAM_NORM
#include "stlqpp/runtime/cpu/cblas_f32_gemm.h"
#endif

namespace stlqpp {
namespace {

template <int M>
Status ValidateInput(const ProductCodebooks<M>& codebooks,
                     const ReconstructionNormInput<M>& input,
                     std::uint32_t& real_count,
                     std::uint32_t& node_count) {
  const auto& codes = input.codes;
  const auto m = codebooks.shape().m;
  if (codes.m != m || codes.cid >= codebooks.shape().nlist) {
    return Status::InvalidArgument(
        "reconstruction norm CID exceeds native IVF nlist");
  }
  const std::uint64_t reals =
      static_cast<std::uint64_t>(codes.real_root_count) +
      codes.linked_real_count;
  const std::uint64_t nodes =
      static_cast<std::uint64_t>(codes.virtual_count) + reals;
  if (reals > std::numeric_limits<std::uint32_t>::max() ||
      nodes > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "reconstruction norm cluster exceeds canonical u32 rows");
  }
  real_count = static_cast<std::uint32_t>(reals);
  node_count = static_cast<std::uint32_t>(nodes);
  if (input.parent_local_1based.size() != real_count ||
      input.coefficients_layer_major.size() !=
          static_cast<std::size_t>(node_count) * m ||
      input.root_scales.size() != m ||
      input.linkage_scales.size() != m ||
      codes.virtual_codes_small.size() !=
          static_cast<std::size_t>(codes.virtual_count) * (m - 1) ||
      codes.real_codes_small.size() !=
          static_cast<std::size_t>(real_count) * (m - 1) ||
      codes.edge_code0.size() != real_count) {
    return Status::InvalidArgument(
        "reconstruction norm cluster arrays disagree with role counts");
  }
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    const std::uint32_t node = codes.virtual_count + real;
    const std::uint32_t parent = input.parent_local_1based[real];
    if (real < codes.real_root_count) {
      if (parent != 0U) {
        return Status::InvalidArgument(
            "reconstruction norm real root unexpectedly has a parent");
      }
    } else if (parent == 0U || parent > node) {
      return Status::InvalidArgument(
          "reconstruction norm linked parent is not causally earlier");
    }
  }
  for (const float scale : input.root_scales) {
    if (!std::isfinite(scale)) {
      return Status::InvalidArgument(
          "reconstruction norm root scale is not finite");
    }
  }
  for (const float scale : input.linkage_scales) {
    if (!std::isfinite(scale)) {
      return Status::InvalidArgument(
          "reconstruction norm linkage scale is not finite");
    }
  }
  return Status::Ok();
}

template <int M>
Status ValidateFloatInput(const ProductCodebooks<M>& codebooks,
                          const FloatReconstructionNormInput<M>& input,
                          std::uint32_t& real_count,
                          std::uint32_t& node_count) {
  const auto& codes = input.codes;
  const auto m = codebooks.shape().m;
  if (codes.m != m || codes.cid >= codebooks.shape().nlist) {
    return Status::InvalidArgument(
        "float reconstruction norm CID exceeds native IVF nlist");
  }
  const std::uint64_t reals =
      static_cast<std::uint64_t>(codes.real_root_count) +
      codes.linked_real_count;
  const std::uint64_t nodes =
      static_cast<std::uint64_t>(codes.virtual_count) + reals;
  if (reals > std::numeric_limits<std::uint32_t>::max() ||
      nodes > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "float reconstruction norm cluster exceeds canonical u32 rows");
  }
  real_count = static_cast<std::uint32_t>(reals);
  node_count = static_cast<std::uint32_t>(nodes);
  if (input.parent_local_1based.size() != real_count ||
      input.coefficients_layer_major.size() !=
          static_cast<std::size_t>(node_count) * m ||
      codes.virtual_codes_small.size() !=
          static_cast<std::size_t>(codes.virtual_count) * (m - 1U) ||
      codes.real_codes_small.size() !=
          static_cast<std::size_t>(real_count) * (m - 1U) ||
      codes.edge_code0.size() != real_count) {
    return Status::InvalidArgument(
        "float reconstruction norm arrays disagree with role counts");
  }
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    const std::uint32_t node = codes.virtual_count + real;
    const std::uint32_t parent = input.parent_local_1based[real];
    if ((real < codes.real_root_count && parent != 0U) ||
        (real >= codes.real_root_count &&
         (parent == 0U || parent > node))) {
      return Status::InvalidArgument(
          "float reconstruction norm parent stream is not causal");
    }
  }
  for (const float coefficient : input.coefficients_layer_major) {
    if (!std::isfinite(coefficient)) {
      return Status::InvalidArgument(
          "float reconstruction coefficient is not finite");
    }
  }
  return Status::Ok();
}

template <int M>
std::uint8_t RootSmallCode(const CanonicalClusterCodeView<M>& codes,
                           std::uint32_t root_node,
                           std::uint32_t layer) noexcept {
  const auto m = M == 0 ? codes.m : M;
  if (root_node < codes.virtual_count) {
    return codes.virtual_codes_small[
        static_cast<std::size_t>(root_node) * (m - 1) + layer - 1U];
  }
  const std::uint32_t real = root_node - codes.virtual_count;
  return codes.real_codes_small[
      static_cast<std::size_t>(real) * (m - 1) + layer - 1U];
}

}  // namespace

template <int M>
StatusOr<std::vector<float>> ReconstructionNormComputer<M>::Compute(
    const ReconstructionNormInput<M>& input) const {
  std::uint32_t real_count = 0U;
  std::uint32_t node_count = 0U;
  if (const Status status =
          ValidateInput(*codebooks_, input, real_count, node_count);
      !status.ok()) {
    return status;
  }
  const std::uint32_t m = codebooks_->shape().m;
  const std::uint32_t dimension = codebooks_->shape().dimension;
  const std::uint32_t root_count =
      input.codes.virtual_count + input.codes.real_root_count;
  std::vector<float> reconstructions(
      static_cast<std::size_t>(node_count) * dimension, 0.0F);

  for (std::uint32_t root = 0U; root < root_count; ++root) {
    float* destination = reconstructions.data() +
        static_cast<std::size_t>(root) * dimension;
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      const std::int32_t q = input.coefficients_layer_major[
          static_cast<std::size_t>(layer) * node_count + root];
      const float weight = input.root_scales[layer] *
                           static_cast<float>(q);
      const float* atom = layer == 0U
          ? codebooks_->root0().column(input.codes.cid)
          : codebooks_->root_small(layer).column(
                RootSmallCode(input.codes, root, layer));
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        destination[axis] += weight * atom[axis];
      }
    }
  }

  for (std::uint32_t linked = 0U;
       linked < input.codes.linked_real_count; ++linked) {
    const std::uint32_t real = input.codes.real_root_count + linked;
    const std::uint32_t node = input.codes.virtual_count + real;
    const std::uint32_t parent = input.parent_local_1based[real] - 1U;
    float* destination = reconstructions.data() +
        static_cast<std::size_t>(node) * dimension;
    const float* parent_values = reconstructions.data() +
        static_cast<std::size_t>(parent) * dimension;
    std::copy(parent_values, parent_values + dimension, destination);
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      const std::int32_t q = input.coefficients_layer_major[
          static_cast<std::size_t>(layer) * node_count + node];
      const float weight = input.linkage_scales[layer] *
                           static_cast<float>(q);
      const std::uint8_t code = layer == 0U
          ? input.codes.edge_code0[real]
          : input.codes.real_codes_small[
                static_cast<std::size_t>(real) * (m - 1) + layer - 1U];
      const float* atom = codebooks_->edge(layer).column(code);
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        destination[axis] += weight * atom[axis];
      }
    }
  }

  std::vector<float> output(real_count);
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    const float* reconstruction = reconstructions.data() +
        static_cast<std::size_t>(input.codes.virtual_count + real) *
            dimension;
    float norm2 = 0.0F;
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      norm2 += reconstruction[axis] * reconstruction[axis];
    }
    output[real] = norm2;
  }
  return output;
}

template <int M>
StatusOr<std::vector<float>> FloatReconstructionNormComputer<M>::Compute(
    const FloatReconstructionNormInput<M>& input) const {
  std::uint32_t real_count = 0U;
  std::uint32_t node_count = 0U;
  if (const Status status = ValidateFloatInput(
          *codebooks_, input, real_count, node_count);
      !status.ok()) {
    return status;
  }
  const std::uint32_t m = codebooks_->shape().m;
  const std::uint32_t dimension = codebooks_->shape().dimension;
  const std::uint32_t root_count =
      input.codes.virtual_count + input.codes.real_root_count;
  std::vector<float> reconstructions(
      static_cast<std::size_t>(node_count) * dimension, 0.0F);

  for (std::uint32_t root = 0U; root < root_count; ++root) {
    float* destination = reconstructions.data() +
        static_cast<std::size_t>(root) * dimension;
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      const float weight = input.coefficients_layer_major[
          static_cast<std::size_t>(layer) * node_count + root];
      const float* atom = layer == 0U
          ? codebooks_->root0().column(input.codes.cid)
          : codebooks_->root_small(layer).column(
                RootSmallCode(input.codes, root, layer));
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        destination[axis] += weight * atom[axis];
      }
    }
  }
  for (std::uint32_t linked = 0U;
       linked < input.codes.linked_real_count; ++linked) {
    const std::uint32_t real = input.codes.real_root_count + linked;
    const std::uint32_t node = input.codes.virtual_count + real;
    const std::uint32_t parent = input.parent_local_1based[real] - 1U;
    float* destination = reconstructions.data() +
        static_cast<std::size_t>(node) * dimension;
    const float* parent_values = reconstructions.data() +
        static_cast<std::size_t>(parent) * dimension;
    std::copy(parent_values, parent_values + dimension, destination);
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      const float weight = input.coefficients_layer_major[
          static_cast<std::size_t>(layer) * node_count + node];
      const std::uint8_t code = layer == 0U
          ? input.codes.edge_code0[real]
          : input.codes.real_codes_small[
                static_cast<std::size_t>(real) * (m - 1U) + layer - 1U];
      const float* atom = codebooks_->edge(layer).column(code);
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        destination[axis] += weight * atom[axis];
      }
    }
  }

  std::vector<float> output(real_count);
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    const float* reconstruction = reconstructions.data() +
        static_cast<std::size_t>(input.codes.virtual_count + real) *
            dimension;
    float norm2 = 0.0F;
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      norm2 += reconstruction[axis] * reconstruction[axis];
    }
    output[real] = norm2;
  }
  return output;
}

namespace {

float NonNegativeSquaredNorm(float value) noexcept {
  // The selected-Gram recurrence is the DEV production path.  Its FP32
  // parent/local/cross cancellation can land just below zero even though the
  // represented quantity is a squared norm.  Preserve every valid positive
  // result bit-for-bit and restore only the mathematical lower bound; NaN and
  // infinities deliberately remain visible to the publication validator.
  return value < 0.0F ? 0.0F : value;
}

template <int M, typename Coefficient>
StatusOr<std::vector<float>> ComputeSelectedAtomNorms(
    const SelectedAtomGramView<M>& grams,
    const CanonicalClusterCodeView<M>& codes,
    ArrayView<const std::uint32_t> parents,
    Coefficient coefficient) {
  const auto m = grams.m();
  const std::uint32_t real_count =
      codes.real_root_count + codes.linked_real_count;
  const std::uint32_t root_count =
      codes.virtual_count + codes.real_root_count;
  const std::uint32_t node_count = root_count + codes.linked_real_count;
  if (grams.m() != codes.m || grams.root_count() != root_count ||
      grams.node_count() != node_count ||
      parents.size() != real_count) {
    return Status::InvalidArgument(
        "selected-atom norm inputs disagree on canonical extents");
  }
  std::vector<float> virtual_norms(codes.virtual_count, 0.0F);
  std::vector<float> real_norms(real_count, 0.0F);
  const auto local_norm = [&](std::uint32_t node, bool root) {
    float value = 0.0F;
    for (std::uint32_t layer = 0U; layer < m; ++layer) {
      const float left = coefficient(node, layer, root);
      const float diagonal = root
          ? grams.roots().At(node, layer, layer)
          : grams.linkage().At(node, layer, layer);
      value += left * left * diagonal;
      for (std::uint32_t other = layer + 1U; other < m; ++other) {
        const float gram = root
            ? grams.roots().At(node, layer, other)
            : grams.linkage().At(node, layer, other);
        value += 2.0F * left * coefficient(node, other, root) *
            gram;
      }
    }
    return value;
  };
  for (std::uint32_t root = 0U; root < root_count; ++root) {
    const float norm = NonNegativeSquaredNorm(local_norm(root, true));
    if (root < codes.virtual_count) {
      virtual_norms[root] = norm;
    } else {
      real_norms[root - codes.virtual_count] = norm;
    }
  }
  for (std::uint32_t linked = 0U;
       linked < codes.linked_real_count; ++linked) {
    const std::uint32_t real = codes.real_root_count + linked;
    const std::uint32_t node = root_count + linked;
    const std::uint32_t parent_1based = parents[real];
    if (parent_1based == 0U || parent_1based > node) {
      return Status::InvalidArgument(
          "selected-atom norm parent is not causally earlier");
    }
    const std::uint32_t parent = parent_1based - 1U;
    float cross = 0.0F;
    std::uint32_t ancestor = parent;
    for (;;) {
      if (ancestor < root_count) {
        for (std::uint32_t layer = 0U; layer < m; ++layer) {
          float dot = 0.0F;
          for (std::uint32_t other = 0U; other < m; ++other) {
            dot += coefficient(node, other, false) *
                grams.RootLinkageAt(
                    ancestor, layer, node, other);
          }
          cross += coefficient(ancestor, layer, true) * dot;
        }
        break;
      }
      for (std::uint32_t layer = 0U; layer < m; ++layer) {
        const float left = coefficient(ancestor, layer, false);
        for (std::uint32_t other = 0U; other < m; ++other) {
          cross += left * coefficient(node, other, false) *
              grams.LinkageCrossAt(
                  ancestor, layer, node, other);
        }
      }
      const std::uint32_t ancestor_real =
          ancestor - codes.virtual_count;
      if (ancestor_real >= real || parents[ancestor_real] == 0U) {
        return Status::InvalidArgument(
            "selected-atom norm ancestor chain is invalid");
      }
      ancestor = parents[ancestor_real] - 1U;
    }
    const float parent_norm = parent < codes.virtual_count
        ? virtual_norms[parent]
        : real_norms[parent - codes.virtual_count];
    real_norms[real] = NonNegativeSquaredNorm(
        parent_norm + local_norm(node, false) + 2.0F * cross);
  }
  return real_norms;
}

}  // namespace

template <int M>
StatusOr<std::vector<float>>
SelectedAtomReconstructionNormComputer<M>::Compute(
    const SelectedAtomGramView<M>& grams,
    const ReconstructionNormInput<M>& input) const {
  const auto m = grams.m();
  const std::uint32_t node_count = grams.node_count();
  if (input.coefficients_layer_major.size() !=
          static_cast<std::size_t>(node_count) * m ||
      input.root_scales.size() != m || input.linkage_scales.size() != m) {
    return Status::InvalidArgument(
        "selected-atom quantized norm coefficient extent is invalid");
  }
  const auto coefficient = [&](std::uint32_t node, std::uint32_t layer,
                               bool root) {
    const float scale = root ? input.root_scales[layer]
                             : input.linkage_scales[layer];
    return scale * static_cast<float>(input.coefficients_layer_major[
        static_cast<std::size_t>(layer) * node_count + node]);
  };
  return ComputeSelectedAtomNorms<M>(
      grams, input.codes, input.parent_local_1based, coefficient);
}

template <int M>
StatusOr<std::vector<float>>
SelectedAtomReconstructionNormComputer<M>::ComputeFloat(
    const SelectedAtomGramView<M>& grams,
    const FloatReconstructionNormInput<M>& input) const {
  return ComputeFloatImpl(grams, input, false);
}

template <int M>
StatusOr<std::vector<float>>
SelectedAtomReconstructionNormComputer<M>::ComputeFloatValidated(
    const SelectedAtomGramView<M>& grams,
    const FloatReconstructionNormInput<M>& input) const {
  return ComputeFloatImpl(grams, input, true);
}

template <int M>
StatusOr<std::vector<float>>
SelectedAtomReconstructionNormComputer<M>::ComputeFloatImpl(
    const SelectedAtomGramView<M>& grams,
    const FloatReconstructionNormInput<M>& input,
    bool validated_coefficients) const {
  const auto m = grams.m();
  const std::uint32_t node_count = grams.node_count();
  if (input.coefficients_layer_major.size() !=
      static_cast<std::size_t>(node_count) * m) {
    return Status::InvalidArgument(
        "selected-atom float norm coefficient extent is invalid");
  }
  if (!validated_coefficients) {
    for (const float value : input.coefficients_layer_major) {
      if (!std::isfinite(value)) {
        return Status::InvalidArgument(
            "selected-atom float norm coefficient is not finite");
      }
    }
  }
#ifndef NDEBUG
  if (validated_coefficients) {
    for (const float value : input.coefficients_layer_major) {
      assert(std::isfinite(value));
    }
  }
#endif
  const auto coefficient = [&](std::uint32_t node, std::uint32_t layer,
                               bool) {
    return input.coefficients_layer_major[
        static_cast<std::size_t>(layer) * node_count + node];
  };
  return ComputeSelectedAtomNorms<M>(
      grams, input.codes, input.parent_local_1based, coefficient);
}

template class ReconstructionNormComputer<0>;
template class ReconstructionNormComputer<5>;
template class ReconstructionNormComputer<12>;
template class ReconstructionNormComputer<6>;
template class ReconstructionNormComputer<10>;
template class FloatReconstructionNormComputer<0>;
template class FloatReconstructionNormComputer<5>;
template class FloatReconstructionNormComputer<12>;
template class FloatReconstructionNormComputer<6>;
template class FloatReconstructionNormComputer<10>;
template class SelectedAtomReconstructionNormComputer<0>;
template class SelectedAtomReconstructionNormComputer<5>;
template class SelectedAtomReconstructionNormComputer<12>;
template class SelectedAtomReconstructionNormComputer<6>;
template class SelectedAtomReconstructionNormComputer<10>;

namespace {
#if !STLQPP_HAS_CBLAS_GRAM_NORM
class ScalarGramGemm final : public F32GemmBackend {
 public:
  Status MultiplyTransposedLeft(const TransposedLeftGemmF32& op) const override {
    for (std::uint32_t col = 0; col < op.query_count; ++col)
      for (std::uint32_t row = 0; row < op.left.columns; ++row) {
        float dot = 0.0F;
        for (std::uint32_t axis = 0; axis < op.left.rows; ++axis)
          dot += op.left.column(row)[axis] *
                 op.right[static_cast<std::size_t>(col) * op.right_leading_dimension + axis];
        op.output[static_cast<std::size_t>(col) * op.output_leading_dimension + row] = dot;
      }
    return Status::Ok();
  }
};
#endif
CanonicalClusterCodeView<0> GramCodes(std::uint32_t m,
                                     const RuntimeCanonicalClusterCodeView& c) {
  return {m, c.cid, c.virtual_count, c.real_root_count, c.linked_real_count,
          c.virtual_codes_small, c.real_codes_small, c.edge_code0};
}
}  // namespace

StatusOr<RuntimeGramReconstructionNormComputer>
RuntimeGramReconstructionNormComputer::Create(RuntimeProductCodebookView model) {
#if STLQPP_HAS_CBLAS_GRAM_NORM
  const CblasF32Gemm gemm(1U);
#else
  const ScalarGramGemm gemm;
#endif
  auto grams = SelectedAtomGramTables<0>::Create(model, gemm);
  if (!grams.ok()) return grams.status();
  return RuntimeGramReconstructionNormComputer(std::move(grams).value());
}

StatusOr<std::vector<float>> RuntimeGramReconstructionNormComputer::Compute(
    const RuntimeReconstructionNormInput& input) const {
  if (const auto s = ValidateRuntimeNormInput(grams_.norm_model_view().shape, input); !s.ok())
    return s;
  auto cross = grams_.BuildRootCrossBlock(input.codes.cid, 1U);
  if (!cross.ok()) return cross.status();
  const auto codes = GramCodes(grams_.norm_model_view().shape.m, input.codes);
  auto selected = grams_.SelectCluster(codes, cross.value());
  if (!selected.ok()) return selected.status();
  return SelectedAtomReconstructionNormComputer<0>{}.Compute(selected.value(),
      {codes, input.parent_local_1based, input.coefficients_layer_major,
       input.root_scales, input.linkage_scales});
}

StatusOr<std::vector<float>> RuntimeGramReconstructionNormComputer::ComputeFloat(
    const RuntimeFloatReconstructionNormInput& input) const {
  if (const auto s = ValidateRuntimeNormInput(grams_.norm_model_view().shape, input); !s.ok())
    return s;
  auto cross = grams_.BuildRootCrossBlock(input.codes.cid, 1U);
  if (!cross.ok()) return cross.status();
  const auto codes = GramCodes(grams_.norm_model_view().shape.m, input.codes);
  auto selected = grams_.SelectCluster(codes, cross.value());
  if (!selected.ok()) return selected.status();
  return SelectedAtomReconstructionNormComputer<0>{}.ComputeFloat(selected.value(),
      {codes, input.parent_local_1based, input.coefficients_layer_major});
}

}  // namespace stlqpp
