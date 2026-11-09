#include "stlqpp/training/linkage/virtual/virtual_root_encoder.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

std::uint64_t VirtualSampleId(std::uint32_t cid,
                              std::uint32_t ordinal) noexcept {
  return (static_cast<std::uint64_t>(cid) << 32U) | ordinal;
}

template <int M>
void Reconstruct(const ProductCodebooks<M>& codebooks,
                 const BasicEncoding<M>& encoding,
                 std::uint32_t row,
                 float* output) noexcept {
  const std::uint32_t dimension = codebooks.shape().dimension;
  const float* root = codebooks.root0().column(encoding.root_code(row));
  const float root_coefficient = encoding.coefficient(row, 0U);
  for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
    output[axis] = root_coefficient * root[axis];
  }
  for (std::uint32_t layer = 1U;
       layer < codebooks.shape().m; ++layer) {
    const float* atom = codebooks.root_small(layer).column(
        encoding.small_code(row, layer));
    const float coefficient = encoding.coefficient(row, layer);
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      output[axis] += coefficient * atom[axis];
    }
  }
}

}  // namespace

template <int M>
StatusOr<EncodedVirtualRoots> EncodeVirtualRoots(
    std::uint32_t cid,
    ColumnMajorMatrixF32View unrotated_real,
    ColumnMajorMatrixF32View rotated_real,
    const UmapVirtualRootPlan& plan,
    const ProductCodebooks<M>& codebooks,
    const VirtualRootSupportEncoder<M>& encoder,
    std::uint32_t seed) {
  const ModelShape& shape = codebooks.shape();
  if (cid >= shape.nlist || unrotated_real.rows != shape.dimension ||
      rotated_real.rows != shape.dimension ||
      unrotated_real.columns != plan.real_count() ||
      rotated_real.columns != plan.real_count() ||
      unrotated_real.values == nullptr || rotated_real.values == nullptr ||
      plan.root_count() == 0U) {
    return Status::InvalidArgument(
        "virtual-root encoder input does not match model or plan");
  }
  auto unrotated = plan.Materialize(unrotated_real);
  if (!unrotated.ok()) {
    return unrotated.status();
  }
  auto rotated = plan.Materialize(rotated_real);
  if (!rotated.ok()) {
    return rotated.status();
  }

  EncodedVirtualRoots output;
  output.dimension_ = shape.dimension;
  output.unrotated_exact_ = std::move(unrotated).value();
  output.rotated_exact_ = std::move(rotated).value();
  output.sample_ids_.reserve(plan.root_count());
  for (std::uint32_t root = 0U; root < plan.root_count(); ++root) {
    output.sample_ids_.push_back(VirtualSampleId(cid, root));
  }
  auto encoded = encoder.Encode(
      {output.rotated_exact_.data(), shape.dimension, plan.root_count()},
      output.sample_ids_, cid, seed);
  if (!encoded.ok()) {
    return encoded.status();
  }
  assert(encoded.value().sample_count() == plan.root_count());
  output.small_codes_.resize(
      static_cast<std::size_t>(plan.root_count()) * (shape.m - 1U));
  output.coefficients_.resize(
      static_cast<std::size_t>(plan.root_count()) * shape.m);
  output.squared_errors_.resize(plan.root_count());
  output.reconstructions_.resize(
      static_cast<std::size_t>(shape.dimension) * plan.root_count());
  for (std::uint32_t root = 0U; root < plan.root_count(); ++root) {
    assert(encoded.value().root_code(root) == cid);
    for (std::uint32_t layer = 0U;
         layer < codebooks.shape().m; ++layer) {
      const float coefficient = encoded.value().coefficient(root, layer);
      output.coefficients_[static_cast<std::size_t>(root) * shape.m + layer] =
          coefficient;
      if (layer != 0U) {
        output.small_codes_[
            static_cast<std::size_t>(root) * (shape.m - 1U) + layer - 1U] =
            encoded.value().small_code(root, layer);
      }
    }
    output.squared_errors_[root] = encoded.value().squared_error(root);
    Reconstruct(codebooks, encoded.value(), root,
                output.reconstructions_.data() +
                    static_cast<std::size_t>(root) * shape.dimension);
  }
  return output;
}

template StatusOr<EncodedVirtualRoots> EncodeVirtualRoots<0>(
    std::uint32_t,
    ColumnMajorMatrixF32View,
    ColumnMajorMatrixF32View,
    const UmapVirtualRootPlan&,
    const ProductCodebooks<0>&,
    const VirtualRootSupportEncoder<0>&,
    std::uint32_t);
template StatusOr<EncodedVirtualRoots> EncodeVirtualRoots<5>(
    std::uint32_t,
    ColumnMajorMatrixF32View,
    ColumnMajorMatrixF32View,
    const UmapVirtualRootPlan&,
    const ProductCodebooks<5>&,
    const VirtualRootSupportEncoder<5>&,
    std::uint32_t);
template StatusOr<EncodedVirtualRoots> EncodeVirtualRoots<12>(
    std::uint32_t,
    ColumnMajorMatrixF32View,
    ColumnMajorMatrixF32View,
    const UmapVirtualRootPlan&,
    const ProductCodebooks<12>&,
    const VirtualRootSupportEncoder<12>&,
    std::uint32_t);
template StatusOr<EncodedVirtualRoots> EncodeVirtualRoots<6>(
    std::uint32_t,
    ColumnMajorMatrixF32View,
    ColumnMajorMatrixF32View,
    const UmapVirtualRootPlan&,
    const ProductCodebooks<6>&,
    const VirtualRootSupportEncoder<6>&,
    std::uint32_t);
template StatusOr<EncodedVirtualRoots> EncodeVirtualRoots<10>(
    std::uint32_t,
    ColumnMajorMatrixF32View,
    ColumnMajorMatrixF32View,
    const UmapVirtualRootPlan&,
    const ProductCodebooks<10>&,
    const VirtualRootSupportEncoder<10>&,
    std::uint32_t);

}  // namespace stlqpp
