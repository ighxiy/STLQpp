#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace stlqpp {
namespace {

float SquaredNorm(const float* values, std::uint32_t count) noexcept {
  float result = 0.0F;
  for (std::uint32_t index = 0U; index < count; ++index) {
    result += values[index] * values[index];
  }
  return result;
}

}  // namespace

StatusOr<RuntimeLinkageCodebookAlgebra> RuntimeLinkageCodebookAlgebra::Create(
    ModelShape shape, ColumnMajorMatrixF32View atoms,
    const F32GemmBackend& gemm) {
  if (const auto status = shape.Validate(); !status.ok()) return status;
  if (atoms.values == nullptr || atoms.rows != shape.dimension ||
      atoms.columns != shape.m * 256U)
    return Status::InvalidArgument("runtime linkage atom shape mismatch");
  for (std::size_t i = 0;
       i < static_cast<std::size_t>(atoms.rows) * atoms.columns; ++i) {
    if (!std::isfinite(atoms.values[i]))
      return Status::InvalidArgument("runtime linkage atom is not finite");
  }
  RuntimeLinkageCodebookAlgebra output;
  if (const auto status = output.Initialize(shape, atoms, gemm); !status.ok())
    return status;
  return output;
}

Status RuntimeLinkageCodebookAlgebra::Initialize(
    ModelShape shape, ColumnMajorMatrixF32View source,
    const F32GemmBackend& gemm) {
  shape_ = shape;
  const std::size_t value_count =
      static_cast<std::size_t>(source.rows) * source.columns;
  atoms_.assign(source.values, source.values + value_count);

  squared_norms_.resize(atom_count());
  inverse_norms_.resize(atom_count());
  for (std::uint32_t atom = 0U; atom < atom_count(); ++atom) {
    const float norm2 = SquaredNorm(source.column(atom), source.rows);
    squared_norms_[atom] = norm2;
    inverse_norms_[atom] = norm2 > 0.0F ? 1.0F / std::sqrt(norm2) : 0.0F;
  }

  gram_.resize(static_cast<std::size_t>(atom_count()) * atom_count());
  const Status status = gemm.MultiplyTransposedLeft({
      .left = {atoms_.data(), source.rows, atom_count()},
      .right = atoms_.data(),
      .query_count = atom_count(),
      .right_leading_dimension = source.rows,
      .output = gram_.data(),
      .output_leading_dimension = atom_count(),
  });
  if (!status.ok()) {
    return status;
  }
  return Status::Ok();
}

template <int M>
StatusOr<LinkageCodebookAlgebra<M>> LinkageCodebookAlgebra<M>::Create(
    const ProductCodebooks<M>& codebooks, const F32GemmBackend& gemm) {
  assert(M == 0 || codebooks.shape().m == static_cast<std::uint32_t>(M));
  LinkageCodebookAlgebra<M> output;
  if (const auto status =
          output.Initialize(codebooks.shape(), codebooks.edge_flat(), gemm);
      !status.ok())
    return status;
  return output;
}

template class LinkageCodebookAlgebra<0>;
template class LinkageCodebookAlgebra<5>;
template class LinkageCodebookAlgebra<12>;
template class LinkageCodebookAlgebra<6>;
template class LinkageCodebookAlgebra<10>;

}  // namespace stlqpp
