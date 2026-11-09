#include "stlqpp/training/linkage/rsd/linkage_projection.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace stlqpp {
namespace {

struct NormContext final {
  ColumnMajorMatrixF32View residuals;
  float* squared_norms = nullptr;
};

void ComputeNorm(void* opaque,
                 std::uint32_t,
                 std::uint32_t sample) noexcept {
  auto& context = *static_cast<NormContext*>(opaque);
  const float* residual = context.residuals.column(sample);
  float norm2 = 0.0F;
  for (std::uint32_t axis = 0U; axis < context.residuals.rows; ++axis) {
    norm2 += residual[axis] * residual[axis];
  }
  context.squared_norms[sample] = norm2;
}

}  // namespace

template <int M>
StatusOr<LinkageProjection<M>> LinkageProjection<M>::Create(
    ColumnMajorMatrixF32View residuals,
    const LinkageCodebookAlgebra<M>& algebra,
    const F32GemmBackend& gemm,
    const ParallelForExecutor& parallel) {
  LinkageProjection<M> output;
  if (const Status status = output.Recompute(
          residuals, algebra, gemm, parallel); !status.ok()) {
    return status;
  }
  return output;
}

template <int M>
Status LinkageProjection<M>::Recompute(
    ColumnMajorMatrixF32View residuals,
    const LinkageCodebookAlgebra<M>& algebra,
    const F32GemmBackend& gemm,
    const ParallelForExecutor& parallel) {
  if (residuals.values == nullptr || residuals.columns == 0U ||
      residuals.rows != algebra.shape().dimension) {
    return Status::InvalidArgument(
        "linkage residuals must be a non-empty contiguous model-space matrix");
  }
  const std::size_t value_count =
      static_cast<std::size_t>(residuals.rows) * residuals.columns;
  for (std::size_t index = 0U; index < value_count; ++index) {
    if (!std::isfinite(residuals.values[index])) {
      return Status::InvalidArgument(
          "linkage residuals contain a non-finite value");
    }
  }

  atom_count_ = algebra.atom_count();
  residuals_ = residuals;
  squared_norms_.resize(residuals.columns);
  NormContext norm_context{
      .residuals = residuals,
      .squared_norms = squared_norms_.data(),
  };
  parallel.Run(residuals.columns, &norm_context, ComputeNorm);

  atom_dots_.resize(
      static_cast<std::size_t>(algebra.atom_count()) * residuals.columns);
  const Status status = gemm.MultiplyTransposedLeft({
      .left = algebra.atoms(),
      .right = residuals.values,
      .query_count = residuals.columns,
      .right_leading_dimension = residuals.rows,
      .output = atom_dots_.data(),
      .output_leading_dimension = algebra.atom_count(),
  });
  return status;
}

template class LinkageProjection<0>;
template class LinkageProjection<5>;
template class LinkageProjection<12>;
template class LinkageProjection<6>;
template class LinkageProjection<10>;

}  // namespace stlqpp
