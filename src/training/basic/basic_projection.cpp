#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/basic/basic_projection.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace stlqpp {
namespace {

struct NormContext final {
  ColumnMajorMatrixF32View samples;
  float* norms = nullptr;
};

void ComputeNorm(void* opaque,
                 std::uint32_t,
                 std::uint32_t sample) noexcept {
  auto& context = *static_cast<NormContext*>(opaque);
  const float* vector = context.samples.column(sample);
  float norm = 0.0F;
  for (std::uint32_t row = 0U; row < context.samples.rows; ++row) {
    norm += vector[row] * vector[row];
  }
  context.norms[sample] = norm;
}

}  // namespace

template <int M>
StatusOr<BasicProjection<M>> BasicProjection<M>::Create(
    ColumnMajorMatrixF32View samples,
    const BasicCodebookAlgebra<M>& algebra,
    const F32GemmBackend& gemm,
    const ParallelForExecutor& parallel) {
  BasicProjection<M> output;
  if (const Status status = output.Recompute(
          samples, algebra, gemm, parallel);
      !status.ok()) {
    return status;
  }
  return output;
}

Status RuntimeBasicProjection::Recompute(
    ColumnMajorMatrixF32View samples,
    const RuntimeBasicCodebookAlgebra& algebra,
    const F32GemmBackend& gemm,
    const ParallelForExecutor& parallel) {
  if (samples.values == nullptr || samples.rows != algebra.shape().dimension ||
      samples.columns == 0U) {
    return Status::InvalidArgument(
        "basic input must be a non-empty contiguous model-space matrix");
  }
#ifndef NDEBUG
  const std::size_t value_count =
      static_cast<std::size_t>(samples.rows) * samples.columns;
  for (std::size_t index = 0U; index < value_count; ++index) {
    if (!std::isfinite(samples.values[index])) {
      return Status::InvalidArgument(
          "basic input contains a non-finite value");
    }
  }
#endif

  m_ = algebra.shape().m;
  samples_ = samples;
  external_squared_norms_ = nullptr;
  external_atom_dots_ = nullptr;
  external_atom_stride_ = 0U;
  external_small_offset_ = 0U;
  squared_norms_.resize(samples.columns);
  NormContext norm_context{
      .samples = samples,
      .norms = squared_norms_.data(),
  };
  parallel.Run(samples.columns, &norm_context, ComputeNorm);

  const std::uint32_t small_count = algebra.small_count();
  small_dots_.resize(
      static_cast<std::size_t>(small_count) * samples.columns);
  const Status status = gemm.MultiplyTransposedLeft({
      .left = algebra.small_flat(),
      .right = samples.values,
      .query_count = samples.columns,
      .right_leading_dimension = samples.rows,
      .output = small_dots_.data(),
      .output_leading_dimension = small_count,
  });
  return status;
}

Status RuntimeBasicProjection::BindFullAtomProjection(
    ColumnMajorMatrixF32View samples,
    ArrayView<const float> squared_norms,
    ArrayView<const float> atom_dots,
    std::uint32_t atom_stride,
    std::uint32_t small_offset) {
  const std::uint32_t required_small =
      (m_ - 1U) * 256U;
  if (m_ < 2U || m_ > stlqpp::kMaximumModelLayers || samples.values == nullptr || samples.columns == 0U ||
      squared_norms.size() != samples.columns ||
      atom_stride < small_offset + required_small ||
      atom_dots.size() <
          static_cast<std::size_t>(samples.columns) * atom_stride) {
    return Status::InvalidArgument(
        "full atom projection does not cover the basic small layers");
  }
  samples_ = samples;
  external_squared_norms_ = squared_norms.data();
  external_atom_dots_ = atom_dots.data();
  external_atom_stride_ = atom_stride;
  external_small_offset_ = small_offset;
  return Status::Ok();
}

StatusOr<RuntimeBasicProjection> RuntimeBasicProjection::Create(
    ColumnMajorMatrixF32View samples, const RuntimeBasicCodebookAlgebra& algebra,
    const F32GemmBackend& gemm, const ParallelForExecutor& parallel) {
  RuntimeBasicProjection output;
  if (const auto status = output.Recompute(samples, algebra, gemm, parallel);
      !status.ok()) return status;
  return output;
}

template class BasicProjection<0>;
template class BasicProjection<5>;
template class BasicProjection<12>;
template class BasicProjection<6>;
template class BasicProjection<10>;

}  // namespace stlqpp
