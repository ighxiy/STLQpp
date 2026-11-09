#include "stlqpp/training/initialization/spherical_rvq_residual.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace stlqpp {
namespace {

constexpr float kMinimumSquaredNorm = 1.0e-12F;

Status ValidateCodebook(ColumnMajorMatrixF32View codebook,
                        std::uint32_t dimension,
                        std::uint32_t entries) {
  if (codebook.values == nullptr || codebook.rows != dimension ||
      codebook.columns != entries) {
    return Status::InvalidArgument(
        "spherical RVQ prior codebook has the wrong shape");
  }
  const std::size_t count =
      static_cast<std::size_t>(dimension) * entries;
  for (std::size_t index = 0U; index < count; ++index) {
    if (!std::isfinite(codebook.values[index])) {
      return Status::InvalidArgument(
          "spherical RVQ prior codebook contains a non-finite value");
    }
  }
  return Status::Ok();
}

template <typename Scalar>
struct ProjectContext final {
  ColumnMajorMatrixView<Scalar> input;
  std::uint64_t first_id = 0U;
  const std::vector<ColumnMajorMatrixF32View>* codebooks = nullptr;
  const RvqAssignmentStore* assignments = nullptr;
  std::uint32_t prior_layer_count = 0U;
  float* output = nullptr;
};

template <typename Scalar>
void ProjectOne(void* opaque,
                std::uint32_t,
                std::uint32_t column) noexcept {
  auto& context = *static_cast<ProjectContext<Scalar>*>(opaque);
  const std::uint32_t dimension = context.input.rows;
  const Scalar* source = context.input.column(column);
  float* residual = context.output +
      static_cast<std::size_t>(column) * dimension;

  float squared_norm = 0.0F;
  for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
    const float value = static_cast<float>(source[axis]);
    residual[axis] = value;
    squared_norm += value * value;
  }
  if (squared_norm > kMinimumSquaredNorm) {
    const float inverse_norm = 1.0F / std::sqrt(squared_norm);
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      residual[axis] *= inverse_norm;
    }
  }

  const std::uint64_t row = context.first_id + column;
  for (std::uint32_t layer = 0U;
       layer < context.prior_layer_count; ++layer) {
    const std::uint32_t code = context.assignments->code(layer, row);
    const float* center = (*context.codebooks)[layer].column(code);
    float dot = 0.0F;
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      dot += residual[axis] * center[axis];
    }
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      residual[axis] -= dot * center[axis];
    }
  }

  squared_norm = 0.0F;
  for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
    squared_norm += residual[axis] * residual[axis];
  }
  if (squared_norm > kMinimumSquaredNorm) {
    const float inverse_norm = 1.0F / std::sqrt(squared_norm);
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      residual[axis] *= inverse_norm;
    }
  }
}

}  // namespace

template <typename Scalar>
StatusOr<SphericalRvqResidualProjector<Scalar>>
SphericalRvqResidualProjector<Scalar>::Create(
    const ModelShape& shape,
    ArrayView<const ColumnMajorMatrixF32View> codebooks,
    std::uint32_t prior_layer_count,
    const RvqAssignmentStore* assignments,
    const ParallelForExecutor& parallel) {
  if (const Status status = shape.Validate(); !status.ok()) {
    return status;
  }
  if (codebooks.size() != shape.m ||
      prior_layer_count >= shape.m ||
      (prior_layer_count != 0U && assignments == nullptr)) {
    return Status::InvalidArgument(
        "spherical RVQ residual request has an invalid product shape");
  }
  for (std::uint32_t layer = 0U; layer < prior_layer_count; ++layer) {
    const std::uint32_t entries = layer == 0U ? shape.nlist : 256U;
    if (const Status status =
            ValidateCodebook(codebooks[layer], shape.dimension, entries);
        !status.ok()) {
      return status;
    }
    if (!assignments->complete(layer)) {
      return Status::FailedPrecondition(
          "spherical RVQ residual needs a closed prior assignment layer");
    }
  }

  SphericalRvqResidualProjector<Scalar> output;
  output.shape_ = shape;
  output.codebooks_.assign(codebooks.begin(), codebooks.end());
  output.prior_layer_count_ = prior_layer_count;
  output.assignments_ = assignments;
  output.parallel_ = &parallel;
  return output;
}

template <typename Scalar>
StatusOr<SphericalRvqResidualBlock>
SphericalRvqResidualProjector<Scalar>::Project(
    const BasicInputBlock<Scalar>& input) const {
  if (input.dimension != shape_.dimension || input.count == 0U ||
      input.values.size() !=
          static_cast<std::size_t>(input.dimension) * input.count ||
      (assignments_ != nullptr &&
       (input.first_id > assignments_->row_count() ||
        input.count > assignments_->row_count() - input.first_id))) {
    return Status::InvalidArgument(
        "spherical RVQ input block disagrees with the registered source");
  }

  SphericalRvqResidualBlock output;
  output.first_id = input.first_id;
  output.dimension = input.dimension;
  output.count = input.count;
  output.values.resize(
      static_cast<std::size_t>(input.dimension) * input.count);
  ProjectContext<Scalar> context{
      .input = input.view(),
      .first_id = input.first_id,
      .codebooks = &codebooks_,
      .assignments = assignments_,
      .prior_layer_count = prior_layer_count_,
      .output = output.values.data(),
  };
  parallel_->Run(input.count, &context, ProjectOne<Scalar>);
  return output;
}

template class SphericalRvqResidualProjector<std::uint8_t>;
template class SphericalRvqResidualProjector<float>;

}  // namespace stlqpp
