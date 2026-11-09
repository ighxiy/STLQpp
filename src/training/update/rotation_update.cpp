#include "stlqpp/training/update/rotation_update.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {
namespace {

Status ValidateBlock(ColumnMajorMatrixF32View block,
                     std::uint32_t dimension,
                     std::uint32_t columns,
                     bool validated_values = false, float* maximum_abs = nullptr) {
  if (block.values == nullptr || block.rows != dimension ||
      block.columns != columns || columns == 0U) {
    return Status::InvalidArgument("rotation update block has an invalid shape");
  }
  const std::size_t count =
      static_cast<std::size_t>(block.rows) * block.columns;
  if (maximum_abs != nullptr) *maximum_abs = 0.0F;
  if (!validated_values || maximum_abs != nullptr) {
    for (std::size_t index = 0U; index < count; ++index) {
      if (!std::isfinite(block.values[index])) {
        return Status::InvalidArgument(
            "rotation update block contains non-finite data");
      }
      if (maximum_abs != nullptr) *maximum_abs = std::max(*maximum_abs, std::fabs(block.values[index]));
    }
  }
#ifndef NDEBUG
  if (validated_values) {
    for (std::size_t index = 0U; index < count; ++index) {
      assert(std::isfinite(block.values[index]));
    }
  }
#endif
  return Status::Ok();
}

}  // namespace

RotationCrossCovariance::RotationCrossCovariance(std::uint32_t dimension)
    : dimension_(dimension),
      values_(static_cast<std::size_t>(dimension) * dimension, 0.0F) {}

StatusOr<RotationCrossCovariance> RotationCrossCovariance::Create(
    std::uint32_t dimension) {
  if (dimension == 0U) {
    return Status::InvalidArgument(
        "rotation cross-covariance dimension must be positive");
  }
  return RotationCrossCovariance(dimension);
}

Status RotationCrossCovariance::Accumulate(
    ColumnMajorMatrixF32View unrotated_exact,
    ColumnMajorMatrixF32View model_reconstructions,
    const F32CrossCovarianceBackend& backend) {
  return AccumulateImpl(
      unrotated_exact, model_reconstructions, backend, false);
}

Status RotationCrossCovariance::AccumulateValidatedReconstruction(
    ColumnMajorMatrixF32View unrotated_exact,
    ColumnMajorMatrixF32View model_reconstructions,
    const F32CrossCovarianceBackend& backend) {
  return AccumulateImpl(
      unrotated_exact, model_reconstructions, backend, true);
}

Status RotationCrossCovariance::ValidateReconstructionPair(
    ColumnMajorMatrixF32View unrotated_exact, ColumnMajorMatrixF32View model_reconstructions,
    bool validated_reconstruction, float* maximum_source_abs) const {
  if (const Status status = ValidateBlock(
          unrotated_exact, dimension_, model_reconstructions.columns, false, maximum_source_abs);
      !status.ok()) {
    return status;
  }
  if (const Status status = ValidateBlock(
          model_reconstructions, dimension_, unrotated_exact.columns,
          validated_reconstruction);
      !status.ok()) {
    return status;
  }
  return Status::Ok();
}

Status RotationCrossCovariance::AccumulateImpl(
    ColumnMajorMatrixF32View unrotated_exact,
    ColumnMajorMatrixF32View model_reconstructions,
    const F32CrossCovarianceBackend& backend,
    bool validated_reconstruction) {
  if (const auto status = ValidateReconstructionPair(unrotated_exact,
          model_reconstructions, validated_reconstruction); !status.ok()) return status;
  if (const Status status = backend.Accumulate({
          .left = model_reconstructions,
          .right = unrotated_exact,
          .output = values_.data(),
          .output_leading_dimension = dimension_,
      });
      !status.ok()) {
    return status;
  }
  sample_count_ += unrotated_exact.columns;
  return Status::Ok();
}

Status RotationCrossCovariance::MergeFrom(
    const RotationCrossCovariance& shard) {
  if (shard.dimension_ != dimension_ ||
      shard.values_.size() != values_.size()) {
    return Status::InvalidArgument(
        "rotation covariance shard has an incompatible shape");
  }
  for (std::size_t index = 0U; index < values_.size(); ++index) {
    values_[index] += shard.values_[index];
  }
  sample_count_ += shard.sample_count_;
  return Status::Ok();
}

StatusOr<RotationUpdateResult> RotationUpdater::Solve(
    const RotationCrossCovariance& covariance) const {
  if (covariance.sample_count() == 0U) {
    return Status::FailedPrecondition(
        "rotation update requires at least one training vector");
  }
  const std::uint32_t dimension = covariance.dimension();
  const std::size_t matrix_size =
      static_cast<std::size_t>(dimension) * dimension;
  std::vector<float> singular_values(dimension);
  std::vector<float> left(matrix_size);
  std::vector<float> right_transposed(matrix_size);
  if (const Status status = svd_->Solve({
          .matrix = covariance.matrix().values,
          .order = dimension,
          .singular_values = singular_values.data(),
          .left_vectors = left.data(),
          .right_vectors_transposed = right_transposed.data(),
          .allow_phase_parallelism = true,
      });
      !status.ok()) {
    return status;
  }

  RotationUpdateResult result;
  result.dimension_ = dimension;
  result.values_.assign(matrix_size, 0.0F);
  if (const auto status = gemm_->MultiplyLeft({
          .left = {left.data(), dimension, dimension},
          .right = right_transposed.data(), .query_count = dimension,
          .right_leading_dimension = dimension, .output = result.values_.data(),
          .output_leading_dimension = dimension}); !status.ok()) return status;
  return result;
}

}  // namespace stlqpp
