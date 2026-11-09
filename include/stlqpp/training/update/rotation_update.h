#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/f32_cross_covariance.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/square_svd_f32.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

struct RoundLinkageAnalysisAccess;

class RotationCrossCovariance final {
 public:
  [[nodiscard]] static StatusOr<RotationCrossCovariance> Create(
      std::uint32_t dimension);

  [[nodiscard]] Status Accumulate(
      ColumnMajorMatrixF32View unrotated_exact,
      ColumnMajorMatrixF32View model_reconstructions,
      const F32CrossCovarianceBackend& backend);

  // DEV accumulates one FP32 covariance per static OpenMP CID shard, then
  // reduces those matrices in ascending shard order before the SVD.
  [[nodiscard]] Status MergeFrom(const RotationCrossCovariance& shard);

  [[nodiscard]] std::uint32_t dimension() const noexcept {
    return dimension_;
  }
  [[nodiscard]] std::uint64_t sample_count() const noexcept {
    return sample_count_;
  }
  [[nodiscard]] ColumnMajorMatrixF32View matrix() const noexcept {
    return {values_.data(), dimension_, dimension_};
  }

 private:
  friend struct RoundLinkageAnalysisAccess;

  [[nodiscard]] Status AccumulateValidatedReconstruction(
      ColumnMajorMatrixF32View unrotated_exact,
      ColumnMajorMatrixF32View model_reconstructions,
      const F32CrossCovarianceBackend& backend);
  [[nodiscard]] Status ValidateReconstructionPair(
      ColumnMajorMatrixF32View unrotated_exact,
      ColumnMajorMatrixF32View model_reconstructions,
      bool validated_reconstruction, float* maximum_source_abs = nullptr) const;
  [[nodiscard]] Status AccumulateImpl(
      ColumnMajorMatrixF32View unrotated_exact,
      ColumnMajorMatrixF32View model_reconstructions,
      const F32CrossCovarianceBackend& backend,
      bool validated_reconstruction);

  explicit RotationCrossCovariance(std::uint32_t dimension);

  std::uint32_t dimension_ = 0U;
  std::uint64_t sample_count_ = 0U;
  std::vector<float> values_;
};

class RotationUpdateResult final {
 public:
  [[nodiscard]] ColumnMajorMatrixF32View rotation() const noexcept {
    return {values_.data(), dimension_, dimension_};
  }

 private:
  friend class RotationUpdater;
  std::uint32_t dimension_ = 0U;
  std::vector<float> values_;
};

class RotationUpdater final {
 public:
  RotationUpdater(const SquareSvdF32& svd, const F32GemmBackend& gemm) noexcept
      : svd_(&svd), gemm_(&gemm) {}

  [[nodiscard]] StatusOr<RotationUpdateResult> Solve(
      const RotationCrossCovariance& covariance) const;

 private:
  const SquareSvdF32* svd_ = nullptr;
  const F32GemmBackend* gemm_ = nullptr;
};

}  // namespace stlqpp
