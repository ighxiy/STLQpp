#include "stlqpp/training/basic/forced_root_global_greedy.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

float SquaredNorm(const float* values, std::uint32_t count) noexcept {
  float sum = 0.0F;
#pragma omp simd reduction(+ : sum)
  for (std::uint32_t index = 0U; index < count; ++index) {
    sum += values[index] * values[index];
  }
  return sum;
}

}  // namespace

std::vector<float> ComputeVirtualRootSampleNorms2(
    ColumnMajorMatrixF32View samples) {
  std::vector<float> norms(samples.columns);
  for (std::uint32_t sample = 0U; sample < samples.columns; ++sample) {
    norms[sample] = SquaredNorm(samples.column(sample), samples.rows);
  }
  return norms;
}

template <int M>
StatusOr<ForcedRootGlobalGreedyResult<M>>
InitializeForcedRootGlobalGreedy(
    ColumnMajorMatrixF32View samples,
    std::uint32_t fixed_root,
    const BasicCodebookAlgebra<M>& algebra,
    const F32GemmBackend& gemm) {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  const auto m = algebra.shape().m;
  if (samples.values == nullptr || samples.columns == 0U ||
      samples.rows != algebra.shape().dimension ||
      fixed_root >= algebra.shape().nlist) {
    return Status::InvalidArgument(
        "forced-root global-greedy input does not match its algebra");
  }

  const std::uint32_t small_count = algebra.small_count();
  const std::uint32_t full_count = algebra.shape().nlist + small_count;
  ForcedRootGlobalGreedyResult<M> output;
  output.root_dots.resize(samples.columns);
  output.small_dots.resize(static_cast<std::size_t>(small_count) * samples.columns);
  Status status = gemm.MultiplyTransposedLeft({
      .left = algebra.small_flat(),
      .right = samples.values,
      .query_count = samples.columns,
      .right_leading_dimension = samples.rows,
      .output = output.small_dots.data(),
      .output_leading_dimension = small_count,
  });
  if (!status.ok()) return status;
  status = gemm.MultiplyTransposedLeft({
      .left = {algebra.root0().column(fixed_root), samples.rows, 1U},
      .right = samples.values,
      .query_count = samples.columns,
      .right_leading_dimension = samples.rows,
      .output = output.root_dots.data(),
      .output_leading_dimension = 1U,
  });
  if (!status.ok()) return status;

  output.small_codes.resize(
      static_cast<std::size_t>(samples.columns) * (m - 1U));
  const std::uint32_t residual_count = algebra.has_dense_full_gram()
      ? full_count
      : small_count;
  const std::uint32_t small_begin = algebra.has_dense_full_gram()
      ? algebra.shape().nlist
      : 0U;
  const float root_inverse_norm = algebra.root_inverse_norm(fixed_root);
  std::vector<float> remaining(residual_count);
  for (std::uint32_t sample = 0U; sample < samples.columns; ++sample) {
    float* residual = remaining.data();
    std::copy_n(output.small_dots.data() +
                    static_cast<std::size_t>(sample) * small_count,
                small_count, residual + small_begin);
    const float root_coefficient =
        output.root_dots[sample] * root_inverse_norm * root_inverse_norm;
    if (algebra.has_dense_full_gram()) {
#pragma omp simd
      for (std::uint32_t flat = small_begin; flat < residual_count; ++flat) {
        residual[flat] -= root_coefficient *
            algebra.dense_full_dot(flat, fixed_root);
      }
    } else {
#pragma omp simd
      for (std::uint32_t flat = 0U; flat < residual_count; ++flat) {
        residual[flat] -= root_coefficient *
            algebra.root_small_dot(fixed_root, flat);
      }
    }

    std::uint32_t available = (1U << m) - 1U;
    available &= ~1U;
    for (std::uint32_t step = 1U;
         step < m; ++step) {
      std::uint32_t best_flat = 0U;
      float best_absolute = -std::numeric_limits<float>::infinity();
      float best_adjusted = 0.0F;
      for (std::uint32_t flat = small_begin;
           flat < residual_count; ++flat) {
        const std::uint32_t small_flat = flat - small_begin;
        const std::uint32_t layer = small_flat / 256U + 1U;
        if ((available & (1U << layer)) == 0U) {
          continue;
        }
        const float adjusted =
            residual[flat] * algebra.small_inverse_norm(small_flat);
        const float absolute = std::fabs(adjusted);
        if (absolute > best_absolute) {
          best_absolute = absolute;
          best_adjusted = adjusted;
          best_flat = small_flat;
        }
      }
      const std::uint32_t layer = best_flat / 256U + 1U;
      output.small_codes[
          static_cast<std::size_t>(sample) * (m - 1U) + layer - 1U] =
          static_cast<std::uint8_t>(best_flat & 255U);
      const float coefficient =
          best_adjusted * algebra.small_inverse_norm(best_flat);
      available &= ~(1U << layer);
      if (step + 1U == m) continue;
      if (algebra.has_dense_full_gram()) {
        for (std::uint32_t next_layer = 1U;
             next_layer < m; ++next_layer) {
          if ((available & (1U << next_layer)) == 0U) continue;
          const std::uint32_t begin = small_begin + (next_layer - 1U) * 256U;
#pragma omp simd
          for (std::uint32_t code = 0U; code < 256U; ++code) {
            residual[begin + code] -= coefficient *
                algebra.dense_full_dot(begin + code, small_begin + best_flat);
          }
        }
      } else {
        for (std::uint32_t next_layer = 1U;
             next_layer < m; ++next_layer) {
          if ((available & (1U << next_layer)) == 0U) continue;
          const std::uint32_t begin = (next_layer - 1U) * 256U;
#pragma omp simd
          for (std::uint32_t code = 0U; code < 256U; ++code) {
            residual[begin + code] -= coefficient *
                algebra.small_dot(begin + code, best_flat);
          }
        }
      }
    }
  }
  return output;
}

template StatusOr<ForcedRootGlobalGreedyResult<0>>
InitializeForcedRootGlobalGreedy<0>(
    ColumnMajorMatrixF32View,
    std::uint32_t,
    const BasicCodebookAlgebra<0>&,
    const F32GemmBackend&);
template StatusOr<ForcedRootGlobalGreedyResult<5>>
InitializeForcedRootGlobalGreedy<5>(
    ColumnMajorMatrixF32View,
    std::uint32_t,
    const BasicCodebookAlgebra<5>&,
    const F32GemmBackend&);
template StatusOr<ForcedRootGlobalGreedyResult<12>>
InitializeForcedRootGlobalGreedy<12>(
    ColumnMajorMatrixF32View,
    std::uint32_t,
    const BasicCodebookAlgebra<12>&,
    const F32GemmBackend&);
template StatusOr<ForcedRootGlobalGreedyResult<6>>
InitializeForcedRootGlobalGreedy<6>(
    ColumnMajorMatrixF32View,
    std::uint32_t,
    const BasicCodebookAlgebra<6>&,
    const F32GemmBackend&);
template StatusOr<ForcedRootGlobalGreedyResult<10>>
InitializeForcedRootGlobalGreedy<10>(
    ColumnMajorMatrixF32View,
    std::uint32_t,
    const BasicCodebookAlgebra<10>&,
    const F32GemmBackend&);

}  // namespace stlqpp
