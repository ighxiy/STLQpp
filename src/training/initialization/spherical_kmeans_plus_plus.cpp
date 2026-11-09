#include "stlqpp/training/initialization/spherical_kmeans_plus_plus.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace stlqpp {
namespace {

constexpr float kNormalizationEpsilon = 1.0e-6F;
constexpr std::uint32_t kDotUpdateColumns = 256U;

struct DotUpdate final {
  ColumnMajorMatrixF32View candidates;
  const float* center = nullptr;
  float* best_dots = nullptr;
  float* distances = nullptr;
};

void UpdateDots(void* opaque,
                std::uint32_t,
                std::uint32_t tile) noexcept {
  auto& update = *static_cast<DotUpdate*>(opaque);
  const std::uint32_t first = tile * kDotUpdateColumns;
  const std::uint32_t count = std::min(
      kDotUpdateColumns, update.candidates.columns - first);
  // GCC's OpenMP SIMD lowering needs this invariant pointer outside the
  // reduction; accessing update.center[axis] there leaves a scalar dot loop.
  const float* center = update.center;
  const int dimension = static_cast<int>(update.candidates.rows);
  // Independent columns share a scheduling callback, not an FP reduction.
  // Keep DEV's per-column SIMD dot and the sequential sampling CDF unchanged.
  for (std::uint32_t column = first; column < first + count; ++column) {
    const float* candidate = update.candidates.column(column);
    float dot = 0.0F;
#pragma omp simd reduction(+:dot)
    for (int axis = 0; axis < dimension; ++axis) {
      dot += candidate[axis] * center[axis];
    }
    if (dot > update.best_dots[column]) {
      update.best_dots[column] = dot;
      update.distances[column] = std::max(0.0F, 2.0F - 2.0F * dot);
    }
  }
}

struct CenterNormalization final {
  float* centers = nullptr;
  std::uint32_t dimension = 0U;
};

void NormalizeCenter(void* opaque,
                     std::uint32_t,
                     std::uint32_t column) noexcept {
  auto& normalization = *static_cast<CenterNormalization*>(opaque);
  float* center = normalization.centers +
      static_cast<std::size_t>(column) * normalization.dimension;
  float squared_norm = 0.0F;
#pragma omp simd reduction(+:squared_norm)
  for (std::uint32_t axis = 0U; axis < normalization.dimension; ++axis) {
    squared_norm += center[axis] * center[axis];
  }
  const float norm = std::sqrt(
      std::max(squared_norm, kNormalizationEpsilon));
  if (norm > kNormalizationEpsilon) {
    const float inverse = 1.0F / norm;
#pragma omp simd
    for (std::uint32_t axis = 0U; axis < normalization.dimension; ++axis) {
      center[axis] *= inverse;
    }
  }
}

}  // namespace

StatusOr<std::vector<float>> ReduceSphericalKmeansPlusPlus(
    ColumnMajorMatrixF32View candidates,
    std::uint32_t center_count,
    std::mt19937& random,
    const ParallelForExecutor& parallel) {
  if (candidates.values == nullptr || candidates.rows == 0U ||
      candidates.columns == 0U || center_count == 0U) {
    return Status::InvalidArgument(
        "spherical k-means++ candidates or center count are invalid");
  }
  const std::size_t values = static_cast<std::size_t>(candidates.rows) *
      candidates.columns;
  for (std::size_t index = 0U; index < values; ++index) {
    if (!std::isfinite(candidates.values[index])) {
      return Status::InvalidArgument(
          "spherical k-means++ candidates are non-finite");
    }
  }

  std::vector<float> centers(
      static_cast<std::size_t>(candidates.rows) * center_count);
  std::vector<float> best_dots(
      candidates.columns, -std::numeric_limits<float>::infinity());
  std::vector<float> distances(candidates.columns, 0.0F);
  std::uniform_int_distribution<int> uniform_index(
      0, static_cast<int>(candidates.columns - 1U));

  for (std::uint32_t center_index = 0U; center_index < center_count;
       ++center_index) {
    std::uint32_t selected =
        static_cast<std::uint32_t>(uniform_index(random));
    if (center_index != 0U) {
      double total = 0.0;
      for (const float distance : distances) {
        total += static_cast<double>(distance);
      }
      if (total > static_cast<double>(kNormalizationEpsilon)) {
        std::uniform_real_distribution<double> threshold(0.0, total);
        const double target = threshold(random);
        double cumulative = 0.0;
        for (std::uint32_t candidate = 0U;
             candidate < candidates.columns; ++candidate) {
          cumulative += static_cast<double>(distances[candidate]);
          if (cumulative >= target) {
            selected = candidate;
            break;
          }
        }
      }
    }
    float* destination = centers.data() +
        static_cast<std::size_t>(center_index) * candidates.rows;
    std::copy_n(candidates.column(selected), candidates.rows, destination);

    DotUpdate update{
        .candidates = candidates,
        .center = destination,
        .best_dots = best_dots.data(),
        .distances = distances.data(),
    };
    if (center_index + 1U < center_count) {
      parallel.Run(1U + (candidates.columns - 1U) / kDotUpdateColumns,
                   &update, UpdateDots);
    }
  }
  CenterNormalization normalization{
      .centers = centers.data(),
      .dimension = candidates.rows,
  };
  parallel.Run(center_count, &normalization, NormalizeCenter);
  return centers;
}

}  // namespace stlqpp
