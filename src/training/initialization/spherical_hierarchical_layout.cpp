#include "stlqpp/training/initialization/spherical_hierarchical_layout.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace stlqpp {
namespace {

constexpr float kNormalizationEpsilon = 1.0e-6F;

std::uint32_t RoundDownPowerOfTwo(std::uint32_t value) noexcept {
  std::uint32_t result = 1U;
  while (result <= value / 2U) {
    result *= 2U;
  }
  return result;
}

std::uint32_t RoundUpPowerOfTwo(std::uint32_t value) noexcept {
  std::uint32_t result = 1U;
  while (result < value) {
    result *= 2U;
  }
  return result;
}

struct CoarseBuildContext final {
  ColumnMajorMatrixF32View fine;
  SphericalHierarchicalLayout layout;
  float* coarse = nullptr;
};

void BuildOneCoarse(void* opaque,
                    std::uint32_t,
                    std::uint32_t coarse_id) noexcept {
  auto& context = *static_cast<CoarseBuildContext*>(opaque);
  float* output = context.coarse +
      static_cast<std::size_t>(coarse_id) * context.fine.rows;
  const std::uint32_t group_size = context.layout.group_size(coarse_id);
  if (group_size == 0U) {
    return;
  }
  const std::uint32_t first = context.layout.group_begin(coarse_id);
  for (std::uint32_t local = 0U; local < group_size; ++local) {
    const float* center = context.fine.column(first + local);
#pragma omp simd
    for (std::uint32_t axis = 0U; axis < context.fine.rows; ++axis) {
      output[axis] += center[axis];
    }
  }
  float squared_norm = 0.0F;
#pragma omp simd reduction(+:squared_norm)
  for (std::uint32_t axis = 0U; axis < context.fine.rows; ++axis) {
    squared_norm += output[axis] * output[axis];
  }
  const float norm = std::sqrt(
      std::max(squared_norm, kNormalizationEpsilon));
  const float inverse_norm =
      norm > kNormalizationEpsilon ? 1.0F / norm : 1.0F;
#pragma omp simd
  for (std::uint32_t axis = 0U; axis < context.fine.rows; ++axis) {
    output[axis] *= inverse_norm;
  }
}

}  // namespace

std::uint32_t SphericalHierarchicalLayout::group_size(
    std::uint32_t coarse) const noexcept {
  const std::uint64_t first =
      static_cast<std::uint64_t>(coarse) * fine_capacity;
  if (first >= center_count) {
    return 0U;
  }
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(
      fine_capacity, static_cast<std::uint64_t>(center_count) - first));
}

SphericalHierarchicalLayout MakeSphericalHierarchicalLayout(
    std::uint32_t center_count,
    SphericalHierarchicalSplit split) noexcept {
  if (center_count == 0U) {
    return {};
  }
  std::uint32_t coarse_count = split.fixed_coarse_count;
  if (coarse_count == 0U) {
    coarse_count = static_cast<std::uint32_t>(std::lround(
        std::sqrt(static_cast<double>(center_count))));
    coarse_count = std::clamp(coarse_count, split.minimum_coarse_count,
                              split.maximum_coarse_count);
    if (split.round_to_power_of_two) {
      const std::uint32_t down = RoundDownPowerOfTwo(coarse_count);
      const std::uint32_t up = RoundUpPowerOfTwo(coarse_count);
      coarse_count = up - coarse_count <= coarse_count - down ? up : down;
      coarse_count = std::clamp(coarse_count, split.minimum_coarse_count,
                                split.maximum_coarse_count);
    }
  }
  coarse_count = std::max(1U, coarse_count);
  coarse_count = std::min(coarse_count, center_count);
  return {
      .center_count = center_count,
      .coarse_count = coarse_count,
      .fine_capacity =
          (center_count + coarse_count - 1U) / coarse_count,
  };
}

SphericalHierarchicalLayout MakeRegisteredSphericalHierarchicalLayout(
    std::uint32_t center_count) noexcept {
  return MakeSphericalHierarchicalLayout(center_count, {});
}

StatusOr<std::vector<float>> BuildSphericalCoarseCenters(
    ColumnMajorMatrixF32View fine,
    const SphericalHierarchicalLayout& layout,
    const ParallelForExecutor& parallel) {
  if (fine.values == nullptr || fine.rows == 0U ||
      fine.columns != layout.center_count || layout.center_count == 0U ||
      layout.coarse_count == 0U || layout.fine_capacity == 0U) {
    return Status::InvalidArgument(
        "spherical hierarchical center layout is invalid");
  }
  std::vector<float> coarse(
      static_cast<std::size_t>(fine.rows) * layout.coarse_count, 0.0F);
  CoarseBuildContext context{
      .fine = fine,
      .layout = layout,
      .coarse = coarse.data(),
  };
  parallel.Run(layout.coarse_count, &context, BuildOneCoarse);
  return coarse;
}

}  // namespace stlqpp
