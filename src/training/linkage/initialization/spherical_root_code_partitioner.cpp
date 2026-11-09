#include "stlqpp/training/linkage/initialization/spherical_root_code_partitioner.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace stlqpp {
namespace {

constexpr std::uint32_t kEdgeCodeCount = 256U;
constexpr std::uint32_t kMaximumIterations = 30U;
constexpr float kTolerance = 1.0e-6F;
constexpr float kEpsilon = 1.0e-6F;
constexpr std::uint32_t kSeedXor = 0xC001D00DU;

struct NormalizeContext final {
  ColumnMajorMatrixF32View input;
  float* output = nullptr;
};

void NormalizeOne(void* opaque,
                  std::uint32_t,
                  std::uint32_t column) noexcept {
  auto& context = *static_cast<NormalizeContext*>(opaque);
  const float* source = context.input.column(column);
  float* destination = context.output +
      static_cast<std::size_t>(column) * context.input.rows;
  float squared_norm = 0.0F;
#pragma omp simd reduction(+ : squared_norm)
  for (std::uint32_t axis = 0U; axis < context.input.rows; ++axis) {
    squared_norm += source[axis] * source[axis];
  }
  const float norm = std::sqrt(std::max(squared_norm, kEpsilon));
  const float inverse = norm > kEpsilon ? 1.0F / norm : 1.0F;
  for (std::uint32_t axis = 0U; axis < context.input.rows; ++axis) {
    destination[axis] = source[axis] * inverse;
  }
}

float Dot(const float* left,
          const float* right,
          std::uint32_t dimension) noexcept {
  float value = 0.0F;
#pragma omp simd reduction(+ : value)
  for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
    value += left[axis] * right[axis];
  }
  return value;
}

struct DistanceContext final {
  const float* samples = nullptr;
  const float* center = nullptr;
  std::uint32_t dimension = 0U;
  std::vector<float>* best_dot = nullptr;
  std::vector<float>* distances = nullptr;
};

void UpdateDistance(void* opaque,
                    std::uint32_t,
                    std::uint32_t sample) noexcept {
  auto& context = *static_cast<DistanceContext*>(opaque);
  const float dot = Dot(
      context.samples + static_cast<std::size_t>(sample) * context.dimension,
      context.center, context.dimension);
  if (dot > (*context.best_dot)[sample]) {
    (*context.best_dot)[sample] = dot;
    (*context.distances)[sample] = std::max(0.0F, 2.0F - 2.0F * dot);
  }
}

struct AssignmentContext final {
  const float* samples = nullptr;
  const float* centers = nullptr;
  std::uint32_t dimension = 0U;
  std::vector<std::uint8_t>* assignments = nullptr;
  std::vector<float>* dots = nullptr;
};

void AssignOne(void* opaque,
               std::uint32_t,
               std::uint32_t sample) noexcept {
  auto& context = *static_cast<AssignmentContext*>(opaque);
  const float* vector = context.samples +
      static_cast<std::size_t>(sample) * context.dimension;
  std::uint32_t best = 0U;
  float best_dot = -std::numeric_limits<float>::infinity();
  for (std::uint32_t center = 0U; center < kEdgeCodeCount; ++center) {
    const float dot = Dot(
        vector,
        context.centers +
            static_cast<std::size_t>(center) * context.dimension,
        context.dimension);
    if (dot > best_dot) {
      best_dot = dot;
      best = center;
    }
  }
  (*context.assignments)[sample] = static_cast<std::uint8_t>(best);
  (*context.dots)[sample] = best_dot;
}

struct CenterUpdateContext final {
  const float* samples = nullptr;
  const std::uint32_t* offsets = nullptr;
  const std::uint32_t* order = nullptr;
  const std::uint32_t* fallback = nullptr;
  std::uint32_t dimension = 0U;
  float* centers = nullptr;
};

void UpdateOneCenter(void* opaque,
                     std::uint32_t,
                     std::uint32_t center) noexcept {
  auto& context = *static_cast<CenterUpdateContext*>(opaque);
  float* destination = context.centers +
      static_cast<std::size_t>(center) * context.dimension;
  const std::uint32_t begin = context.offsets[center];
  const std::uint32_t end = context.offsets[center + 1U];
  if (begin == end) {
    const float* source = context.samples +
        static_cast<std::size_t>(context.fallback[center]) *
            context.dimension;
    std::copy_n(source, context.dimension, destination);
    return;
  }

  std::fill_n(destination, context.dimension, 0.0F);
  for (std::uint32_t position = begin; position < end; ++position) {
    const float* source = context.samples +
        static_cast<std::size_t>(context.order[position]) *
            context.dimension;
    for (std::uint32_t axis = 0U; axis < context.dimension; ++axis) {
      destination[axis] += source[axis];
    }
  }
  const float inverse_count = 1.0F / static_cast<float>(end - begin);
  float squared_norm = 0.0F;
#pragma omp simd reduction(+ : squared_norm)
  for (std::uint32_t axis = 0U; axis < context.dimension; ++axis) {
    destination[axis] *= inverse_count;
    squared_norm += destination[axis] * destination[axis];
  }
  const float inverse_norm =
      1.0F / std::sqrt(std::max(squared_norm, kEpsilon));
  for (std::uint32_t axis = 0U; axis < context.dimension; ++axis) {
    destination[axis] *= inverse_norm;
  }
}

Status ValidateRootBook(ColumnMajorMatrixF32View roots) {
  if (roots.values == nullptr || roots.rows == 0U || roots.columns == 0U) {
    return Status::InvalidArgument(
        "spherical root partition requires a non-empty root codebook");
  }
  const std::size_t count =
      static_cast<std::size_t>(roots.rows) * roots.columns;
  for (std::size_t index = 0U; index < count; ++index) {
    if (!std::isfinite(roots.values[index])) {
      return Status::InvalidArgument(
          "spherical root partition received a non-finite codeword");
    }
  }
  return Status::Ok();
}

}  // namespace

StatusOr<RootToEdgeCodeMap> SphericalRootCodePartitioner::Build(
    ColumnMajorMatrixF32View roots,
    std::uint32_t training_seed) const {
  return BuildImpl(roots, training_seed, nullptr);
}

StatusOr<std::vector<float>> SphericalRootCodePartitioner::BuildEdgeSeed(
    ColumnMajorMatrixF32View roots,
    std::uint32_t training_seed) const {
  std::vector<float> centers;
  auto map = BuildImpl(roots, training_seed, &centers);
  if (!map.ok()) return map.status();
  return centers;
}

StatusOr<RootToEdgeCodeMap> SphericalRootCodePartitioner::BuildImpl(
    ColumnMajorMatrixF32View roots,
    std::uint32_t training_seed,
    std::vector<float>* centers_output) const {
  if (const Status status = ValidateRootBook(roots); !status.ok()) {
    return status;
  }
  if (roots.columns == kEdgeCodeCount ||
      (roots.columns < kEdgeCodeCount && centers_output == nullptr)) {
    if (centers_output != nullptr) {
      centers_output->resize(
          static_cast<std::size_t>(roots.rows) * kEdgeCodeCount);
      for (std::uint32_t center = 0U; center < kEdgeCodeCount; ++center) {
        std::copy_n(roots.column(center), roots.rows,
                    centers_output->data() +
                        static_cast<std::size_t>(center) * roots.rows);
      }
    }
    std::vector<std::uint8_t> identity(roots.columns);
    std::iota(identity.begin(), identity.end(), std::uint8_t{0U});
    return RootToEdgeCodeMap::Create(roots.columns, identity);
  }

  std::vector<float> normalized(
      static_cast<std::size_t>(roots.rows) * roots.columns);
  NormalizeContext normalize{roots, normalized.data()};
  parallel_->Run(roots.columns, &normalize, NormalizeOne);

  std::mt19937 random(training_seed ^ kSeedXor);
  std::uniform_int_distribution<std::uint32_t> uniform_index(
      0U, roots.columns - 1U);
  std::vector<float> centers(
      static_cast<std::size_t>(roots.rows) * kEdgeCodeCount);
  const std::uint32_t first = uniform_index(random);
  std::copy_n(
      normalized.data() + static_cast<std::size_t>(first) * roots.rows,
      roots.rows, centers.data());

  std::vector<float> best_dot(
      roots.columns, -std::numeric_limits<float>::infinity());
  std::vector<float> distances(roots.columns, 0.0F);
  for (std::uint32_t center = 0U; center < kEdgeCodeCount; ++center) {
    if (center != 0U) {
      double total = 0.0;
      for (const float distance : distances) {
        total += static_cast<double>(distance);
      }
      std::uint32_t selected = uniform_index(random);
      if (total > static_cast<double>(kEpsilon)) {
        std::uniform_real_distribution<double> threshold(0.0, total);
        const double target = threshold(random);
        double cumulative = 0.0;
        for (std::uint32_t sample = 0U;
             sample < roots.columns; ++sample) {
          cumulative += static_cast<double>(distances[sample]);
          if (cumulative >= target) {
            selected = sample;
            break;
          }
        }
      }
      std::copy_n(
          normalized.data() +
              static_cast<std::size_t>(selected) * roots.rows,
          roots.rows,
          centers.data() + static_cast<std::size_t>(center) * roots.rows);
    }
    DistanceContext distance_context{
        .samples = normalized.data(),
        .center = centers.data() +
            static_cast<std::size_t>(center) * roots.rows,
        .dimension = roots.rows,
        .best_dot = &best_dot,
        .distances = &distances,
    };
    parallel_->Run(roots.columns, &distance_context, UpdateDistance);
  }

  std::vector<std::uint8_t> assignments(roots.columns, 0U);
  std::vector<float> dots(roots.columns, 0.0F);
  std::vector<std::uint32_t> offsets(kEdgeCodeCount + 1U, 0U);
  std::vector<std::uint32_t> order(roots.columns, 0U);
  std::vector<std::uint32_t> cursor(kEdgeCodeCount + 1U, 0U);
  std::vector<std::uint32_t> fallback(kEdgeCodeCount, 0U);
  float previous_cost = std::numeric_limits<float>::infinity();
  for (std::uint32_t iteration = 0U;
       iteration < kMaximumIterations; ++iteration) {
    AssignmentContext assignment_context{
        .samples = normalized.data(),
        .centers = centers.data(),
        .dimension = roots.rows,
        .assignments = &assignments,
        .dots = &dots,
    };
    parallel_->Run(roots.columns, &assignment_context, AssignOne);

    double total_cost = 0.0;
    for (const float dot : dots) {
      total_cost += static_cast<double>(1.0F - dot);
    }
    const float current_cost = static_cast<float>(
        total_cost / static_cast<double>(roots.columns));

    std::fill(offsets.begin(), offsets.end(), 0U);
    for (const std::uint8_t code : assignments) {
      ++offsets[static_cast<std::size_t>(code) + 1U];
    }
    for (std::uint32_t code = 0U; code < kEdgeCodeCount; ++code) {
      offsets[code + 1U] += offsets[code];
    }
    cursor = offsets;
    for (std::uint32_t sample = 0U;
         sample < roots.columns; ++sample) {
      const std::uint32_t code = assignments[sample];
      order[cursor[code]++] = sample;
    }
    for (std::uint32_t code = 0U; code < kEdgeCodeCount; ++code) {
      fallback[code] = uniform_index(random);
    }
    CenterUpdateContext update_context{
        .samples = normalized.data(),
        .offsets = offsets.data(),
        .order = order.data(),
        .fallback = fallback.data(),
        .dimension = roots.rows,
        .centers = centers.data(),
    };
    parallel_->Run(kEdgeCodeCount, &update_context, UpdateOneCenter);

    if (std::abs(previous_cost - current_cost) < kTolerance) {
      break;
    }
    previous_cost = current_cost;
  }
  if (centers_output != nullptr) *centers_output = std::move(centers);
  return RootToEdgeCodeMap::Create(roots.columns, assignments);
}

}  // namespace stlqpp
