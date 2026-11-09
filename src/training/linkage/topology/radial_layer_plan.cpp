#include "stlqpp/training/linkage/topology/radial_layer_plan.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace stlqpp {

StatusOr<RadialLayerPlan> RadialLayerPlan::Build(
    ColumnMajorMatrixF32View exact_real,
    ArrayView<const float> native_center,
    RadialLayerPolicy policy) {
  if (exact_real.values == nullptr || exact_real.rows == 0U ||
      exact_real.columns == 0U ||
      native_center.size() != exact_real.rows ||
      policy.layer_count == 0U || !std::isfinite(policy.root_fraction) ||
      policy.root_fraction <= 0.0 || policy.root_fraction > 1.0) {
    return Status::InvalidArgument(
        "radial layer plan has an invalid matrix or policy");
  }
  const int dimension = static_cast<int>(exact_real.rows);
  const int count = static_cast<int>(exact_real.columns);
  float center_norm = 0.0F;
#pragma omp simd reduction(+:center_norm)
  for (int axis = 0; axis < dimension; ++axis) {
    center_norm += native_center[static_cast<std::size_t>(axis)] *
        native_center[static_cast<std::size_t>(axis)];
  }
  if (!(center_norm > 0.0F)) {
    return Status::InvalidArgument(
        "radial layer plan requires a nonzero native center");
  }
  center_norm = std::sqrt(std::max(center_norm, 1.0e-12F));

  std::vector<float> cosine(static_cast<std::size_t>(count), 0.0F);
  for (int row = 0; row < count; ++row) {
    const float* vector = exact_real.column(
        static_cast<std::uint32_t>(row));
    float vector_norm2 = 0.0F;
    float dot = 0.0F;
#pragma omp simd reduction(+:vector_norm2,dot)
    for (int axis = 0; axis < dimension; ++axis) {
      const float value = vector[axis];
      vector_norm2 += value * value;
      dot += value * native_center[static_cast<std::size_t>(axis)];
    }
    const float denominator = std::max(
        center_norm * std::sqrt(std::max(vector_norm2, 1.0e-12F)),
        1.0e-12F);
    cosine[static_cast<std::size_t>(row)] = dot / denominator;
  }
  std::vector<int> order(static_cast<std::size_t>(count), 0);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int left, int right) {
              return cosine[static_cast<std::size_t>(left)] >
                  cosine[static_cast<std::size_t>(right)];
            });

  RadialLayerPlan output;
  output.local_ids_.resize(policy.layer_count);
  const float root_fraction = static_cast<float>(policy.root_fraction);
  int root_count = static_cast<int>(std::floor(
      root_fraction * static_cast<float>(count)));
  root_count = std::max(1, std::min(root_count, count));
  output.local_ids_[0U].reserve(static_cast<std::size_t>(root_count));
  for (int index = 0; index < root_count; ++index) {
    output.local_ids_[0U].push_back(static_cast<std::uint32_t>(
        order[static_cast<std::size_t>(index)]));
  }
  const int remaining = count - root_count;
  if (policy.layer_count == 1U) {
    output.local_ids_[0U].clear();
    output.local_ids_[0U].reserve(static_cast<std::size_t>(count));
    for (const int local_id : order) {
      output.local_ids_[0U].push_back(
          static_cast<std::uint32_t>(local_id));
    }
  } else {
    const int trailing_layers = static_cast<int>(policy.layer_count) - 1;
    const int base = remaining / trailing_layers;
    const int remainder = remaining % trailing_layers;
    int begin = root_count;
    for (int layer = 1; layer <= trailing_layers; ++layer) {
      const int layer_count = base + (layer <= remainder ? 1 : 0);
      auto& destination = output.local_ids_[static_cast<std::size_t>(layer)];
      destination.reserve(static_cast<std::size_t>(layer_count));
      for (int index = begin; index < begin + layer_count; ++index) {
        destination.push_back(static_cast<std::uint32_t>(
            order[static_cast<std::size_t>(index)]));
      }
      begin += layer_count;
    }
  }

  return output;
}

std::vector<RadialLayerView> RadialLayerPlan::Views() const {
  std::vector<RadialLayerView> output;
  output.reserve(local_ids_.size());
  for (const auto& layer : local_ids_) {
    output.push_back({layer});
  }
  return output;
}

}  // namespace stlqpp
