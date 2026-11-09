#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/training/linkage/candidates/cpu_exact_parent_candidates.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

struct RadialLayerPolicy final {
  std::uint32_t layer_count = 0U;
  double root_fraction = 0.0;
};

// Deterministic cosine-to-native-center partition used by the registered
// good-cluster topology. Processing order is deliberately not constructed
// here: DEV interleaves each layer shuffle with that layer's tiny-CPU ICM
// draws, so the causal builder must consume the cluster stream layer by layer.
class RadialLayerPlan final {
 public:
  [[nodiscard]] static StatusOr<RadialLayerPlan> Build(
      ColumnMajorMatrixF32View exact_real,
      ArrayView<const float> native_center,
      RadialLayerPolicy policy);

  [[nodiscard]] std::uint32_t layer_count() const noexcept {
    return static_cast<std::uint32_t>(local_ids_.size());
  }
  [[nodiscard]] std::vector<RadialLayerView> Views() const;

 private:
  std::vector<std::vector<std::uint32_t>> local_ids_;
};

}  // namespace stlqpp
