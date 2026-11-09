#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/training/linkage/candidates/cpu_exact_knn.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

struct UmapVirtualRootPolicy final {
  // Zero consumes the complete prepared row. Otherwise only this ordered
  // prefix participates in the UMAP membership graph.
  std::uint32_t neighbor_count = 0U;
  std::uint32_t local_connectivity = 1U;
  float overlap_threshold = 0.9F;
  bool prefer_density_peaks = true;
  std::uint32_t anchor_neighbor_k = 16U;
};

// Query-independent barycentric plan derived from one exact real-row graph.
// The same plan materializes roots in source and current model coordinates;
// linear rotation therefore cannot change root identity or neighborhood.
class UmapVirtualRootPlan final {
 public:
  [[nodiscard]] static StatusOr<UmapVirtualRootPlan> Build(
      const PreparedKnnRows& real_graph,
      std::uint32_t requested_roots,
      UmapVirtualRootPolicy policy);

  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return real_count_;
  }
  [[nodiscard]] std::uint32_t root_count() const noexcept {
    return static_cast<std::uint32_t>(seed_local_ids_.size());
  }
  [[nodiscard]] ArrayView<const std::uint32_t> seed_local_ids() const
      noexcept {
    return seed_local_ids_;
  }

  [[nodiscard]] StatusOr<std::vector<float>> Materialize(
      ColumnMajorMatrixF32View real_vectors) const;

 private:
  std::uint32_t real_count_ = 0U;
  std::vector<std::uint32_t> seed_local_ids_;
  std::vector<std::uint32_t> offsets_;
  std::vector<std::uint32_t> source_local_ids_;
  std::vector<float> raw_weights_;
  std::vector<float> inverse_weight_sums_;
};

}  // namespace stlqpp
