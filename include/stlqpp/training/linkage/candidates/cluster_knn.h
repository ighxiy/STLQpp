#pragma once

#include "stlqpp/training/linkage/candidates/cpu_exact_knn.h"

namespace stlqpp {

// Search preparation is shared across M and both numerical executors.
enum class BadClusterKnnBackend : std::uint8_t { kCpuExact, kAuto, kHnsw };

struct ClusterKnnPolicy final {
  BadClusterKnnBackend backend = BadClusterKnnBackend::kCpuExact;
  std::uint32_t cpu_exact_max_points = 150000U;
  std::uint32_t hnsw_m = 24U;
  std::uint32_t hnsw_ef_construction_cap = 0U;
  // Zero retains DEV's k_graph + 1 search width.
  std::uint32_t hnsw_ef_search = 0U;
};

[[nodiscard]] StatusOr<PreparedKnnRows> BuildBadClusterKnn(
    const CpuExactKnnProvider& exact, ColumnMajorMatrixF32View points,
    std::uint32_t k, const ClusterKnnPolicy& policy,
    ExactKnnCpuBudget* cpu_budget = nullptr);

}  // namespace stlqpp
