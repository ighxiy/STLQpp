#include "stlqpp/training/linkage/candidates/cluster_knn.h"

#include "hnswlib/hnswlib.h"
#include <algorithm>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace stlqpp {

struct HnswKnnRowsAccess final {
  static PreparedKnnRows Make(std::uint32_t rows, std::uint32_t k) {
    PreparedKnnRows output;
    output.row_count_ = rows;
    output.neighbors_per_row_ = k;
    output.producer_guaranteed_unique_ = true;
    output.local_ids_.resize(static_cast<std::size_t>(rows) * k);
    output.squared_distances_.resize(static_cast<std::size_t>(rows) * k);
    return output;
  }
  static void Store(PreparedKnnRows& output, std::uint32_t row,
                    const std::vector<std::pair<float, std::uint32_t>>& values) {
    const auto start = static_cast<std::size_t>(row) * output.neighbors_per_row_;
    for (std::size_t i = 0; i < values.size(); ++i) {
      output.local_ids_[start + i] = values[i].second;
      output.squared_distances_[start + i] = values[i].first;
    }
  }
};

namespace {
StatusOr<PreparedKnnRows> HnswRows(
    ColumnMajorMatrixF32View points, std::uint32_t requested_k,
    const ClusterKnnPolicy& policy) {
  if (!points.values || points.rows == 0 || points.columns == 0 ||
      requested_k == 0 || policy.hnsw_m < 2)
    return Status::InvalidArgument("HNSW KNN requires nonempty points and positive k/M");
  const auto k = std::min(requested_k, points.columns - 1U);
  try {
    auto result = HnswKnnRowsAccess::Make(points.columns, k);
    if (k == 0) return result;
    hnswlib::L2Space space(points.rows);
    // DEV graph preparation builds sequentially in local-ID order. The
    // library's default seed is 100; construction ef inherits graph k.
    auto efc = std::min(k, points.columns);
    if (policy.hnsw_ef_construction_cap != 0)
      efc = std::min(efc, policy.hnsw_ef_construction_cap);
    hnswlib::HierarchicalNSW<float> index(&space, points.columns,
                                        policy.hnsw_m, std::max(1U, efc));
    for (std::uint32_t row = 0; row < points.columns; ++row)
      index.addPoint(points.column(row), row);
    const auto want = std::min(points.columns, k + 1U);
    index.setEf(std::max(want, policy.hnsw_ef_search));
    std::vector<std::pair<float, std::uint32_t>> values;
    values.reserve(k);
    for (std::uint32_t row = 0; row < points.columns; ++row) {
      auto queue = index.searchKnn(points.column(row), want);
      values.clear();
      // Match DEV's heap consumption and self exclusion, including ties.
      while (!queue.empty() && values.size() < k) {
        const auto value = queue.top(); queue.pop();
        if (value.second == row) continue;
        values.emplace_back(value.first, static_cast<std::uint32_t>(value.second));
      }
      if (values.size() != k)
        return Status::DataLoss("HNSW KNN returned insufficient neighbors");
      std::reverse(values.begin(), values.end());
      HnswKnnRowsAccess::Store(result, row, values);
    }
    return result;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("HNSW cluster graph allocation failed");
  } catch (const std::exception& error) {
    return Status::Internal(std::string("HNSW cluster graph failed: ") + error.what());
  }
}
}  // namespace

StatusOr<PreparedKnnRows> BuildBadClusterKnn(
    const CpuExactKnnProvider& exact, ColumnMajorMatrixF32View points,
    std::uint32_t k, const ClusterKnnPolicy& policy,
    ExactKnnCpuBudget* cpu_budget) {
  if (policy.backend == BadClusterKnnBackend::kAuto && policy.cpu_exact_max_points == 0)
    return Status::InvalidArgument("auto KNN requires a positive exact point threshold");
  if (policy.backend == BadClusterKnnBackend::kCpuExact ||
      (policy.backend == BadClusterKnnBackend::kAuto &&
       points.columns <= policy.cpu_exact_max_points))
    return exact.BuildSymmetric(points, k, cpu_budget);
  return HnswRows(points, k, policy);
}

}  // namespace stlqpp
