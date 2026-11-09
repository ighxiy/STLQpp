#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/search/query_batch.h"
#include <filesystem>
#include <memory>
#include <vector>

namespace stlqpp {

struct IvfHnswOptions final {
  bool enabled = false;
  std::uint32_t m = 32U;
  std::uint32_t ef_construction = 200U;
  std::uint32_t ef_search = 32U;
  std::filesystem::path cache_root = {};
};

// Shared routing control, independent of reconstruction code length. Search
// ef is fixed when opening this object; concurrent queries never mutate it.
class IvfHnswRouter final {
 public:
  ~IvfHnswRouter();
  [[nodiscard]] const std::filesystem::path& index_path() const noexcept;
  [[nodiscard]] static StatusOr<std::unique_ptr<IvfHnswRouter>> Open(
      ColumnMajorMatrixF32View roots, const IvfHnswOptions& options);
  // Query-owner path. Shapes/buffers were checked at the search boundary;
  // writes nprobe route IDs and only their entries in the nlist-sized dot row.
  [[nodiscard]] Status RouteQueryTrusted(
      const float* query, std::uint32_t nprobe,
      std::uint32_t* routes, float* root_dots) const noexcept;
  [[nodiscard]] Status RouteBlock(
      const TrustedQueryBatchView& queries, std::uint32_t nprobe,
      const ParallelForExecutor& parallel, std::vector<std::uint32_t>& routes,
      float* root_dots, std::vector<Status>& worker_failures) const;
 private:
  struct Impl;
  explicit IvfHnswRouter(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
}  // namespace stlqpp
