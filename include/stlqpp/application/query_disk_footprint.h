#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/storage/index_manifest.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stlqpp {

// Logical query payloads only. Whole files retain headers/padding; registered
// container payloads may specify a range. No hashing or directory traversal.
class QueryDiskFootprint final {
 public:
  void AddFile(const std::string& role, const std::filesystem::path& path) {
    AddRange(role, path, 0, std::filesystem::file_size(path));
  }
  void AddRange(const std::string& role, const std::filesystem::path& path,
                std::uint64_t offset, std::uint64_t bytes) {
    const auto resolved = std::filesystem::canonical(path);
    const auto size = std::filesystem::file_size(resolved);
    if (offset > size || bytes > size - offset)
      throw std::runtime_error("query payload exceeds file: " + resolved.string());
    // Charge only uncovered intervals, even when a whole file and an embedded
    // payload are both referenced. Attribution follows first registration.
    auto& ranges = ranges_[resolved];
    std::sort(ranges.begin(), ranges.end());
    const auto end = offset + bytes;
    auto cursor = offset;
    std::uint64_t added = 0;
    for (const auto& range : ranges) {
      if (range.second <= cursor || range.first >= end) continue;
      if (range.first > cursor) added += range.first - cursor;
      cursor = std::max(cursor, std::min(end, range.second));
    }
    added += end - cursor;
    if (added > std::numeric_limits<std::uint64_t>::max() - total_)
      throw std::runtime_error("query payload total exceeds uint64");
    totals_[role] += added;
    total_ += added;
    ranges.emplace_back(offset, end);
  }
  std::uint64_t bytes() const noexcept { return total_; }
  std::uint64_t role_bytes(const std::string& role) const {
    const auto it = totals_.find(role);
    return it == totals_.end() ? 0 : it->second;
  }
  std::string Summary(const std::string& representation, const std::string& norm,
                      const std::string& routing, std::uint64_t real,
                      std::uint64_t virtual_rows = 0) const {
    std::ostringstream out;
    out << "scope=minimal_query_data representation=" << representation
        << " norm=" << norm << " routing=" << routing
        << " real_rows=" << real << " virtual_rows=" << virtual_rows;
    for (const auto* role : {"model", "directory", "codes_and_ids", "parents",
                             "coefficients", "norm", "routing_extra"})
      out << ' ' << role << "_bytes=" << role_bytes(role);
    out << " minimal_query_disk_bytes=" << total_ << std::fixed << std::setprecision(6)
        << " MiB=" << double(total_) / (1024.0 * 1024.0)
        << " GiB=" << double(total_) / (1024.0 * 1024.0 * 1024.0);
    return out.str();
  }
 private:
  std::map<std::filesystem::path,
           std::vector<std::pair<std::uint64_t, std::uint64_t>>> ranges_;
  std::map<std::string, std::uint64_t> totals_;
  std::uint64_t total_ = 0;
};
// Inventory follows the effective manifest and norm preparation result.
[[nodiscard]] StatusOr<QueryDiskFootprint> CollectQueryDiskFootprint(
    const std::filesystem::path& index, const IndexManifest& manifest,
    bool native, bool scaled, bool normless,
    const std::vector<std::filesystem::path>& norm_payloads = {},
    const std::filesystem::path& routing_index = {});
}  // namespace stlqpp
