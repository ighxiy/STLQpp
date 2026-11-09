#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {

// Frozen mapping from the wide C_root[0] domain used by init-linkage to the
// byte C_one[0] domain trained from that forest. Construction of the mapping
// is a separate initialization algorithm; moment accumulation only consumes
// this immutable result.
class RootToEdgeCodeMap final {
 public:
  [[nodiscard]] static StatusOr<RootToEdgeCodeMap> Create(
      std::uint32_t root_count,
      ArrayView<const std::uint8_t> edge_codes) {
    if (root_count == 0U || edge_codes.size() != root_count) {
      return Status::InvalidArgument(
          "root-to-edge mapping must cover every root code");
    }
    return RootToEdgeCodeMap(std::vector<std::uint8_t>(
        edge_codes.begin(), edge_codes.end()));
  }

  [[nodiscard]] std::uint32_t root_count() const noexcept {
    return static_cast<std::uint32_t>(edge_codes_.size());
  }

  [[nodiscard]] std::uint8_t edge_code(
      std::uint32_t root_code) const noexcept {
    assert(root_code < edge_codes_.size());
    return edge_codes_[root_code];
  }

  [[nodiscard]] ArrayView<const std::uint8_t> codes() const noexcept {
    return edge_codes_;
  }

 private:
  explicit RootToEdgeCodeMap(std::vector<std::uint8_t> edge_codes) noexcept
      : edge_codes_(std::move(edge_codes)) {}

  std::vector<std::uint8_t> edge_codes_;
};

}  // namespace stlqpp
