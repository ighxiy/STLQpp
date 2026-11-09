#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"

#include <array>
#include <cstdint>

namespace stlqpp {

template <int M>
class QueryTableBatch;

template <int M>
struct PreparedQueryTableView final {
  const float* root0 = nullptr;
  std::uint32_t root0_entries = 0U;
  std::array<const float*, M - 1> root_small{};
  std::array<const float*, M> edge{};
  std::uint32_t residual_entries = 0U;
  ArrayView<const std::uint32_t> selected_routes = {};
};

template <int M>
class TrustedQueryTables final {
 public:
  [[nodiscard]] static StatusOr<TrustedQueryTables<M>> Create(
      const PreparedQueryTableView<M>& source);

  [[nodiscard]] ArrayView<const std::uint32_t> selected_routes() const noexcept {
    return view_.selected_routes;
  }
  [[nodiscard]] float root0(std::uint32_t cid) const noexcept {
    return view_.root0[cid];
  }
  [[nodiscard]] ArrayView<const float> root0_values() const noexcept {
    return {view_.root0, view_.root0_entries};
  }
  [[nodiscard]] std::uint32_t root0_entries() const noexcept {
    return view_.root0_entries;
  }
  [[nodiscard]] float root_small(std::uint32_t layer,
                                 std::uint8_t code) const noexcept {
    return view_.root_small[layer - 1U][code];
  }
  [[nodiscard]] float edge(std::uint32_t layer,
                           std::uint8_t code) const noexcept {
    return view_.edge[layer][code];
  }

 private:
  friend class QueryTableBatch<M>;

  explicit TrustedQueryTables(PreparedQueryTableView<M> view) noexcept
      : view_(view) {}

  PreparedQueryTableView<M> view_;
};

extern template class TrustedQueryTables<5>;
extern template class TrustedQueryTables<10>;

}  // namespace stlqpp
