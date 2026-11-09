#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/search/query_tables.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

template <int M>
class QueryTableBuilder;

// Query-owner views borrow this batch. Physical storage matches the three
// codebook GEMM outputs and is reused across query blocks.
template <int M>
class QueryTableBatch final {
 public:
  static_assert(M >= 2 && M <= 20);

  [[nodiscard]] static StatusOr<QueryTableBatch<M>> Create(
      const ModelShape& shape,
      std::uint32_t query_capacity);

  QueryTableBatch(QueryTableBatch&&) noexcept = default;
  QueryTableBatch& operator=(QueryTableBatch&&) noexcept = default;
  QueryTableBatch(const QueryTableBatch&) = delete;
  QueryTableBatch& operator=(const QueryTableBatch&) = delete;

  [[nodiscard]] std::uint32_t nlist() const noexcept { return nlist_; }
  [[nodiscard]] std::uint32_t dimension() const noexcept {
    return dimension_;
  }
  [[nodiscard]] std::uint32_t query_capacity() const noexcept {
    return query_capacity_;
  }
  [[nodiscard]] std::uint32_t query_count() const noexcept {
    return query_count_;
  }

  [[nodiscard]] TrustedQueryTables<M> query(
      std::uint32_t query_index) const noexcept {
    assert(query_index < query_count_);
    PreparedQueryTableView<M> view;
    view.root0 = root0_.data() +
        static_cast<std::size_t>(query_index) * nlist_;
    view.root0_entries = nlist_;
    if (route_count_ != 0U) view.selected_routes = {
        routes_.data() + static_cast<std::size_t>(query_index) * route_count_, route_count_};
    view.residual_entries = 256U;
    const float* root_small = root_small_.data() +
        static_cast<std::size_t>(query_index) * (M - 1) * 256U;
    const float* edge = edge_.data() +
        static_cast<std::size_t>(query_index) * M * 256U;
    for (std::uint32_t layer = 0U;
         layer < static_cast<std::uint32_t>(M - 1); ++layer) {
      view.root_small[layer] = root_small +
          static_cast<std::size_t>(layer) * 256U;
    }
    for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(M);
         ++layer) {
      view.edge[layer] = edge + static_cast<std::size_t>(layer) * 256U;
    }
    return TrustedQueryTables<M>(view);
  }

 private:
  friend class QueryTableBuilder<M>;
  friend class ExactDenseSearcher<M>;

  QueryTableBatch(std::uint32_t dimension,
                  std::uint32_t nlist,
                  std::uint32_t query_capacity)
      : dimension_(dimension),
        nlist_(nlist),
        query_capacity_(query_capacity),
        root0_(static_cast<std::size_t>(nlist) * query_capacity),
        root_small_(static_cast<std::size_t>(M - 1) * 256U *
                    query_capacity),
        edge_(static_cast<std::size_t>(M) * 256U * query_capacity) {}

  std::uint32_t dimension_ = 0U;
  std::uint32_t nlist_ = 0U;
  std::uint32_t query_capacity_ = 0U;
  std::uint32_t query_count_ = 0U;
  std::vector<float> root0_;
  std::vector<std::uint32_t> routes_;
  std::uint32_t route_count_ = 0U;
  std::vector<float> root_small_;
  std::vector<float> edge_;
};

extern template class QueryTableBatch<5>;
extern template class QueryTableBatch<10>;

}  // namespace stlqpp
