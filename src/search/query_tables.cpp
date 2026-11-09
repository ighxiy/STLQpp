#include "stlqpp/search/query_tables.h"

#include <cmath>
#include <cstdint>

namespace stlqpp {

template <int M>
StatusOr<TrustedQueryTables<M>> TrustedQueryTables<M>::Create(
    const PreparedQueryTableView<M>& source) {
  static_assert(M >= 2 && M <= 20);
  if (source.root0 == nullptr ||
      source.root0_entries == 0U || source.residual_entries != 256U) {
    return Status::InvalidArgument("query table metadata is invalid");
  }
  if (!source.selected_routes.empty() && source.selected_routes.data() == nullptr)
    return Status::InvalidArgument("selected IVF route pointer is null");
  for (std::size_t i = 0; i < source.selected_routes.size(); ++i) {
    if (source.selected_routes[i] >= source.root0_entries)
      return Status::InvalidArgument("selected IVF route exceeds nlist");
    for (std::size_t j = 0; j < i; ++j)
      if (source.selected_routes[i] == source.selected_routes[j])
        return Status::InvalidArgument("selected IVF routes contain duplicate CID");
  }
  for (const float* table : source.root_small) {
    if (table == nullptr) {
      return Status::InvalidArgument("query is missing a root codebook table");
    }
  }
  for (const float* table : source.edge) {
    if (table == nullptr) {
      return Status::InvalidArgument("query is missing an edge codebook table");
    }
  }
  for (std::uint32_t cid = 0U; cid < source.root0_entries; ++cid) {
    if (!std::isfinite(source.root0[cid])) {
      return Status::InvalidArgument("root query table contains a non-finite value");
    }
  }
  for (const float* table : source.root_small) {
    for (std::uint32_t code = 0U; code < source.residual_entries; ++code) {
      if (!std::isfinite(table[code])) {
        return Status::InvalidArgument(
            "root-small query table contains a non-finite value");
      }
    }
  }
  for (const float* table : source.edge) {
    for (std::uint32_t code = 0U; code < source.residual_entries; ++code) {
      if (!std::isfinite(table[code])) {
        return Status::InvalidArgument(
            "edge query table contains a non-finite value");
      }
    }
  }
  return TrustedQueryTables<M>(source);
}

template class TrustedQueryTables<5>;
template class TrustedQueryTables<12>;
template class TrustedQueryTables<6>;
template class TrustedQueryTables<10>;

}  // namespace stlqpp
