#include "stlqpp/search/query_batch.h"

#include <cassert>
#include <cmath>

namespace stlqpp {

Status QueryBatchView::Validate(std::uint32_t expected_dimension) const {
  if (count == 0U) {
    return Status::InvalidArgument("query batch must not be empty");
  }
  if (data == nullptr) {
    return Status::InvalidArgument("query data pointer must not be null");
  }
  if (dimension != expected_dimension) {
    return Status::InvalidArgument("query dimension does not match index dimension");
  }
  if (stride_floats < dimension) {
    return Status::InvalidArgument("query stride is smaller than dimension");
  }
  for (std::uint32_t query = 0U; query < count; ++query) {
    const float* row = data + static_cast<std::size_t>(query) * stride_floats;
    for (std::uint32_t coordinate = 0U; coordinate < dimension;
         ++coordinate) {
      if (!std::isfinite(row[coordinate])) {
        return Status::InvalidArgument("query contains a non-finite value");
      }
    }
  }
  return Status::Ok();
}

StatusOr<TrustedQueryBatchView> TrustedQueryBatchView::Create(
    const QueryBatchView& source,
    std::uint32_t expected_dimension) {
  if (const Status status = source.Validate(expected_dimension); !status.ok()) {
    return status;
  }
  return TrustedQueryBatchView(source);
}

TrustedQueryBatchView TrustedQueryBatchView::Slice(
    std::uint32_t first,
    std::uint32_t count) const noexcept {
  assert(first <= view_.count && count <= view_.count - first);
  QueryBatchView slice = view_;
  slice.data += static_cast<std::size_t>(first) * view_.stride_floats;
  slice.count = count;
  return TrustedQueryBatchView(slice);
}

}  // namespace stlqpp
