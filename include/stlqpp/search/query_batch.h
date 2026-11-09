#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"

#include <cstddef>
#include <cstdint>

namespace stlqpp {

template <int M>
class ExactDenseSearcher;

struct QueryBatchView final {
  const float* data = nullptr;
  std::uint32_t count = 0;
  std::uint32_t dimension = 0;
  std::size_t stride_floats = 0;

  [[nodiscard]] Status Validate(std::uint32_t expected_dimension) const;
};

// Result of the request-boundary validation. Slices preserve the proof and
// are the only query views accepted by table-generation internals.
class TrustedQueryBatchView final {
 public:
  [[nodiscard]] static StatusOr<TrustedQueryBatchView> Create(
      const QueryBatchView& source,
      std::uint32_t expected_dimension);

  [[nodiscard]] const float* data() const noexcept { return view_.data; }
  [[nodiscard]] std::uint32_t count() const noexcept { return view_.count; }
  [[nodiscard]] std::uint32_t dimension() const noexcept {
    return view_.dimension;
  }
  [[nodiscard]] std::size_t stride_floats() const noexcept {
    return view_.stride_floats;
  }
  [[nodiscard]] TrustedQueryBatchView Slice(
      std::uint32_t first,
      std::uint32_t count) const noexcept;

 private:
  template <int M>
  friend class ExactDenseSearcher;
  friend class RuntimeExactDenseSearcher;

  [[nodiscard]] static TrustedQueryBatchView FromTrustedTransform(
      const float* data,
      std::uint32_t count,
      std::uint32_t dimension) noexcept {
    return TrustedQueryBatchView({
        .data = data,
        .count = count,
        .dimension = dimension,
        .stride_floats = dimension,
    });
  }

  explicit TrustedQueryBatchView(QueryBatchView view) noexcept : view_(view) {}

  QueryBatchView view_;
};

}  // namespace stlqpp
