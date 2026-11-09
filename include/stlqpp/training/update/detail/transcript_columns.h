#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "stlqpp/training/linkage/topology/cluster_topology_state.h"

namespace stlqpp::detail {

// A serialization-only view of existing tuple storage. The numerical owner
// retains its fixed layout; this view never repacks or allocates rows.
template <typename Byte>
class TranscriptSupports final {
 public:
  template <int M>
  explicit TranscriptSupports(std::vector<LinkageCodeTuple<M>>& rows) noexcept
      : data_(reinterpret_cast<Byte*>(rows.data())),
        count_(rows.size()),
        stride_(sizeof(LinkageCodeTuple<M>)),
        first_offset_(offsetof(LinkageCodeTuple<M>, first)),
        trailing_offset_(offsetof(LinkageCodeTuple<M>, trailing)) {
    static_assert(std::is_standard_layout_v<LinkageCodeTuple<M>>);
  }
  template <int M>
  explicit TranscriptSupports(
      const std::vector<LinkageCodeTuple<M>>& rows) noexcept
      : data_(reinterpret_cast<Byte*>(rows.data())),
        count_(rows.size()),
        stride_(sizeof(LinkageCodeTuple<M>)),
        first_offset_(offsetof(LinkageCodeTuple<M>, first)),
        trailing_offset_(offsetof(LinkageCodeTuple<M>, trailing)) {
    static_assert(std::is_const_v<Byte>);
    static_assert(std::is_standard_layout_v<LinkageCodeTuple<M>>);
  }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] std::uint32_t first(std::size_t row) const noexcept {
    assert(row < count_);
    std::uint32_t value;
    std::memcpy(&value, data_ + row * stride_ + first_offset_, sizeof(value));
    return value;
  }
  void set_first(std::size_t row, std::uint32_t value) const noexcept {
    static_assert(!std::is_const_v<Byte>);
    assert(row < count_);
    std::memcpy(data_ + row * stride_ + first_offset_, &value, sizeof(value));
  }
  [[nodiscard]] Byte* trailing(std::size_t row) const noexcept {
    assert(row < count_);
    return data_ + row * stride_ + trailing_offset_;
  }

 private:
  Byte* data_;
  std::size_t count_;
  std::size_t stride_;
  std::size_t first_offset_;
  std::size_t trailing_offset_;
};

using TranscriptSupportView = TranscriptSupports<const std::uint8_t>;
using TranscriptSupportOutput = TranscriptSupports<std::uint8_t>;

struct TranscriptTopologyView final {
  std::uint32_t m;
  std::uint32_t cid;
  std::uint32_t dimension;
  std::uint32_t virtual_count;
  ArrayView<const LinkageNodeRole> roles;
  ArrayView<const std::optional<LinkageParentReference>> parents;
  ArrayView<const std::uint32_t> depths;
  TranscriptSupportView supports;
  ArrayView<const float> coefficients;
};

template <int M>
TranscriptTopologyView ViewTranscriptTopology(
    const ClusterTopologyResult<M>& source) noexcept {
  return {source.m,           source.cid,
          source.dimension,   source.virtual_count,
          source.roles,       source.parents,
          source.depths,      TranscriptSupportView(source.supports),
          source.coefficients};
}

struct TranscriptDecodeColumns final {
  ArrayView<LinkageNodeRole> roles;
  ArrayView<std::optional<LinkageParentReference>> parents;
  ArrayView<std::uint32_t> depths;
  TranscriptSupportOutput supports;
  ArrayView<float> coefficients;
};

}  // namespace stlqpp::detail
