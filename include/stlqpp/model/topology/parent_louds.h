#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/numeric/bit_ops.h"
#include "stlqpp/core/status.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

// Canonical succinct forest representation. Actual nodes are numbered 1..N
// in BFS/causal order; parent 0 is the implicit super-root. The bit stream is
// the LOUDS degree sequence for nodes 0..N and always has 2*N+1 bits.
[[nodiscard]] Status EncodeParentLouds(
    ArrayView<const std::uint32_t> parent_local_1based,
    std::vector<std::uint64_t>& words);

[[nodiscard]] Status DecodeParentLouds(
    ArrayView<const std::uint64_t> words,
    std::uint32_t node_count,
    std::vector<std::uint32_t>& parent_local_1based);

// Validates the complete degree stream without expanding one parent word per
// node. root_count is the exact number of leading children of the implicit
// super-root.
[[nodiscard]] Status ValidateParentLouds(
    ArrayView<const std::uint64_t> words,
    std::uint32_t node_count,
    std::uint32_t root_count);

// Query-hot sequential decoder for an already validated canonical LOUDS
// stream. It emits the same one-based parent convention as DecodeParentLouds.
class ParentLoudsSequentialDecoder final {
 public:
  explicit ParentLoudsSequentialDecoder(
      ArrayView<const std::uint64_t> words) noexcept;

  [[nodiscard]] std::uint32_t NextParent1Based() noexcept;
  void Skip(std::uint32_t count) noexcept;
  void InitializeAfterValidatedRootPrefix(
      std::uint32_t root_count) noexcept;

 private:
  const std::uint64_t* words_ = nullptr;
  std::size_t next_word_ = 0U;
  std::uint64_t one_positions_ = 0U;
  std::uint64_t word_base_ = 0U;
  std::uint32_t emitted_children_ = 0U;
};

#if defined(_MSC_VER)
#define STLQPP_LOUDS_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define STLQPP_LOUDS_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define STLQPP_LOUDS_ALWAYS_INLINE inline
#endif

STLQPP_LOUDS_ALWAYS_INLINE
std::uint32_t ParentLoudsSequentialDecoder::NextParent1Based() noexcept {
  // Every one bit emits a child. Its parent is the number of preceding
  // zeros: select1(child) - child, with zero-based bit/child positions.
  // Enumerating ones avoids visiting zero-degree parents individually.
  // The storage boundary has already validated zero tail padding and the
  // caller consumes at most node_count children.
  while (one_positions_ == 0U) {
    word_base_ = static_cast<std::uint64_t>(next_word_) * 64U;
    one_positions_ = words_[next_word_++];
  }
  const std::uint32_t bit = CountTrailingZerosNonZero(one_positions_);
  one_positions_ &= one_positions_ - 1U;
  return static_cast<std::uint32_t>(word_base_ + bit - emitted_children_++);
}

#undef STLQPP_LOUDS_ALWAYS_INLINE

[[nodiscard]] constexpr std::uint64_t ParentLoudsBitCount(
    std::uint32_t node_count) noexcept {
  return 2U * static_cast<std::uint64_t>(node_count) + 1U;
}

[[nodiscard]] constexpr std::uint64_t ParentLoudsWordCount(
    std::uint32_t node_count) noexcept {
  return (ParentLoudsBitCount(node_count) + 63U) / 64U;
}

}  // namespace stlqpp
