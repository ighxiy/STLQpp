#include "stlqpp/model/topology/parent_louds.h"
#include "parent_louds_internal.h"

#include "stlqpp/core/numeric/bit_ops.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <cassert>
#include <bit>
#include <vector>

namespace stlqpp {

namespace {

template <bool kValidated>
Status EncodeParentLoudsImpl(
    ArrayView<const std::uint32_t> parent_local_1based,
    std::vector<std::uint64_t>& words) {
  if (parent_local_1based.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return Status::ResourceExhausted(
        "ParentLOUDS cluster exceeds the u32 node domain");
  }
  const std::uint32_t node_count =
      static_cast<std::uint32_t>(parent_local_1based.size());
  std::vector<std::uint32_t> degree(
      static_cast<std::size_t>(node_count) + 1U, 0U);
  std::uint32_t previous_parent = 0U;
  for (std::uint32_t local = 0U; local < node_count; ++local) {
    const std::uint32_t parent = parent_local_1based[local];
    const std::uint32_t node_id = local + 1U;
    if constexpr (kValidated) {
      assert(parent < node_id);
      assert(parent >= previous_parent);
    } else {
      if (parent >= node_id) {
        return Status::DataLoss(
            "ParentLOUDS input is not in causal parent-before-child order");
      }
      if (parent < previous_parent) {
        return Status::DataLoss(
            "ParentLOUDS input is not in canonical BFS order");
      }
    }
    previous_parent = parent;
    ++degree[parent];
  }

  const std::uint64_t word_count64 = ParentLoudsWordCount(node_count);
  if (word_count64 > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted(
        "ParentLOUDS word extent exceeds size_t");
  }
  words.assign(static_cast<std::size_t>(word_count64), 0U);
  std::uint64_t bit_position = 0U;
  for (std::uint32_t node = 0U; node <= node_count; ++node) {
    for (std::uint32_t child = 0U; child < degree[node]; ++child) {
      words[static_cast<std::size_t>(bit_position / 64U)] |=
          std::uint64_t{1U} << (bit_position % 64U);
      ++bit_position;
    }
    ++bit_position;
  }
  assert(bit_position == ParentLoudsBitCount(node_count));
  return Status::Ok();
}

}  // namespace

Status EncodeParentLouds(
    ArrayView<const std::uint32_t> parents,
    std::vector<std::uint64_t>& words) {
  return EncodeParentLoudsImpl<false>(parents, words);
}

Status internal::EncodeValidatedParentLouds(
    ArrayView<const std::uint32_t> parents,
    std::vector<std::uint64_t>& words) {
  return EncodeParentLoudsImpl<true>(parents, words);
}

Status DecodeParentLouds(
    ArrayView<const std::uint64_t> words,
    std::uint32_t node_count,
    std::vector<std::uint32_t>& parent_local_1based) {
  const std::uint64_t bit_count = ParentLoudsBitCount(node_count);
  const std::uint64_t word_count = ParentLoudsWordCount(node_count);
  if (words.size() != word_count) {
    return Status::DataLoss(
        "ParentLOUDS word count does not match its node count");
  }

  parent_local_1based.assign(node_count, 0U);
  std::uint64_t previous_zero_plus_one = 0U;
  std::uint32_t node_id = 0U;
  std::uint32_t next_child = 1U;
  for (std::uint64_t word_index = 0U; word_index < word_count;
       ++word_index) {
    const std::uint64_t base = word_index * 64U;
    const std::uint32_t live_bits = static_cast<std::uint32_t>(
        bit_count - base < 64U ? bit_count - base : 64U);
    const std::uint64_t live_mask = live_bits == 64U
        ? std::numeric_limits<std::uint64_t>::max()
        : (std::uint64_t{1U} << live_bits) - 1U;
    const std::uint64_t word = words[static_cast<std::size_t>(word_index)];
    if ((word & ~live_mask) != 0U) {
      return Status::DataLoss("ParentLOUDS has nonzero tail padding");
    }
    std::uint64_t zeros = (~word) & live_mask;
    while (zeros != 0U) {
      if (node_id > node_count) {
        return Status::DataLoss("ParentLOUDS contains too many nodes");
      }
      const std::uint32_t bit = CountTrailingZerosNonZero(zeros);
      const std::uint64_t zero_position = base + bit;
      const std::uint64_t degree = zero_position - previous_zero_plus_one;
      for (std::uint64_t child = 0U; child < degree; ++child) {
        if (next_child > node_count ||
            (node_id != 0U && node_id >= next_child)) {
          return Status::DataLoss(
              "ParentLOUDS is not a causal BFS forest");
        }
        parent_local_1based[next_child - 1U] = node_id;
        ++next_child;
      }
      previous_zero_plus_one = zero_position + 1U;
      ++node_id;
      zeros &= zeros - 1U;
    }
  }
  if (node_id != node_count + 1U || next_child != node_count + 1U) {
    return Status::DataLoss(
        "ParentLOUDS degree totals do not match its node count");
  }
  return Status::Ok();
}

Status ValidateParentLouds(
    ArrayView<const std::uint64_t> words,
    std::uint32_t node_count,
    std::uint32_t root_count) {
  if (root_count > node_count ||
      words.size() != ParentLoudsWordCount(node_count)) {
    return Status::DataLoss(
        "ParentLOUDS word/root count does not match its node count");
  }
  const std::uint64_t bit_count = ParentLoudsBitCount(node_count);
  std::uint64_t previous_zero_plus_one = 0U;
  std::uint32_t node_id = 0U;
  std::uint32_t next_child = 1U;
  for (std::size_t word_index = 0U; word_index < words.size();
       ++word_index) {
    const std::uint64_t base = word_index * 64U;
    const std::uint32_t live_bits = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(64U, bit_count - base));
    const std::uint64_t live_mask = live_bits == 64U
        ? std::numeric_limits<std::uint64_t>::max()
        : (std::uint64_t{1U} << live_bits) - 1U;
    const std::uint64_t word = words[word_index];
    if ((word & ~live_mask) != 0U) {
      return Status::DataLoss("ParentLOUDS has nonzero tail padding");
    }
    std::uint64_t zeros = (~word) & live_mask;
    while (zeros != 0U) {
      if (node_id > node_count) {
        return Status::DataLoss("ParentLOUDS contains too many nodes");
      }
      const std::uint32_t bit = CountTrailingZerosNonZero(zeros);
      const std::uint64_t zero_position = base + bit;
      const std::uint64_t degree = zero_position - previous_zero_plus_one;
      for (std::uint64_t child = 0U; child < degree; ++child) {
        if (next_child > node_count ||
            (node_id != 0U && node_id >= next_child) ||
            (next_child <= root_count && node_id != 0U) ||
            (next_child > root_count && node_id == 0U)) {
          return Status::DataLoss(
              "ParentLOUDS is not the registered causal BFS forest");
        }
        ++next_child;
      }
      previous_zero_plus_one = zero_position + 1U;
      ++node_id;
      zeros &= zeros - 1U;
    }
  }
  if (node_id != node_count + 1U || next_child != node_count + 1U) {
    return Status::DataLoss(
        "ParentLOUDS degree totals do not match its node count");
  }
  return Status::Ok();
}

ParentLoudsSequentialDecoder::ParentLoudsSequentialDecoder(
    ArrayView<const std::uint64_t> words) noexcept
    : words_(words.data()) {}

void ParentLoudsSequentialDecoder::Skip(std::uint32_t count) noexcept {
  while (count != 0U) {
    while (one_positions_ == 0U) {
      word_base_ = static_cast<std::uint64_t>(next_word_) * 64U;
      one_positions_ = words_[next_word_++];
    }
    const std::uint32_t available =
        static_cast<std::uint32_t>(std::popcount(one_positions_));
    const std::uint32_t consumed = std::min(count, available);
    if (consumed == available) {
      one_positions_ = 0U;
    } else {
      for (std::uint32_t child = 0U; child < consumed; ++child) {
        one_positions_ &= one_positions_ - 1U;
      }
    }
    count -= consumed;
    emitted_children_ += consumed;
  }
}

void ParentLoudsSequentialDecoder::InitializeAfterValidatedRootPrefix(
    std::uint32_t root_count) noexcept {
  // A validated canonical stream starts with root_count one bits and the
  // implicit-super-root terminator. Position at the first linked child
  // without enumerating the known root prefix.
  const std::uint64_t next_bit =
      static_cast<std::uint64_t>(root_count) + 1U;
  const std::size_t word = static_cast<std::size_t>(next_bit >> 6U);
  const std::uint32_t offset = static_cast<std::uint32_t>(next_bit & 63U);
  word_base_ = static_cast<std::uint64_t>(word) << 6U;
  one_positions_ = words_[word] &
      (~std::uint64_t{0U} << offset);
  next_word_ = word + 1U;
  emitted_children_ = root_count;
}

}  // namespace stlqpp
