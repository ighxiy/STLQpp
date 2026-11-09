#include "stlqpp/model/codec/canonical_huffman.h"

#include "canonical_huffman_internal.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

constexpr std::uint32_t kMaximumCodeLength = 32U;

struct HuffmanNode final {
  std::uint64_t frequency = 0U;
  std::uint16_t minimum_symbol = 0U;
  std::uint16_t symbol = 0U;
  std::uint32_t left = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t right = std::numeric_limits<std::uint32_t>::max();

  [[nodiscard]] bool is_leaf() const noexcept {
    return left == std::numeric_limits<std::uint32_t>::max();
  }
};

struct HuffmanNodeLess final {
  const std::vector<HuffmanNode>* nodes = nullptr;

  [[nodiscard]] bool operator()(std::uint32_t lhs,
                                std::uint32_t rhs) const noexcept {
    const HuffmanNode& a = (*nodes)[lhs];
    const HuffmanNode& b = (*nodes)[rhs];
    if (a.frequency != b.frequency) {
      return a.frequency > b.frequency;
    }
    return a.minimum_symbol > b.minimum_symbol;
  }
};

Status AssignCodeLengths(const std::vector<HuffmanNode>& nodes,
                         std::uint32_t node_index,
                         std::uint32_t depth,
                         std::vector<std::uint8_t>& lengths) {
  const HuffmanNode& node = nodes[node_index];
  if (node.is_leaf()) {
    if (depth > kMaximumCodeLength) {
      return Status::ResourceExhausted(
          "canonical Huffman tree exceeds the 32-bit format limit");
    }
    lengths[node.symbol] = static_cast<std::uint8_t>(depth);
    return Status::Ok();
  }
  if (const Status status =
          AssignCodeLengths(nodes, node.left, depth + 1U, lengths);
      !status.ok()) {
    return status;
  }
  return AssignCodeLengths(nodes, node.right, depth + 1U, lengths);
}

StatusOr<std::vector<std::uint8_t>> BuildCodeLengths(
    ArrayView<const std::uint64_t> frequencies) {
  std::vector<std::uint8_t> lengths(frequencies.size(), 0U);
  std::vector<HuffmanNode> nodes;
  nodes.reserve(frequencies.size() * 2U);
  HuffmanNodeLess less{&nodes};
  std::priority_queue<std::uint32_t, std::vector<std::uint32_t>,
                      HuffmanNodeLess>
      queue(less);

  for (std::uint32_t symbol = 0U; symbol < frequencies.size(); ++symbol) {
    if (frequencies[symbol] == 0U) {
      continue;
    }
    nodes.push_back({.frequency = frequencies[symbol],
                     .minimum_symbol = static_cast<std::uint16_t>(symbol),
                     .symbol = static_cast<std::uint16_t>(symbol)});
    queue.push(static_cast<std::uint32_t>(nodes.size() - 1U));
  }
  if (queue.empty()) {
    return lengths;
  }
  if (queue.size() == 1U) {
    lengths[nodes[queue.top()].symbol] = 1U;
    return lengths;
  }

  while (queue.size() > 1U) {
    const std::uint32_t left = queue.top();
    queue.pop();
    const std::uint32_t right = queue.top();
    queue.pop();
    const HuffmanNode& a = nodes[left];
    const HuffmanNode& b = nodes[right];
    if (a.frequency >
        std::numeric_limits<std::uint64_t>::max() - b.frequency) {
      return Status::ResourceExhausted(
          "canonical Huffman frequency sum overflows u64");
    }
    nodes.push_back({
        .frequency = a.frequency + b.frequency,
        .minimum_symbol = std::min(a.minimum_symbol, b.minimum_symbol),
        .left = left,
        .right = right,
    });
    queue.push(static_cast<std::uint32_t>(nodes.size() - 1U));
  }
  if (const Status status =
          AssignCodeLengths(nodes, queue.top(), 0U, lengths);
      !status.ok()) {
    return status;
  }
  return lengths;
}

struct CanonicalEncodingTable final {
  std::vector<std::uint32_t> codes;
  std::vector<std::uint8_t> lengths;
};

CanonicalEncodingTable MakeCanonicalEncodingTable(
    std::vector<std::uint8_t> lengths) {
  std::array<std::uint32_t, kMaximumCodeLength + 1U> counts{};
  for (const std::uint8_t length : lengths) {
    if (length != 0U) {
      ++counts[length];
    }
  }
  std::array<std::uint32_t, kMaximumCodeLength + 1U> next_codes{};
  std::uint32_t code = 0U;
  for (std::uint32_t bits = 1U; bits <= kMaximumCodeLength; ++bits) {
    code = (code + counts[bits - 1U]) << 1U;
    next_codes[bits] = code;
  }

  CanonicalEncodingTable table;
  table.codes.resize(lengths.size(), 0U);
  table.lengths = std::move(lengths);
  for (std::uint32_t symbol = 0U; symbol < table.lengths.size(); ++symbol) {
    const std::uint8_t length = table.lengths[symbol];
    if (length != 0U) {
      table.codes[symbol] = next_codes[length]++;
    }
  }
  return table;
}

}  // namespace

namespace {

template <bool ValidateSymbols>
StatusOr<EncodedCanonicalHuffman> EncodeCanonicalHuffmanImpl(
    ArrayView<const std::uint8_t> symbols,
    std::uint32_t alphabet_size) {
  if (alphabet_size == 0U || alphabet_size > 256U) {
    return Status::InvalidArgument(
        "canonical Huffman alphabet must contain 1..256 symbols");
  }
  std::vector<std::uint64_t> frequencies(alphabet_size, 0U);
  if constexpr (ValidateSymbols) {
    for (const std::uint8_t symbol : symbols) {
      if (symbol >= alphabet_size) {
        return Status::InvalidArgument(
            "canonical Huffman input symbol exceeds the alphabet");
      }
      ++frequencies[symbol];
    }
  } else {
    for (const std::uint8_t symbol : symbols) {
      assert(symbol < alphabet_size);
      ++frequencies[symbol];
    }
  }
  auto lengths_result = BuildCodeLengths(frequencies);
  if (!lengths_result.ok()) {
    return lengths_result.status();
  }
  CanonicalEncodingTable table =
      MakeCanonicalEncodingTable(std::move(lengths_result).value());

  std::uint64_t total_bits = 0U;
  for (std::uint32_t symbol = 0U; symbol < alphabet_size; ++symbol) {
    const std::uint64_t length = table.lengths[symbol];
    if (length != 0U && frequencies[symbol] >
                            (std::numeric_limits<std::uint64_t>::max() -
                             total_bits) /
                                length) {
      return Status::ResourceExhausted(
          "canonical Huffman payload bit count overflows u64");
    }
    total_bits += frequencies[symbol] * length;
  }
  const std::uint64_t total_bytes =
      total_bits / 8U + (total_bits % 8U == 0U ? 0U : 1U);
  if (total_bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted(
        "canonical Huffman payload exceeds addressable memory");
  }

  EncodedCanonicalHuffman output;
  output.code_lengths = table.lengths;
  output.payload.reserve(static_cast<std::size_t>(total_bytes));
  std::uint64_t pending = 0U;
  std::uint32_t pending_bits = 0U;
  for (const std::uint8_t symbol : symbols) {
    const std::uint32_t length = table.lengths[symbol];
    pending = (pending << length) | table.codes[symbol];
    pending_bits += length;
    while (pending_bits >= 8U) {
      const std::uint32_t shift = pending_bits - 8U;
      output.payload.push_back(
          static_cast<std::uint8_t>((pending >> shift) & 0xFFU));
      pending_bits -= 8U;
      if (pending_bits == 0U) {
        pending = 0U;
      } else {
        pending &= (std::uint64_t{1U} << pending_bits) - 1U;
      }
    }
  }
  if (pending_bits != 0U) {
    output.payload.push_back(
        static_cast<std::uint8_t>(pending << (8U - pending_bits)));
  }
  return output;
}

}  // namespace

StatusOr<EncodedCanonicalHuffman> EncodeCanonicalHuffmanMsbFirst(
    ArrayView<const std::uint8_t> symbols,
    std::uint32_t alphabet_size) {
  return EncodeCanonicalHuffmanImpl<true>(symbols, alphabet_size);
}

StatusOr<EncodedCanonicalHuffman> internal::EncodeCanonicalHuffmanMsbFirstTrusted(
    ArrayView<const std::uint8_t> symbols,
    std::uint32_t alphabet_size) {
  return EncodeCanonicalHuffmanImpl<false>(symbols, alphabet_size);
}

StatusOr<CanonicalHuffmanDecoder> CanonicalHuffmanDecoder::Create(
    ArrayView<const std::uint8_t> code_lengths) {
  if (code_lengths.empty() || code_lengths.size() > 256U) {
    return Status::InvalidArgument(
        "canonical Huffman alphabet must contain 1..256 symbols");
  }

  CanonicalHuffmanDecoder output;
  output.alphabet_size_ = static_cast<std::uint32_t>(code_lengths.size());
  for (const std::uint8_t length : code_lengths) {
    output.max_code_length_ =
        std::max(output.max_code_length_, static_cast<std::uint32_t>(length));
  }
  if (output.max_code_length_ == 0U) {
    return Status::DataLoss("canonical Huffman table contains no symbols");
  }
  if (output.max_code_length_ > kMaximumCodeLength) {
    return Status::Unimplemented(
        "canonical Huffman code lengths above 32 bits are unsupported");
  }

  output.length_counts_.assign(output.max_code_length_ + 1U, 0U);
  for (const std::uint8_t length : code_lengths) {
    if (length != 0U) {
      ++output.length_counts_[length];
    }
  }

  std::uint64_t available = 1U;
  for (std::uint32_t bits = 1U; bits <= output.max_code_length_; ++bits) {
    available *= 2U;
    if (output.length_counts_[bits] > available) {
      return Status::DataLoss("canonical Huffman table is oversubscribed");
    }
    available -= output.length_counts_[bits];
  }

  output.first_codes_.assign(output.max_code_length_ + 1U, 0U);
  output.first_symbol_offsets_.assign(output.max_code_length_ + 1U, 0U);
  std::uint64_t first_code = 0U;
  std::uint32_t first_symbol = 0U;
  for (std::uint32_t bits = 1U; bits <= output.max_code_length_; ++bits) {
    first_code =
        (first_code + output.length_counts_[bits - 1U]) * 2U;
    output.first_codes_[bits] = first_code;
    output.first_symbol_offsets_[bits] = first_symbol;
    first_symbol += output.length_counts_[bits];
  }

  output.symbols_by_length_.reserve(first_symbol);
  for (std::uint32_t bits = 1U; bits <= output.max_code_length_; ++bits) {
    for (std::uint32_t symbol = 0U; symbol < code_lengths.size(); ++symbol) {
      if (code_lengths[symbol] == bits) {
        output.symbols_by_length_.push_back(
            static_cast<std::uint8_t>(symbol));
      }
    }
  }

  for (std::uint32_t bits = 1U;
       bits <= std::min(kFastBits, output.max_code_length_); ++bits) {
    const std::uint32_t repetitions = 1U << (kFastBits - bits);
    const std::uint32_t count = output.length_counts_[bits];
    const std::uint32_t symbol_offset = output.first_symbol_offsets_[bits];
    for (std::uint32_t rank = 0U; rank < count; ++rank) {
      const std::uint32_t prefix = static_cast<std::uint32_t>(
          (output.first_codes_[bits] + rank) << (kFastBits - bits));
      for (std::uint32_t suffix = 0U; suffix < repetitions; ++suffix) {
        output.fast_symbols_[prefix + suffix] =
            output.symbols_by_length_[symbol_offset + rank];
        output.fast_lengths_[prefix + suffix] =
            static_cast<std::uint8_t>(bits);
      }
    }
  }
  return output;
}

Status CanonicalHuffmanDecoder::DecodeMsbFirst(
    ArrayView<const std::uint8_t> payload,
    ArrayView<std::uint8_t> output) const {
  if (output.empty()) {
    return payload.empty()
        ? Status::Ok()
        : Status::DataLoss("zero-symbol Huffman stream contains payload bytes");
  }
  if (payload.empty()) {
    return Status::DataLoss("Huffman payload is truncated");
  }

  std::size_t bit_position = 0U;
  for (std::uint8_t& decoded : output) {
    const std::size_t remaining_bits = payload.size() * 8U - bit_position;
    if (remaining_bits >= kFastBits) {
      const std::size_t byte_position = bit_position / 8U;
      const std::uint32_t bit_in_byte =
          static_cast<std::uint32_t>(bit_position % 8U);
      std::uint16_t window = static_cast<std::uint16_t>(
          static_cast<std::uint32_t>(payload[byte_position]) << 8U);
      if (byte_position + 1U < payload.size()) {
        window = static_cast<std::uint16_t>(
            static_cast<std::uint32_t>(window) |
            static_cast<std::uint32_t>(payload[byte_position + 1U]));
      }
      const std::uint8_t prefix = static_cast<std::uint8_t>(
          (window >> (8U - bit_in_byte)) & 0xFFU);
      const std::uint8_t length = fast_lengths_[prefix];
      if (length != 0U) {
        decoded = fast_symbols_[prefix];
        bit_position += length;
        continue;
      }
    }

    std::uint64_t code = 0U;
    bool found = false;
    for (std::uint32_t bits = 1U; bits <= max_code_length_; ++bits) {
      const std::size_t byte_position = bit_position / 8U;
      if (byte_position >= payload.size()) {
        return Status::DataLoss("Huffman payload is truncated");
      }
      const std::uint32_t bit_in_byte =
          static_cast<std::uint32_t>(bit_position % 8U);
      const std::uint8_t bit = static_cast<std::uint8_t>(
          (payload[byte_position] >> (7U - bit_in_byte)) & 1U);
      ++bit_position;
      code = code * 2U + bit;

      const std::uint64_t first = first_codes_[bits];
      const std::uint32_t count = length_counts_[bits];
      if (code >= first && code - first < count) {
        const std::uint64_t offset =
            static_cast<std::uint64_t>(first_symbol_offsets_[bits]) +
            (code - first);
        decoded = symbols_by_length_[static_cast<std::size_t>(offset)];
        found = true;
        break;
      }
    }
    if (!found) {
      return Status::DataLoss("Huffman payload contains an invalid codeword");
    }
  }

  const std::size_t required_bytes = (bit_position + 7U) / 8U;
  if (payload.size() != required_bytes) {
    return Status::DataLoss("Huffman payload contains trailing bytes");
  }
  const std::uint32_t padding =
      static_cast<std::uint32_t>(required_bytes * 8U - bit_position);
  if (padding != 0U) {
    const std::uint8_t padding_mask = static_cast<std::uint8_t>(
        (static_cast<std::uint32_t>(1U) << padding) - 1U);
    if ((payload.back() & padding_mask) != 0U) {
      return Status::DataLoss("Huffman payload has nonzero padding bits");
    }
  }
  return Status::Ok();
}

}  // namespace stlqpp
