#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>

namespace stlqpp {

// Owning cold-path representation produced before a stream is appended to the
// canonical index. Empty input has an all-zero length table and no payload.
struct EncodedCanonicalHuffman final {
  std::vector<std::uint8_t> code_lengths;
  std::vector<std::uint8_t> payload;
};

// Builds the same deterministic canonical code used by the decoder below and
// emits an MSB-first, minimally padded payload. Equal-frequency tree nodes are
// ordered by their smallest symbol, making the result independent of worker
// scheduling and standard-library heap implementation details.
[[nodiscard]] StatusOr<EncodedCanonicalHuffman> EncodeCanonicalHuffmanMsbFirst(
    ArrayView<const std::uint8_t> symbols,
    std::uint32_t alphabet_size);

// Validated cold-path decoder for the canonical, MSB-first Huffman streams
// stored by the registered scalar coefficient codec. Query kernels never see
// this object: coefficients are decoded once while the resident index loads.
class CanonicalHuffmanDecoder final {
 public:
  [[nodiscard]] static StatusOr<CanonicalHuffmanDecoder> Create(
      ArrayView<const std::uint8_t> code_lengths);

  // Decodes exactly output.size() symbols and requires the payload to be the
  // minimal byte representation emitted by the encoder. The final byte may
  // contain only zero padding bits.
  [[nodiscard]] Status DecodeMsbFirst(
      ArrayView<const std::uint8_t> payload,
      ArrayView<std::uint8_t> output) const;

  [[nodiscard]] std::uint32_t alphabet_size() const noexcept {
    return alphabet_size_;
  }
  [[nodiscard]] std::uint32_t max_code_length() const noexcept {
    return max_code_length_;
  }

 private:
  static constexpr std::uint32_t kFastBits = 8U;

  std::uint32_t alphabet_size_ = 0U;
  std::uint32_t max_code_length_ = 0U;
  std::vector<std::uint32_t> length_counts_;
  std::vector<std::uint64_t> first_codes_;
  std::vector<std::uint32_t> first_symbol_offsets_;
  std::vector<std::uint8_t> symbols_by_length_;
  std::array<std::uint8_t, 1U << kFastBits> fast_symbols_{};
  std::array<std::uint8_t, 1U << kFastBits> fast_lengths_{};
};

}  // namespace stlqpp
