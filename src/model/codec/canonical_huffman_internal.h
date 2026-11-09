#pragma once

#include "stlqpp/model/codec/canonical_huffman.h"

namespace stlqpp::internal {

// Only for the scalar coefficient encoder after it has bounded every
// quantized coefficient to the configured alphabet.
[[nodiscard]] StatusOr<EncodedCanonicalHuffman>
EncodeCanonicalHuffmanMsbFirstTrusted(
    ArrayView<const std::uint8_t> symbols,
    std::uint32_t alphabet_size);

}  // namespace stlqpp::internal
