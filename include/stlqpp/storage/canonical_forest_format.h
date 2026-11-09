#pragma once

#include <cstdint>

namespace stlqpp {

inline constexpr std::uint32_t kCanonicalForestEncodingVersion = 1U;
inline constexpr std::uint32_t kCanonicalPayloadHeaderBytes = 32U;
inline constexpr std::uint32_t kCanonicalDirectoryHeaderBytes = 64U;
inline constexpr std::uint32_t kCanonicalDirectoryEntryBytes = 80U;
inline constexpr std::uint32_t kCompletionOrderDirectoryEncodingVersion = 2U;
inline constexpr std::uint32_t kCompletionOrderDirectoryEntryBytes = 88U;
inline constexpr std::uint32_t kCanonicalBaselineCoefficientAlphabetSize =
    63U;
inline constexpr std::uint32_t kCanonicalHuffmanLengthBits = 6U;
inline constexpr std::uint32_t kCanonicalRecordAlignment = 8U;

[[nodiscard]] constexpr std::uint64_t CanonicalHuffmanLengthBytes(
    std::uint32_t alphabet_size) noexcept {
  return (static_cast<std::uint64_t>(alphabet_size) *
              kCanonicalHuffmanLengthBits +
          7U) /
      8U;
}

[[nodiscard]] constexpr std::uint64_t CanonicalCoefficientHeaderBytes(
    std::uint32_t m,
    std::uint32_t alphabet_size =
        kCanonicalBaselineCoefficientAlphabetSize) noexcept {
  return 8U + 8U * m +
      2U * CanonicalHuffmanLengthBytes(alphabet_size);
}

// Coefficient records carry their alphabet implicitly in the two packed
// code-length extents. Zero means that the record cannot represent any
// supported signed coefficient width.
[[nodiscard]] constexpr std::uint32_t CanonicalCoefficientAlphabetFromRecord(
    std::uint32_t m,
    std::uint64_t record_bytes,
    std::uint64_t root_payload_bytes,
    std::uint64_t linkage_payload_bytes) noexcept {
  const std::uint64_t fixed = 8U + 8U * m;
  if (record_bytes < fixed + root_payload_bytes + linkage_payload_bytes) {
    return 0U;
  }
  const std::uint64_t packed_pair =
      record_bytes - fixed - root_payload_bytes - linkage_payload_bytes;
  std::uint32_t matched = 0U;
  for (std::uint32_t bits = 2U; bits <= 8U; ++bits) {
    const std::uint32_t limit = (1U << (bits - 1U)) - 1U;
    const std::uint32_t alphabet = 2U * limit + 1U;
    if (2U * CanonicalHuffmanLengthBytes(alphabet) == packed_pair) {
      if (matched != 0U) return 0U;
      matched = alphabet;
    }
  }
  return matched;
}

[[nodiscard]] constexpr std::uint64_t AlignCanonicalRecord(
    std::uint64_t offset) noexcept {
  return (offset + kCanonicalRecordAlignment - 1U) &
         ~(static_cast<std::uint64_t>(kCanonicalRecordAlignment) - 1U);
}

struct CanonicalCidEntry final {
  std::uint32_t cid = 0U;
  std::uint32_t real_count = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  std::uint64_t codes_offset = 0U;
  std::uint64_t codes_bytes = 0U;
  std::uint64_t parents_offset = 0U;
  std::uint64_t parents_bytes = 0U;
  std::uint64_t coefficients_offset = 0U;
  std::uint64_t coefficients_bytes = 0U;
  std::uint64_t norm8_offset = 0U;
  std::uint64_t norm8_bytes = 0U;
  std::uint64_t float_coefficients_offset = 0U;  // directory v2
};

}  // namespace stlqpp
