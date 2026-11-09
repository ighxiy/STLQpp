#pragma once

#include <cstdint>
#include <cstring>
#include <limits>

namespace stlqpp {

static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4U);

[[nodiscard]] inline std::uint32_t LoadLittleEndianU32(
    const std::uint8_t* source) noexcept {
  std::uint32_t value = 0U;
  for (std::uint32_t byte = 0U; byte < 4U; ++byte) {
    value |= static_cast<std::uint32_t>(source[byte]) << (8U * byte);
  }
  return value;
}

[[nodiscard]] inline std::uint64_t LoadLittleEndianU64(
    const std::uint8_t* source) noexcept {
  std::uint64_t value = 0U;
  for (std::uint32_t byte = 0U; byte < 8U; ++byte) {
    value |= static_cast<std::uint64_t>(source[byte]) << (8U * byte);
  }
  return value;
}

[[nodiscard]] inline float LoadLittleEndianF32(
    const std::uint8_t* source) noexcept {
  const std::uint32_t bits = LoadLittleEndianU32(source);
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline void StoreLittleEndianU32(std::uint32_t value,
                                 std::uint8_t* destination) noexcept {
  for (std::uint32_t byte = 0U; byte < 4U; ++byte) {
    destination[byte] =
        static_cast<std::uint8_t>((value >> (8U * byte)) & 0xffU);
  }
}

inline void StoreLittleEndianU64(std::uint64_t value,
                                 std::uint8_t* destination) noexcept {
  for (std::uint32_t byte = 0U; byte < 8U; ++byte) {
    destination[byte] =
        static_cast<std::uint8_t>((value >> (8U * byte)) & 0xffU);
  }
}

inline void StoreLittleEndianF32(float value,
                                 std::uint8_t* destination) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  StoreLittleEndianU32(bits, destination);
}

}  // namespace stlqpp
