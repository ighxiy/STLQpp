#pragma once

#include <cassert>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace stlqpp {

[[nodiscard]] inline std::uint32_t CountTrailingZerosNonZero(
    std::uint64_t value) noexcept {
  assert(value != 0U);
#if defined(_MSC_VER)
  unsigned long index = 0U;
  _BitScanForward64(&index, static_cast<unsigned __int64>(value));
  return static_cast<std::uint32_t>(index);
#else
  return static_cast<std::uint32_t>(__builtin_ctzll(value));
#endif
}

}  // namespace stlqpp
