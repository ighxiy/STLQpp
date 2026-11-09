#include "stlqpp/platform/cpu_features.h"

#if defined(_MSC_VER)
#include <intrin.h>
#elif !defined(__GNUC__)
#error "STLQpp CPU feature detection requires MSVC or GCC"
#endif

#include <cstdint>

namespace stlqpp {

CpuFeatures DetectCpuFeatures() noexcept {
#if defined(_MSC_VER)
  int registers[4] = {};
  __cpuidex(registers, 0, 0);
  const int maximum_leaf = registers[0];
  if (maximum_leaf < 1) {
    return {};
  }
  __cpuidex(registers, 1, 0);
  constexpr int kFmaBit = 1 << 12;
  constexpr int kOsxsaveBit = 1 << 27;
  constexpr int kAvxBit = 1 << 28;
  const bool cpu_avx = (registers[2] & kAvxBit) != 0;
  const bool osxsave = (registers[2] & kOsxsaveBit) != 0;
  const bool cpu_fma = (registers[2] & kFmaBit) != 0;
  const std::uint64_t xcr0 = osxsave ? _xgetbv(0) : 0U;
  const bool avx_register_state =
      cpu_avx && osxsave && (xcr0 & 0x6U) == 0x6U;
  bool avx2 = false;
  if (maximum_leaf >= 7) {
    __cpuidex(registers, 7, 0);
    constexpr int kAvx2Bit = 1 << 5;
    avx2 = avx_register_state && (registers[1] & kAvx2Bit) != 0;
  }
  return {
      .avx_register_state = avx_register_state,
      .avx2 = avx2,
      .fma = avx_register_state && cpu_fma,
  };
#else
  __builtin_cpu_init();
  return {
      .avx_register_state = __builtin_cpu_supports("avx") != 0,
      .avx2 = __builtin_cpu_supports("avx2") != 0,
      .fma = __builtin_cpu_supports("fma") != 0,
  };
#endif
}

}  // namespace stlqpp
