#pragma once

namespace stlqpp {

struct CpuFeatures final {
  bool avx_register_state = false;
  bool avx2 = false;
  bool fma = false;
};

[[nodiscard]] CpuFeatures DetectCpuFeatures() noexcept;

}  // namespace stlqpp
