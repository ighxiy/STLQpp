#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/model/model_shape.h"

#include <cstdint>

namespace stlqpp {

struct SearchOptions final {
  std::uint32_t nprobe = 1;
  std::uint32_t top_k = 100;

  [[nodiscard]] Status Validate(const ModelShape& shape) const;
};

}  // namespace stlqpp
