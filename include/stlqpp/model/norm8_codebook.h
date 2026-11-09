#pragma once

#include <cstdint>

namespace stlqpp {

// Immutable index-owned reconstruction-norm table. Resident CIDs store only
// one-byte labels and borrow this table when a typed scan view is formed.
struct Norm8CodebookView final {
  const float* centers = nullptr;
  std::uint32_t center_count = 0U;
};

}  // namespace stlqpp
