#pragma once

#include <cstdint>

namespace stlqpp {

// Canonical storage remains ParentLOUDS in both modes. This choice controls
// only the resident scan representation built while opening the index.
enum class ParentAccessMode : std::uint8_t {
  kMaterialized = 0U,
  kNativeLouds,
};

enum class MaterializedParentWidth : std::uint8_t {
  kNone = 0U,
  kU16,
  kU32,
  kMixed,
};

}  // namespace stlqpp
