#pragma once

#include "stlqpp/core/array_view.h"

#include <cstdint>
#include <optional>

namespace stlqpp {

enum class LinkageParentKind : std::uint8_t {
  kVirtual = 0,
  kReal = 1,
};

struct LinkageParentReference final {
  LinkageParentKind kind = LinkageParentKind::kReal;
  std::uint32_t local_id = 0U;

  friend bool operator==(LinkageParentReference left,
                         LinkageParentReference right) noexcept {
    return left.kind == right.kind && left.local_id == right.local_id;
  }
};

enum class LinkageNodeRole : std::uint8_t {
  kRoot = 0,
  kLinked = 1,
};

struct TypedForestView final {
  std::uint32_t virtual_count = 0U;
  ArrayView<const LinkageNodeRole> real_roles;
  ArrayView<const std::optional<LinkageParentReference>> real_parents;
};

}  // namespace stlqpp
