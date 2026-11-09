#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"

#include <filesystem>
#include <string_view>

namespace stlqpp {

// Resolves existing symlink components and normalizes a possibly nonexistent
// tail. This is intended for destructive-operation admission, not for display
// or persisted artifact identity.
[[nodiscard]] StatusOr<std::filesystem::path> ResolveFilesystemPath(
    const std::filesystem::path& path);

// Rejects equality and either ancestor/descendant relation after resolving
// both paths. Component comparison is used; lexical string prefixes are not.
[[nodiscard]] Status ValidateDisjointFilesystemPaths(
    const std::filesystem::path& first,
    std::string_view first_role,
    const std::filesystem::path& second,
    std::string_view second_role);

// A derived index may live in source/derived-index-cache/<identity>. Its
// immutable shards are hard-linked, never overwritten in the source. All
// other overlapping layouts remain forbidden.
[[nodiscard]] Status ValidateIndexDerivationPaths(
    const std::filesystem::path& source,
    const std::filesystem::path& destination);

}  // namespace stlqpp
