#pragma once

#include "stlqpp/core/status_or.h"

#include <filesystem>

namespace stlqpp {

// Allocates the same user-visible date/sequence namespace as DEV's
// MakeUniqueDatedPath. The returned path is not created; the checkpoint
// publisher remains the create-only authority and detects a concurrent race.
[[nodiscard]] StatusOr<std::filesystem::path> NextDatedCheckpointPath(
    const std::filesystem::path& base_path);

}  // namespace stlqpp
