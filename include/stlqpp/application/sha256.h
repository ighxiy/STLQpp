#pragma once

#include "stlqpp/core/status_or.h"

#include <filesystem>
#include <string>
#include <string_view>

namespace stlqpp::app {

// Portable SHA-256 used only at application identity boundaries. It is not a
// training or search hot-path primitive.
[[nodiscard]] std::string Sha256Bytes(std::string_view bytes);
[[nodiscard]] StatusOr<std::string> Sha256File(
    const std::filesystem::path& path);
// Hashes the executable image of this process through the operating-system
// identity path. This deliberately does not trust argv[0], which may be a
// bare PATH lookup, a symlink spelling or a caller-controlled value.
[[nodiscard]] StatusOr<std::string> Sha256CurrentExecutable();

}  // namespace stlqpp::app
