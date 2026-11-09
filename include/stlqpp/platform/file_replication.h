#pragma once

#include "stlqpp/core/status_or.h"

#include <filesystem>

namespace stlqpp {

enum class FileReplicationMethod {
  kCopy,
  kCopyOnWriteClone,
  kHardLink,
};

// Creates destination from source without replacing an existing file. On
// platforms/filesystems that support copy-on-write cloning, unchanged payload
// blocks are shared until either file is modified. Otherwise this falls back
// to a portable byte copy.
[[nodiscard]] StatusOr<FileReplicationMethod> ReplicateFile(
    const std::filesystem::path& source,
    const std::filesystem::path& destination);

// Shares an immutable source inode when the destination filesystem permits
// it. Canonical shards are create-only, so deletion or renaming of either
// pathname does not invalidate the other. Cross-device and unsupported cases
// retain the clone/copy fallback above.
[[nodiscard]] StatusOr<FileReplicationMethod> ReplicateImmutableFile(
    const std::filesystem::path& source,
    const std::filesystem::path& destination);

}  // namespace stlqpp
