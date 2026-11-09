#include "stlqpp/platform/file_replication.h"

#include <filesystem>
#include <string>
#include <system_error>

namespace stlqpp {

StatusOr<FileReplicationMethod> ReplicateFile(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) {
  std::error_code error;
  if (!std::filesystem::copy_file(
          source, destination, std::filesystem::copy_options::none, error)) {
    return Status::Internal(
        "failed to copy file from " + source.string() + " to " +
        destination.string() + ": " + error.message());
  }
  return FileReplicationMethod::kCopy;
}

StatusOr<FileReplicationMethod> ReplicateImmutableFile(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) {
  std::error_code error;
  std::filesystem::create_hard_link(source, destination, error);
  if (!error) {
    return FileReplicationMethod::kHardLink;
  }
  return ReplicateFile(source, destination);
}

}  // namespace stlqpp
