#include "stlqpp/platform/file_replication.h"

#include <cerrno>
#include <filesystem>
#include <string>
#include <system_error>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace stlqpp {
namespace {

Status CopyFile(const std::filesystem::path& source,
                const std::filesystem::path& destination) {
  std::error_code error;
  if (!std::filesystem::copy_file(
          source, destination, std::filesystem::copy_options::none, error)) {
    return Status::Internal(
        "failed to copy file from " + source.string() + " to " +
        destination.string() + ": " + error.message());
  }
  return Status::Ok();
}

}  // namespace

StatusOr<FileReplicationMethod> ReplicateFile(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) {
#if defined(__linux__)
  const int input = open(source.c_str(), O_RDONLY | O_CLOEXEC);
  if (input < 0) {
    return Status::Internal(
        "failed to open source file for replication: " + source.string());
  }
  const int output = open(destination.c_str(),
                          O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
  if (output < 0) {
    close(input);
    return Status::Internal(
        "failed to create destination file for replication: " +
        destination.string());
  }
  const bool cloned = ioctl(output, FICLONE, input) == 0;
  const int output_close = close(output);
  const int input_close = close(input);
  if (cloned) {
    if (output_close != 0 || input_close != 0) {
      return Status::Internal("failed to close cloned file descriptors");
    }
    return FileReplicationMethod::kCopyOnWriteClone;
  }
  std::error_code remove_error;
  std::filesystem::remove(destination, remove_error);
  if (remove_error) {
    return Status::Internal(
        "failed to remove unsupported clone destination: " +
        remove_error.message());
  }
#endif
  if (const Status status = CopyFile(source, destination); !status.ok()) {
    return status;
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
