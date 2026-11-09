#include "stlqpp/platform/positioned_file.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

namespace stlqpp {

struct ReadOnlyPositionedFile::State final {
  int descriptor = -1;
  std::uint64_t bytes = 0U;
  std::filesystem::path path;

  ~State() {
    if (descriptor >= 0) {
      close(descriptor);
    }
  }
};

struct PositionedFile::State final {
  int descriptor = -1;
  std::uint64_t bytes = 0U;
  std::filesystem::path path;

  ~State() {
    if (descriptor >= 0) {
      close(descriptor);
    }
  }
};

namespace {

Status NativeError(const char* operation,
                   const std::filesystem::path& path) {
  return Status::Internal(std::string(operation) + " failed for " +
                          path.string() + ": errno=" +
                          std::to_string(errno) + " (" +
                          std::strerror(errno) + ")");
}

}  // namespace

ReadOnlyPositionedFile::ReadOnlyPositionedFile() = default;
ReadOnlyPositionedFile::~ReadOnlyPositionedFile() = default;
ReadOnlyPositionedFile::ReadOnlyPositionedFile(
    ReadOnlyPositionedFile&& other) noexcept = default;
ReadOnlyPositionedFile& ReadOnlyPositionedFile::operator=(
    ReadOnlyPositionedFile&& other) noexcept = default;

ReadOnlyPositionedFile::ReadOnlyPositionedFile(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

StatusOr<ReadOnlyPositionedFile> ReadOnlyPositionedFile::Open(
    const std::filesystem::path& path) {
  if (path.empty()) {
    return Status::InvalidArgument(
        "read-only positioned file path must not be empty");
  }
  auto state = std::make_unique<State>();
  state->descriptor = open(path.c_str(), O_RDONLY);
  if (state->descriptor < 0) {
    return NativeError("open", path);
  }
  struct stat metadata {};
  if (fstat(state->descriptor, &metadata) != 0) {
    return NativeError("fstat", path);
  }
  if (metadata.st_size < 0) {
    return Status::DataLoss("positioned file reports a negative size");
  }
  state->path = path;
  state->bytes = static_cast<std::uint64_t>(metadata.st_size);
  return ReadOnlyPositionedFile(std::move(state));
}

Status ReadOnlyPositionedFile::ReadAt(
    std::uint64_t offset,
    ArrayView<std::uint8_t> bytes) const {
  if (state_ == nullptr || offset > state_->bytes ||
      bytes.size() > state_->bytes - offset) {
    return Status::InvalidArgument("positioned read exceeds the file");
  }
  std::uint8_t* cursor = bytes.data();
  std::size_t remaining = bytes.size();
  std::uint64_t position = offset;
  constexpr std::size_t kMaxRead =
      static_cast<std::size_t>(std::numeric_limits<ssize_t>::max());
  while (remaining != 0U) {
    const std::size_t request = std::min(remaining, kMaxRead);
    const ssize_t read = pread(state_->descriptor, cursor, request,
                               static_cast<off_t>(position));
    if (read < 0) {
      if (errno == EINTR) {
        continue;
      }
      return NativeError("pread", state_->path);
    }
    if (read == 0) {
      return Status::DataLoss("pread reached an unexpected end of file");
    }
    const std::size_t advanced = static_cast<std::size_t>(read);
    cursor += advanced;
    remaining -= advanced;
    position += advanced;
  }
  return Status::Ok();
}

std::uint64_t ReadOnlyPositionedFile::size() const noexcept {
  return state_ == nullptr ? 0U : state_->bytes;
}

PositionedFile::PositionedFile() = default;
PositionedFile::~PositionedFile() = default;
PositionedFile::PositionedFile(PositionedFile&& other) noexcept = default;
PositionedFile& PositionedFile::operator=(PositionedFile&& other) noexcept =
    default;

PositionedFile::PositionedFile(std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

StatusOr<PositionedFile> PositionedFile::Create(
    const std::filesystem::path& path,
    std::uint64_t byte_count) {
  if (path.empty()) {
    return Status::InvalidArgument("positioned file path must not be empty");
  }
  if (byte_count >
      static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
    return Status::ResourceExhausted(
        "positioned file exceeds the POSIX offset domain");
  }
  auto state = std::make_unique<State>();
  state->descriptor =
      open(path.c_str(), O_CREAT | O_TRUNC | O_RDWR, S_IRUSR | S_IWUSR |
                                                    S_IRGRP | S_IROTH);
  if (state->descriptor < 0) {
    return NativeError("open", path);
  }
  state->path = path;
  state->bytes = byte_count;
  if (ftruncate(state->descriptor, static_cast<off_t>(byte_count)) != 0) {
    return NativeError("ftruncate", path);
  }
  return PositionedFile(std::move(state));
}

Status PositionedFile::WriteAt(
    std::uint64_t offset,
    ArrayView<const std::uint8_t> bytes) const {
  if (state_ == nullptr || offset > state_->bytes ||
      bytes.size() > state_->bytes - offset) {
    return Status::InvalidArgument(
        "positioned write exceeds the pre-sized file");
  }
  const std::uint8_t* cursor = bytes.data();
  std::size_t remaining = bytes.size();
  std::uint64_t position = offset;
  constexpr std::size_t kMaxWrite =
      static_cast<std::size_t>(std::numeric_limits<ssize_t>::max());
  while (remaining != 0U) {
    const std::size_t request = std::min(remaining, kMaxWrite);
    const ssize_t written = pwrite(state_->descriptor, cursor, request,
                                   static_cast<off_t>(position));
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return NativeError("pwrite", state_->path);
    }
    if (written == 0) {
      return Status::Internal("pwrite made no forward progress");
    }
    const std::size_t advanced = static_cast<std::size_t>(written);
    cursor += advanced;
    remaining -= advanced;
    position += advanced;
  }
  return Status::Ok();
}

Status PositionedFile::Flush() const {
  if (state_ == nullptr) {
    return Status::FailedPrecondition("positioned file is not open");
  }
  if (fsync(state_->descriptor) != 0) {
    return NativeError("fsync", state_->path);
  }
  return Status::Ok();
}

std::uint64_t PositionedFile::size() const noexcept {
  return state_ == nullptr ? 0U : state_->bytes;
}

}  // namespace stlqpp
