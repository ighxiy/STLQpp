#include "stlqpp/platform/sequential_file_writer.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

namespace stlqpp {

struct SequentialFileWriter::State final {
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

SequentialFileWriter::SequentialFileWriter() = default;
SequentialFileWriter::~SequentialFileWriter() = default;
SequentialFileWriter::SequentialFileWriter(
    SequentialFileWriter&& other) noexcept = default;
SequentialFileWriter& SequentialFileWriter::operator=(
    SequentialFileWriter&& other) noexcept = default;
SequentialFileWriter::SequentialFileWriter(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

StatusOr<SequentialFileWriter> SequentialFileWriter::Create(
    const std::filesystem::path& path) {
  if (path.empty()) {
    return Status::InvalidArgument(
        "sequential file path must not be empty");
  }
  auto state = std::make_unique<State>();
  state->descriptor = open(
      path.c_str(), O_CREAT | O_TRUNC | O_WRONLY,
      S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (state->descriptor < 0) {
    return NativeError("open", path);
  }
  state->path = path;
  return SequentialFileWriter(std::move(state));
}

StatusOr<SequentialFileWriter> SequentialFileWriter::OpenAppend(
    const std::filesystem::path& path, std::uint64_t committed_bytes) {
  if (path.empty() || committed_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
    return Status::InvalidArgument(
        "sequential append path or cursor is invalid");
  }
  auto state = std::make_unique<State>();
  std::error_code extent_error;
  const std::uint64_t actual =
      std::filesystem::file_size(path, extent_error);
  if (extent_error) {
    return Status::Internal(
        "failed to inspect sequential append file " + path.string());
  }
  if (actual < committed_bytes) {
    return Status::DataLoss(
        "sequential append file is shorter than its committed cursor");
  }
  state->descriptor = open(path.c_str(), O_WRONLY);
  if (state->descriptor < 0) return NativeError("open", path);
  if (ftruncate(state->descriptor, static_cast<off_t>(committed_bytes)) != 0 ||
      lseek(state->descriptor, static_cast<off_t>(committed_bytes), SEEK_SET) <
          0) {
    return NativeError("truncate/seek", path);
  }
  state->bytes = committed_bytes;
  state->path = path;
  return SequentialFileWriter(std::move(state));
}

Status SequentialFileWriter::Append(
    ArrayView<const std::uint8_t> bytes) {
  if (state_ == nullptr || state_->descriptor < 0 ||
      (bytes.data() == nullptr && !bytes.empty())) {
    return Status::FailedPrecondition(
        "sequential file is closed or received invalid bytes");
  }
  if (bytes.size() > std::numeric_limits<std::uint64_t>::max() -
                         state_->bytes) {
    return Status::ResourceExhausted(
        "sequential file cursor overflows u64");
  }
  const std::uint8_t* cursor = bytes.data();
  std::size_t remaining = bytes.size();
  constexpr std::size_t kMaximumWrite =
      static_cast<std::size_t>(std::numeric_limits<ssize_t>::max());
  while (remaining != 0U) {
    const std::size_t request = std::min(remaining, kMaximumWrite);
    const ssize_t written = write(state_->descriptor, cursor, request);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return NativeError("write", state_->path);
    }
    if (written == 0) {
      return Status::Internal("write made no forward progress");
    }
    const std::size_t advanced = static_cast<std::size_t>(written);
    cursor += advanced;
    remaining -= advanced;
  }
  state_->bytes += bytes.size();
  return Status::Ok();
}

Status SequentialFileWriter::Flush() {
  if (state_ == nullptr || state_->descriptor < 0) {
    return Status::FailedPrecondition(
        "sequential file cannot flush a closed writer");
  }
  if (fsync(state_->descriptor) != 0) {
    return NativeError("fsync", state_->path);
  }
  return Status::Ok();
}

Status SequentialFileWriter::Close() {
  if (state_ == nullptr || state_->descriptor < 0) {
    return Status::FailedPrecondition(
        "sequential file cannot be closed twice");
  }
  if (close(state_->descriptor) != 0) {
    state_->descriptor = -1;
    return NativeError("close", state_->path);
  }
  state_->descriptor = -1;
  return Status::Ok();
}

std::uint64_t SequentialFileWriter::size() const noexcept {
  return state_ == nullptr ? 0U : state_->bytes;
}

}  // namespace stlqpp
