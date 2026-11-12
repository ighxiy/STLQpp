#include "stlqpp/platform/sequential_file_writer.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace stlqpp {

struct SequentialFileWriter::State final {
  HANDLE file = INVALID_HANDLE_VALUE;
  std::uint64_t bytes = 0U;
  std::filesystem::path path;

  ~State() {
    if (file != INVALID_HANDLE_VALUE) {
      CloseHandle(file);
    }
  }
};

namespace {

Status NativeError(const char* operation,
                   const std::filesystem::path& path) {
  return Status::Internal(std::string(operation) + " failed for " +
                          path.string() + ": GetLastError=" +
                          std::to_string(GetLastError()));
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
  state->file = CreateFileW(
      path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
      FILE_ATTRIBUTE_NORMAL, nullptr);
  if (state->file == INVALID_HANDLE_VALUE) {
    return NativeError("CreateFileW", path);
  }
  state->path = path;
  return SequentialFileWriter(std::move(state));
}

StatusOr<SequentialFileWriter> SequentialFileWriter::OpenAppend(
    const std::filesystem::path& path, std::uint64_t committed_bytes) {
  if (path.empty() || committed_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<LONGLONG>::max())) {
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
  state->file = CreateFileW(
      path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
      FILE_ATTRIBUTE_NORMAL, nullptr);
  if (state->file == INVALID_HANDLE_VALUE) {
    return NativeError("CreateFileW", path);
  }
  LARGE_INTEGER offset;
  offset.QuadPart = static_cast<LONGLONG>(committed_bytes);
  if (!SetFilePointerEx(state->file, offset, nullptr, FILE_BEGIN) ||
      !SetEndOfFile(state->file)) {
    return NativeError("SetFilePointerEx/SetEndOfFile", path);
  }
  state->bytes = committed_bytes;
  state->path = path;
  return SequentialFileWriter(std::move(state));
}

Status SequentialFileWriter::Append(
    ArrayView<const std::uint8_t> bytes) {
  if (state_ == nullptr || state_->file == INVALID_HANDLE_VALUE ||
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
  constexpr std::size_t kMaximumWrite = 1U << 30U;
  while (remaining != 0U) {
    const DWORD request =
        static_cast<DWORD>(std::min(remaining, kMaximumWrite));
    DWORD written = 0U;
    if (!WriteFile(state_->file, cursor, request, &written, nullptr)) {
      return NativeError("WriteFile", state_->path);
    }
    if (written == 0U) {
      return Status::Internal("WriteFile made no forward progress");
    }
    cursor += written;
    remaining -= written;
  }
  state_->bytes += bytes.size();
  return Status::Ok();
}

Status SequentialFileWriter::Flush() {
  if (state_ == nullptr || state_->file == INVALID_HANDLE_VALUE) {
    return Status::FailedPrecondition(
        "sequential file cannot flush a closed writer");
  }
  if (!FlushFileBuffers(state_->file)) {
    return NativeError("FlushFileBuffers", state_->path);
  }
  return Status::Ok();
}

Status SequentialFileWriter::Close() {
  if (state_ == nullptr || state_->file == INVALID_HANDLE_VALUE) {
    return Status::FailedPrecondition(
        "sequential file cannot be closed twice");
  }
  if (!CloseHandle(state_->file)) {
    state_->file = INVALID_HANDLE_VALUE;
    return NativeError("CloseHandle", state_->path);
  }
  state_->file = INVALID_HANDLE_VALUE;
  return Status::Ok();
}

std::uint64_t SequentialFileWriter::size() const noexcept {
  return state_ == nullptr ? 0U : state_->bytes;
}

}  // namespace stlqpp
