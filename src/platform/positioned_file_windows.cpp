#include "stlqpp/platform/positioned_file.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace stlqpp {

struct ReadOnlyPositionedFile::State final {
  HANDLE file = INVALID_HANDLE_VALUE;
  std::uint64_t bytes = 0U;
  std::filesystem::path path;

  ~State() {
    if (file != INVALID_HANDLE_VALUE) {
      CloseHandle(file);
    }
  }
};

struct PositionedFile::State final {
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
  const DWORD error = GetLastError();
  return Status::Internal(std::string(operation) + " failed for " +
                          path.string() + ": GetLastError=" +
                          std::to_string(error));
}

Status NativeError(const char* operation,
                   const std::filesystem::path& path,
                   DWORD error) {
  return Status::Internal(std::string(operation) + " failed for " +
                          path.string() + ": GetLastError=" +
                          std::to_string(error));
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
  state->file = CreateFileW(
      path.c_str(), GENERIC_READ,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
  if (state->file == INVALID_HANDLE_VALUE) {
    return NativeError("CreateFileW", path);
  }
  LARGE_INTEGER size{};
  if (!GetFileSizeEx(state->file, &size)) {
    return NativeError("GetFileSizeEx", path);
  }
  if (size.QuadPart < 0) {
    return Status::DataLoss("positioned file reports a negative size");
  }
  state->path = path;
  state->bytes = static_cast<std::uint64_t>(size.QuadPart);
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
  constexpr std::size_t kMaxRead = 1U << 30U;
  while (remaining != 0U) {
    const DWORD request = static_cast<DWORD>(
        std::min(remaining, kMaxRead));
    OVERLAPPED operation{};
    operation.Offset = static_cast<DWORD>(position);
    operation.OffsetHigh = static_cast<DWORD>(position >> 32U);
    operation.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (operation.hEvent == nullptr) {
      return NativeError("CreateEventW", state_->path);
    }
    const BOOL issued = ReadFile(state_->file, cursor, request, nullptr,
                                 &operation);
    if (!issued) {
      const DWORD issue_error = GetLastError();
      if (issue_error != ERROR_IO_PENDING) {
        CloseHandle(operation.hEvent);
        return NativeError("ReadFile", state_->path, issue_error);
      }
    }
    DWORD read = 0U;
    if (!GetOverlappedResult(state_->file, &operation, &read, TRUE)) {
      const DWORD completion_error = GetLastError();
      CloseHandle(operation.hEvent);
      return NativeError(
          "GetOverlappedResult", state_->path, completion_error);
    }
    CloseHandle(operation.hEvent);
    if (read == 0U) {
      return Status::DataLoss("ReadFile reached an unexpected end of file");
    }
    cursor += read;
    remaining -= read;
    position += read;
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
      static_cast<std::uint64_t>(std::numeric_limits<LONGLONG>::max())) {
    return Status::ResourceExhausted(
        "positioned file exceeds the Windows offset domain");
  }
  auto state = std::make_unique<State>();
  state->file = CreateFileW(
      path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr,
      CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
  if (state->file == INVALID_HANDLE_VALUE) {
    return NativeError("CreateFileW", path);
  }
  state->path = path;
  state->bytes = byte_count;
  LARGE_INTEGER end{};
  end.QuadPart = static_cast<LONGLONG>(byte_count);
  if (!SetFilePointerEx(state->file, end, nullptr, FILE_BEGIN) ||
      !SetEndOfFile(state->file)) {
    return NativeError("SetEndOfFile", path);
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
  constexpr std::size_t kMaxWrite = 1U << 30U;
  while (remaining != 0U) {
    const DWORD request = static_cast<DWORD>(
        std::min(remaining, kMaxWrite));
    OVERLAPPED operation{};
    operation.Offset = static_cast<DWORD>(position);
    operation.OffsetHigh = static_cast<DWORD>(position >> 32U);
    operation.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (operation.hEvent == nullptr) {
      return NativeError("CreateEventW", state_->path);
    }
    const BOOL issued = WriteFile(state_->file, cursor, request, nullptr,
                                  &operation);
    if (!issued) {
      const DWORD issue_error = GetLastError();
      if (issue_error != ERROR_IO_PENDING) {
        CloseHandle(operation.hEvent);
        return NativeError("WriteFile", state_->path, issue_error);
      }
    }
    DWORD written = 0U;
    if (!GetOverlappedResult(state_->file, &operation, &written, TRUE)) {
      const DWORD completion_error = GetLastError();
      CloseHandle(operation.hEvent);
      return NativeError(
          "GetOverlappedResult", state_->path, completion_error);
    }
    CloseHandle(operation.hEvent);
    if (written == 0U) {
      return Status::Internal("WriteFile made no forward progress");
    }
    cursor += written;
    remaining -= written;
    position += written;
  }
  return Status::Ok();
}

Status PositionedFile::Flush() const {
  if (state_ == nullptr) {
    return Status::FailedPrecondition("positioned file is not open");
  }
  if (!FlushFileBuffers(state_->file)) {
    return NativeError("FlushFileBuffers", state_->path);
  }
  return Status::Ok();
}

std::uint64_t PositionedFile::size() const noexcept {
  return state_ == nullptr ? 0U : state_->bytes;
}

}  // namespace stlqpp
