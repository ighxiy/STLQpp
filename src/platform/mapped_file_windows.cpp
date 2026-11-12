#include "stlqpp/platform/mapped_file.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <limits>
#include <string>
#include <utility>

namespace stlqpp {

struct ReadOnlyMappedFile::State final {
  HANDLE file = INVALID_HANDLE_VALUE;
  HANDLE mapping = nullptr;
  void* address = nullptr;
  std::size_t bytes = 0U;

  ~State() {
    if (address != nullptr) {
      UnmapViewOfFile(address);
    }
    if (mapping != nullptr) {
      CloseHandle(mapping);
    }
    if (file != INVALID_HANDLE_VALUE) {
      CloseHandle(file);
    }
  }
};

namespace {

[[nodiscard]] Status NativeError(const char* operation,
                                 const std::filesystem::path& path) {
  return Status::DataLoss(std::string(operation) + " failed for " +
                          path.string() + ": GetLastError=" +
                          std::to_string(GetLastError()));
}

}  // namespace

ReadOnlyMappedFile::ReadOnlyMappedFile() = default;
ReadOnlyMappedFile::~ReadOnlyMappedFile() = default;
ReadOnlyMappedFile::ReadOnlyMappedFile(ReadOnlyMappedFile&& other) noexcept =
    default;
ReadOnlyMappedFile& ReadOnlyMappedFile::operator=(
    ReadOnlyMappedFile&& other) noexcept = default;

ReadOnlyMappedFile::ReadOnlyMappedFile(std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

StatusOr<ReadOnlyMappedFile> ReadOnlyMappedFile::Open(
    const std::filesystem::path& path) {
  auto state = std::make_unique<State>();
  // Permit concurrent flush handles and file replacement, as POSIX opens do.
  // Publication releases mapped views before moving the containing directory.
  state->file = CreateFileW(path.c_str(), GENERIC_READ,
                            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                            nullptr);
  if (state->file == INVALID_HANDLE_VALUE) {
    return NativeError("CreateFileW", path);
  }
  LARGE_INTEGER file_size {};
  if (!GetFileSizeEx(state->file, &file_size)) {
    return NativeError("GetFileSizeEx", path);
  }
  if (file_size.QuadPart < 0 ||
      static_cast<unsigned long long>(file_size.QuadPart) >
          static_cast<unsigned long long>(
              std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted("mapped file exceeds the host address domain");
  }
  state->bytes = static_cast<std::size_t>(file_size.QuadPart);
  if (state->bytes != 0U) {
    state->mapping = CreateFileMappingW(state->file, nullptr, PAGE_READONLY,
                                        0, 0, nullptr);
    if (state->mapping == nullptr) {
      return NativeError("CreateFileMappingW", path);
    }
    state->address = MapViewOfFile(state->mapping, FILE_MAP_READ, 0, 0, 0);
    if (state->address == nullptr) {
      return NativeError("MapViewOfFile", path);
    }
  }
  return ReadOnlyMappedFile(std::move(state));
}

const std::uint8_t* ReadOnlyMappedFile::data() const noexcept {
  return state_ == nullptr
      ? nullptr
      : static_cast<const std::uint8_t*>(state_->address);
}

std::size_t ReadOnlyMappedFile::size() const noexcept {
  return state_ == nullptr ? 0U : state_->bytes;
}

}  // namespace stlqpp
