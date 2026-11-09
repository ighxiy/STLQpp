#include "stlqpp/platform/mapped_file.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace stlqpp {

struct ReadOnlyMappedFile::State final {
  int descriptor = -1;
  void* address = nullptr;
  std::size_t bytes = 0U;

  ~State() {
    if (address != nullptr) {
      munmap(address, bytes);
    }
    if (descriptor >= 0) {
      close(descriptor);
    }
  }
};

namespace {

[[nodiscard]] Status NativeError(const char* operation,
                                 const std::filesystem::path& path) {
  return Status::DataLoss(std::string(operation) + " failed for " +
                          path.string() + ": errno=" +
                          std::to_string(errno) + " (" +
                          std::strerror(errno) + ")");
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
  state->descriptor = open(path.c_str(), O_RDONLY);
  if (state->descriptor < 0) {
    return NativeError("open", path);
  }
  struct stat metadata {};
  if (fstat(state->descriptor, &metadata) != 0) {
    return NativeError("fstat", path);
  }
  if (metadata.st_size < 0 ||
      static_cast<std::uint64_t>(metadata.st_size) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted("mapped file exceeds the host address domain");
  }
  state->bytes = static_cast<std::size_t>(metadata.st_size);
  if (state->bytes != 0U) {
    state->address = mmap(nullptr, state->bytes, PROT_READ, MAP_SHARED,
                          state->descriptor, 0);
    if (state->address == MAP_FAILED) {
      state->address = nullptr;
      return NativeError("mmap", path);
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
