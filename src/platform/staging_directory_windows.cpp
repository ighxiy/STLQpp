#include "stlqpp/platform/staging_directory.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <random>
#include <string>
#include <system_error>
#include <utility>

namespace stlqpp {

struct StagingDirectory::State final {
  std::filesystem::path destination;
  std::filesystem::path staging;
  bool durable = true;
  bool published = false;
  bool cleanup_on_failure = true;

  ~State() {
    if (cleanup_on_failure && !published && !staging.empty()) {
      std::error_code error;
      std::filesystem::remove_all(staging, error);
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

std::filesystem::path MakeStagingPath(
    const std::filesystem::path& destination,
    std::uint64_t token) {
  std::filesystem::path name = destination.filename();
  name += L".staging.";
  name += std::to_wstring(token);
  return destination.parent_path() / name;
}

Status SyncFile(const std::filesystem::path& path) {
  const HANDLE file = CreateFileW(
      path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    return NativeError("CreateFileW", path);
  }
  if (!FlushFileBuffers(file)) {
    const Status status = NativeError("FlushFileBuffers", path);
    CloseHandle(file);
    return status;
  }
  CloseHandle(file);
  return Status::Ok();
}

}  // namespace

StagingDirectory::StagingDirectory() = default;
StagingDirectory::~StagingDirectory() = default;
StagingDirectory::StagingDirectory(StagingDirectory&& other) noexcept =
    default;
StagingDirectory& StagingDirectory::operator=(
    StagingDirectory&& other) noexcept = default;

StagingDirectory::StagingDirectory(std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

StatusOr<StagingDirectory> StagingDirectory::Create(
    const std::filesystem::path& destination) {
  return CreateImpl(destination, true);
}

StatusOr<StagingDirectory> StagingDirectory::CreateTransient(
    const std::filesystem::path& destination) {
  return CreateImpl(destination, false);
}

StatusOr<StagingDirectory> StagingDirectory::CreateImpl(
    const std::filesystem::path& destination,
    bool durable) {
  if (destination.empty() || destination.filename().empty()) {
    return Status::InvalidArgument(
        "staging destination must name a directory");
  }
  std::error_code error;
  if (std::filesystem::exists(destination, error) || error) {
    return Status::FailedPrecondition(
        "staging destination already exists or cannot be inspected: " +
        destination.string());
  }
  const std::filesystem::path parent = destination.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, error);
    if (error) {
      return Status::Internal("failed to create staging parent directory");
    }
  }
  auto state = std::make_unique<State>();
  state->destination = destination;
  state->durable = durable;
  std::random_device random;
  for (std::uint32_t attempt = 0U; attempt < 32U; ++attempt) {
    const std::uint64_t token =
        (static_cast<std::uint64_t>(random()) << 32U) ^ random();
    const std::filesystem::path candidate =
        MakeStagingPath(destination, token);
    if (std::filesystem::create_directory(candidate, error)) {
      state->staging = candidate;
      break;
    }
    if (error) {
      return Status::Internal("failed to create sibling staging directory " +
                              candidate.string() + ": " + error.message());
    }
  }
  if (state->staging.empty()) {
    return Status::ResourceExhausted(
        "could not allocate a unique sibling staging directory");
  }
  return StagingDirectory(std::move(state));
}

StatusOr<bool> StagingDirectory::SharesFilesystem(
    const std::filesystem::path& first, const std::filesystem::path& second) {
  const auto existing =
      [](std::filesystem::path path) -> StatusOr<std::filesystem::path> {
    if (path.empty()) path = ".";
    std::error_code error;
    while (!std::filesystem::exists(path, error)) {
      if (error)
        return Status::Internal("cannot inspect publication filesystem: " +
                                error.message());
      path = path.has_parent_path() ? path.parent_path()
                                    : std::filesystem::path(".");
    }
    if (error)
      return Status::Internal("cannot inspect publication filesystem: " +
                              error.message());
    return path;
  };
  auto a = existing(first), b = existing(second);
  if (!a.ok()) return a.status();
  if (!b.ok()) return b.status();
  const auto first_absolute = std::filesystem::absolute(a.value());
  const auto second_absolute = std::filesystem::absolute(b.value());
  wchar_t first_volume[MAX_PATH]{}, second_volume[MAX_PATH]{};
  if (!GetVolumePathNameW(first_absolute.c_str(), first_volume, MAX_PATH))
    return NativeError("GetVolumePathNameW", a.value());
  if (!GetVolumePathNameW(second_absolute.c_str(), second_volume, MAX_PATH))
    return NativeError("GetVolumePathNameW", b.value());
  return _wcsicmp(first_volume, second_volume) == 0;
}

StatusOr<StagingDirectory> StagingDirectory::OpenExisting(
    const std::filesystem::path& staging,
    const std::filesystem::path& destination) {
  std::error_code error;
  if (destination.empty() || destination.filename().empty() ||
      std::filesystem::exists(destination, error) || error)
    return Status::FailedPrecondition(
        "publication destination already exists or is invalid");
  if (!std::filesystem::is_directory(staging, error) || error)
    return Status::DataLoss("restartable publication directory is missing");
  if (!destination.parent_path().empty()) {
    std::filesystem::create_directories(destination.parent_path(), error);
    if (error)
      return Status::Internal("cannot create publication parent: " +
                              error.message());
  }
  auto same_filesystem = SharesFilesystem(staging, destination.parent_path());
  if (!same_filesystem.ok()) return same_filesystem.status();
  if (!same_filesystem.value())
    return Status::FailedPrecondition(
        "atomic publication requires staging on the output filesystem");
  auto state = std::make_unique<State>();
  state->staging = staging;
  state->destination = destination;
  state->cleanup_on_failure = false;
  return StagingDirectory(std::move(state));
}

const std::filesystem::path& StagingDirectory::path() const noexcept {
  static const std::filesystem::path empty;
  return state_ == nullptr ? empty : state_->staging;
}

Status StagingDirectory::Publish() {
  if (state_ == nullptr || state_->published) {
    return Status::FailedPrecondition(
        "staging directory is not publishable");
  }
  std::error_code error;
  if (state_->durable) {
    for (std::filesystem::recursive_directory_iterator iterator(
             state_->staging, error), end;
         iterator != end && !error; iterator.increment(error)) {
      if (iterator->is_regular_file(error)) {
        if (error) {
          break;
        }
        if (const Status status = SyncFile(iterator->path()); !status.ok()) {
          return status;
        }
      }
    }
    if (error) {
      return Status::Internal("failed to enumerate staging files for sync");
    }
  }
  if (!MoveFileExW(state_->staging.c_str(), state_->destination.c_str(),
                   state_->durable ? MOVEFILE_WRITE_THROUGH : 0U)) {
    return NativeError("MoveFileExW", state_->destination);
  }
  state_->published = true;
  return Status::Ok();
}

}  // namespace stlqpp
