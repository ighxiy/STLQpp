#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"

#include <filesystem>
#include <memory>

namespace stlqpp {

// Owns a sibling staging directory. Persistent publication makes every file
// durable before the atomic rename. Transient publication keeps the same
// all-or-nothing visibility inside one process without forcing temporary
// training worksets to storage. Unpublished staging data is removed by the
// destructor in either case.
class StagingDirectory final {
 public:
  StagingDirectory();
  ~StagingDirectory();

  StagingDirectory(const StagingDirectory&) = delete;
  StagingDirectory& operator=(const StagingDirectory&) = delete;
  StagingDirectory(StagingDirectory&& other) noexcept;
  StagingDirectory& operator=(StagingDirectory&& other) noexcept;

  // Checks existing ancestors without creating either path.
  [[nodiscard]] static StatusOr<bool> SharesFilesystem(
      const std::filesystem::path& first, const std::filesystem::path& second);

  // Borrow a restartable directory. Failure leaves it intact; Publish renames
  // it atomically.
  [[nodiscard]] static StatusOr<StagingDirectory> OpenExisting(
      const std::filesystem::path& staging,
      const std::filesystem::path& destination);
  [[nodiscard]] static StatusOr<StagingDirectory> Create(
      const std::filesystem::path& destination);
  [[nodiscard]] static StatusOr<StagingDirectory> CreateTransient(
      const std::filesystem::path& destination);

  [[nodiscard]] const std::filesystem::path& path() const noexcept;
  [[nodiscard]] Status Publish();

 private:
  struct State;
  explicit StagingDirectory(std::unique_ptr<State> state) noexcept;
  [[nodiscard]] static StatusOr<StagingDirectory> CreateImpl(
      const std::filesystem::path& destination,
      bool durable);

  std::unique_ptr<State> state_;
};

}  // namespace stlqpp
