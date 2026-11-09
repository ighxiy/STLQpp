#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace stlqpp {

// Immutable file with thread-safe positioned reads. Consumers own buffer
// lifetime; this platform boundary owns native handles and exact-range I/O.
class ReadOnlyPositionedFile final {
 public:
  ReadOnlyPositionedFile();
  ~ReadOnlyPositionedFile();

  ReadOnlyPositionedFile(const ReadOnlyPositionedFile&) = delete;
  ReadOnlyPositionedFile& operator=(const ReadOnlyPositionedFile&) = delete;
  ReadOnlyPositionedFile(ReadOnlyPositionedFile&& other) noexcept;
  ReadOnlyPositionedFile& operator=(ReadOnlyPositionedFile&& other) noexcept;

  [[nodiscard]] static StatusOr<ReadOnlyPositionedFile> Open(
      const std::filesystem::path& path);

  [[nodiscard]] Status ReadAt(
      std::uint64_t offset,
      ArrayView<std::uint8_t> bytes) const;
  [[nodiscard]] std::uint64_t size() const noexcept;

 private:
  struct State;
  explicit ReadOnlyPositionedFile(std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

// Pre-sized file with thread-safe disjoint positioned writes. Algorithms own
// the non-overlap contract; this platform boundary owns only native file APIs.
class PositionedFile final {
 public:
  PositionedFile();
  ~PositionedFile();

  PositionedFile(const PositionedFile&) = delete;
  PositionedFile& operator=(const PositionedFile&) = delete;
  PositionedFile(PositionedFile&& other) noexcept;
  PositionedFile& operator=(PositionedFile&& other) noexcept;

  [[nodiscard]] static StatusOr<PositionedFile> Create(
      const std::filesystem::path& path,
      std::uint64_t byte_count);

  [[nodiscard]] Status WriteAt(
      std::uint64_t offset,
      ArrayView<const std::uint8_t> bytes) const;
  [[nodiscard]] Status Flush() const;
  [[nodiscard]] std::uint64_t size() const noexcept;

 private:
  struct State;
  explicit PositionedFile(std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

}  // namespace stlqpp
