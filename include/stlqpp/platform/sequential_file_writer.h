#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace stlqpp {

// Portable append-only file boundary. It owns partial-write handling and a
// monotonically increasing u64 cursor; record layout remains an algorithm
// concern. Close is explicit so publication never depends on a destructor to
// report an I/O failure.
class SequentialFileWriter final {
 public:
  SequentialFileWriter();
  ~SequentialFileWriter();
  SequentialFileWriter(SequentialFileWriter&&) noexcept;
  SequentialFileWriter& operator=(SequentialFileWriter&&) noexcept;
  SequentialFileWriter(const SequentialFileWriter&) = delete;
  SequentialFileWriter& operator=(const SequentialFileWriter&) = delete;

  [[nodiscard]] static StatusOr<SequentialFileWriter> Create(
      const std::filesystem::path& path);
  // Reopens an append-only checkpoint payload at its last committed byte.
  // Any uncommitted tail is truncated before the cursor is exposed.
  [[nodiscard]] static StatusOr<SequentialFileWriter> OpenAppend(
      const std::filesystem::path& path, std::uint64_t committed_bytes);

  [[nodiscard]] Status Append(ArrayView<const std::uint8_t> bytes);
  [[nodiscard]] Status Flush();
  [[nodiscard]] Status Close();
  [[nodiscard]] std::uint64_t size() const noexcept;

 private:
  struct State;
  explicit SequentialFileWriter(std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

}  // namespace stlqpp
