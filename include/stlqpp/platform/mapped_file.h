#pragma once

#include "stlqpp/core/status_or.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace stlqpp {

class ReadOnlyMappedFile final {
 public:
  ReadOnlyMappedFile();
  ~ReadOnlyMappedFile();

  ReadOnlyMappedFile(const ReadOnlyMappedFile&) = delete;
  ReadOnlyMappedFile& operator=(const ReadOnlyMappedFile&) = delete;
  ReadOnlyMappedFile(ReadOnlyMappedFile&& other) noexcept;
  ReadOnlyMappedFile& operator=(ReadOnlyMappedFile&& other) noexcept;

  [[nodiscard]] static StatusOr<ReadOnlyMappedFile> Open(
      const std::filesystem::path& path);

  [[nodiscard]] const std::uint8_t* data() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept { return size() == 0U; }

 private:
  struct State;
  explicit ReadOnlyMappedFile(std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

}  // namespace stlqpp
