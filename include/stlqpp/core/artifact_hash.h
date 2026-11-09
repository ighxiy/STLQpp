#pragma once

#include <cstdint>
#include <string>

namespace stlqpp {

class ArtifactHash64 final {
 public:
  constexpr ArtifactHash64() noexcept = default;

  [[nodiscard]] static constexpr ArtifactHash64 FromValue(
      std::uint64_t value) noexcept {
    return ArtifactHash64(value);
  }

  [[nodiscard]] constexpr std::uint64_t value() const noexcept {
    return value_;
  }
  [[nodiscard]] constexpr bool empty() const noexcept { return value_ == 0U; }
  [[nodiscard]] std::string ToHex() const;

  friend constexpr bool operator==(ArtifactHash64 lhs,
                                   ArtifactHash64 rhs) noexcept {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(ArtifactHash64 lhs,
                                   ArtifactHash64 rhs) noexcept {
    return !(lhs == rhs);
  }

 private:
  explicit constexpr ArtifactHash64(std::uint64_t value) noexcept
      : value_(value) {}

  std::uint64_t value_ = 0U;
};

}  // namespace stlqpp
