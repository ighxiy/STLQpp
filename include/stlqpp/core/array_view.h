#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace stlqpp {

// Non-owning contiguous range for supported compilers whose C++20 library
// predates std::span. Ownership never crosses this boundary.
template <typename T>
class ArrayView final {
 public:
  using element_type = T;
  using iterator = T*;

  constexpr ArrayView() noexcept = default;
  constexpr ArrayView(T* data, std::size_t size) noexcept
      : data_(data), size_(size) {}

  template <typename U, std::size_t N,
            std::enable_if_t<std::is_convertible_v<U*, T*>, int> = 0>
  constexpr ArrayView(std::array<U, N>& values) noexcept
      : data_(values.data()), size_(N) {}

  template <typename U, std::size_t N,
            std::enable_if_t<std::is_convertible_v<const U*, T*>, int> = 0>
  constexpr ArrayView(const std::array<U, N>& values) noexcept
      : data_(values.data()), size_(N) {}

  template <typename U, typename Allocator,
            std::enable_if_t<std::is_convertible_v<U*, T*>, int> = 0>
  ArrayView(std::vector<U, Allocator>& values) noexcept
      : data_(values.data()), size_(values.size()) {}

  template <typename U, typename Allocator,
            std::enable_if_t<std::is_convertible_v<const U*, T*>, int> = 0>
  ArrayView(const std::vector<U, Allocator>& values) noexcept
      : data_(values.data()), size_(values.size()) {}

  [[nodiscard]] constexpr T* data() const noexcept { return data_; }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0U; }
  [[nodiscard]] constexpr T& operator[](std::size_t index) const noexcept {
    return data_[index];
  }
  [[nodiscard]] constexpr T& back() const noexcept {
    return data_[size_ - 1U];
  }
  [[nodiscard]] constexpr iterator begin() const noexcept { return data_; }
  [[nodiscard]] constexpr iterator end() const noexcept {
    return size_ == 0U ? data_ : data_ + size_;
  }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0U;
};

}  // namespace stlqpp
