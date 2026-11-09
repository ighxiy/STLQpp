#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/training/basic/runtime_basic_encoding.h"

namespace stlqpp {

namespace detail {
struct BasicEncodingAccess;
}  // namespace detail

template <int M>
class BasicEncoding final {
 public:
  static_assert(M >= 2 && M <= 20);

  BasicEncoding(BasicEncoding&&) noexcept = default;
  BasicEncoding& operator=(BasicEncoding&&) noexcept = default;
  BasicEncoding(const BasicEncoding&) = delete;
  BasicEncoding& operator=(const BasicEncoding&) = delete;

  [[nodiscard]] std::uint32_t sample_count() const noexcept {
    return sample_count_;
  }
  [[nodiscard]] std::uint32_t root_code(std::uint32_t sample) const noexcept {
    assert(sample < sample_count_);
    return root_codes_[sample];
  }
  [[nodiscard]] std::uint8_t small_code(std::uint32_t sample,
                                        std::uint32_t layer) const noexcept {
    assert(sample < sample_count_);
    assert(layer > 0U && layer < static_cast<std::uint32_t>(M));
    return small_codes_[static_cast<std::size_t>(sample) * (M - 1) +
                        (layer - 1U)];
  }
  [[nodiscard]] float coefficient(std::uint32_t sample,
                                  std::uint32_t layer) const noexcept {
    assert(sample < sample_count_);
    assert(layer < static_cast<std::uint32_t>(M));
    return coefficients_[static_cast<std::size_t>(sample) * M + layer];
  }
  [[nodiscard]] float squared_error(std::uint32_t sample) const noexcept {
    assert(sample < sample_count_);
    return squared_errors_[sample];
  }
  [[nodiscard]] ArrayView<const std::uint32_t> root_codes() const noexcept {
    return root_codes_;
  }
  [[nodiscard]] ArrayView<const std::uint8_t> small_codes_row_major()
      const noexcept {
    return small_codes_;
  }
  [[nodiscard]] ArrayView<const float> coefficients_row_major() const noexcept {
    return coefficients_;
  }
  [[nodiscard]] ArrayView<const float> squared_errors() const noexcept {
    return squared_errors_;
  }
  [[nodiscard]] ArrayView<const float> initial_squared_errors() const noexcept {
    return initial_squared_errors_;
  }

 private:
  friend struct detail::BasicEncodingAccess;

  explicit BasicEncoding(std::uint32_t sample_count)
      : sample_count_(sample_count),
        root_codes_(sample_count),
        small_codes_(static_cast<std::size_t>(sample_count) * (M - 1)),
        coefficients_(static_cast<std::size_t>(sample_count) * M),
        initial_squared_errors_(sample_count),
        squared_errors_(sample_count) {}

  explicit BasicEncoding(RuntimeBasicEncoding encoding) noexcept
      : sample_count_(static_cast<std::uint32_t>(encoding.roots.size())),
        root_codes_(std::move(encoding.roots)),
        small_codes_(std::move(encoding.codes)),
        coefficients_(std::move(encoding.coefficients)),
        initial_squared_errors_(std::move(encoding.raw_errors)),
        squared_errors_(std::move(encoding.errors)) {}

  std::uint32_t sample_count_ = 0U;
  std::vector<std::uint32_t> root_codes_;
  std::vector<std::uint8_t> small_codes_;
  std::vector<float> coefficients_;
  std::vector<float> initial_squared_errors_;
  std::vector<float> squared_errors_;
};

// Width-independent ownership at the executor/pipeline boundary. Arrays retain
// their actual M/M-1 strides; transport does not allocate or convert rows.
class BasicEncodingBatch {
 public:
  explicit BasicEncodingBatch(RuntimeBasicEncoding encoding)
      : encoding_(std::move(encoding)) {}
  BasicEncodingBatch(BasicEncodingBatch&&) noexcept = default;
  BasicEncodingBatch& operator=(BasicEncodingBatch&&) noexcept = default;
  BasicEncodingBatch(const BasicEncodingBatch&) = delete;
  BasicEncodingBatch& operator=(const BasicEncodingBatch&) = delete;
  [[nodiscard]] std::uint32_t sample_count() const noexcept {
    return static_cast<std::uint32_t>(encoding_.roots.size());
  }
  [[nodiscard]] std::uint32_t m() const noexcept { return encoding_.m; }
  [[nodiscard]] std::uint32_t root_code(std::uint32_t sample) const noexcept {
    assert(sample < sample_count());
    return encoding_.roots[sample];
  }
  [[nodiscard]] std::uint8_t small_code(std::uint32_t sample,
                                        std::uint32_t layer) const noexcept {
    assert(sample < sample_count() && layer > 0U && layer < m());
    return encoding_
        .codes[static_cast<std::size_t>(sample) * (m() - 1U) + layer - 1U];
  }
  [[nodiscard]] float coefficient(std::uint32_t sample,
                                  std::uint32_t layer) const noexcept {
    assert(sample < sample_count() && layer < m());
    return encoding_
        .coefficients[static_cast<std::size_t>(sample) * m() + layer];
  }
  [[nodiscard]] float squared_error(std::uint32_t sample) const noexcept {
    assert(sample < sample_count());
    return encoding_.errors[sample];
  }
  [[nodiscard]] ArrayView<const std::uint32_t> root_codes() const noexcept {
    return encoding_.roots;
  }
  [[nodiscard]] ArrayView<const std::uint8_t> small_codes_row_major()
      const noexcept {
    return encoding_.codes;
  }
  [[nodiscard]] ArrayView<const float> coefficients_row_major() const noexcept {
    return encoding_.coefficients;
  }
  [[nodiscard]] ArrayView<const float> squared_errors() const noexcept {
    return encoding_.errors;
  }
  [[nodiscard]] ArrayView<const float> initial_squared_errors() const noexcept {
    return encoding_.raw_errors;
  }

 private:
  friend struct detail::BasicEncodingAccess;
  RuntimeBasicEncoding encoding_;
};

// Runtime-width numerical adapter; fixed numerical encoders keep their existing
// BasicEncoding<M> accessors and compile-time row strides.
template <>
class BasicEncoding<0> final : public BasicEncodingBatch {
 public:
  using BasicEncodingBatch::BasicEncodingBatch;
};

}  // namespace stlqpp
