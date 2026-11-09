#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "stlqpp/model/model_shape.h"

namespace stlqpp {

namespace detail {
struct LinkageEncodingAccess;
}

template <int M>
class CpuLinkageRsdEncoder;

template <int M>
class LinkageEncoding final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  [[nodiscard]] std::uint32_t m() const noexcept {
    if constexpr (M == 0)
      return m_;
    else
      return M;
  }

  LinkageEncoding(LinkageEncoding&&) noexcept = default;
  LinkageEncoding& operator=(LinkageEncoding&&) noexcept = default;
  LinkageEncoding(const LinkageEncoding&) = delete;
  LinkageEncoding& operator=(const LinkageEncoding&) = delete;

  [[nodiscard]] std::uint32_t sample_count() const noexcept {
    return sample_count_;
  }
  [[nodiscard]] std::uint8_t code(std::uint32_t sample,
                                  std::uint32_t layer) const noexcept {
    assert(sample < sample_count_);
    assert(layer < m());
    return codes_[static_cast<std::size_t>(sample) * m() + layer];
  }
  [[nodiscard]] float coefficient(std::uint32_t sample,
                                  std::uint32_t layer) const noexcept {
    assert(sample < sample_count_);
    assert(layer < m());
    return coefficients_[static_cast<std::size_t>(sample) * m() + layer];
  }
  [[nodiscard]] float squared_error(std::uint32_t sample) const noexcept {
    assert(sample < sample_count_);
    return squared_errors_[sample];
  }

 private:
  friend class CpuLinkageRsdEncoder<M>;
  friend struct detail::LinkageEncodingAccess;

  explicit LinkageEncoding(std::uint32_t sample_count,
                           std::uint32_t m = FixedModelLayers<M>())
      : m_(m),
        sample_count_(sample_count),
        codes_(static_cast<std::size_t>(sample_count) * m),
        coefficients_(static_cast<std::size_t>(sample_count) * m),
        squared_errors_(sample_count) {}

  std::uint32_t m_ = M;
  std::uint32_t sample_count_ = 0U;
  std::vector<std::uint8_t> codes_;
  std::vector<float> coefficients_;
  std::vector<float> squared_errors_;
};

}  // namespace stlqpp
