#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "stlqpp/model/linkage_code_tuple.h"
#include "stlqpp/model/model_shape.h"

namespace stlqpp {

template <int M>
class CpuRootOnlyLinkageEncoder;

template <int M>
class RootOnlyLinkageEncoding final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  [[nodiscard]] std::uint32_t m() const noexcept {
    if constexpr (M == 0)
      return m_;
    else
      return M;
  }

  RootOnlyLinkageEncoding(RootOnlyLinkageEncoding&&) noexcept = default;
  RootOnlyLinkageEncoding& operator=(RootOnlyLinkageEncoding&&) noexcept =
      default;
  RootOnlyLinkageEncoding(const RootOnlyLinkageEncoding&) = delete;
  RootOnlyLinkageEncoding& operator=(const RootOnlyLinkageEncoding&) = delete;

  [[nodiscard]] std::uint32_t sample_count() const noexcept {
    return sample_count_;
  }
  [[nodiscard]] const LinkageCodeTuple<M>& support(
      std::uint32_t sample) const noexcept {
    assert(sample < sample_count_);
    return supports_[sample];
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
  friend class CpuRootOnlyLinkageEncoder<M>;

  explicit RootOnlyLinkageEncoding(std::uint32_t sample_count,
                                   std::uint32_t m = FixedModelLayers<M>())
      : m_(m),
        sample_count_(sample_count),
        supports_(sample_count),
        coefficients_(static_cast<std::size_t>(sample_count) * m),
        squared_errors_(sample_count) {}

  std::uint32_t m_ = M;
  std::uint32_t sample_count_ = 0U;
  std::vector<LinkageCodeTuple<M>> supports_;
  std::vector<float> coefficients_;
  std::vector<float> squared_errors_;
};

}  // namespace stlqpp
