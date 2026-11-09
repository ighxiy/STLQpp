#pragma once

#include <cstdint>

#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/linkage/rsd/linkage_encoding.h"

namespace stlqpp::detail {

struct LinkageEncodingAccess final {
  template <int M>
  [[nodiscard]] static LinkageEncoding<M> Create(
      std::uint32_t sample_count, std::uint32_t m = FixedModelLayers<M>()) {
    return LinkageEncoding<M>(sample_count, m);
  }

  template <int M>
  [[nodiscard]] static std::uint8_t* Codes(
      LinkageEncoding<M>* encoding) noexcept {
    return encoding->codes_.data();
  }

  template <int M>
  [[nodiscard]] static float* Coefficients(
      LinkageEncoding<M>* encoding) noexcept {
    return encoding->coefficients_.data();
  }

  template <int M>
  [[nodiscard]] static float* SquaredErrors(
      LinkageEncoding<M>* encoding) noexcept {
    return encoding->squared_errors_.data();
  }
};

}  // namespace stlqpp::detail
