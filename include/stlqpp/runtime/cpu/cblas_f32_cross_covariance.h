#pragma once

#include "stlqpp/runtime/cpu/f32_cross_covariance.h"

namespace stlqpp {

class CblasF32CrossCovariance final : public F32CrossCovarianceBackend {
 public:
  CblasF32CrossCovariance() = default;

  [[nodiscard]] Status Accumulate(
      const CrossCovarianceF32& operation) const override;
};

}  // namespace stlqpp
