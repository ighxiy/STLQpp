#pragma once

#include <cstdint>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/model_shape.h"

namespace stlqpp {

// One contiguous array per field. Active strides are M and M-1, not a maximum
// template length and not one heap allocation per row.
struct RuntimeBasicEncoding final {
  std::uint32_t m = 0U;
  std::vector<std::uint32_t> roots;
  std::vector<std::uint8_t> codes;
  std::vector<float> coefficients;
  std::vector<float> raw_errors;
  std::vector<float> errors;

  [[nodiscard]] Status Resize(std::uint32_t layers, std::uint32_t rows) {
    if (layers < kMinimumModelLayers || layers > kMaximumModelLayers)
      return Status::InvalidArgument("runtime basic M must be in [2,24]");
    m = layers;
    roots.resize(rows);
    codes.resize(static_cast<std::size_t>(rows) * (m - 1U));
    coefficients.resize(static_cast<std::size_t>(rows) * m);
    raw_errors.resize(rows);
    errors.resize(rows);
    return Status::Ok();
  }
  [[nodiscard]] Status Validate(std::uint32_t layers, std::uint32_t rows,
                                std::uint32_t nlist) const {
    if (m != layers || roots.size() != rows || errors.size() != rows ||
        raw_errors.size() != rows ||
        codes.size() != static_cast<std::size_t>(rows) * (m - 1U) ||
        coefficients.size() != static_cast<std::size_t>(rows) * m)
      return Status::InvalidArgument(
          "runtime basic buffers disagree with shape");
    for (const auto root : roots) {
      if (root >= nlist)
        return Status::InvalidArgument("runtime basic root exceeds codebook");
    }
    return Status::Ok();
  }
};

}  // namespace stlqpp
