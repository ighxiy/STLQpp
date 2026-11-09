#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "stlqpp/model/model_layers.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/basic/basic_encoding.h"

namespace stlqpp::detail {

// Construction access shared by semantic and production BasicEncoding
// backends. It is deliberately outside the public product API, but must be
// available to independently compiled CPU and CUDA implementations.
struct BasicEncodingAccess final {
  // Transfer only vector ownership; numerical producers and consumers keep
  // their typed width, while queues and execution interfaces do not.
  template <int M>
  [[nodiscard]] static BasicEncodingBatch Release(BasicEncoding<M>&& input) {
    if constexpr (M == 0) {
      return BasicEncodingBatch(std::move(input.encoding_));
    } else {
      return BasicEncodingBatch(RuntimeBasicEncoding{
          .m = M,
          .roots = std::move(input.root_codes_),
          .codes = std::move(input.small_codes_),
          .coefficients = std::move(input.coefficients_),
          .raw_errors = std::move(input.initial_squared_errors_),
          .errors = std::move(input.squared_errors_),
      });
    }
  }

  template <int M>
  [[nodiscard]] static StatusOr<BasicEncodingBatch> Release(
      StatusOr<BasicEncoding<M>> input) {
    if (!input.ok()) return input.status();
    return Release(std::move(input).value());
  }

  template <int M>
  [[nodiscard]] static BasicEncoding<M> Adopt(BasicEncodingBatch&& input) {
    assert(M == 0 || input.m() == M);
    return BasicEncoding<M>(std::move(input.encoding_));
  }

  template <int M>
  [[nodiscard]] static BasicEncoding<M> Create(
      std::uint32_t sample_count,
      [[maybe_unused]] std::uint32_t m = FixedModelLayers<M>()) {
    if constexpr (M == 0) {
      RuntimeBasicEncoding encoding;
      encoding.m = m;
      BasicEncoding<0> output(std::move(encoding));
      Resize(&output, sample_count);
      return output;
    } else {
      return BasicEncoding<M>(sample_count);
    }
  }

  template <int M>
  static void Resize(BasicEncoding<M>* output, std::uint32_t sample_count) {
    if constexpr (M == 0) {
      auto& e = output->encoding_;
      e.roots.resize(sample_count);
      e.codes.resize(static_cast<std::size_t>(sample_count) * (e.m - 1U));
      e.coefficients.resize(static_cast<std::size_t>(sample_count) * e.m);
      e.raw_errors.resize(sample_count);
      e.errors.resize(sample_count);
    } else {
      output->sample_count_ = sample_count;
      output->root_codes_.resize(sample_count);
      output->small_codes_.resize(static_cast<std::size_t>(sample_count) *
                                  (M - 1));
      output->coefficients_.resize(static_cast<std::size_t>(sample_count) * M);
      output->initial_squared_errors_.resize(sample_count);
      output->squared_errors_.resize(sample_count);
    }
  }

  template <int M, std::size_t Capacity = kModelScratchLayers<M>>
  static void Store(
      BasicEncoding<M>* output, std::uint32_t sample,
      const std::array<std::uint8_t, Capacity - 1U>& codes,
      const std::array<float, Capacity>& coefficients,
      std::uint32_t root, float error) noexcept {
    const std::uint32_t m = [&] {
      if constexpr (M == 0)
        return output->m();
      else
        return static_cast<std::uint32_t>(M);
    }();
    Roots(output)[sample] = root;
    std::copy_n(codes.data(), m - 1U,
                Codes(output) + static_cast<std::size_t>(sample) * (m - 1U));
    std::copy_n(coefficients.data(), m,
                Coefficients(output) + static_cast<std::size_t>(sample) * m);
    InitialErrors(output)[sample] = error;
    Errors(output)[sample] = error;
  }

  template <int M>
  [[nodiscard]] static std::uint32_t* Roots(BasicEncoding<M>* output) noexcept {
    if constexpr (M == 0)
      return output->encoding_.roots.data();
    else
      return output->root_codes_.data();
  }

  template <int M>
  [[nodiscard]] static std::uint8_t* Codes(BasicEncoding<M>* output) noexcept {
    if constexpr (M == 0)
      return output->encoding_.codes.data();
    else
      return output->small_codes_.data();
  }

  template <int M>
  [[nodiscard]] static float* Coefficients(BasicEncoding<M>* output) noexcept {
    if constexpr (M == 0)
      return output->encoding_.coefficients.data();
    else
      return output->coefficients_.data();
  }

  template <int M>
  [[nodiscard]] static float* Errors(BasicEncoding<M>* output) noexcept {
    if constexpr (M == 0)
      return output->encoding_.errors.data();
    else
      return output->squared_errors_.data();
  }

  template <int M>
  [[nodiscard]] static float* InitialErrors(BasicEncoding<M>* output) noexcept {
    if constexpr (M == 0)
      return output->encoding_.raw_errors.data();
    else
      return output->initial_squared_errors_.data();
  }
};

}  // namespace stlqpp::detail
