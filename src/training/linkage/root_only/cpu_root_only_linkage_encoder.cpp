#include "stlqpp/training/linkage/root_only/cpu_root_only_linkage_encoder.h"

#include "stlqpp/training/basic/basic_projection.h"
#include "stlqpp/training/basic/support_least_squares.h"
#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"

#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace stlqpp {
namespace {

constexpr float kAcceptanceEpsilon = 1.0e-6F;

float Dot(const float* left,
          const float* right,
          std::uint32_t dimension) noexcept {
  float sum = 0.0F;
  for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
    sum += left[axis] * right[axis];
  }
  return sum;
}

template <int M, std::uint32_t Capacity>
[[nodiscard]] bool Refit(
    const BasicCodebookAlgebra<M>& algebra,
    const LinkageCodeTuple<M>& support,
    float root_dot,
    const float* small_dots,
    float residual_squared_norm,
    float* coefficients,
    float* squared_error) noexcept {
  using LeastSquares = std::conditional_t<M == 0, RuntimeBasicSupportLeastSquares, BasicSupportLeastSquares<M>>;
  if (!detail::SolveBasicSupport<M, Capacity>(
          algebra, support.first, support.trailing.data(), root_dot,
          small_dots, coefficients, nullptr)) {
    return false;
  }
  *squared_error = LeastSquares::SquaredCost(
      algebra, support.first, support.trailing.data(), root_dot, small_dots,
      coefficients, residual_squared_norm);
  return std::isfinite(*squared_error);
}

template <int M>
[[nodiscard]] std::uint32_t FlatAtom(
    const BasicCodebookAlgebra<M>& algebra,
    const LinkageCodeTuple<M>& support,
    std::uint32_t layer) noexcept {
  return algebra.small_offset(layer) + support.trailing[layer - 1U];
}

}  // namespace

template <int M>
struct CpuRootOnlyLinkageEncoder<M>::EncodeContext final {
  const BasicCodebookAlgebra<M>* algebra = nullptr;
  const BasicProjection<M>* projection = nullptr;
  const std::uint8_t* icm_layer_orders = nullptr;
  std::uint32_t order_stride = M;
  std::uint32_t icm_rounds = 1U;
  RootOnlyLinkageEncoding<M>* output = nullptr;
  std::atomic<std::uint32_t>* failures = nullptr;
};

template <int M>
template <std::uint32_t Capacity>
void CpuRootOnlyLinkageEncoder<M>::EncodeSample(
    void* opaque,
    std::uint32_t,
    std::uint32_t sample) noexcept {
  auto& context = *static_cast<EncodeContext*>(opaque);
  const BasicCodebookAlgebra<M>& algebra = *context.algebra;
  const auto m = algebra.shape().m;
  const BasicProjection<M>& projection = *context.projection;
  const float* residual = projection.samples().column(sample);
  const float* small_dots = projection.small_dots(sample);
  const ColumnMajorMatrixF32View roots = algebra.root0();

  LinkageCodeTuple<M> support;
  float selected_root_dot = 0.0F;
  float best_root_score = -std::numeric_limits<float>::infinity();
  for (std::uint32_t root = 0U; root < roots.columns; ++root) {
    const float root_dot = Dot(residual, roots.column(root), roots.rows);
    const float score =
        std::fabs(root_dot) / std::sqrt(algebra.root_norm2(root));
    if (score > best_root_score ||
        (score == best_root_score && root < support.first)) {
      best_root_score = score;
      support.first = root;
      selected_root_dot = root_dot;
    }
  }

  std::array<float, (Capacity - 1U) * 256U> remaining_dots{};
  const float root_alpha =
      selected_root_dot / algebra.root_norm2(support.first);
  for (std::uint32_t flat = 0U; flat < algebra.small_count(); ++flat) {
    remaining_dots[flat] =
        small_dots[flat] -
        root_alpha * algebra.root_small_dot(support.first, flat);
  }

  std::array<bool, Capacity> available{};
  for (std::uint32_t layer = 1U;
       layer < m; ++layer) {
    available[layer] = true;
  }
  for (std::uint32_t step = 1U;
       step < m; ++step) {
    float best_score = -std::numeric_limits<float>::infinity();
    std::uint32_t best_flat = 0U;
    std::uint32_t best_layer = 1U;
    for (std::uint32_t layer = 1U;
         layer < m; ++layer) {
      if (!available[layer]) {
        continue;
      }
      const std::uint32_t begin = algebra.small_offset(layer);
      for (std::uint32_t code = 0U; code < 256U; ++code) {
        const std::uint32_t flat = begin + code;
        const float score =
            std::fabs(remaining_dots[flat] *
                      algebra.small_inverse_norm(flat));
        if (score > best_score ||
            (score == best_score && flat < best_flat)) {
          best_score = score;
          best_flat = flat;
          best_layer = layer;
        }
      }
    }
    support.trailing[best_layer - 1U] = static_cast<std::uint8_t>(
        best_flat - algebra.small_offset(best_layer));
    available[best_layer] = false;
    if (step + 1U == m) break;
    const float alpha =
        remaining_dots[best_flat] / algebra.small_norm2(best_flat);
    for (std::uint32_t layer = 1U;
         layer < m; ++layer) {
      if (!available[layer]) continue;
      const std::uint32_t begin = algebra.small_offset(layer);
      for (std::uint32_t code = 0U; code < 256U; ++code) {
        const std::uint32_t flat = begin + code;
        remaining_dots[flat] -= alpha * algebra.small_dot(flat, best_flat);
      }
    }
  }

  std::array<float, Capacity> coefficients{};
  float squared_error = 0.0F;
  if (!Refit<M, Capacity>(algebra, support, selected_root_dot, small_dots,
             projection.squared_norm(sample), coefficients.data(),
             &squared_error)) {
    context.failures->fetch_add(1U, std::memory_order_relaxed);
    return;
  }

  const std::uint8_t* order = context.icm_layer_orders +
      static_cast<std::size_t>(sample) * context.order_stride;
  for (std::uint32_t round = 0U; round < context.icm_rounds; ++round) {
  for (std::uint32_t position = 0U;
       position < m; ++position) {
    const std::uint32_t layer = order[position];
    LinkageCodeTuple<M> trial_support = support;
    float trial_root_dot = selected_root_dot;

    if (layer == 0U) {
      float best_score = -std::numeric_limits<float>::infinity();
      std::uint32_t best_root = support.first;
      for (std::uint32_t root = 0U; root < roots.columns; ++root) {
        const float root_dot = Dot(residual, roots.column(root), roots.rows);
        float remaining_dot = root_dot;
        for (std::uint32_t other = 1U;
             other < m; ++other) {
          remaining_dot -= coefficients[other] * algebra.root_small_dot(
              root, FlatAtom(algebra, support, other));
        }
        const float score =
            std::fabs(remaining_dot) / std::sqrt(algebra.root_norm2(root));
        if (score > best_score ||
            (score == best_score && root < best_root)) {
          best_score = score;
          best_root = root;
          trial_root_dot = root_dot;
        }
      }
      if (best_root == support.first) {
        continue;
      }
      trial_support.first = best_root;
    } else {
      const std::uint8_t old_code = support.trailing[layer - 1U];
      std::uint8_t best_code = old_code;
      float best_score = -std::numeric_limits<float>::infinity();
      const std::uint32_t begin = algebra.small_offset(layer);
      for (std::uint32_t code = 0U; code < 256U; ++code) {
        const std::uint32_t atom = begin + code;
        float remaining_dot = small_dots[atom] -
            coefficients[0] *
                algebra.root_small_dot(support.first, atom);
        for (std::uint32_t other = 1U;
             other < m; ++other) {
          if (other == layer) {
            continue;
          }
          remaining_dot -= coefficients[other] *
              algebra.small_dot(atom, FlatAtom(algebra, support, other));
        }
        const float score =
            std::fabs(remaining_dot * algebra.small_inverse_norm(atom));
        if (score > best_score ||
            (score == best_score && code < best_code)) {
          best_score = score;
          best_code = static_cast<std::uint8_t>(code);
        }
      }
      if (best_code == old_code) {
        continue;
      }
      trial_support.trailing[layer - 1U] = best_code;
    }

    std::array<float, Capacity> trial_coefficients{};
    float trial_error = 0.0F;
    if (Refit<M, Capacity>(algebra, trial_support, trial_root_dot, small_dots,
              projection.squared_norm(sample), trial_coefficients.data(),
              &trial_error) &&
        trial_error + kAcceptanceEpsilon < squared_error) {
      support = trial_support;
      selected_root_dot = trial_root_dot;
      coefficients = trial_coefficients;
      squared_error = trial_error;
    }
  }
  }

  context.output->supports_[sample] = support;
  const std::size_t coefficient_base = static_cast<std::size_t>(sample) * m;
  for (std::uint32_t layer = 0U;
       layer < m; ++layer) {
    context.output->coefficients_[coefficient_base + layer] =
        coefficients[layer];
  }
  context.output->squared_errors_[sample] = squared_error;
}

template <int M>
StatusOr<RootOnlyLinkageEncoding<M>>
CpuRootOnlyLinkageEncoder<M>::Encode(
    ColumnMajorMatrixF32View residuals,
    ArrayView<const std::uint8_t> icm_layer_orders) const {
  auto projection = BasicProjection<M>::Create(
      residuals, algebra_, gemm_, parallel_);
  if (!projection.ok()) {
    return projection.status();
  }
  const Status order_status = ValidateIcmLayerOrders<M>(
      residuals.columns, icm_layer_orders, algebra_.shape().m);
  if (!order_status.ok()) {
    return order_status;
  }
  return EncodeProjected(projection.value(), icm_layer_orders, algebra_.shape().m);
}

template <int M>
StatusOr<RootOnlyLinkageEncoding<M>>
CpuRootOnlyLinkageEncoder<M>::EncodeValidatedSharedOrder(
    ColumnMajorMatrixF32View residuals,
    ArrayView<const std::uint8_t> icm_layer_order) const {
  assert(ValidateIcmLayerOrders<M>(1U, icm_layer_order, algebra_.shape().m).ok());
  auto projection = BasicProjection<M>::Create(
      residuals, algebra_, gemm_, parallel_);
  if (!projection.ok()) return projection.status();
  return EncodeProjected(projection.value(), icm_layer_order, 0U);
}

template <int M>
StatusOr<RootOnlyLinkageEncoding<M>>
CpuRootOnlyLinkageEncoder<M>::EncodeProjected(
    const BasicProjection<M>& projection,
    ArrayView<const std::uint8_t> icm_layer_orders,
    std::uint32_t order_stride) const {

  RootOnlyLinkageEncoding<M> output(projection.sample_count(), algebra_.shape().m);
  std::atomic<std::uint32_t> failures{0U};
  EncodeContext context{
      .algebra = &algebra_,
      .projection = &projection,
      .icm_layer_orders = icm_layer_orders.data(),
      .order_stride = order_stride,
      .icm_rounds = icm_rounds_,
      .output = &output,
      .failures = &failures,
  };
  if constexpr (M == 0) {
    parallel_.Run(projection.sample_count(), &context,
        algebra_.shape().m > kLegacyRuntimeScratchLayers
        ? EncodeSample<kWideRuntimeScratchLayers> : EncodeSample<kLegacyRuntimeScratchLayers>);
  } else {
    parallel_.Run(projection.sample_count(), &context, EncodeSample<kModelScratchLayers<M>>);
  }
  if (failures.load(std::memory_order_relaxed) != 0U) {
    return Status::FailedPrecondition(
        "root-only support is singular after registered FP32 retries");
  }
  return output;
}

template class CpuRootOnlyLinkageEncoder<0>;
template class CpuRootOnlyLinkageEncoder<5>;
template class CpuRootOnlyLinkageEncoder<12>;
template class CpuRootOnlyLinkageEncoder<6>;
template class CpuRootOnlyLinkageEncoder<10>;

}  // namespace stlqpp
