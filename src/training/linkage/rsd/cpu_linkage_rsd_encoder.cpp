#include "stlqpp/training/linkage/rsd/cpu_linkage_rsd_encoder.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "stlqpp/training/linkage/rsd/detail/linkage_rsd_kernel.h"
#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"
#include "stlqpp/training/linkage/rsd/ranked_single_deviation.h"

namespace stlqpp {

template <int M>
struct CpuLinkageRsdEncoder<M>::EncodeContext final {
  const LinkageCodebookAlgebra<M>* algebra = nullptr;
  const LinkageProjection<M>* projection = nullptr;
  const std::uint8_t* icm_layer_orders = nullptr;
  std::uint32_t maximum_rank = 2U;
  LeastSquaresCostMode cost_mode = LeastSquaresCostMode::kNormalEquation;
  std::uint32_t icm_rounds = 1U;
  LinkageEncoding<M>* output = nullptr;
  std::atomic<std::uint32_t>* failures = nullptr;
};

template <int M>
template <std::uint32_t Capacity>
void CpuLinkageRsdEncoder<M>::EncodeSample(void* opaque, std::uint32_t,
                                           std::uint32_t sample) noexcept {
  auto& context = *static_cast<EncodeContext*>(opaque);
  const float* atom_dots = context.projection->atom_dots(sample);
  const std::uint32_t m = M == 0 ? context.algebra->shape().m : M;

  LinkageSolution<M, Capacity> solution;
  const std::uint8_t* order =
      context.icm_layer_orders + static_cast<std::size_t>(sample) * m;
  if (!detail::EncodeRsdOne(*context.algebra, atom_dots,
                            context.projection->squared_norm(sample), order,
                            &solution, context.maximum_rank, context.cost_mode,
                            context.icm_rounds)) {
    context.failures->fetch_add(1U, std::memory_order_relaxed);
    return;
  }

  const std::size_t base = static_cast<std::size_t>(sample) * m;
  for (std::uint32_t layer = 0U; layer < m; ++layer) {
    context.output->codes_[base + layer] = solution.support.codes[layer];
    context.output->coefficients_[base + layer] = solution.coefficients[layer];
  }
  context.output->squared_errors_[sample] = solution.squared_error;
}

template <int M>
StatusOr<LinkageEncoding<M>> CpuLinkageRsdEncoder<M>::Encode(
    const LinkageProjection<M>& projection,
    ArrayView<const std::uint8_t> icm_layer_orders) const {
  if (!IsSupportedRsdRank(maximum_rank_)) {
    return Status::InvalidArgument("linkage RSD rank must be 1, 2 or 3");
  }
  const Status order_status = ValidateIcmLayerOrders<M>(
      projection.sample_count(), icm_layer_orders, algebra_.shape().m);
  if (!order_status.ok()) {
    return order_status;
  }
  return EncodeValidatedOrders(projection, icm_layer_orders);
}

template <int M>
StatusOr<LinkageEncoding<M>> CpuLinkageRsdEncoder<M>::EncodeValidatedOrders(
    const LinkageProjection<M>& projection,
    ArrayView<const std::uint8_t> icm_layer_orders) const {
  if (!IsSupportedRsdRank(maximum_rank_)) {
    return Status::InvalidArgument("linkage RSD rank must be 1, 2 or 3");
  }
  assert(ValidateIcmLayerOrders<M>(projection.sample_count(), icm_layer_orders,
                                   algebra_.shape().m)
             .ok());

  LinkageEncoding<M> output(projection.sample_count(), algebra_.shape().m);
  std::atomic<std::uint32_t> failures{0U};
  EncodeContext context{
      .algebra = &algebra_,
      .projection = &projection,
      .icm_layer_orders = icm_layer_orders.data(),
      .maximum_rank = maximum_rank_,
      .cost_mode = cost_mode_,
      .icm_rounds = icm_rounds_,
      .output = &output,
      .failures = &failures,
  };
  if constexpr (M == 0) {
    if (algebra_.shape().m > kLegacyRuntimeScratchLayers)
      parallel_.Run(projection.sample_count(), &context, EncodeSample<kWideRuntimeScratchLayers>);
    else
      parallel_.Run(projection.sample_count(), &context, EncodeSample<kLegacyRuntimeScratchLayers>);
  } else {
    parallel_.Run(projection.sample_count(), &context, EncodeSample<kModelScratchLayers<M>>);
  }
  if (failures.load(std::memory_order_relaxed) != 0U) {
    return Status::FailedPrecondition(
        "linkage rsd contains a support singular after registered retries");
  }
  return output;
}

template class CpuLinkageRsdEncoder<0>;
template class CpuLinkageRsdEncoder<5>;
template class CpuLinkageRsdEncoder<12>;
template class CpuLinkageRsdEncoder<6>;
template class CpuLinkageRsdEncoder<10>;

}  // namespace stlqpp
