#include "stlqpp/training/linkage/replay/cpu_fixed_edge_replay_encoder.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "stlqpp/training/linkage/rsd/cpu_linkage_rsd_encoder.h"
#include "stlqpp/training/linkage/rsd/ranked_single_deviation.h"

namespace stlqpp {

template <int M>
StatusOr<LinkageEncoding<M>> CpuFixedEdgeReplayEncoder<M>::Encode(
    ColumnMajorMatrixF32View residuals,
    ArrayView<const std::uint8_t> icm_layer_orders) const {
  return EncodeImpl(residuals, icm_layer_orders, false);
}

template <int M>
StatusOr<LinkageEncoding<M>>
CpuFixedEdgeReplayEncoder<M>::EncodeResidualsValidated(
    ColumnMajorMatrixF32View residuals,
    ArrayView<const std::uint8_t> icm_layer_orders) const {
  return EncodeImpl(residuals, icm_layer_orders, true);
}

template <int M>
StatusOr<LinkageEncoding<M>> CpuFixedEdgeReplayEncoder<M>::EncodeImpl(
    ColumnMajorMatrixF32View residuals,
    ArrayView<const std::uint8_t> icm_layer_orders,
    bool orders_validated) const {
  if (!IsSupportedRsdRank(maximum_rank_)) {
    return Status::InvalidArgument("fixed-edge replay RSD rank is unsupported");
  }
  auto projection_result =
      LinkageProjection<M>::Create(residuals, algebra_, gemm_, parallel_);
  if (!projection_result.ok()) {
    return projection_result.status();
  }
  LinkageProjection<M> projection = std::move(projection_result).value();
  CpuLinkageRsdEncoder<M> encoder(algebra_, parallel_, maximum_rank_,
                                  cost_mode_, icm_rounds_);
  return orders_validated
             ? encoder.EncodeValidatedOrders(projection, icm_layer_orders)
             : encoder.Encode(projection, icm_layer_orders);
}

template class CpuFixedEdgeReplayEncoder<0>;
template class CpuFixedEdgeReplayEncoder<5>;
template class CpuFixedEdgeReplayEncoder<12>;
template class CpuFixedEdgeReplayEncoder<6>;
template class CpuFixedEdgeReplayEncoder<10>;

}  // namespace stlqpp
