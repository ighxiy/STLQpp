#include "stlqpp/training/basic/cuda_fixed_root_replay_encoder.h"

#include "stlqpp/training/basic/basic_schedule.h"

#include <utility>

namespace stlqpp {

template <int M>
StatusOr<CudaFixedRootReplayEncoder<M>>
CudaFixedRootReplayEncoder<M>::Create(
    const BasicCodebookAlgebra<M>& algebra,
    CudaContext& context) {
  auto encoder = CudaBasicEncoder<M>::Create(algebra, context);
  if (!encoder.ok()) {
    return encoder.status();
  }
  return CudaFixedRootReplayEncoder<M>(std::move(encoder).value());
}

template <int M>
StatusOr<BasicEncoding<M>> CudaFixedRootReplayEncoder<M>::Encode(
    ColumnMajorMatrixF32View samples,
    ArrayView<const std::uint32_t> fixed_roots,
    ArrayView<const std::uint64_t> global_sample_ids,
    std::uint32_t seed) const {
  return encoder_.EncodeFixedRoots(
      samples, fixed_roots, global_sample_ids,
      BasicSchedule::Training(), seed);
}

template class CudaFixedRootReplayEncoder<0>;
template class CudaFixedRootReplayEncoder<5>;
template class CudaFixedRootReplayEncoder<12>;
template class CudaFixedRootReplayEncoder<6>;
template class CudaFixedRootReplayEncoder<10>;

}  // namespace stlqpp
