#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/training/basic/basic_encoder_cuda.h"
#include "stlqpp/training/basic/large_root_algebra.h"
#include "stlqpp/training/basic/fixed_root_replay_encoder.h"

#include <utility>

namespace stlqpp {

template <int M>
class CudaFixedRootReplayEncoder final : public FixedRootReplayEncoder<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<CudaFixedRootReplayEncoder<M>> Create(
      const BasicCodebookAlgebra<M>& algebra,
      CudaContext& context);

  CudaFixedRootReplayEncoder(CudaFixedRootReplayEncoder&&) noexcept = default;
  CudaFixedRootReplayEncoder& operator=(
      CudaFixedRootReplayEncoder&&) noexcept = default;
  CudaFixedRootReplayEncoder(const CudaFixedRootReplayEncoder&) = delete;
  CudaFixedRootReplayEncoder& operator=(
      const CudaFixedRootReplayEncoder&) = delete;

  [[nodiscard]] StatusOr<BasicEncoding<M>> Encode(
      ColumnMajorMatrixF32View samples,
      ArrayView<const std::uint32_t> fixed_roots,
      ArrayView<const std::uint64_t> global_sample_ids,
      std::uint32_t seed) const override;

  [[nodiscard]] std::uint64_t model_resident_bytes() const noexcept {
    return encoder_.model_resident_bytes();
  }

 private:
  explicit CudaFixedRootReplayEncoder(
      CudaBasicEncoder<M> encoder) noexcept
      : encoder_(std::move(encoder)) {}

  mutable CudaBasicEncoder<M> encoder_;
};

extern template class CudaFixedRootReplayEncoder<5>;
extern template class CudaFixedRootReplayEncoder<10>;

}  // namespace stlqpp
