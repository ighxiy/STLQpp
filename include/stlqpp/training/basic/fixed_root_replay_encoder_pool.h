#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/training/basic/fixed_root_replay_encoder.h"

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace stlqpp {

// Bounded execution owner for virtual-root fixed-CID encoding. CPU cluster
// workers share this pool; they do not replicate one complete large-root CUDA
// model per outer worker.
template <int M>
class FixedRootReplayEncoderPool final : public FixedRootReplayEncoder<M> {
 public:
  static_assert(M >= 2 && M <= 20);

  [[nodiscard]] static StatusOr<
      std::unique_ptr<FixedRootReplayEncoderPool<M>>>
  Create(std::vector<std::unique_ptr<FixedRootReplayEncoder<M>>> encoders);

  [[nodiscard]] StatusOr<BasicEncoding<M>> Encode(
      ColumnMajorMatrixF32View samples,
      ArrayView<const std::uint32_t> fixed_roots,
      ArrayView<const std::uint64_t> global_sample_ids,
      std::uint32_t seed) const override;

 private:
  explicit FixedRootReplayEncoderPool(
      std::vector<std::unique_ptr<FixedRootReplayEncoder<M>>> encoders);

  std::vector<std::unique_ptr<FixedRootReplayEncoder<M>>> encoders_;
  mutable std::mutex mutex_;
  mutable std::condition_variable available_;
  mutable std::vector<std::size_t> free_;
};

extern template class FixedRootReplayEncoderPool<5>;
extern template class FixedRootReplayEncoderPool<10>;

}  // namespace stlqpp
