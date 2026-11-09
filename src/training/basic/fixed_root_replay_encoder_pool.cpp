#include "stlqpp/training/basic/fixed_root_replay_encoder_pool.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace stlqpp {

template <int M>
StatusOr<std::unique_ptr<FixedRootReplayEncoderPool<M>>>
FixedRootReplayEncoderPool<M>::Create(
    std::vector<std::unique_ptr<FixedRootReplayEncoder<M>>> encoders) {
  if (encoders.empty()) {
    return Status::InvalidArgument(
        "fixed-root encoder pool requires at least one backend");
  }
  for (const auto& encoder : encoders) {
    if (encoder == nullptr) {
      return Status::InvalidArgument(
          "fixed-root encoder pool contains a null backend");
    }
  }
  return std::unique_ptr<FixedRootReplayEncoderPool<M>>(
      new FixedRootReplayEncoderPool<M>(std::move(encoders)));
}

template <int M>
FixedRootReplayEncoderPool<M>::FixedRootReplayEncoderPool(
    std::vector<std::unique_ptr<FixedRootReplayEncoder<M>>> encoders)
    : encoders_(std::move(encoders)) {
  free_.reserve(encoders_.size());
  for (std::size_t index = encoders_.size(); index != 0U; --index) {
    free_.push_back(index - 1U);
  }
}

template <int M>
StatusOr<BasicEncoding<M>> FixedRootReplayEncoderPool<M>::Encode(
    ColumnMajorMatrixF32View samples,
    ArrayView<const std::uint32_t> fixed_roots,
    ArrayView<const std::uint64_t> global_sample_ids,
    std::uint32_t seed) const {
  std::size_t index = 0U;
  {
    std::unique_lock lock(mutex_);
    available_.wait(lock, [this]() { return !free_.empty(); });
    index = free_.back();
    free_.pop_back();
  }
  StatusOr<BasicEncoding<M>> result = encoders_[index]->Encode(
      samples, fixed_roots, global_sample_ids, seed);
  {
    std::lock_guard lock(mutex_);
    free_.push_back(index);
  }
  available_.notify_one();
  return result;
}

template class FixedRootReplayEncoderPool<5>;
template class FixedRootReplayEncoderPool<12>;
template class FixedRootReplayEncoderPool<6>;
template class FixedRootReplayEncoderPool<10>;

}  // namespace stlqpp
