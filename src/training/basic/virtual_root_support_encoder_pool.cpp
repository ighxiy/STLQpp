#include "stlqpp/training/basic/virtual_root_support_encoder_pool.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace stlqpp {

template <int M>
StatusOr<std::unique_ptr<VirtualRootSupportEncoderPool<M>>>
VirtualRootSupportEncoderPool<M>::Create(
    std::vector<std::unique_ptr<VirtualRootSupportEncoder<M>>> encoders) {
  return Create(std::move(encoders), {});
}

template <int M>
StatusOr<std::unique_ptr<VirtualRootSupportEncoderPool<M>>>
VirtualRootSupportEncoderPool<M>::Create(
    std::vector<std::unique_ptr<VirtualRootSupportEncoder<M>>> encoders,
    std::shared_ptr<ExclusiveIndexPool> shared_lanes) {
  if (encoders.empty()) {
    return Status::InvalidArgument(
        "virtual-root encoder pool requires at least one backend");
  }
  for (const auto& encoder : encoders) {
    if (encoder == nullptr) {
      return Status::InvalidArgument(
          "virtual-root encoder pool contains a null backend");
    }
  }
  if (shared_lanes != nullptr && shared_lanes->size() != encoders.size()) {
    return Status::InvalidArgument(
        "virtual-root encoder count does not match shared lane count");
  }
  return std::unique_ptr<VirtualRootSupportEncoderPool<M>>(
      new VirtualRootSupportEncoderPool<M>(
          std::move(encoders), std::move(shared_lanes)));
}

template <int M>
VirtualRootSupportEncoderPool<M>::VirtualRootSupportEncoderPool(
    std::vector<std::unique_ptr<VirtualRootSupportEncoder<M>>> encoders,
    std::shared_ptr<ExclusiveIndexPool> shared_lanes)
    : encoders_(std::move(encoders)),
      shared_lanes_(std::move(shared_lanes)) {
  if (shared_lanes_ != nullptr) {
    return;
  }
  free_.reserve(encoders_.size());
  for (std::size_t index = encoders_.size(); index != 0U; --index) {
    free_.push_back(index - 1U);
  }
}

template <int M>
StatusOr<BasicEncoding<M>> VirtualRootSupportEncoderPool<M>::Encode(
    ColumnMajorMatrixF32View samples,
    ArrayView<const std::uint64_t> sample_ids,
    std::uint32_t fixed_root,
    std::uint32_t seed) const {
  std::size_t index = 0U;
  if (shared_lanes_ != nullptr) {
    auto acquired = shared_lanes_->Acquire();
    if (!acquired.ok()) return acquired.status();
    index = acquired.value();
  } else {
    std::unique_lock lock(mutex_);
    available_.wait(lock, [this]() { return !free_.empty(); });
    index = free_.back();
    free_.pop_back();
  }
  StatusOr<BasicEncoding<M>> result =
      encoders_[index]->Encode(samples, sample_ids, fixed_root, seed);
  if (shared_lanes_ != nullptr) {
    shared_lanes_->Release(index);
  } else {
    std::lock_guard lock(mutex_);
    free_.push_back(index);
    available_.notify_one();
  }
  return result;
}

template class VirtualRootSupportEncoderPool<0>;
template class VirtualRootSupportEncoderPool<5>;
template class VirtualRootSupportEncoderPool<12>;
template class VirtualRootSupportEncoderPool<6>;
template class VirtualRootSupportEncoderPool<10>;

}  // namespace stlqpp
