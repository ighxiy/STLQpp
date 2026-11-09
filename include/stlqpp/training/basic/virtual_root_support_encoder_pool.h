#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/exclusive_index_pool.h"
#include "stlqpp/training/basic/virtual_root_support_encoder.h"

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace stlqpp {

// Bounded owner for virtual-root encoders shared by outer cluster workers.
template <int M>
class VirtualRootSupportEncoderPool final
    : public VirtualRootSupportEncoder<M> {
 public:
  [[nodiscard]] static StatusOr<
      std::unique_ptr<VirtualRootSupportEncoderPool<M>>>
  Create(std::vector<std::unique_ptr<VirtualRootSupportEncoder<M>>> encoders);

  [[nodiscard]] static StatusOr<
      std::unique_ptr<VirtualRootSupportEncoderPool<M>>>
  Create(std::vector<std::unique_ptr<VirtualRootSupportEncoder<M>>> encoders,
         std::shared_ptr<ExclusiveIndexPool> shared_lanes);

  [[nodiscard]] StatusOr<BasicEncoding<M>> Encode(
      ColumnMajorMatrixF32View samples,
      ArrayView<const std::uint64_t> sample_ids,
      std::uint32_t fixed_root,
      std::uint32_t seed) const override;

 private:
  explicit VirtualRootSupportEncoderPool(
      std::vector<std::unique_ptr<VirtualRootSupportEncoder<M>>> encoders,
      std::shared_ptr<ExclusiveIndexPool> shared_lanes = {});

  std::vector<std::unique_ptr<VirtualRootSupportEncoder<M>>> encoders_;
  std::shared_ptr<ExclusiveIndexPool> shared_lanes_;
  mutable std::mutex mutex_;
  mutable std::condition_variable available_;
  mutable std::vector<std::size_t> free_;
};

extern template class VirtualRootSupportEncoderPool<5>;
extern template class VirtualRootSupportEncoderPool<10>;

}  // namespace stlqpp
