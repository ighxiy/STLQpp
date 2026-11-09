#pragma once

#include "stlqpp/training/checkpoint/checkpoint_replay_loader.h"
#include "stlqpp/index/emit/base_list_reader.h"

namespace stlqpp {

// Training replay is anchored by the previous checkpoint forest. Fixed-root
// basic emission preserves forest CID membership, while list publication may
// retain a different stable-ID order. One list-major read is therefore
// gathered into checkpoint order before deterministic per-ID ICM orders are
// derived from the same identities.
template <int M>
class StableIdReplaySource final
    : public TrainingReplayRealClusterSource {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  explicit StableIdReplaySource(
      const BaseListReader<M>& current_basic) noexcept
      : current_basic_(&current_basic) {}

  [[nodiscard]] StatusOr<TrainingReplayRealCluster> Load(
      std::uint32_t cid,
      ArrayView<const std::uint64_t> stable_ids,
      std::uint32_t round_seed) override;

 private:
  const BaseListReader<M>* current_basic_ = nullptr;
};

extern template class StableIdReplaySource<5>;
extern template class StableIdReplaySource<10>;

}  // namespace stlqpp
