#pragma once

#include "stlqpp/index/emit/base_list_reader.h"
#include "stlqpp/training/checkpoint/checkpoint_replay_loader.h"

namespace stlqpp {

// Concrete bounded real-row source for checkpoint replay. One list-major CID
// is decoded at a time; stable IDs alone determine the edge ICM permutation.
// Basic code/coeff streams are carried into the bounded replay window so
// real roots reuse the already-encoded final/basic support verbatim.
template <int M>
class BaseListReplaySource final
    : public TrainingReplayRealClusterSource {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  explicit BaseListReplaySource(const BaseListReader<M>& lists) noexcept
      : lists_(&lists) {}

  [[nodiscard]] StatusOr<TrainingReplayRealCluster> Load(
      std::uint32_t cid,
      ArrayView<const std::uint64_t> stable_ids,
      std::uint32_t round_seed) override;

 private:
  const BaseListReader<M>* lists_ = nullptr;
};

extern template class BaseListReplaySource<5>;
extern template class BaseListReplaySource<10>;

}  // namespace stlqpp
