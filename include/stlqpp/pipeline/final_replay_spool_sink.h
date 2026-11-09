#pragma once

#include "stlqpp/pipeline/final_forest_window_sink.h"
#include "stlqpp/training/linkage/replay/fixed_topology_replay_pipeline.h"

#include <cstdint>

namespace stlqpp {

// Ordered bridge from the final exact fixed-topology replay to the canonical
// prepared spool. This is pipeline composition, not training or storage
// policy: it joins typed products from both offline leaves without making
// either library depend on the other.
template <int M>
class FinalReplaySpoolSink final
    : public FixedTopologyReplayWindowSink<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  explicit FinalReplaySpoolSink(
      FinalForestWindowSink<M>& forest_sink) noexcept
      : forest_sink_(&forest_sink) {}

  [[nodiscard]] bool RequiresRealReconstructions() const noexcept override {
    return false;
  }

  [[nodiscard]] Status Commit(
      std::uint64_t sequence,
      ArrayView<const FixedTopologyReplayRequest> requests,
      ArrayView<const FixedTopologyReplayResult<M>> results) override;

 private:
  FinalForestWindowSink<M>* forest_sink_ = nullptr;
  std::uint64_t next_sequence_ = 0U;
};

extern template class FinalReplaySpoolSink<5>;
extern template class FinalReplaySpoolSink<10>;

}  // namespace stlqpp
