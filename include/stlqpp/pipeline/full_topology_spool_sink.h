#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/pipeline/final_forest_window_sink.h"
#include "stlqpp/training/linkage/topology/full_topology_pipeline.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace stlqpp {

struct FullTopologySpoolWindowOptions final {
  std::uint32_t maximum_clusters = 64U;
  std::uint64_t target_real_rows = 4096U;
};

// Converts ordered owned full-topology products into bounded windows for the
// shared canonical spool ingress. It retains only the final static streams;
// exact vectors and reconstructions are released after each Commit call.
template <int M>
class FullTopologySpoolSink final
    : public FullTopologyClusterSink<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<FullTopologySpoolSink<M>> Create(
      FinalForestWindowSink<M>& destination,
      FullTopologySpoolWindowOptions options,
      std::uint32_t first_sequence = 0U);

  FullTopologySpoolSink(FullTopologySpoolSink&&) noexcept = default;
  FullTopologySpoolSink& operator=(FullTopologySpoolSink&&) noexcept =
      default;
  FullTopologySpoolSink(const FullTopologySpoolSink&) = delete;
  FullTopologySpoolSink& operator=(const FullTopologySpoolSink&) = delete;

  [[nodiscard]] bool RequiresRotatedExactReal()
      const noexcept override {
    return false;
  }
  [[nodiscard]] Status PrepareForCommit(
      std::uint64_t sequence,
      FullTopologyClusterCommitPayload* payload,
      FullTopologyClusterProduct<M>* product) override;
  [[nodiscard]] std::uint64_t PreparedResidentBytes(
      std::uint64_t sequence) const noexcept override;
  [[nodiscard]] bool RequiresOrderedCommit() const noexcept override {
    return destination_->RequiresOrderedCommit();
  }
  [[nodiscard]] Status Commit(
      std::uint64_t sequence,
      FullTopologyClusterCommitPayload payload,
      FullTopologyClusterProduct<M> product) override;

  [[nodiscard]] Status Finish();

 private:
  struct PreparationState;

  FullTopologySpoolSink(
      FinalForestWindowSink<M>& destination,
      FullTopologySpoolWindowOptions options,
      std::uint32_t first_sequence) noexcept
      : destination_(&destination), options_(options),
        next_sequence_(first_sequence) {}

  [[nodiscard]] Status Flush();

  FinalForestWindowSink<M>* destination_ = nullptr;
  FullTopologySpoolWindowOptions options_;
  std::shared_ptr<PreparationState> preparation_;
  std::vector<PreparedCanonicalCluster<M>> pending_;
  std::uint64_t pending_real_rows_ = 0U;
  std::uint64_t next_sequence_ = 0U;
  bool finished_ = false;
  bool failed_ = false;
};

extern template class FullTopologySpoolSink<5>;
extern template class FullTopologySpoolSink<10>;

}  // namespace stlqpp
