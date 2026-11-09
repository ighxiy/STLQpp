#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_pipeline.h"
#include "stlqpp/training/update/root_only_initialization_transcript.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

struct InitializationBadClusterResult final {
  std::vector<std::uint8_t> mask;
  double quantile_baseline = 0.0;
  double minimum_squared_error = 0.0;
  double maximum_squared_error = 0.0;
  double mean_squared_error = 0.0;
  std::uint64_t real_rows = 0U;
};

// Ordered first-pass initializer sink. It derives the frozen bad-cluster mask
// and stages the exact root-only forest needed by the weighted C_one pass.
template <int M>
class RootOnlyInitializationUpdateSink final
    : public RootOnlyInitializationClusterSink<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<RootOnlyInitializationUpdateSink<M>> Create(
      RootOnlyInitializationTranscriptWriter<M>& transcript,
      std::uint32_t nlist,
      double good_cluster_fraction);

  RootOnlyInitializationUpdateSink(
      RootOnlyInitializationUpdateSink&&) noexcept = default;
  RootOnlyInitializationUpdateSink& operator=(
      RootOnlyInitializationUpdateSink&&) noexcept = default;
  RootOnlyInitializationUpdateSink(
      const RootOnlyInitializationUpdateSink&) = delete;
  RootOnlyInitializationUpdateSink& operator=(
      const RootOnlyInitializationUpdateSink&) = delete;

  [[nodiscard]] Status Commit(
      std::uint64_t sequence,
      RootOnlyInitializationClusterWork<M> work,
      ClusterTopologyResult<M> topology) override;

  [[nodiscard]] StatusOr<InitializationBadClusterResult> Finish();

 private:
  RootOnlyInitializationUpdateSink(
      RootOnlyInitializationTranscriptWriter<M>& transcript,
      std::uint32_t nlist,
      double good_cluster_fraction) noexcept
      : transcript_(&transcript),
        good_cluster_fraction_(good_cluster_fraction),
        sums_(nlist, 0.0),
        counts_(nlist, 0U) {}

  RootOnlyInitializationTranscriptWriter<M>* transcript_ = nullptr;
  double good_cluster_fraction_ = 0.0;
  std::vector<double> sums_;
  std::vector<std::uint64_t> counts_;
  double minimum_squared_error_ = 0.0;
  double maximum_squared_error_ = 0.0;
  bool has_squared_error_ = false;
  std::uint32_t next_cid_ = 0U;
  bool finished_ = false;
};

extern template class RootOnlyInitializationUpdateSink<5>;
extern template class RootOnlyInitializationUpdateSink<10>;

}  // namespace stlqpp
