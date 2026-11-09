#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/training/linkage/replay/fixed_topology_replay_executor.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace stlqpp {

template <int M>
class CheckpointReplayWindowLoader;
class CheckpointReplayWindowLoaderBase;
template <int M>
class FixedTopologyReplayPipeline;

// One owned gather slot. All row streams are contiguous across its clusters,
// so clearing a slot retains the high-water allocation for its next load.
// Width is window metadata; none of these buffers depend on a numerical
// specialization. Append requires a uniform M until the next Clear.
class FixedTopologyReplayWindow final {
 public:
  void Clear() noexcept;
  [[nodiscard]] Status Append(
      const FixedTopologyReplayRequest& request);
  [[nodiscard]] std::vector<FixedTopologyReplayRequest> Requests() const;

  [[nodiscard]] std::uint32_t cluster_count() const noexcept {
    return static_cast<std::uint32_t>(clusters_.size());
  }
  [[nodiscard]] std::uint64_t real_row_count() const noexcept {
    return real_sample_ids_.size();
  }

 private:
  template <int M>
  friend class CheckpointReplayWindowLoader;
  friend class CheckpointReplayWindowLoaderBase;
  template <int M>
  friend class FixedTopologyReplayPipeline;

  [[nodiscard]] Status AppendImpl(
      const FixedTopologyReplayRequest& request,
      bool copy_matrices);

  struct ClusterSlice final {
    std::uint32_t cid = 0U;
    std::uint32_t dimension = 0U;
    std::uint32_t real_begin = 0U;
    std::uint32_t real_count = 0U;
    std::uint32_t virtual_begin = 0U;
    std::uint32_t virtual_count = 0U;
    std::uint32_t seed = 0U;
  };

  std::uint32_t m_ = 0U;
  std::vector<ClusterSlice> clusters_;
  std::vector<float> exact_real_;
  std::vector<float> unrotated_exact_real_;
  std::vector<std::uint64_t> real_sample_ids_;
  std::vector<std::uint64_t> virtual_sample_ids_;
  std::vector<std::uint8_t> basic_small_codes_;
  std::vector<float> basic_coefficients_;
  std::vector<std::uint8_t> frozen_virtual_small_codes_;
  std::vector<float> frozen_virtual_coefficients_;
  std::vector<LinkageNodeRole> roles_;
  std::vector<std::optional<LinkageParentReference>> parents_;
  std::vector<std::uint32_t> depths_;
  std::vector<std::uint8_t> edge_icm_layer_orders_;
  // Set only after the checkpoint loader has validated and assembled every
  // cluster. Public Append and Clear revoke this provenance.
  bool validated_checkpoint_payload_ = false;
};

// Loader is called only by the producer thread. It appends one complete
// immutable window and returns false at end of stream.
class FixedTopologyReplayWindowLoader {
 public:
  virtual ~FixedTopologyReplayWindowLoader() = default;
  [[nodiscard]] virtual StatusOr<bool> LoadNext(
      FixedTopologyReplayWindow* window) = 0;
};

// Sink is called in sequence order by the consumer thread after a complete
// window has crossed every depth barrier.
template <int M>
class ReplayRoundUpdateSink;

template <int M>
class FixedTopologyReplayWindowSink {
 public:
  virtual ~FixedTopologyReplayWindowSink() = default;
  // Custom sinks retain the historical complete result unless they explicitly
  // declare that real reconstructions are not consumed.
  [[nodiscard]] virtual bool RequiresRealReconstructions() const noexcept {
    return true;
  }
  [[nodiscard]] virtual Status Commit(
      std::uint64_t sequence,
      ArrayView<const FixedTopologyReplayRequest> requests,
      ArrayView<const FixedTopologyReplayResult<M>> results) = 0;

 private:
  friend class FixedTopologyReplayPipeline<M>;
  friend class ReplayRoundUpdateSink<M>;
  // Only a successful executor result in the same immutable window may use
  // this boundary. Independent sinks retain Commit's full checks.
  [[nodiscard]] virtual Status CommitValidatedFromExecutor(
      std::uint64_t sequence,
      ArrayView<const FixedTopologyReplayRequest> requests,
      ArrayView<const FixedTopologyReplayResult<M>> results) {
    return Commit(sequence, requests, results);
  }
};

struct FixedTopologyReplayPipelineStatistics final {
  std::uint64_t windows = 0U;
  std::uint64_t clusters = 0U;
  std::uint64_t real_rows = 0U;
  std::uint64_t linked_real_rows = 0U;
  std::uint64_t depth_sum = 0U;
  std::uint32_t maximum_depth = 0U;
  double squared_error_sum = 0.0;
  float minimum_squared_error = 0.0F;
  float maximum_squared_error = 0.0F;
  bool has_squared_error = false;
  double wall_seconds = 0.0;
  double loader_seconds = 0.0;
  double request_seconds = 0.0;
  double executor_seconds = 0.0;
  double sink_seconds = 0.0;
};

// Three reusable host slots overlap mapped gather/load, semantic replay and
// single-threaded ordered publication without copying request matrices.
// The executor remains the sole owner of
// replay mathematics and topology causality.
template <int M>
class FixedTopologyReplayPipeline final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<FixedTopologyReplayPipelineStatistics> Run(
      FixedTopologyReplayWindowLoader& loader,
      const FixedTopologyReplayExecutor<M>& executor,
      FixedTopologyReplayWindowSink<M>& sink,
      bool collect_quality_statistics = false);
};

extern template class FixedTopologyReplayPipeline<5>;
extern template class FixedTopologyReplayPipeline<10>;

}  // namespace stlqpp
