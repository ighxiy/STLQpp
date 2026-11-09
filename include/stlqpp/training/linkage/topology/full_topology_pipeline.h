#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/training/linkage/topology/full_topology_cluster_builder.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace stlqpp {

template <int M>
class FullTopologyClusterSource {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  virtual ~FullTopologyClusterSource() = default;
  FullTopologyClusterSource(FullTopologyClusterSource&&) noexcept = default;
  FullTopologyClusterSource& operator=(
      FullTopologyClusterSource&&) noexcept = default;
  FullTopologyClusterSource(const FullTopologyClusterSource&) = delete;
  FullTopologyClusterSource& operator=(const FullTopologyClusterSource&) =
      delete;

  [[nodiscard]] virtual StatusOr<
      std::optional<FullTopologyClusterWork<M>>> Next() = 0;
  // Checkpoint resume moves the source to the first uncommitted CID. Generic
  // sources retain correctness by consuming the prefix; list-backed sources
  // override this with an O(1) cursor update.
  [[nodiscard]] virtual Status SkipTo(std::uint32_t next_cid) {
    for (std::uint32_t cid = 0U; cid < next_cid; ++cid) {
      auto skipped = Next();
      if (!skipped.ok()) return skipped.status();
      if (!skipped.value().has_value() || skipped.value()->cid != cid) {
        return Status::DataLoss(
            "full-topology source cannot restore the checkpoint prefix");
      }
    }
    return Status::Ok();
  }

  // Skip one already committed CID while resuming a non-prefix checkpoint.
  // The default consumes from the current position; list sources only move a
  // cursor.
  [[nodiscard]] virtual Status SkipCompletedCid(std::uint32_t cid) {
    auto skipped = Next();
    if (!skipped.ok()) return skipped.status();
    if (!skipped.value().has_value() || skipped.value()->cid != cid)
      return Status::DataLoss("full-topology source cannot skip committed CID");
    return Status::Ok();
  }

 protected:
  FullTopologyClusterSource() = default;
};

// The exact subset of one completed cluster that remains live until ordered
// publication. Topology construction has already consumed every other field
// from FullTopologyClusterWork before this payload is formed. Rotated exact
// rows are present only when the concrete sink declares that it consumes them.
// Width-neutral publication payload; numerical support stays in the product.
struct FullTopologyClusterCommitPayload final {

  std::uint32_t cid = 0U;
  std::uint32_t dimension = 0U;
  std::vector<float> rotated_exact_real;
  std::vector<std::uint64_t> real_sample_ids;
  std::vector<float> source_real_norm2;

  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return static_cast<std::uint32_t>(real_sample_ids.size());
  }
};

// Ordered publication boundary. Calls are source ordered even when cluster
// construction finishes out of order. `payload` and `product` remain live
// only for the duration of this call. The product's construction-only real
// reconstruction matrix has already been released.
template <int M>
class FullTopologyClusterSink {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  virtual ~FullTopologyClusterSink() = default;
  FullTopologyClusterSink(FullTopologyClusterSink&&) noexcept = default;
  FullTopologyClusterSink& operator=(
      FullTopologyClusterSink&&) noexcept = default;
  FullTopologyClusterSink(const FullTopologyClusterSink&) = delete;
  FullTopologyClusterSink& operator=(const FullTopologyClusterSink&) = delete;

  [[nodiscard]] virtual bool RequiresRotatedExactReal()
      const noexcept = 0;
  // Optional per-CID quality handoff before construction-only error rows are
  // released. Called on the owning worker; distinct CIDs may run concurrently.
  [[nodiscard]] virtual bool RequiresQualityBeforeRelease()
      const noexcept { return false; }
  [[nodiscard]] virtual Status ObserveQualityBeforeRelease(
      const ClusterTopologyResult<M>&) { return Status::Ok(); }
  // Optional worker-thread transformation before ordered publication. The
  // default keeps the raw payload/product contract used by training
  // checkpoints. Final publication overrides this to release float forest
  // state after producing its bounded prepared canonical cluster.
  [[nodiscard]] virtual Status PrepareForCommit(
      std::uint64_t,
      FullTopologyClusterCommitPayload*,
      FullTopologyClusterProduct<M>*) {
    return Status::Ok();
  }
  // Includes any prepared data held by the sink between PrepareForCommit and
  // ordered Commit. Called only after preparation has completed.
  [[nodiscard]] virtual std::uint64_t PreparedResidentBytes(
      std::uint64_t) const noexcept { return 0U; }
  [[nodiscard]] virtual bool RequiresOrderedCommit() const noexcept {
    return true;
  }
  [[nodiscard]] virtual Status Commit(
      std::uint64_t sequence,
      FullTopologyClusterCommitPayload payload,
      FullTopologyClusterProduct<M> product) = 0;

 protected:
  FullTopologyClusterSink() = default;
};

struct FullTopologyPipelineOptions final {
  std::uint32_t maximum_inflight_clusters = 0U;
  // Ordered sinks reserve this reorder-cache budget before starting work.
  // Completion-order sinks use the inflight and prefetch count bounds instead:
  // every received result is committed before a new computation slot is filled.
  // Zero uses the internal default. Sizes are reconciled on completion, so a
  // larger result may temporarily exceed the budget; new admissions then stop.
  // A single larger CID may occupy the cache alone.
  std::uint64_t maximum_result_cache_bytes = 0U;
  std::uint32_t first_sequence = 0U;
  std::uint64_t committed_real_rows = 0U;
  // Final-build resume; empty keeps prefix semantics.
  std::vector<std::uint8_t> completed_cids = {};
  // DEV linkage_async_io_depth bounds source-prefetched work independently
  // from executing/completed clusters retained for ordered publication.
  // Zero keeps direct typed callers on the historical overall bound.
  std::uint32_t maximum_prefetched_clusters = 0U;
  std::uint32_t progress_total_clusters = 0U;
  // DEV computes the pre-C_one linkage summary only when its corresponding
  // metric path is enabled. Final-base callers enable this for the ordinary
  // summary; training callers may leave it disabled without adding an O(N)
  // diagnostics pass to the topology hot path.
  bool collect_quality_statistics = false;
};

struct FullTopologyWorkerStatistics final {
  std::uint64_t clusters = 0U;
  std::uint64_t real_rows = 0U;
  std::uint64_t virtual_roots = 0U;
  std::uint64_t unused_virtual_roots_removed = 0U;
  std::uint64_t linked_real_rows = 0U;
  std::uint64_t depth_sum = 0U;
  std::uint32_t maximum_depth = 0U;
  double squared_error_sum = 0.0;
  float minimum_squared_error = 0.0F;
  float maximum_squared_error = 0.0F;
  bool has_squared_error = false;
  // Summed worker service intervals, not disjoint pipeline wall time.
  double topology_seconds = 0.0;
  double preparation_seconds = 0.0;
  double build_seconds = 0.0;
};

struct FullTopologyPipelineStatistics final {
  std::uint64_t clusters = 0U;
  std::uint64_t real_rows = 0U;
  // Counts include checkpoint rows; quality and service statistics describe
  // only the work performed by this invocation.
  std::uint64_t resumed_real_rows = 0U;
  std::uint64_t virtual_roots = 0U;
  std::uint64_t unused_virtual_roots_removed = 0U;
  double wall_seconds = 0.0;
  double source_seconds = 0.0;
  double sink_seconds = 0.0;
  double longest_ordered_wait_seconds = 0.0;
  std::uint64_t peak_result_cache_bytes = 0U;
  std::uint64_t peak_reserved_result_bytes = 0U;
  std::uint32_t peak_completed_ahead = 0U;
  std::uint64_t linked_real_rows = 0U;
  std::uint64_t depth_sum = 0U;
  std::uint32_t maximum_depth = 0U;
  double squared_error_sum = 0.0;
  float minimum_squared_error = 0.0F;
  float maximum_squared_error = 0.0F;
  bool has_squared_error = false;
  std::vector<FullTopologyWorkerStatistics> workers;
};

// Bounded, source-ordered outer topology pipeline. Each executor is bound to
// exactly one worker thread, allowing it to own non-thread-safe root-encoding
// scratch while all builders may still share the thread-safe linkage
// scheduler. Construction may complete out of order; forest publication may
// not.
template <int M>
class FullTopologyPipeline final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<FullTopologyPipelineStatistics> Run(
      FullTopologyClusterSource<M>& source,
      FullTopologyClusterSink<M>& sink,
      ArrayView<FullTopologyClusterExecutor<M>*> executors,
      const FullTopologyPolicy& policy,
      FullTopologyPipelineOptions options);
};

extern template class FullTopologyPipeline<5>;
extern template class FullTopologyPipeline<10>;

}  // namespace stlqpp
