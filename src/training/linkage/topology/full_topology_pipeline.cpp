#include "stlqpp/training/linkage/topology/full_topology_pipeline.h"

#include "stlqpp/runtime/concurrency/bounded_queue.h"
#include "stlqpp/runtime/cpu/openmp_affinity.h"
#include "stlqpp/runtime/telemetry/logger.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

template <int M>
struct FullTopologyWorkItem final {
  std::uint64_t sequence = 0U;
  std::uint64_t reserved_result_bytes = 0U;
  FullTopologyClusterWork<M> work;
};

template <int M>
struct FullTopologyCompletedItem final {
  std::uint64_t sequence = 0U;
  std::uint64_t reserved_result_bytes = 0U;
  std::uint64_t resident_bytes = 0U;
  std::uint32_t real_count = 0U;
  FullTopologyClusterCommitPayload payload;
  FullTopologyClusterProduct<M> product;
};

template <int M>
std::uint64_t ResultResidentBytes(
    const FullTopologyClusterCommitPayload& payload,
    const FullTopologyClusterProduct<M>& product) noexcept {
  const auto& topology = product.topology;
  std::uint64_t bytes = sizeof(FullTopologyCompletedItem<M>);
  bytes += static_cast<std::uint64_t>(payload.rotated_exact_real.capacity() +
                                      payload.source_real_norm2.capacity()) * sizeof(float);
  bytes += static_cast<std::uint64_t>(payload.real_sample_ids.capacity()) * sizeof(std::uint64_t);
  bytes += topology.roles.capacity() * sizeof(LinkageNodeRole);
  bytes += topology.parents.capacity() * sizeof(std::optional<LinkageParentReference>);
  bytes += static_cast<std::uint64_t>(topology.depths.capacity()) * sizeof(std::uint32_t);
  bytes += topology.supports.capacity() * sizeof(LinkageCodeTuple<M>);
  bytes += static_cast<std::uint64_t>(topology.coefficients.capacity() +
                                      topology.squared_errors.capacity() +
                                      topology.real_reconstructions.capacity()) * sizeof(float);
  if (product.virtual_roots.has_value()) {
    bytes += product.virtual_roots->resident_bytes();
  }
  return bytes;
}

template <int M>
std::uint64_t ReserveResultBytes(
    const FullTopologyClusterWork<M>& work,
    const FullTopologyClusterSink<M>& sink,
    const FullTopologyPolicy& policy) noexcept {
  // Admit using the expected topology/prepared size and the actual capacity
  // of source buffers that may be moved into the completed result. Results
  // are reconciled against their measured size after completion, without
  // making a valid result fail merely because an estimate was too small.
  const std::uint64_t rows = work.real_count();
  const std::uint64_t virtual_roots =
      work.bad_cluster && policy.virtual_enabled
          ? std::min<std::uint64_t>(rows, policy.maximum_virtual_roots)
          : 0U;
  const std::uint64_t per_row = 512U +
      (sink.RequiresRotatedExactReal()
           ? static_cast<std::uint64_t>(work.dimension) * sizeof(float)
           : 0U);
  constexpr std::uint64_t kOverhead = 16U * 1024U;
  const std::uint64_t retained_input_bytes =
      static_cast<std::uint64_t>(work.real_sample_ids.capacity()) *
          sizeof(std::uint64_t) +
      static_cast<std::uint64_t>(work.source_real_norm2.capacity()) *
          sizeof(float) +
      (sink.RequiresRotatedExactReal()
           ? static_cast<std::uint64_t>(work.rotated_exact_real.capacity()) *
                 sizeof(float)
           : 0U);
  if (retained_input_bytes >
          std::numeric_limits<std::uint64_t>::max() - kOverhead ||
      rows > (std::numeric_limits<std::uint64_t>::max() - kOverhead -
              retained_input_bytes) /
                 per_row ||
      virtual_roots >
          (std::numeric_limits<std::uint64_t>::max() - kOverhead -
           retained_input_bytes - rows * per_row) / 512U) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return kOverhead + retained_input_bytes + rows * per_row +
      virtual_roots * 512U;
}

template <int M>
FullTopologyClusterCommitPayload TakeCommitPayload(
    FullTopologyClusterWork<M>* work,
    bool retain_rotated_exact_real) {
  return {
      .cid = work->cid,
      .dimension = work->dimension,
      .rotated_exact_real = retain_rotated_exact_real
          ? std::move(work->rotated_exact_real)
          : std::vector<float>{},
      .real_sample_ids = std::move(work->real_sample_ids),
      .source_real_norm2 = std::move(work->source_real_norm2),
  };
}

template <int M>
Status ValidateWork(const FullTopologyClusterWork<M>& work,
                    std::uint64_t expected_sequence) {
  const std::uint64_t real_count = work.real_sample_ids.size();
  const std::uint64_t vector_count =
      static_cast<std::uint64_t>(work.dimension) * real_count;
  const bool valid_unrotated = work.unrotated_exact_real.size() ==
          vector_count ||
      (!work.bad_cluster && work.unrotated_exact_real.empty());
  const bool valid_source_norm2 = work.source_real_norm2.empty() ||
      work.source_real_norm2.size() == real_count;
  if (work.cid != expected_sequence || work.dimension == 0U ||
      vector_count > static_cast<std::uint64_t>(
                         std::numeric_limits<std::size_t>::max()) ||
      !valid_unrotated || !valid_source_norm2 ||
      work.rotated_exact_real.size() != vector_count ||
      work.basic_real_reconstructions.size() != vector_count ||
      work.basic_supports.size() != real_count ||
      work.basic_coefficients.size() != real_count * work.m ||
      work.basic_squared_errors.size() != real_count) {
    return Status::InvalidArgument(
        "full-topology source returned an invalid or unordered cluster");
  }
  for (const LinkageCodeTuple<M>& support : work.basic_supports) {
    if (support.first != work.cid) {
      return Status::InvalidArgument(
          "full-topology source changed the native basic CID");
    }
  }
  if (real_count == 0U && work.bad_cluster) {
    return Status::InvalidArgument(
        "empty full-topology cluster carries bad-cluster state");
  }
  return Status::Ok();
}

// The admission budget and ordered commit cursor do not depend on support
// width. Typed queues retain ownership of work/products; this control object
// only tracks their sequence and byte reservations at cluster boundaries.
class FullPipelineControl final {
 public:
  FullPipelineControl(FullTopologyPipelineOptions options,
                      std::uint32_t workers,
                      bool requires_ordered_commit,
                      FullTopologyPipelineStatistics& statistics,
                      std::chrono::steady_clock::time_point start) noexcept
      : options_(options),
        statistics_(statistics),
        cache_limit_(options.maximum_result_cache_bytes == 0U
                         ? 512ULL * 1024ULL * 1024ULL
                         : options.maximum_result_cache_bytes),
        requires_ordered_commit_(requires_ordered_commit),
        next_sequence_(options.first_sequence),
        next_to_commit_(options.first_sequence),
        completed_clusters_(statistics.clusters),
        committed_(options.completed_cids),
        exact_knn_cpu_budget_(workers),
        last_progress_(start),
        live_progress_(ProgressStreamIsInteractive()) {}

  [[nodiscard]] bool failed() const noexcept {
    return failed_.load(std::memory_order_acquire);
  }
  void Fail(Status status) {
    bool expected = false;
    if (failed_.compare_exchange_strong(expected, true)) {
      exact_knn_cpu_budget_.Cancel();
      std::lock_guard lock(failure_mutex_);
      failure_ = std::move(status);
    }
  }
  [[nodiscard]] Status failure() {
    std::lock_guard lock(failure_mutex_);
    return failure_;
  }
  [[nodiscard]] bool WorkerFinished() noexcept {
    return exact_knn_cpu_budget_.RetireWorker();
  }
  [[nodiscard]] ExactKnnCpuBudget& exact_knn_cpu_budget() noexcept {
    return exact_knn_cpu_budget_;
  }

  [[nodiscard]] std::uint64_t next_sequence() const noexcept {
    return next_sequence_;
  }
  void Fetched() noexcept { ++next_sequence_; }
  [[nodiscard]] std::uint64_t next_to_commit() const noexcept {
    return next_to_commit_;
  }
  [[nodiscard]] bool HasAdmissionSlot() const noexcept {
    return active_ < options_.maximum_inflight_clusters;
  }
  [[nodiscard]] bool CanReadSource() const noexcept {
    return !requires_ordered_commit_ || reserved_cache_bytes_ < cache_limit_;
  }
  [[nodiscard]] bool CanAdmit(std::uint64_t bytes) const noexcept {
    // Only ordered sinks can accumulate results behind a slow CID after
    // releasing their computation slots. Completion-order sinks commit each
    // received result before refilling: the inflight count bounds their work
    // and results together. A reorder-cache budget must not idle their workers
    // merely because several executing clusters have large estimated results.
    return !requires_ordered_commit_ || reserved_cache_bytes_ == 0U ||
        (reserved_cache_bytes_ < cache_limit_ &&
         bytes <= cache_limit_ - reserved_cache_bytes_);
  }
  void Admitted(std::uint64_t bytes) noexcept {
    reserved_cache_bytes_ += bytes;
    statistics_.peak_reserved_result_bytes = std::max(
        statistics_.peak_reserved_result_bytes, reserved_cache_bytes_);
    ++active_;
  }
  [[nodiscard]] bool HasActive() const noexcept { return active_ != 0U; }
  [[nodiscard]] Status Completed(std::uint64_t sequence,
                                 std::uint64_t resident_bytes,
                                 std::uint64_t& reserved_bytes) {
    assert(active_ != 0U && sequence >= next_to_commit_);
    (void)sequence;
    --active_;
    if (resident_bytes > reserved_bytes) {
      const std::uint64_t extra = resident_bytes - reserved_bytes;
      if (extra > std::numeric_limits<std::uint64_t>::max() -
                      reserved_cache_bytes_) {
        return Status::ResourceExhausted(
            "full-topology result cache accounting overflowed");
      }
      reserved_cache_bytes_ += extra;
      reserved_bytes = resident_bytes;
      statistics_.peak_reserved_result_bytes = std::max(
          statistics_.peak_reserved_result_bytes, reserved_cache_bytes_);
    }
    completed_cache_bytes_ += resident_bytes;
    statistics_.peak_result_cache_bytes = std::max(
        statistics_.peak_result_cache_bytes, completed_cache_bytes_);
    ++completed_clusters_;
    return Status::Ok();
  }
  void RecordedArrival(bool has_next, std::size_t ready_count) {
    if (!has_next) {
      if (!ordered_wait_begin_.has_value()) {
        ordered_wait_begin_ = std::chrono::steady_clock::now();
      }
      statistics_.peak_completed_ahead = std::max(
          statistics_.peak_completed_ahead,
          static_cast<std::uint32_t>(ready_count));
    } else if (ordered_wait_begin_.has_value()) {
      statistics_.longest_ordered_wait_seconds = std::max(
          statistics_.longest_ordered_wait_seconds,
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() -
              *ordered_wait_begin_).count());
      ordered_wait_begin_.reset();
    }
  }
  void RemovedReady(std::uint64_t resident_bytes) noexcept {
    completed_cache_bytes_ -= resident_bytes;
  }
  void Committed(std::uint64_t sequence,
                 std::uint64_t reserved_bytes) noexcept {
    if (committed_.empty()) {
      if (sequence == next_to_commit_) ++next_to_commit_;
    } else {
      committed_[sequence] = 1U;
      while (next_to_commit_ < committed_.size() && committed_[next_to_commit_])
        ++next_to_commit_;
    }
    reserved_cache_bytes_ -= reserved_bytes;
  }
  void ReportProgress() {
    if (options_.progress_total_clusters != 0U && live_progress_) {
      const auto now = std::chrono::steady_clock::now();
      if (completed_clusters_ <= 16U ||
          completed_clusters_ % 16U == 0U ||
          now - last_progress_ >= std::chrono::seconds(1)) {
        LogRawProgress(ProgressText());
        last_progress_ = now;
      }
    }
  }
  void FinishProgress() const {
    if (options_.progress_total_clusters != 0U) {
      LogRawProgressDone(ProgressText());
    }
  }

 private:
  // Completion is observed on the existing result queue, independently of
  // the ordered sink cursor. "Committed" means accepted by the sink; a spool
  // window/checkpoint may be flushed later by that sink.
  [[nodiscard]] std::string ProgressText() const {
    return "Linkage build: completed " + std::to_string(completed_clusters_) +
        "/" + std::to_string(options_.progress_total_clusters) +
        " clusters, committed " + std::to_string(statistics_.clusters) +
        ", inflight " + std::to_string(active_) +
        ", waiting " + std::to_string(completed_clusters_ - statistics_.clusters) +
        " (next CID " + std::to_string(next_to_commit_) + ")";
  }
  FullTopologyPipelineOptions options_;
  FullTopologyPipelineStatistics& statistics_;
  std::uint64_t cache_limit_ = 0U;
  bool requires_ordered_commit_ = true;
  std::uint64_t next_sequence_ = 0U;
  std::uint64_t next_to_commit_ = 0U;
  std::uint64_t completed_clusters_ = 0U;
  std::vector<std::uint8_t> committed_;
  std::uint32_t active_ = 0U;
  std::uint64_t reserved_cache_bytes_ = 0U;
  std::uint64_t completed_cache_bytes_ = 0U;
  std::optional<std::chrono::steady_clock::time_point> ordered_wait_begin_;
  std::atomic<bool> failed_{false};
  ExactKnnCpuBudget exact_knn_cpu_budget_;
  std::mutex failure_mutex_;
  Status failure_ = Status::Ok();
  std::chrono::steady_clock::time_point last_progress_;
  bool live_progress_ = false;
};

}  // namespace

template <int M>
StatusOr<FullTopologyPipelineStatistics> FullTopologyPipeline<M>::Run(
    FullTopologyClusterSource<M>& source,
    FullTopologyClusterSink<M>& sink,
    ArrayView<FullTopologyClusterExecutor<M>*> executors,
    const FullTopologyPolicy& policy,
    FullTopologyPipelineOptions options) {
  if (executors.empty() || options.maximum_inflight_clusters == 0U ||
      options.maximum_inflight_clusters < executors.size()) {
    return Status::InvalidArgument(
        "full-topology pipeline has an invalid worker or inflight bound");
  }
  for (std::size_t worker = 0U; worker < executors.size(); ++worker) {
    if (executors[worker] == nullptr) {
      return Status::InvalidArgument(
          "full-topology pipeline contains a null executor");
    }
    for (std::size_t previous = 0U; previous < worker; ++previous) {
      if (executors[worker] == executors[previous]) {
        return Status::InvalidArgument(
            "full-topology workers must own distinct executors");
      }
    }
  }

  if (!sink.RequiresOrderedCommit() && options.completed_cids.empty()) {
    options.completed_cids.assign(options.progress_total_clusters, 0U);
    if (options.first_sequence <= options.completed_cids.size())
      std::fill_n(options.completed_cids.begin(), options.first_sequence, 1U);
  }
  if (!options.completed_cids.empty()) {
    std::size_t first_missing = 0U;
    while (first_missing < options.completed_cids.size() &&
           options.completed_cids[first_missing] == 1U)
      ++first_missing;
    if (first_missing != options.first_sequence ||
        (options.progress_total_clusters != 0U &&
         options.completed_cids.size() != options.progress_total_clusters) ||
        std::any_of(options.completed_cids.begin(),
                    options.completed_cids.end(),
                    [](auto value) { return value > 1U; })) {
      return Status::InvalidArgument(
          "full-topology completion bitmap is invalid");
    }
  }
  const std::size_t capacity = options.maximum_inflight_clusters;
  const std::size_t prefetch_capacity = std::min<std::size_t>(
      capacity, options.maximum_prefetched_clusters == 0U
                    ? capacity : options.maximum_prefetched_clusters);
  const bool retain_rotated_exact_real =
      sink.RequiresRotatedExactReal();
  BoundedQueue<FullTopologyWorkItem<M>> work_queue(prefetch_capacity);
  BoundedQueue<FullTopologyCompletedItem<M>> result_queue(capacity);
  FullTopologyPipelineStatistics statistics;
  if (options.first_sequence > options.progress_total_clusters &&
      options.progress_total_clusters != 0U) {
    return Status::InvalidArgument(
        "full-topology checkpoint cursor exceeds the progress domain");
  }
  statistics.clusters = options.completed_cids.empty()
                            ? options.first_sequence
                            : static_cast<std::uint32_t>(
                                  std::count(options.completed_cids.begin(),
                                             options.completed_cids.end(), 1U));
  statistics.real_rows = options.committed_real_rows;
  statistics.resumed_real_rows = options.committed_real_rows;
  statistics.workers.resize(executors.size());
  const auto wall_begin = std::chrono::steady_clock::now();
  FullPipelineControl control(
      options, static_cast<std::uint32_t>(executors.size()),
      sink.RequiresOrderedCommit(), statistics, wall_begin);
  auto fail = [&](Status status) {
    control.Fail(std::move(status));
    work_queue.Close();
    result_queue.Close();
  };
  std::vector<std::thread> workers;
  workers.reserve(executors.size());
  for (std::size_t worker = 0U; worker < executors.size(); ++worker) {
    workers.emplace_back([&, worker]() {
      if (const Status status = RestoreNativeThreadAffinity(); !status.ok()) {
        fail(status);
        if (control.WorkerFinished()) {
          result_queue.Close();
        }
        return;
      }
      FullTopologyWorkerStatistics& local = statistics.workers[worker];
      while (!control.failed()) {
        auto item = work_queue.Pop();
        if (!item.has_value()) {
          break;
        }
        const std::uint32_t real_count = item->work.real_count();
        const auto begin = std::chrono::steady_clock::now();
        StatusOr<FullTopologyClusterProduct<M>> product = [&]()
            -> StatusOr<FullTopologyClusterProduct<M>> {
          if (item->work.real_count() != 0U) {
            auto request = item->work.request();
            request.exact_knn_cpu_budget = &control.exact_knn_cpu_budget();
            return executors[worker]->Build(request, policy);
          }
          FullTopologyClusterProduct<M> empty;
          empty.topology.m = item->work.m;
          empty.topology.cid = item->work.cid;
          empty.topology.dimension = item->work.dimension;
          return empty;
        }();
        local.topology_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        if (!product.ok()) {
          fail(product.status());
          break;
        }
        const std::uint64_t sequence = item->sequence;
        const std::uint64_t reserved_result_bytes =
            item->reserved_result_bytes;
        FullTopologyClusterCommitPayload payload =
            TakeCommitPayload(
                &item->work, retain_rotated_exact_real);
        item.reset();
        FullTopologyClusterProduct<M> completed_product =
            std::move(product).value();
        const std::uint32_t generated_virtual_roots =
            completed_product.topology.virtual_count;
        if (options.collect_quality_statistics) {
          const ClusterTopologyResult<M>& topology =
              completed_product.topology;
          if (topology.roles.size() != real_count ||
              topology.depths.size() != real_count ||
              topology.squared_errors.size() != real_count) {
            fail(Status::DataLoss(
                "full-topology quality streams changed cluster shape"));
            break;
          }
          for (std::uint32_t row = 0U; row < real_count; ++row) {
            const float error = topology.squared_errors[row];
            if (!std::isfinite(error) || error < 0.0F) {
              fail(Status::DataLoss(
                  "full-topology produced an invalid squared error"));
              break;
            }
            local.linked_real_rows +=
                topology.roles[row] == LinkageNodeRole::kLinked ? 1U : 0U;
            local.depth_sum += topology.depths[row];
            local.maximum_depth = std::max(
                local.maximum_depth, topology.depths[row]);
            local.squared_error_sum += error;
            if (!local.has_squared_error) {
              local.minimum_squared_error = error;
              local.maximum_squared_error = error;
              local.has_squared_error = true;
            } else {
              local.minimum_squared_error = std::min(
                  local.minimum_squared_error, error);
              local.maximum_squared_error = std::max(
                  local.maximum_squared_error, error);
            }
          }
          if (control.failed()) break;
        }
        if (sink.RequiresQualityBeforeRelease()) {
          if (const Status status = sink.ObserveQualityBeforeRelease(
                  completed_product.topology);
              !status.ok()) {
            fail(status);
            break;
          }
        }
        std::vector<float>().swap(
            completed_product.topology.real_reconstructions);
        std::vector<float>().swap(
            completed_product.topology.squared_errors);
        const auto preparation_begin = std::chrono::steady_clock::now();
        if (const Status status = sink.PrepareForCommit(
                sequence, &payload, &completed_product);
            !status.ok()) {
          fail(status);
          break;
        }
        local.preparation_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - preparation_begin).count();
        if (completed_product.topology.virtual_count >
            generated_virtual_roots) {
          fail(Status::DataLoss(
              "full-topology sink increased the virtual-root count"));
          break;
        }
        local.virtual_roots += completed_product.topology.virtual_count;
        local.unused_virtual_roots_removed +=
            generated_virtual_roots -
            completed_product.topology.virtual_count;
        local.build_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        ++local.clusters;
        local.real_rows += real_count;
        const std::uint64_t resident_bytes =
            ResultResidentBytes(payload, completed_product) +
            sink.PreparedResidentBytes(sequence);
        if (!result_queue.Push({
                .sequence = sequence,
                .reserved_result_bytes = reserved_result_bytes,
                .resident_bytes = resident_bytes,
                .real_count = real_count,
                .payload = std::move(payload),
                .product = std::move(completed_product),
            })) {
          break;
        }
      }
      if (control.WorkerFinished()) {
        result_queue.Close();
      }
    });
  }

  bool source_exhausted = false;
  std::optional<FullTopologyWorkItem<M>> deferred_work;
  std::map<std::uint64_t, FullTopologyCompletedItem<M>> ready;

  auto fill = [&]() {
    while (!control.failed() && !source_exhausted &&
           control.HasAdmissionSlot()) {
      if (!deferred_work.has_value()) {
        while (control.next_sequence() < options.completed_cids.size() &&
               options.completed_cids[control.next_sequence()]) {
          const auto cid = static_cast<std::uint32_t>(control.next_sequence());
          control.Fetched();
          if (auto status = source.SkipCompletedCid(cid); !status.ok()) {
            fail(status);
            return;
          }
        }
        if (!control.CanReadSource()) return;
        const auto source_begin = std::chrono::steady_clock::now();
        auto next = source.Next();
        statistics.source_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - source_begin).count();
        if (!next.ok()) {
          fail(next.status());
          return;
        }
        if (!next.value().has_value()) {
          source_exhausted = true;
          work_queue.Close();
          return;
        }
        FullTopologyClusterWork<M> work =
            std::move(*next.value());
        if (const Status status = ValidateWork(work, control.next_sequence());
            !status.ok()) {
          fail(status);
          return;
        }
        const std::uint64_t reservation =
            ReserveResultBytes(work, sink, policy);
        deferred_work.emplace(FullTopologyWorkItem<M>{
            .sequence = control.next_sequence(),
            .reserved_result_bytes = reservation,
            .work = std::move(work),
        });
        control.Fetched();
      }
      const std::uint64_t reservation =
          deferred_work->reserved_result_bytes;
      if (!control.CanAdmit(reservation)) return;
      if (!work_queue.Push(std::move(*deferred_work))) {
        return;
      }
      deferred_work.reset();
      control.Admitted(reservation);
    }
  };

  fill();
  control.ReportProgress();
  while (!control.failed() &&
         (control.HasActive() || !ready.empty() || deferred_work.has_value())) {
    auto completed = result_queue.Pop();
    if (!completed.has_value()) {
      if (!control.failed()) {
        fail(Status::Internal(
            "full-topology workers stopped before completing all clusters"));
      }
      break;
    }
    if (const Status status = control.Completed(
            completed->sequence, completed->resident_bytes,
            completed->reserved_result_bytes);
        !status.ok()) {
      fail(status);
      break;
    }
    if (!sink.RequiresOrderedCommit()) {
      control.RemovedReady(completed->resident_bytes);
      const auto sequence = completed->sequence;
      const auto real_count = completed->real_count;
      const auto reservation = completed->reserved_result_bytes;
      const auto begin = std::chrono::steady_clock::now();
      auto status = sink.Commit(sequence, std::move(completed->payload),
                                std::move(completed->product));
      statistics.sink_seconds += std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - begin)
                                     .count();
      if (!status.ok()) {
        fail(status);
        break;
      }
      ++statistics.clusters;
      statistics.real_rows += real_count;
      control.Committed(sequence, reservation);
      fill();
      control.ReportProgress();
      continue;
    }
    const auto insertion = ready.emplace(
        completed->sequence, std::move(*completed));
    if (!insertion.second) {
      fail(Status::Internal("full-topology completed a CID more than once"));
      break;
    }
    control.RecordedArrival(
        ready.find(control.next_to_commit()) != ready.end(), ready.size());

    for (auto next = ready.find(control.next_to_commit()); next != ready.end();
         next = ready.find(control.next_to_commit())) {
      auto item = std::move(next->second);
      ready.erase(next);
      control.RemovedReady(item.resident_bytes);
      const std::uint32_t real_count = item.real_count;
      const auto sink_begin = std::chrono::steady_clock::now();
      const Status status = sink.Commit(
          control.next_to_commit(), std::move(item.payload),
          std::move(item.product));
      statistics.sink_seconds += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - sink_begin).count();
      if (!status.ok()) {
        fail(status);
        break;
      }
      ++statistics.clusters;
      statistics.real_rows += real_count;
      control.Committed(item.sequence, item.reserved_result_bytes);
    }
    if (!control.failed()) {
      fill();
      control.ReportProgress();
    }
  }

  work_queue.Close();
  result_queue.Close();
  for (std::thread& worker : workers) {
    worker.join();
  }
  if (control.failed()) return control.failure();
  control.FinishProgress();
  for (const FullTopologyWorkerStatistics& worker : statistics.workers) {
    statistics.virtual_roots += worker.virtual_roots;
    statistics.unused_virtual_roots_removed +=
        worker.unused_virtual_roots_removed;
    statistics.linked_real_rows += worker.linked_real_rows;
    statistics.depth_sum += worker.depth_sum;
    statistics.maximum_depth = std::max(
        statistics.maximum_depth, worker.maximum_depth);
    statistics.squared_error_sum += worker.squared_error_sum;
    if (worker.has_squared_error) {
      if (!statistics.has_squared_error) {
        statistics.minimum_squared_error = worker.minimum_squared_error;
        statistics.maximum_squared_error = worker.maximum_squared_error;
        statistics.has_squared_error = true;
      } else {
        statistics.minimum_squared_error = std::min(
            statistics.minimum_squared_error, worker.minimum_squared_error);
        statistics.maximum_squared_error = std::max(
            statistics.maximum_squared_error, worker.maximum_squared_error);
      }
    }
  }
  statistics.wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall_begin).count();
  return statistics;
}

template class FullTopologyPipeline<0>;
template class FullTopologyPipeline<5>;
template class FullTopologyPipeline<12>;
template class FullTopologyPipeline<6>;
template class FullTopologyPipeline<10>;

}  // namespace stlqpp
