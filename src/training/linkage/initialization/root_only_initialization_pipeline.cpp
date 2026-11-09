#include "stlqpp/training/linkage/initialization/root_only_initialization_pipeline.h"

#include "stlqpp/runtime/concurrency/bounded_queue.h"
#include "stlqpp/runtime/cpu/openmp_affinity.h"
#include "stlqpp/runtime/telemetry/logger.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

template <int M>
struct WorkItem final {
  std::uint64_t sequence = 0U;
  RootOnlyInitializationClusterWork<M> work;
};

template <int M>
struct CompletedItem final {
  std::uint64_t sequence = 0U;
  RootOnlyInitializationClusterWork<M> work;
  ClusterTopologyResult<M> topology;
};

template <int M>
Status ValidateWork(const RootOnlyInitializationClusterWork<M>& work,
                    std::uint64_t expected_sequence) {
  const std::uint64_t rows = work.sample_ids.size();
  const std::uint64_t values =
      static_cast<std::uint64_t>(work.dimension) * rows;
  if (work.cid != expected_sequence || work.dimension == 0U ||
      values > std::numeric_limits<std::size_t>::max() ||
      work.exact_real.size() != values ||
      work.basic_real_reconstructions.size() != values ||
      work.basic_supports.size() != rows ||
      work.basic_coefficients.size() != rows * work.m ||
      work.basic_squared_errors.size() != rows ||
      work.icm_layer_order.size() != static_cast<std::size_t>(work.m)) {
    return Status::InvalidArgument(
        "root-only initialization source returned an invalid cluster");
  }
  for (const LinkageCodeTuple<M>& support : work.basic_supports) {
    if (support.first != work.cid) {
      return Status::InvalidArgument(
          "root-only initialization source changed the native CID");
    }
  }
  return Status::Ok();
}

}  // namespace

template <int M>
StatusOr<RootOnlyInitializationPipelineStatistics>
RootOnlyInitializationPipeline<M>::Run(
    RootOnlyInitializationClusterSource<M>& source,
    RootOnlyInitializationClusterSink<M>& sink,
    ArrayView<RootOnlyInitializationClusterExecutor<M>*> executors,
    const RootOnlyInitializationTopologyPolicy& policy,
    RootOnlyInitializationPipelineOptions options) {
  if (executors.empty() || options.maximum_inflight_clusters == 0U ||
      options.maximum_inflight_clusters < executors.size()) {
    return Status::InvalidArgument(
        "root-only initialization pipeline has invalid bounds");
  }
  for (std::size_t worker = 0U; worker < executors.size(); ++worker) {
    if (executors[worker] == nullptr) {
      return Status::InvalidArgument(
          "root-only initialization pipeline contains a null executor");
    }
    for (std::size_t previous = 0U; previous < worker; ++previous) {
      if (executors[worker] == executors[previous]) {
        return Status::InvalidArgument(
            "root-only initialization workers require distinct executors");
      }
    }
  }

  const std::size_t capacity = options.maximum_inflight_clusters;
  const std::size_t prefetch_capacity = std::min<std::size_t>(
      capacity, options.maximum_prefetched_clusters == 0U
                    ? capacity : options.maximum_prefetched_clusters);
  BoundedQueue<WorkItem<M>> work_queue(prefetch_capacity);
  BoundedQueue<CompletedItem<M>> result_queue(capacity);
  std::atomic<bool> failed{false};
  std::atomic<std::uint32_t> live_workers(
      static_cast<std::uint32_t>(executors.size()));
  std::mutex failure_mutex;
  Status failure = Status::Ok();
  auto fail = [&](Status status) {
    bool expected = false;
    if (failed.compare_exchange_strong(expected, true)) {
      std::lock_guard lock(failure_mutex);
      failure = std::move(status);
    }
    work_queue.Close();
    result_queue.Close();
  };

  RootOnlyInitializationPipelineStatistics statistics;
  statistics.workers.resize(executors.size());
  const auto wall_begin = std::chrono::steady_clock::now();
  auto last_progress = wall_begin;
  const bool live_progress = ProgressStreamIsInteractive();
  std::vector<std::thread> workers;
  workers.reserve(executors.size());
  for (std::size_t worker = 0U; worker < executors.size(); ++worker) {
    workers.emplace_back([&, worker]() {
      if (const Status status = RestoreNativeThreadAffinity(); !status.ok()) {
        fail(status);
        if (live_workers.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
          result_queue.Close();
        }
        return;
      }
      auto& local = statistics.workers[worker];
      while (!failed.load(std::memory_order_acquire)) {
        auto item = work_queue.Pop();
        if (!item.has_value()) {
          break;
        }
        const std::uint32_t rows = item->work.real_count();
        const auto begin = std::chrono::steady_clock::now();
        StatusOr<ClusterTopologyResult<M>> topology = [&]()
            -> StatusOr<ClusterTopologyResult<M>> {
          if (rows != 0U) {
            return executors[worker]->Build(item->work.request(), policy);
          }
          ClusterTopologyResult<M> empty;
          empty.m = item->work.m;
          empty.cid = item->work.cid;
          empty.dimension = item->work.dimension;
          return empty;
        }();
        local.build_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        if (!topology.ok()) {
          fail(topology.status());
          break;
        }
        ++local.clusters;
        local.real_rows += rows;
        if (!result_queue.Push({
                .sequence = item->sequence,
                .work = std::move(item->work),
                .topology = std::move(topology).value(),
            })) {
          break;
        }
      }
      if (live_workers.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
        result_queue.Close();
      }
    });
  }

  std::uint64_t next_sequence = 0U;
  std::uint64_t next_to_commit = 0U;
  std::uint32_t inflight = 0U;
  bool source_exhausted = false;
  std::vector<std::optional<CompletedItem<M>>> reorder(capacity);
  auto fill = [&]() {
    while (!failed.load(std::memory_order_acquire) && !source_exhausted &&
           inflight < options.maximum_inflight_clusters) {
      auto next = source.Next();
      if (!next.ok()) {
        fail(next.status());
        return;
      }
      if (!next.value().has_value()) {
        source_exhausted = true;
        work_queue.Close();
        return;
      }
      auto work = std::move(*next.value());
      if (const Status status = ValidateWork(work, next_sequence);
          !status.ok()) {
        fail(status);
        return;
      }
      if (!work_queue.Push({
              .sequence = next_sequence,
              .work = std::move(work),
          })) {
        return;
      }
      ++next_sequence;
      ++inflight;
    }
  };

  fill();
  while (!failed.load(std::memory_order_acquire) && inflight != 0U) {
    auto completed = result_queue.Pop();
    if (!completed.has_value()) {
      if (!failed.load(std::memory_order_acquire)) {
        fail(Status::Internal(
            "root-only initialization workers stopped early"));
      }
      break;
    }
    assert(completed->sequence >= next_to_commit);
    assert(completed->sequence - next_to_commit < capacity);
    const std::size_t slot =
        static_cast<std::size_t>(completed->sequence % capacity);
    assert(!reorder[slot].has_value());
    reorder[slot] = std::move(*completed);
    while (inflight != 0U) {
      const std::size_t next_slot =
          static_cast<std::size_t>(next_to_commit % capacity);
      if (!reorder[next_slot].has_value()) {
        break;
      }
      auto item = std::move(*reorder[next_slot]);
      reorder[next_slot].reset();
      const std::uint32_t rows = item.work.real_count();
      const Status status = sink.Commit(
          next_to_commit, std::move(item.work), std::move(item.topology));
      if (!status.ok()) {
        fail(status);
        break;
      }
      ++statistics.clusters;
      statistics.real_rows += rows;
      ++next_to_commit;
      --inflight;
      if (options.progress_total_clusters != 0U && live_progress &&
          statistics.clusters < options.progress_total_clusters) {
        const auto now = std::chrono::steady_clock::now();
        if (statistics.clusters <= 16U ||
            statistics.clusters % 16U == 0U ||
            now - last_progress >= std::chrono::seconds(1)) {
          LogRawProgress(
              "Linkage init build: " +
              std::to_string(statistics.clusters) + "/" +
              std::to_string(options.progress_total_clusters) +
              " clusters   ");
          last_progress = now;
        }
      }
      fill();
    }
  }

  work_queue.Close();
  result_queue.Close();
  for (std::thread& worker : workers) {
    worker.join();
  }
  if (failed.load(std::memory_order_acquire)) {
    std::lock_guard lock(failure_mutex);
    return failure;
  }
  if (options.progress_total_clusters != 0U) {
    LogRawProgressDone(
        "Linkage init build: " + std::to_string(statistics.clusters) + "/" +
        std::to_string(options.progress_total_clusters) + " clusters   ");
  }
  statistics.wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall_begin).count();
  return statistics;
}

template class RootOnlyInitializationPipeline<0>;
template class RootOnlyInitializationPipeline<5>;
template class RootOnlyInitializationPipeline<12>;
template class RootOnlyInitializationPipeline<6>;
template class RootOnlyInitializationPipeline<10>;

}  // namespace stlqpp
