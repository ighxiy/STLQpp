#include "stlqpp/training/linkage/replay/fixed_topology_replay_pipeline.h"

#include "stlqpp/runtime/cpu/openmp_affinity.h"
#include "stlqpp/runtime/concurrency/bounded_queue.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

namespace stlqpp {
namespace {

template <typename T>
ArrayView<const T> Slice(const std::vector<T>& values,
                         std::size_t begin,
                         std::size_t count) noexcept {
  return count == 0U
      ? ArrayView<const T>{}
      : ArrayView<const T>{values.data() + begin, count};
}

enum class ReplaySlotState : std::uint8_t {
  kEmpty,
  kFilling,
  kReady,
  kConsuming,
};

struct ReplaySlot final {
  FixedTopologyReplayWindow window;
  ReplaySlotState state = ReplaySlotState::kEmpty;
  std::uint64_t sequence = 0U;
};

// Three reusable windows let loading, encoding and ordered publication overlap.
// Their producer/consumer handoff is independent
// of M. Numerical execution and result ownership stay in the typed caller.
class ReplayWindowControl final {
 public:
  explicit ReplayWindowControl(FixedTopologyReplayWindowLoader& loader) noexcept
      : loader_(loader) {}
  ReplayWindowControl(const ReplayWindowControl&) = delete;
  ReplayWindowControl& operator=(const ReplayWindowControl&) = delete;
  ~ReplayWindowControl() { (void)Stop(); }

  void Start() {
    producer_ = std::thread([this]() { Produce(); });
  }

  [[nodiscard]] StatusOr<ReplaySlot*> Acquire() {
    std::unique_lock lock(mutex_);
    ReplaySlot& slot = slots_[next_to_consume_ % slots_.size()];
    changed_.wait(lock, [&]() {
      return cancelled_ || !producer_status_.ok() ||
          (slot.state == ReplaySlotState::kReady &&
           slot.sequence == next_to_consume_) || producer_finished_;
    });
    if (!producer_status_.ok()) {
      cancelled_ = true;
      return producer_status_;
    }
    if (cancelled_) return nullptr;
    if (slot.state != ReplaySlotState::kReady ||
        slot.sequence != next_to_consume_) return nullptr;
    slot.state = ReplaySlotState::kConsuming;
    ++next_to_consume_;
    return &slot;
  }

  void Release(ReplaySlot& slot, bool success) {
    {
      std::lock_guard lock(mutex_);
      slot.state = ReplaySlotState::kEmpty;
      if (!success) {
        cancelled_ = true;
      }
    }
    changed_.notify_all();
  }

  void Cancel() {
    {
      std::lock_guard lock(mutex_);
      cancelled_ = true;
    }
    changed_.notify_all();
  }

  [[nodiscard]] Status Stop() {
    Cancel();
    if (producer_.joinable()) producer_.join();
    return producer_status_;
  }

  [[nodiscard]] double loader_seconds() const noexcept {
    return loader_seconds_;
  }

 private:
  void Produce() {
    using Clock = std::chrono::steady_clock;
    if (const Status status = RestoreNativeThreadAffinity(); !status.ok()) {
      std::lock_guard lock(mutex_);
      producer_status_ = status;
      producer_finished_ = true;
      changed_.notify_all();
      return;
    }
    std::uint64_t sequence = 0U;
    for (;;) {
      ReplaySlot& slot = slots_[sequence % slots_.size()];
      {
        std::unique_lock lock(mutex_);
        changed_.wait(lock, [&]() {
          return cancelled_ || slot.state == ReplaySlotState::kEmpty;
        });
        if (cancelled_) return;
        slot.state = ReplaySlotState::kFilling;
      }
      slot.window.Clear();
      const Clock::time_point load_begin = Clock::now();
      auto loaded = loader_.LoadNext(&slot.window);
      loader_seconds_ += std::chrono::duration<double>(
          Clock::now() - load_begin).count();
      {
        std::lock_guard lock(mutex_);
        if (cancelled_) {
          slot.state = ReplaySlotState::kEmpty;
          changed_.notify_all();
          return;
        }
        if (!loaded.ok()) {
          producer_status_ = loaded.status();
          producer_finished_ = true;
          slot.state = ReplaySlotState::kEmpty;
          changed_.notify_all();
          return;
        }
        if (!loaded.value()) {
          producer_finished_ = true;
          slot.state = ReplaySlotState::kEmpty;
          changed_.notify_all();
          return;
        }
        if (slot.window.cluster_count() == 0U) {
          producer_status_ = Status::InvalidArgument(
              "fixed-topology replay loader returned an empty window");
          producer_finished_ = true;
          slot.state = ReplaySlotState::kEmpty;
          changed_.notify_all();
          return;
        }
        slot.sequence = sequence;
        slot.state = ReplaySlotState::kReady;
      }
      changed_.notify_all();
      ++sequence;
    }
  }

  FixedTopologyReplayWindowLoader& loader_;
  std::array<ReplaySlot, 3U> slots_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::thread producer_;
  bool cancelled_ = false;
  bool producer_finished_ = false;
  std::uint64_t next_to_consume_ = 0U;
  Status producer_status_ = Status::Ok();
  double loader_seconds_ = 0.0;
};

}  // namespace

void FixedTopologyReplayWindow::Clear() noexcept {
  validated_checkpoint_payload_ = false;
  clusters_.clear();
  exact_real_.clear();
  unrotated_exact_real_.clear();
  real_sample_ids_.clear();
  virtual_sample_ids_.clear();
  basic_small_codes_.clear();
  basic_coefficients_.clear();
  frozen_virtual_small_codes_.clear();
  frozen_virtual_coefficients_.clear();
  roles_.clear();
  parents_.clear();
  depths_.clear();
  edge_icm_layer_orders_.clear();
}

Status FixedTopologyReplayWindow::Append(
    const FixedTopologyReplayRequest& request) {
  validated_checkpoint_payload_ = false;
  return AppendImpl(request, true);
}

Status FixedTopologyReplayWindow::AppendImpl(
    const FixedTopologyReplayRequest& request,
    bool copy_matrices) {
  const std::uint32_t real_count = request.exact_real.columns;
  const std::size_t virtual_count = request.virtual_sample_ids.size();
  const std::uint32_t dimension = request.exact_real.rows;
  const auto m = request.m;
  if (m < kMinimumModelLayers || m > kMaximumModelLayers ||
      (!clusters_.empty() && m != m_)) {
    return Status::InvalidArgument("replay window M mismatch");
  }
  m_ = m;
  if (clusters_.size() >=
          kMaximumReplayWindowClusters ||
      dimension == 0U ||
      virtual_count > std::numeric_limits<std::uint32_t>::max() ||
      request.unrotated_exact_real.rows != dimension ||
      request.unrotated_exact_real.columns != real_count ||
      (real_count != 0U && request.exact_real.values == nullptr) ||
      (real_count != 0U &&
       request.unrotated_exact_real.values == nullptr) ||
      request.real_sample_ids.size() != real_count ||
      request.basic_small_codes.size() !=
          static_cast<std::size_t>(real_count) * (m - 1U) ||
      request.basic_coefficients.size() !=
          static_cast<std::size_t>(real_count) * m ||
      request.frozen_virtual_small_codes.size() !=
          static_cast<std::size_t>(virtual_count) * (m - 1U) ||
      request.frozen_virtual_coefficients.size() !=
          static_cast<std::size_t>(virtual_count) * m ||
      request.frozen_roles.size() != real_count ||
      request.frozen_parents.size() != real_count ||
      request.frozen_depths.size() != real_count ||
      request.edge_icm_layer_orders.size() !=
          static_cast<std::size_t>(real_count) * m ||
      (!clusters_.empty() && clusters_.front().dimension != dimension)) {
    return Status::InvalidArgument(
        "fixed-topology replay loader supplied an invalid cluster");
  }
  const std::uint64_t real_end =
      static_cast<std::uint64_t>(real_sample_ids_.size()) + real_count;
  const std::uint64_t virtual_end =
      static_cast<std::uint64_t>(virtual_sample_ids_.size()) + virtual_count;
  if (real_end > std::numeric_limits<std::uint32_t>::max() ||
      virtual_end > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "fixed-topology replay window row count exceeds uint32");
  }

  clusters_.push_back({
      .cid = request.cid,
      .dimension = dimension,
      .real_begin = static_cast<std::uint32_t>(real_sample_ids_.size()),
      .real_count = real_count,
      .virtual_begin =
          static_cast<std::uint32_t>(virtual_sample_ids_.size()),
      .virtual_count = static_cast<std::uint32_t>(virtual_count),
      .seed = request.seed,
  });
  if (copy_matrices && real_count != 0U) {
    exact_real_.insert(
        exact_real_.end(), request.exact_real.values,
        request.exact_real.values +
            static_cast<std::size_t>(dimension) * real_count);
    unrotated_exact_real_.insert(
        unrotated_exact_real_.end(), request.unrotated_exact_real.values,
        request.unrotated_exact_real.values +
            static_cast<std::size_t>(dimension) * real_count);
  }
  if (real_count != 0U) {
    real_sample_ids_.insert(
        real_sample_ids_.end(), request.real_sample_ids.begin(),
        request.real_sample_ids.end());
    basic_small_codes_.insert(
        basic_small_codes_.end(), request.basic_small_codes.begin(),
        request.basic_small_codes.end());
    basic_coefficients_.insert(
        basic_coefficients_.end(), request.basic_coefficients.begin(),
        request.basic_coefficients.end());
    roles_.insert(
        roles_.end(), request.frozen_roles.begin(),
        request.frozen_roles.end());
    parents_.insert(
        parents_.end(), request.frozen_parents.begin(),
        request.frozen_parents.end());
    depths_.insert(
        depths_.end(), request.frozen_depths.begin(),
        request.frozen_depths.end());
    edge_icm_layer_orders_.insert(
        edge_icm_layer_orders_.end(), request.edge_icm_layer_orders.begin(),
        request.edge_icm_layer_orders.end());
  }
  if (virtual_count != 0U) {
    virtual_sample_ids_.insert(
        virtual_sample_ids_.end(), request.virtual_sample_ids.begin(),
        request.virtual_sample_ids.end());
    frozen_virtual_small_codes_.insert(
        frozen_virtual_small_codes_.end(),
        request.frozen_virtual_small_codes.begin(),
        request.frozen_virtual_small_codes.end());
    frozen_virtual_coefficients_.insert(
        frozen_virtual_coefficients_.end(),
        request.frozen_virtual_coefficients.begin(),
        request.frozen_virtual_coefficients.end());
  }
  return Status::Ok();
}

std::vector<FixedTopologyReplayRequest>
FixedTopologyReplayWindow::Requests() const {
  const auto m = m_;
  std::vector<FixedTopologyReplayRequest> requests;
  requests.reserve(clusters_.size());
  for (const ClusterSlice& cluster : clusters_) {
    const std::size_t real_begin = cluster.real_begin;
    const std::size_t virtual_begin = cluster.virtual_begin;
    requests.push_back({
        .m = m,
        .cid = cluster.cid,
        .exact_real = {
            Slice(exact_real_, real_begin * cluster.dimension,
                  static_cast<std::size_t>(cluster.real_count) *
                      cluster.dimension)
                .data(),
            cluster.dimension,
            cluster.real_count},
        .unrotated_exact_real = {
            Slice(unrotated_exact_real_, real_begin * cluster.dimension,
                  static_cast<std::size_t>(cluster.real_count) *
                      cluster.dimension)
                .data(),
            cluster.dimension,
            cluster.real_count},
        .real_sample_ids =
            Slice(real_sample_ids_, real_begin, cluster.real_count),
        .virtual_sample_ids =
            Slice(virtual_sample_ids_, virtual_begin, cluster.virtual_count),
        .basic_small_codes = Slice(
            basic_small_codes_, real_begin * (m - 1U),
            static_cast<std::size_t>(cluster.real_count) * (m - 1U)),
        .basic_coefficients = Slice(
            basic_coefficients_, real_begin * m,
            static_cast<std::size_t>(cluster.real_count) * m),
        .frozen_virtual_small_codes = Slice(
            frozen_virtual_small_codes_, virtual_begin * (m - 1U),
            static_cast<std::size_t>(cluster.virtual_count) * (m - 1U)),
        .frozen_virtual_coefficients = Slice(
            frozen_virtual_coefficients_, virtual_begin * m,
            static_cast<std::size_t>(cluster.virtual_count) * m),
        .frozen_roles = Slice(roles_, real_begin, cluster.real_count),
        .frozen_parents = Slice(parents_, real_begin, cluster.real_count),
        .frozen_depths = Slice(depths_, real_begin, cluster.real_count),
        .edge_icm_layer_orders = Slice(
            edge_icm_layer_orders_, real_begin * m,
            static_cast<std::size_t>(cluster.real_count) * m),
        .seed = cluster.seed,
    });
  }
  return requests;
}

template <int M>
StatusOr<FixedTopologyReplayPipelineStatistics>
FixedTopologyReplayPipeline<M>::Run(
    FixedTopologyReplayWindowLoader& loader,
    const FixedTopologyReplayExecutor<M>& executor,
    FixedTopologyReplayWindowSink<M>& sink,
    bool collect_quality_statistics) {
  using Clock = std::chrono::steady_clock;
  const Clock::time_point wall_begin = Clock::now();
  ReplayWindowControl control(loader);
  control.Start();

  FixedTopologyReplayPipelineStatistics statistics;
  Status consumer_status = Status::Ok();
  Status sink_status = Status::Ok();
  const bool reconstruct_real = sink.RequiresRealReconstructions();
  struct CompletedWindow {
    ReplaySlot* slot;
    std::vector<FixedTopologyReplayRequest> requests;
    std::vector<FixedTopologyReplayResult<M>> results;
  };
  BoundedQueue<CompletedWindow> completed(1U);
  std::thread publisher([&] {
    sink_status = RestoreNativeThreadAffinity();
    if (!sink_status.ok()) {
      control.Cancel();
      completed.Close();
      return;
    }
    while (auto window = completed.Pop()) {
      const auto begin = Clock::now();
      sink_status = sink.CommitValidatedFromExecutor(
          window->slot->sequence, window->requests, window->results);
      statistics.sink_seconds += std::chrono::duration<double>(
          Clock::now() - begin).count();
      // Release the input storage only after its non-owning request views
      // have been fully consumed. No d-by-n matrix copy is needed.
      control.Release(*window->slot, sink_status.ok());
      if (!sink_status.ok()) {
        completed.Close();
        return;
      }
    }
  });
  for (;;) {
    auto acquired = control.Acquire();
    if (!acquired.ok()) {
      consumer_status = acquired.status();
      break;
    }
    ReplaySlot* slot = acquired.value();
    if (slot == nullptr) break;

    const std::uint32_t cluster_count = slot->window.cluster_count();
    const std::uint64_t real_rows = slot->window.real_row_count();
    const Clock::time_point request_begin = Clock::now();
    auto requests = slot->window.Requests();
    statistics.request_seconds += std::chrono::duration<double>(
        Clock::now() - request_begin).count();
    const Clock::time_point executor_begin = Clock::now();
    auto results = slot->window.validated_checkpoint_payload_
        ? executor.ExecuteWindowFromCheckpoint(
              requests, reconstruct_real)
        : executor.ExecuteWindow(
              requests, reconstruct_real);
    statistics.executor_seconds += std::chrono::duration<double>(
        Clock::now() - executor_begin).count();
    if (results.ok() && collect_quality_statistics) {
      for (const FixedTopologyReplayResult<M>& result : results.value()) {
        const ClusterTopologyResult<M>& topology = result.topology;
        if (topology.roles.size() != topology.squared_errors.size() ||
            topology.roles.size() != topology.depths.size()) {
          consumer_status = Status::DataLoss(
              "fixed-topology replay quality streams changed shape");
          break;
        }
        for (std::size_t row = 0U; row < topology.roles.size(); ++row) {
          const float error = topology.squared_errors[row];
          if (!std::isfinite(error) || error < 0.0F) {
            consumer_status = Status::DataLoss(
                "fixed-topology replay produced an invalid squared error");
            break;
          }
          statistics.linked_real_rows +=
              topology.roles[row] == LinkageNodeRole::kLinked ? 1U : 0U;
          statistics.depth_sum += topology.depths[row];
          statistics.maximum_depth = std::max(
              statistics.maximum_depth, topology.depths[row]);
          statistics.squared_error_sum += error;
          if (!statistics.has_squared_error) {
            statistics.minimum_squared_error = error;
            statistics.maximum_squared_error = error;
            statistics.has_squared_error = true;
          } else {
            statistics.minimum_squared_error = std::min(
                statistics.minimum_squared_error, error);
            statistics.maximum_squared_error = std::max(
                statistics.maximum_squared_error, error);
          }
        }
        if (!consumer_status.ok()) break;
      }
    }
    if (results.ok() && consumer_status.ok()) {
      if (!completed.Push({slot, std::move(requests),
                           std::move(results).value()})) {
        control.Release(*slot, false);
        break;
      }
    } else if (!results.ok()) {
      consumer_status = results.status();
    }

    if (!consumer_status.ok()) {
      control.Release(*slot, false);
      break;
    }
    ++statistics.windows;
    statistics.clusters += cluster_count;
    statistics.real_rows += real_rows;
  }

  completed.Close();
  publisher.join();
  const Status producer_status = control.Stop();
  if (!consumer_status.ok()) {
    return consumer_status;
  }
  if (!sink_status.ok()) return sink_status;
  if (!producer_status.ok()) {
    return producer_status;
  }
  statistics.wall_seconds = std::chrono::duration<double>(
      Clock::now() - wall_begin).count();
  statistics.loader_seconds = control.loader_seconds();
  return statistics;
}

template class FixedTopologyReplayPipeline<0>;
template class FixedTopologyReplayPipeline<5>;
template class FixedTopologyReplayPipeline<12>;
template class FixedTopologyReplayPipeline<6>;
template class FixedTopologyReplayPipeline<10>;

}  // namespace stlqpp
