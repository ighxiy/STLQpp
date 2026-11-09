#include "stlqpp/training/linkage/candidates/cpu_exact_knn.h"

#include "ranked_neighbor_heap.h"
#include "stlqpp/runtime/cpu/openmp_affinity.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
#include <utility>
#include <limits>
#include <string>
#include <vector>

namespace stlqpp {
namespace detail {

struct PreparedKnnRowsAccess final {
  static PreparedKnnRows Finalize(
      std::uint32_t rows,
      std::uint32_t k,
      std::vector<linkage_candidates::RankedNeighbor>* heaps,
      [[maybe_unused]] const std::vector<std::uint32_t>& counts) {
    PreparedKnnRows result;
    result.row_count_ = rows;
    result.neighbors_per_row_ = k;
    result.producer_guaranteed_unique_ = true;
    result.local_ids_.resize(static_cast<std::size_t>(rows) * k);
    result.squared_distances_.resize(static_cast<std::size_t>(rows) * k);
    for (std::uint32_t row = 0U; row < rows; ++row) {
      assert(counts[row] == k);
      linkage_candidates::RankedNeighbor* begin = heaps->data() +
          static_cast<std::size_t>(row) * k;
      std::sort(begin, begin + k,
                linkage_candidates::RankedNeighborLess);
      for (std::uint32_t index = 0U; index < k; ++index) {
        const std::size_t output =
            static_cast<std::size_t>(row) * k + index;
        result.local_ids_[output] = begin[index].local_id;
        result.squared_distances_[output] = begin[index].distance;
      }
    }
    return result;
  }
};

}  // namespace detail
namespace {

constexpr std::uint32_t kSymmetricTile = 256U;

using linkage_candidates::OfferRankedNeighbor;
using linkage_candidates::RankedNeighbor;
using linkage_candidates::RankedNeighborLess;

[[nodiscard]] float SquaredNorm(const float* vector,
                                std::uint32_t dimension) noexcept {
  float result = 0.0F;
#pragma omp simd reduction(+:result)
  for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
    result += vector[axis] * vector[axis];
  }
  return result;
}

[[nodiscard]] std::vector<float> SquaredNorms(
    ColumnMajorMatrixF32View matrix) {
  std::vector<float> result(matrix.columns);
  for (std::uint32_t column = 0U; column < matrix.columns; ++column) {
    result[column] = SquaredNorm(matrix.column(column), matrix.rows);
  }
  return result;
}

Status ValidateMatrix(ColumnMajorMatrixF32View matrix,
                      const char* role) {
  if (matrix.values == nullptr || matrix.rows == 0U || matrix.columns == 0U) {
    return Status::InvalidArgument(
        std::string("exact KNN requires a non-empty ") + role + " matrix");
  }
  return Status::Ok();
}

StatusOr<std::size_t> CheckedHeapCount(std::uint32_t rows,
                                       std::uint32_t k) {
  const std::uint64_t count =
      static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(k);
  if (count > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted("exact KNN heap extent exceeds size_t");
  }
  return static_cast<std::size_t>(count);
}

// GEMM dimensions, arithmetic and heap comparator are identical in both paths.
// Parallel batches own disjoint endpoint tiles, so neither heap needs a lock.
struct SymmetricKnnTiles final {
  using Scratch = std::vector<float>;

  void ReserveScratch(Scratch& dots) const {
    dots.reserve(kSymmetricTile * kSymmetricTile);
  }

  const F32GemmBackend& gemm;
  ColumnMajorMatrixF32View points;
  const std::vector<float>& norms;
  std::vector<RankedNeighbor>& heaps;
  std::vector<std::uint32_t>& counts;
  std::uint32_t k;

  Status Compute(std::uint32_t query_tile, std::uint32_t parent_tile,
                 std::vector<float>& dots) const {
    const auto point_view = points;
    const auto neighbor_count = k;
    const auto* norm_data = norms.data();
    auto* heap_data = heaps.data();
    auto* count_data = counts.data();
    const auto query_begin = query_tile * kSymmetricTile;
    const auto parent_begin = parent_tile * kSymmetricTile;
    const auto query_count = std::min(kSymmetricTile, point_view.columns - query_begin);
    const auto parent_count = std::min(kSymmetricTile, point_view.columns - parent_begin);
    dots.resize(static_cast<std::size_t>(parent_count) * query_count);
    const Status status = parent_tile == query_tile
        ? gemm.MultiplySelfTransposedUpper({
              .input = {point_view.column(query_begin), point_view.rows, query_count},
              .output = dots.data(), .output_leading_dimension = query_count})
        : gemm.MultiplyTransposedLeft({
              .left = {point_view.column(parent_begin), point_view.rows, parent_count},
              .right = point_view.column(query_begin), .query_count = query_count,
              .right_leading_dimension = point_view.rows, .output = dots.data(),
              .output_leading_dimension = parent_count});
    if (!status.ok()) return status;
    for (std::uint32_t query = 0U; query < query_count; ++query) {
      const auto query_id = query_begin + query;
      const auto parent_limit = parent_tile == query_tile ? query : parent_count;
      const float* query_dots = dots.data() + static_cast<std::size_t>(query) * parent_count;
      for (std::uint32_t parent = 0U; parent < parent_limit; ++parent) {
        const auto parent_id = parent_begin + parent;
        const float distance = std::max(
            0.0F, norm_data[query_id] + norm_data[parent_id] - 2.0F * query_dots[parent]);
        OfferRankedNeighbor({distance, parent_id},
            heap_data + static_cast<std::size_t>(query_id) * neighbor_count,
            count_data + query_id, neighbor_count);
        OfferRankedNeighbor({distance, query_id},
            heap_data + static_cast<std::size_t>(parent_id) * neighbor_count,
            count_data + parent_id, neighbor_count);
      }
    }
    return Status::Ok();
  }
};

constexpr std::uint32_t kBipartiteQueryTile = 32U;

// Independent query rows keep the original direct-delta SIMD reduction and
// pair ordering. Each participant owns its candidate scratch and output rows.
struct BipartiteKnnQueries final {
  using Scratch = std::vector<std::pair<float, std::uint32_t>>;
  ColumnMajorMatrixF32View centers;
  ColumnMajorMatrixF32View queries;
  std::uint32_t k;
  std::uint32_t* local_ids;
  float* squared_distances;

  void ReserveScratch(Scratch& ranked) const { ranked.resize(centers.columns); }

  Status Compute(std::uint32_t query_tile, std::uint32_t,
                 Scratch& ranked) const {
    const auto first = query_tile * kBipartiteQueryTile;
    const auto count = std::min(kBipartiteQueryTile, queries.columns - first);
    for (std::uint32_t query = first; query < first + count; ++query) {
      const float* query_values = queries.column(query);
      for (std::uint32_t center = 0U; center < centers.columns; ++center) {
        const float* center_values = centers.column(center);
        float distance = 0.0F;
#pragma omp simd reduction(+:distance)
        for (std::uint32_t axis = 0U; axis < centers.rows; ++axis) {
          const float delta = query_values[axis] - center_values[axis];
          distance += delta * delta;
        }
        ranked[center] = {distance, center};
      }
      std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end());
      const std::size_t output = static_cast<std::size_t>(query) * k;
      for (std::uint32_t index = 0U; index < k; ++index) {
        squared_distances[output + index] = ranked[index].first;
        local_ids[output + index] = ranked[index].second;
      }
    }
    return Status::Ok();
  }
};

class KnnHelperLease final {
 public:
  explicit KnnHelperLease(ExactKnnCpuBudget& budget) noexcept : budget_(budget) {}
  ~KnnHelperLease() { budget_.ReleaseHelpers(held_); }
  KnnHelperLease(const KnnHelperLease&) = delete;
  KnnHelperLease& operator=(const KnnHelperLease&) = delete;
  std::uint32_t Adjust(std::uint32_t maximum) noexcept {
    if (held_ > maximum) {
      budget_.ReleaseHelpers(held_ - maximum);
      held_ = maximum;
    }
    held_ += budget_.AcquireHelpers(held_, maximum);
    return held_;
  }
 private:
  ExactKnnCpuBudget& budget_;
  std::uint32_t held_ = 0U;
};

struct KnnTilePair final { std::uint32_t query; std::uint32_t parent; };

// Created only after a tail call acquires a helper slot. Threads and numerical
// scratch are reused across batches and can grow as outer workers exit. The
// two work types are bound at compile time; there is no per-row callback.
template <typename Work>
class KnnWorkTeam final {
 public:
  KnnWorkTeam(const Work& tiles, ExactKnnCpuBudget& budget,
              typename Work::Scratch owner_dots)
      : tiles_(tiles), budget_(budget), scratch_(budget.workers()) {
    scratch_[0].dots = std::move(owner_dots);
    tiles_.ReserveScratch(scratch_[0].dots);
    helpers_.reserve(budget.workers() - 1U);
  }
  ~KnnWorkTeam() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    work_.notify_all();
    for (auto& thread : helpers_) thread.join();
  }
  KnnWorkTeam(const KnnWorkTeam&) = delete;
  KnnWorkTeam& operator=(const KnnWorkTeam&) = delete;

  Status Run(const std::vector<KnnTilePair>& pairs, std::uint32_t participants) {
    while (helpers_.size() + 1U < participants) {
      const auto slot = static_cast<std::uint32_t>(helpers_.size() + 1U);
      tiles_.ReserveScratch(scratch_[slot].dots);
      helpers_.emplace_back([this, slot] { Worker(slot); });
    }
    phase_failed_.store(false, std::memory_order_relaxed);
    if (participants == 1U) {
      Execute(0U, 1U, pairs, Status::Ok());
      return scratch_[0].status;
    }
    {
      std::lock_guard lock(mutex_);
      pairs_ = &pairs;
      participants_ = participants;
      completed_ = 0U;
      ++generation_;
    }
    work_.notify_all();
    Execute(0U, participants, pairs, Status::Ok());
    {
      std::unique_lock lock(mutex_);
      done_.wait(lock, [&] { return completed_ == participants - 1U; });
    }
    for (std::uint32_t slot = 0U; slot < participants; ++slot)
      if (!scratch_[slot].status.ok()) return scratch_[slot].status;
    return Status::Ok();
  }
 private:
  struct Scratch final { typename Work::Scratch dots; Status status; };
  void Execute(std::uint32_t slot, std::uint32_t participants,
               const std::vector<KnnTilePair>& pairs, const Status& affinity) {
    auto& local = scratch_[slot];
    local.status = affinity;
    if (!local.status.ok()) {
      phase_failed_.store(true, std::memory_order_release);
      return;
    }
    try {
      for (std::size_t pair = slot; pair < pairs.size(); pair += participants) {
        if (phase_failed_.load(std::memory_order_acquire)) break;
        if (budget_.cancelled()) {
          local.status = Status::FailedPrecondition("exact KNN pipeline cancelled");
          break;
        }
        local.status = tiles_.Compute(pairs[pair].query, pairs[pair].parent, local.dots);
        if (!local.status.ok()) break;
      }
    } catch (const std::bad_alloc&) {
      local.status = Status::ResourceExhausted("exact KNN tile allocation failed");
    } catch (const std::exception& error) {
      local.status = Status::Internal(std::string("exact KNN tile failed: ") + error.what());
    }
    if (!local.status.ok()) phase_failed_.store(true, std::memory_order_release);
  }
  void Worker(std::uint32_t slot) {
    const Status affinity = RestoreNativeThreadAffinity();
    std::uint64_t seen = 0U;
    std::unique_lock lock(mutex_);
    for (;;) {
      work_.wait(lock, [&] { return stopping_ || generation_ != seen; });
      if (stopping_) return;
      seen = generation_;
      if (slot >= participants_) continue;
      const auto participants = participants_;
      const auto* pairs = pairs_;
      lock.unlock();
      Execute(slot, participants, *pairs, affinity);
      lock.lock();
      ++completed_;
      done_.notify_one();
    }
  }
  const Work& tiles_;
  ExactKnnCpuBudget& budget_;
  std::vector<Scratch> scratch_;
  std::vector<std::thread> helpers_;
  std::mutex mutex_;
  std::condition_variable work_;
  std::condition_variable done_;
  const std::vector<KnnTilePair>* pairs_ = nullptr;
  std::uint32_t participants_ = 1U;
  std::uint32_t completed_ = 0U;
  std::uint64_t generation_ = 0U;
  bool stopping_ = false;
  std::atomic<bool> phase_failed_{false};
};

Status ComputeTailTiles(const SymmetricKnnTiles& tiles, std::uint32_t first,
                        std::uint32_t tile_count, ExactKnnCpuBudget& budget,
                        KnnHelperLease& lease, std::vector<float> dots) {
  // Circle-method matchings cover every off-diagonal tile pair exactly once.
  // Orient every pair as before (larger tile = query). Prefix query strips
  // already computed serially are skipped; there is no repeated distance work.
  KnnWorkTeam<SymmetricKnnTiles> team(tiles, budget, std::move(dots));
  const auto circle_size = tile_count + tile_count % 2U;
  std::vector<std::uint32_t> ring(circle_size);
  for (std::uint32_t i = 0U; i < circle_size; ++i) ring[i] = i;
  std::vector<KnnTilePair> pairs;
  pairs.reserve(tile_count);
  for (std::uint32_t i = first; i < tile_count; ++i) pairs.push_back({i, i});
  auto run = [&]() {
    if (budget.cancelled())
      return Status::FailedPrecondition("exact KNN pipeline cancelled");
    if (pairs.empty()) return Status::Ok();
    return team.Run(pairs, lease.Adjust(static_cast<std::uint32_t>(pairs.size() - 1U)) + 1U);
  };
  if (const Status status = run(); !status.ok()) return status;
  for (std::uint32_t round = 0U; round + 1U < circle_size; ++round) {
    pairs.clear();
    for (std::uint32_t i = 0U; i < circle_size / 2U; ++i) {
      const auto query = std::max(ring[i], ring[circle_size - 1U - i]);
      const auto parent = std::min(ring[i], ring[circle_size - 1U - i]);
      if (query < tile_count && query >= first) pairs.push_back({query, parent});
    }
    if (const Status status = run(); !status.ok()) return status;
    const auto last = ring.back();
    std::move_backward(ring.begin() + 1, ring.end() - 1, ring.end());
    ring[1] = last;
  }
  return Status::Ok();
}

Status ComputeTailQueries(const BipartiteKnnQueries& queries,
                          std::uint32_t first, std::uint32_t tile_count,
                          ExactKnnCpuBudget& budget, KnnHelperLease& lease,
                          BipartiteKnnQueries::Scratch scratch) {
  KnnWorkTeam<BipartiteKnnQueries> team(queries, budget, std::move(scratch));
  std::vector<KnnTilePair> batch;
  batch.reserve(budget.workers());
  while (first < tile_count) {
    if (budget.cancelled())
      return Status::FailedPrecondition("exact KNN pipeline cancelled");
    const auto count = std::min(budget.workers(), tile_count - first);
    batch.clear();
    for (std::uint32_t tile = 0U; tile < count; ++tile)
      batch.push_back({first + tile, 0U});
    const auto participants = lease.Adjust(count - 1U) + 1U;
    if (auto status = team.Run(batch, participants); !status.ok()) return status;
    first += count;
  }
  return Status::Ok();
}

}  // namespace

StatusOr<PreparedKnnRows> CpuExactKnnProvider::BuildSymmetric(
    ColumnMajorMatrixF32View points, std::uint32_t requested_k,
    ExactKnnCpuBudget* cpu_budget) const {
  const Status validation = ValidateMatrix(points, "symmetric point");
  if (!validation.ok()) return validation;
  if (requested_k == 0U)
    return Status::InvalidArgument("symmetric exact KNN requires k > 0");
  const auto k = std::min(requested_k, points.columns - 1U);
  if (k == 0U) {
    PreparedKnnRows empty;
    empty.row_count_ = points.columns;
    empty.producer_guaranteed_unique_ = true;
    return empty;
  }
  auto heap_count = CheckedHeapCount(points.columns, k);
  if (!heap_count.ok()) return heap_count.status();
  const std::vector<float> norms = SquaredNorms(points);
  std::vector<RankedNeighbor> heaps(heap_count.value());
  std::vector<std::uint32_t> counts(points.columns, 0U);
  std::vector<float> dots;
  const SymmetricKnnTiles tiles{*gemm_, points, norms, heaps, counts, k};
  const auto tile_count = (points.columns - 1U) / kSymmetricTile + 1U;
  for (std::uint32_t query_tile = 0U; query_tile < tile_count; ++query_tile) {
    if (cpu_budget != nullptr) {
      if (cpu_budget->cancelled())
        return Status::FailedPrecondition("exact KNN pipeline cancelled");
      const auto possible = std::min(tile_count / 2U, tile_count - query_tile);
      KnnHelperLease lease(*cpu_budget);
      // Tiny graphs and the final single query strip stay on the serial path.
      if (tile_count >= 4U && possible > 1U && lease.Adjust(possible - 1U) != 0U) {
        try {
          const Status status = ComputeTailTiles(
              tiles, query_tile, tile_count, *cpu_budget, lease, std::move(dots));
          if (!status.ok()) return status;
        } catch (const std::bad_alloc&) {
          return Status::ResourceExhausted("exact KNN tail workspace allocation failed");
        } catch (const std::system_error& error) {
          return Status::ResourceExhausted(std::string("exact KNN helper failed: ") + error.what());
        }
        break;
      }
    }
    for (std::uint32_t parent_tile = 0U; parent_tile <= query_tile; ++parent_tile) {
      const Status status = tiles.Compute(query_tile, parent_tile, dots);
      if (!status.ok()) return status;
    }
  }
  return detail::PreparedKnnRowsAccess::Finalize(points.columns, k, &heaps, counts);
}

StatusOr<PreparedKnnRows> CpuExactKnnProvider::BuildBipartite(
    ColumnMajorMatrixF32View centers,
    ColumnMajorMatrixF32View queries,
    std::uint32_t requested_k, ExactKnnCpuBudget* cpu_budget) const {
  Status validation = ValidateMatrix(centers, "center");
  if (!validation.ok()) {
    return validation;
  }
  validation = ValidateMatrix(queries, "query");
  if (!validation.ok()) {
    return validation;
  }
  if (centers.rows != queries.rows || requested_k == 0U) {
    return Status::InvalidArgument(
        "bipartite exact KNN requires matching dimensions and k > 0");
  }
  const std::uint32_t k = std::min(requested_k, centers.columns);
  auto heap_count = CheckedHeapCount(queries.columns, k);
  if (!heap_count.ok()) {
    return heap_count.status();
  }

  // The frozen producer defines virtual-parent exact KNN with a direct FP32
  // sum of squared coordinate deltas.  The algebraically equivalent
  // norm-plus-dot formulation has different cancellation and rounding and can
  // therefore change the kth candidate.  Preserve the producer's arithmetic
  // and pair ordering here; symmetric real-graph KNN intentionally keeps its
  // separate tiled GEMM definition above.
  PreparedKnnRows result;
  result.row_count_ = queries.columns;
  result.neighbors_per_row_ = k;
  result.local_ids_.resize(heap_count.value());
  result.squared_distances_.resize(heap_count.value());
  BipartiteKnnQueries::Scratch ranked(centers.columns);
  const BipartiteKnnQueries work{centers, queries, k,
      result.local_ids_.data(), result.squared_distances_.data()};
  const auto tile_count = (queries.columns - 1U) / kBipartiteQueryTile + 1U;
  // Check every 256 query rows before borrowing. Small workloads retain the
  // original serial path, even when the pipeline has spare CPU credits.
  const bool enough_work = queries.columns >= kSymmetricTile &&
      static_cast<std::uint64_t>(centers.rows) * centers.columns >= 4096U;
  for (std::uint32_t tile = 0U; tile < tile_count; ++tile) {
    if (cpu_budget != nullptr && tile % (kSymmetricTile / kBipartiteQueryTile) == 0U) {
      if (cpu_budget->cancelled())
        return Status::FailedPrecondition("exact KNN pipeline cancelled");
      KnnHelperLease lease(*cpu_budget);
      const auto possible = std::min(cpu_budget->workers(), tile_count - tile);
      if (enough_work && possible > 1U && lease.Adjust(possible - 1U) != 0U) {
        try {
          if (auto status = ComputeTailQueries(work, tile, tile_count,
                  *cpu_budget, lease, std::move(ranked)); !status.ok())
            return status;
        } catch (const std::bad_alloc&) {
          return Status::ResourceExhausted("exact KNN tail workspace allocation failed");
        } catch (const std::system_error& error) {
          return Status::ResourceExhausted(std::string("exact KNN helper failed: ") + error.what());
        }
        break;
      }
    }
    if (auto status = work.Compute(tile, 0U, ranked); !status.ok()) return status;
  }
  return result;
}

}  // namespace stlqpp
