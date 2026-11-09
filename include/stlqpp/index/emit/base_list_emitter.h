#pragma once

#include "stlqpp/core/artifact_hash.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/index/emit/basic_bucket_store.h"
#include "stlqpp/runtime/cpu/parallel_for.h"

#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

namespace stlqpp {

namespace detail {

// List metadata and retention describe an on-disk width, independently of the
// numerical specialization chosen by its consumer.
struct BaseListMetadata final {
  [[nodiscard]] static StatusOr<BaseListMetadata> Open(
      const std::filesystem::path& directory, std::uint32_t expected_m);
  [[nodiscard]] static Status DiscardRawPayload(
      const std::filesystem::path& directory, std::uint32_t expected_m);
  [[nodiscard]] std::filesystem::path ids_path() const;
  [[nodiscard]] std::filesystem::path codes_path() const;
  [[nodiscard]] std::filesystem::path coefficients_path() const;
  [[nodiscard]] std::filesystem::path raw_path() const;

  std::filesystem::path directory;
  BaseBuildShape shape;
  std::uint32_t m = 0U;
  ArtifactHash64 identity;
  ArtifactHash64 source_model_identity;
  std::vector<std::uint64_t> offsets;
  bool has_raw_payload = true;
};

}  // namespace detail

template <int M>
class BaseListStore final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  BaseListStore() = default;

  [[nodiscard]] static StatusOr<BaseListStore<M>> Open(
      const std::filesystem::path& directory);

  [[nodiscard]] const std::filesystem::path& directory() const noexcept {
    return metadata_.directory;
  }
  [[nodiscard]] std::uint32_t m() const noexcept {
    return M == 0 ? metadata_.m : M;
  }
  [[nodiscard]] const BaseBuildShape& shape() const noexcept {
    return metadata_.shape;
  }
  [[nodiscard]] ArtifactHash64 identity() const noexcept {
    return metadata_.identity;
  }
  [[nodiscard]] ArtifactHash64 source_model_identity() const noexcept {
    return metadata_.source_model_identity;
  }
  [[nodiscard]] std::uint64_t total_vectors() const noexcept {
    return metadata_.offsets.empty() ? 0U : metadata_.offsets.back();
  }
  [[nodiscard]] const std::vector<std::uint64_t>& offsets() const noexcept {
    return metadata_.offsets;
  }
  [[nodiscard]] std::filesystem::path ids_path() const;
  [[nodiscard]] std::filesystem::path codes_path() const;
  [[nodiscard]] std::filesystem::path coefficients_path() const;
  [[nodiscard]] std::filesystem::path raw_path() const;
  [[nodiscard]] bool has_raw_payload() const noexcept {
    return metadata_.has_raw_payload;
  }

  // Staged cleanup boundary. The marker is published before the raw shard is
  // removed so support-only readers can distinguish intentional retention
  // from corruption after a restart.
  [[nodiscard]] static Status DiscardRawPayload(
      const std::filesystem::path& directory);

 private:
  template <int N>
  friend class BaseListReader;

  explicit BaseListStore(detail::BaseListMetadata&& metadata) noexcept
      : metadata_(std::move(metadata)) {}

  detail::BaseListMetadata metadata_;
};

// Optional diagnostics for the phase replay tool. Production callers leave
// this null: no clock reads or per-bucket diagnostics storage are required.
struct BaseListBucketStatistics final {
  std::uint32_t worker_index = 0U;
  std::uint64_t rows = 0U;
  std::uint64_t read_bytes = 0U;
  std::uint64_t write_bytes = 0U;
  std::uint64_t write_calls = 0U;
  double start_sec = 0.0;
  double end_sec = 0.0;
  double read_sec = 0.0;
  double write_sec = 0.0;
  // Buffer allocation, open/close, validation and packing, excluding measured
  // read/write calls; these summed service times are not stage wall times.
  double pack_other_sec = 0.0;
};

struct BaseListEmitStatistics final {
  double setup_wall_sec = 0.0;
  double workers_wall_sec = 0.0;
  double publication_wall_sec = 0.0;
  std::vector<BaseListBucketStatistics> buckets;
};

template <int M>
class ParallelBaseListEmitter final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static Status Emit(
      const BasicBucketStore<M>& buckets,
      const std::filesystem::path& destination,
      const ParallelForExecutor& parallel,
      ArtifactHash64 identity = {},
      ArtifactHash64 source_model_identity = {});
  [[nodiscard]] static Status EmitTransient(
      const BasicBucketStore<M>& buckets,
      const std::filesystem::path& destination,
      const ParallelForExecutor& parallel,
      ArtifactHash64 identity = {},
      ArtifactHash64 source_model_identity = {},
      BaseListEmitStatistics* statistics = nullptr);

 private:
  [[nodiscard]] static Status EmitImpl(
      const BasicBucketStore<M>& buckets,
      const std::filesystem::path& destination,
      const ParallelForExecutor& parallel,
      ArtifactHash64 identity,
      ArtifactHash64 source_model_identity,
      bool durable,
      BaseListEmitStatistics* statistics);
};

using RuntimeBaseListStore = BaseListStore<0>;
using RuntimeParallelBaseListEmitter = ParallelBaseListEmitter<0>;

extern template class BaseListStore<0>;
extern template class ParallelBaseListEmitter<0>;
extern template class BaseListStore<5>;
extern template class BaseListStore<10>;
extern template class ParallelBaseListEmitter<5>;
extern template class ParallelBaseListEmitter<10>;

}  // namespace stlqpp
