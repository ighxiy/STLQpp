#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/artifact_hash.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/model_shape.h"

namespace stlqpp {

namespace detail {
template <int M, typename Scalar>
class RootMomentAndArtifactSink;
}

enum class BaseVectorScalar : std::uint32_t {
  kU8 = 1U,
  kF32 = 4U,
};

struct BaseBuildShape final {
  std::uint32_t dimension = 0U;
  std::uint32_t nlist = 0U;
  std::uint32_t cids_per_bucket = 0U;
  BaseVectorScalar vector_scalar = BaseVectorScalar::kU8;

  [[nodiscard]] Status Validate(std::uint32_t m) const;
  [[nodiscard]] std::uint32_t bucket_count() const noexcept;
  [[nodiscard]] std::uint32_t raw_bytes_per_vector() const noexcept;
};

namespace detail {

// Bucket metadata and its file extents depend on the stored width, not on a
// numerical specialization. The typed reader only selects the expected M.
struct BasicBucketStoreMetadata final {
  [[nodiscard]] static StatusOr<BasicBucketStoreMetadata> Open(
      const std::filesystem::path& directory, std::uint32_t expected_m);
  [[nodiscard]] std::filesystem::path bucket_path(std::uint32_t bucket) const;
  [[nodiscard]] std::uint64_t bucket_rows(std::uint32_t bucket) const noexcept;

  std::filesystem::path directory;
  BaseBuildShape shape;
  std::uint32_t m = 0U;
  std::uint64_t total_vectors = 0U;
  std::vector<std::uint64_t> cid_counts;
};

struct BasicBucketStoreWriterState;

}  // namespace detail

// M=0 selects the runtime-width path; it is not a padded code length.
// Fixed-M callers retain compile-time serialization strides.
template <int M, typename Raw>
struct EncodedBaseBlock final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t first_id = 0U;
  ArrayView<const std::uint32_t> root_cids;
  ArrayView<const std::uint8_t> small_codes_row_major;
  ArrayView<const float> coefficients_row_major;
  const Raw* raw_vectors = nullptr;
  std::uint32_t raw_leading_dimension = 0U;
};

template <int M>
class BasicBucketStore final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  BasicBucketStore() = default;

  [[nodiscard]] static StatusOr<BasicBucketStore<M>> Open(
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
  [[nodiscard]] std::uint64_t total_vectors() const noexcept {
    return metadata_.total_vectors;
  }
  [[nodiscard]] const std::vector<std::uint64_t>& cid_counts() const noexcept {
    return metadata_.cid_counts;
  }
  [[nodiscard]] std::filesystem::path bucket_path(std::uint32_t bucket) const;
  [[nodiscard]] std::uint64_t bucket_rows(std::uint32_t bucket) const noexcept;

 private:
  template <int N>
  friend class ParallelBaseListEmitter;

  explicit BasicBucketStore(detail::BasicBucketStoreMetadata&& metadata) noexcept;

  detail::BasicBucketStoreMetadata metadata_;
};

template <int M>
class BasicBucketStoreWriter final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<BasicBucketStoreWriter<M>> Create(
      const std::filesystem::path& destination, BaseBuildShape shape,
      std::size_t bucket_flush_bytes = 256U * 1024U * 1024U,
      std::uint32_t m = FixedModelLayers<M>());
  [[nodiscard]] static StatusOr<BasicBucketStoreWriter<M>> CreateTransient(
      const std::filesystem::path& destination, BaseBuildShape shape,
      std::size_t bucket_flush_bytes = 256U * 1024U * 1024U,
      std::uint32_t m = FixedModelLayers<M>());
  [[nodiscard]] static StatusOr<BasicBucketStoreWriter<M>> CreateCheckpointed(
      const std::filesystem::path& destination, BaseBuildShape shape,
      ArtifactHash64 identity,
      std::size_t bucket_flush_bytes = 256U * 1024U * 1024U,
      std::uint32_t m = FixedModelLayers<M>());
  [[nodiscard]] static StatusOr<BasicBucketStoreWriter<M>> OpenCheckpointed(
      const std::filesystem::path& destination, BaseBuildShape shape,
      ArtifactHash64 expected_identity,
      std::size_t bucket_flush_bytes = 256U * 1024U * 1024U,
      std::uint32_t m = FixedModelLayers<M>());

  ~BasicBucketStoreWriter();
  BasicBucketStoreWriter(BasicBucketStoreWriter&&) noexcept;
  BasicBucketStoreWriter& operator=(BasicBucketStoreWriter&&) noexcept;
  BasicBucketStoreWriter(const BasicBucketStoreWriter&) = delete;
  BasicBucketStoreWriter& operator=(const BasicBucketStoreWriter&) = delete;

  [[nodiscard]] Status Append(const EncodedBaseBlock<M, std::uint8_t>& block);
  [[nodiscard]] Status Append(const EncodedBaseBlock<M, float>& block);
  [[nodiscard]] Status Commit();
  [[nodiscard]] std::uint64_t next_id() const noexcept;

 private:
  template <int N, typename Scalar>
  friend class detail::RootMomentAndArtifactSink;

  using State = detail::BasicBucketStoreWriterState;
  explicit BasicBucketStoreWriter(std::unique_ptr<State> state) noexcept;

  // Training-only: root moments checked every CID and the pass matched the
  // updater's model shape to this writer before processing any block.
  [[nodiscard]] Status AppendAfterBasicMoments(
      const EncodedBaseBlock<M, std::uint8_t>& block);
  [[nodiscard]] Status AppendAfterBasicMoments(
      const EncodedBaseBlock<M, float>& block);

  template <bool CidsValidated, typename Raw>
  [[nodiscard]] Status AppendTyped(const EncodedBaseBlock<M, Raw>& block);
  [[nodiscard]] static StatusOr<BasicBucketStoreWriter<M>> CreateImpl(
      const std::filesystem::path& destination, BaseBuildShape shape,
      bool durable, std::size_t bucket_flush_bytes, std::uint32_t m);

  std::unique_ptr<State> state_;
};

using RuntimeBasicBucketStore = BasicBucketStore<0>;
using RuntimeBasicBucketStoreWriter = BasicBucketStoreWriter<0>;
template <typename Raw>
using RuntimeEncodedBaseBlock = EncodedBaseBlock<0, Raw>;

extern template class BasicBucketStore<0>;
extern template class BasicBucketStoreWriter<0>;
extern template class BasicBucketStore<5>;
extern template class BasicBucketStore<10>;
extern template class BasicBucketStoreWriter<5>;
extern template class BasicBucketStoreWriter<10>;

}  // namespace stlqpp
