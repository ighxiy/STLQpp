#include "stlqpp/model/model_shape.h"
#include "stlqpp/index/emit/basic_bucket_store.h"

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/platform/staging_directory.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

constexpr std::uint32_t kBucketStoreMagic = 0x51425453U;  // "STBQ"
constexpr std::uint32_t kBucketStoreVersion = 1U;
constexpr std::size_t kBucketStoreHeaderBytes = 48U;
constexpr std::array<std::uint8_t, 8U> kCheckpointMagic = {
    'S', 'T', 'L', 'Q', 'B', 'C', 'K', '1'};
constexpr std::uint32_t kCheckpointVersion = 2U;
constexpr std::size_t kCheckpointHeaderBytes = 64U;

std::filesystem::path BucketFilename(std::uint32_t bucket) {
  std::ostringstream name;
  name << "bucket_" << std::setw(6) << std::setfill('0') << bucket << ".bin";
  return name.str();
}

std::uint64_t RecordBytes(const BaseBuildShape& shape, std::uint32_t m) noexcept {
  return 2U * sizeof(std::uint32_t) + shape.raw_bytes_per_vector() +
      (m - 1U) + m * sizeof(float);
}

void StoreLittleEndianF32Span(const float* source,
                              std::size_t count,
                              std::uint8_t* destination) noexcept {
#if defined(_WIN32) || \
    (defined(__BYTE_ORDER__) && \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
  std::memcpy(destination, source, count * sizeof(float));
#else
  for (std::size_t index = 0U; index < count; ++index) {
    StoreLittleEndianF32(
        source[index], destination + index * sizeof(float));
  }
#endif
}

class UninitializedByteBuffer final {
 public:
  [[nodiscard]] bool Reserve(
      std::size_t requested,
      std::size_t limit = std::numeric_limits<std::size_t>::max()) noexcept {
    if (requested <= capacity_) {
      return true;
    }
    // Blocks append to partially filled buckets. Exact-fit reservation on
    // every block repeatedly copied the entire prefix (quadratic traffic).
    const std::size_t growth = capacity_ > limit - capacity_ / 2U
        ? limit : capacity_ + capacity_ / 2U;
    requested = std::max(requested, std::min(limit, growth));
    std::unique_ptr<std::uint8_t[]> replacement(
        new (std::nothrow) std::uint8_t[requested]);
    if (replacement == nullptr) {
      return false;
    }
    if (size_ != 0U) {
      std::memcpy(replacement.get(), data_.get(), size_);
    }
    data_ = std::move(replacement);
    capacity_ = requested;
    return true;
  }

  [[nodiscard]] std::uint8_t* Append(std::size_t count) noexcept {
    if (count > std::numeric_limits<std::size_t>::max() - size_) {
      return nullptr;
    }
    const std::size_t requested = size_ + count;
    if (requested > capacity_) {
      if (!Reserve(requested)) {
        return nullptr;
      }
    }
    std::uint8_t* appended = data_.get() + size_;
    size_ = requested;
    return appended;
  }

  [[nodiscard]] bool empty() const noexcept { return size_ == 0U; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] const std::uint8_t* data() const noexcept {
    return data_.get();
  }
  void clear() noexcept { size_ = 0U; }

 private:
  std::unique_ptr<std::uint8_t[]> data_;
  std::size_t size_ = 0U;
  std::size_t capacity_ = 0U;
};

Status WriteAll(std::ofstream& stream,
                const std::uint8_t* bytes,
                std::size_t byte_count,
                const char* label) {
  if (byte_count >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    return Status::ResourceExhausted(std::string(label) +
                                     " exceeds streamsize");
  }
  if (byte_count != 0U) {
    stream.write(reinterpret_cast<const char*>(bytes),
                 static_cast<std::streamsize>(byte_count));
  }
  if (!stream.good()) {
    return Status::Internal(std::string("failed to write ") + label);
  }
  return Status::Ok();
}

Status EncodeMetadata(const BaseBuildShape& shape,
                      std::uint32_t m,
                      std::uint64_t total_vectors,
                      const std::vector<std::uint64_t>& cid_counts,
                      std::vector<std::uint8_t>& output) {
  const std::uint64_t count_bytes =
      static_cast<std::uint64_t>(cid_counts.size()) * sizeof(std::uint64_t);
  if (count_bytes > std::numeric_limits<std::size_t>::max() -
                        kBucketStoreHeaderBytes) {
    return Status::ResourceExhausted(
        "basic bucket metadata exceeds addressable memory");
  }
  output.assign(kBucketStoreHeaderBytes +
                    static_cast<std::size_t>(count_bytes),
                0U);
  StoreLittleEndianU32(kBucketStoreMagic, output.data());
  StoreLittleEndianU32(kBucketStoreVersion, output.data() + 4U);
  StoreLittleEndianU32(m, output.data() + 8U);
  StoreLittleEndianU32(shape.dimension, output.data() + 12U);
  StoreLittleEndianU32(shape.nlist, output.data() + 16U);
  StoreLittleEndianU32(shape.cids_per_bucket, output.data() + 20U);
  StoreLittleEndianU32(static_cast<std::uint32_t>(shape.vector_scalar),
                       output.data() + 24U);
  StoreLittleEndianU32(shape.bucket_count(), output.data() + 28U);
  StoreLittleEndianU64(total_vectors, output.data() + 32U);
  StoreLittleEndianU64(count_bytes, output.data() + 40U);
  for (std::size_t cid = 0U; cid < cid_counts.size(); ++cid) {
    StoreLittleEndianU64(cid_counts[cid],
                         output.data() + kBucketStoreHeaderBytes +
                             cid * sizeof(std::uint64_t));
  }
  return Status::Ok();
}

std::filesystem::path CheckpointPath(
    const std::filesystem::path& directory, std::uint64_t generation) {
  return directory /
      (generation % 2U == 0U ? "checkpoint.0.stlqpp"
                             : "checkpoint.1.stlqpp");
}

std::uint64_t Checksum(ArrayView<const std::uint8_t> bytes) noexcept {
  std::uint64_t value = 1469598103934665603ULL;
  for (const std::uint8_t byte : bytes) {
    value ^= byte;
    value *= 1099511628211ULL;
  }
  return value;
}

struct BasicCheckpointState final {
  BaseBuildShape shape;
  ArtifactHash64 identity;
  std::uint64_t generation = 0U;
  std::uint64_t next_id = 0U;
  std::vector<std::uint64_t> cid_counts;
  std::vector<std::uint64_t> bucket_sizes;
};

Status EncodeCheckpoint(std::uint32_t m,
                        const BasicCheckpointState& checkpoint,
                        std::vector<std::uint8_t>* output) {
  const std::uint64_t bucket_bytes =
      static_cast<std::uint64_t>(checkpoint.bucket_sizes.size()) * 8U;
  if (bucket_bytes > std::numeric_limits<std::size_t>::max() -
                         kCheckpointHeaderBytes - sizeof(std::uint64_t)) {
    return Status::ResourceExhausted(
        "base-basic checkpoint metadata exceeds addressable memory");
  }
  output->assign(kCheckpointHeaderBytes +
                     static_cast<std::size_t>(bucket_bytes) +
                     sizeof(std::uint64_t),
                 0U);
  std::copy(kCheckpointMagic.begin(), kCheckpointMagic.end(), output->begin());
  StoreLittleEndianU32(kCheckpointVersion, output->data() + 8U);
  StoreLittleEndianU32(m, output->data() + 12U);
  StoreLittleEndianU32(checkpoint.shape.dimension, output->data() + 16U);
  StoreLittleEndianU32(checkpoint.shape.nlist, output->data() + 20U);
  StoreLittleEndianU32(checkpoint.shape.cids_per_bucket, output->data() + 24U);
  StoreLittleEndianU32(
      static_cast<std::uint32_t>(checkpoint.shape.vector_scalar),
      output->data() + 28U);
  StoreLittleEndianU64(checkpoint.identity.value(), output->data() + 32U);
  StoreLittleEndianU64(checkpoint.generation, output->data() + 40U);
  StoreLittleEndianU64(checkpoint.next_id, output->data() + 48U);
  StoreLittleEndianU32(
      static_cast<std::uint32_t>(checkpoint.bucket_sizes.size()),
      output->data() + 56U);
  std::size_t cursor = kCheckpointHeaderBytes;
  for (const std::uint64_t size : checkpoint.bucket_sizes) {
    StoreLittleEndianU64(size, output->data() + cursor);
    cursor += 8U;
  }
  StoreLittleEndianU64(
      Checksum({output->data(), cursor}), output->data() + cursor);
  return Status::Ok();
}

StatusOr<BasicCheckpointState> ReadCheckpoint(
    const std::filesystem::path& path,
    std::uint32_t expected_m,
    const BaseBuildShape& expected_shape,
    ArtifactHash64 expected_identity) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return Status::DataLoss("base-basic checkpoint slot is unreadable");
  }
  const std::streamoff end = input.tellg();
  if (end < 0 || static_cast<std::uint64_t>(end) >
                     std::numeric_limits<std::size_t>::max()) {
    return Status::DataLoss("base-basic checkpoint slot has invalid size");
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  input.seekg(0);
  if (!bytes.empty() &&
      !input.read(reinterpret_cast<char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()))) {
    return Status::DataLoss("base-basic checkpoint slot is truncated");
  }
  if (bytes.size() < kCheckpointHeaderBytes + 8U) {
    return Status::DataLoss("base-basic checkpoint slot is truncated");
  }
  const std::uint32_t version = LoadLittleEndianU32(bytes.data() + 8U);
  const std::size_t expected_bytes = kCheckpointHeaderBytes +
      (version == 1U ? static_cast<std::size_t>(expected_shape.nlist) * 8U : 0U) +
      static_cast<std::size_t>(expected_shape.bucket_count()) * 8U + 8U;
  if (bytes.size() != expected_bytes ||
      !std::equal(kCheckpointMagic.begin(), kCheckpointMagic.end(),
                  bytes.begin()) ||
      (version != 1U && version != kCheckpointVersion) ||
      LoadLittleEndianU32(bytes.data() + 12U) != expected_m ||
      LoadLittleEndianU32(bytes.data() + 16U) != expected_shape.dimension ||
      LoadLittleEndianU32(bytes.data() + 20U) != expected_shape.nlist ||
      LoadLittleEndianU32(bytes.data() + 24U) !=
          expected_shape.cids_per_bucket ||
      LoadLittleEndianU32(bytes.data() + 28U) !=
          static_cast<std::uint32_t>(expected_shape.vector_scalar) ||
      LoadLittleEndianU64(bytes.data() + 32U) != expected_identity.value() ||
      LoadLittleEndianU32(bytes.data() + 56U) !=
          expected_shape.bucket_count() ||
      LoadLittleEndianU64(bytes.data() + bytes.size() - 8U) !=
          Checksum({bytes.data(), bytes.size() - 8U})) {
    return Status::DataLoss(
        "base-basic checkpoint identity, shape, or checksum changed");
  }
  BasicCheckpointState checkpoint;
  checkpoint.shape = expected_shape;
  checkpoint.identity = expected_identity;
  checkpoint.generation = LoadLittleEndianU64(bytes.data() + 40U);
  checkpoint.next_id = LoadLittleEndianU64(bytes.data() + 48U);
  if (version == 1U) checkpoint.cid_counts.resize(expected_shape.nlist);
  checkpoint.bucket_sizes.resize(expected_shape.bucket_count());
  std::size_t cursor = kCheckpointHeaderBytes;
  std::uint64_t total = 0U;
  for (std::uint64_t& count : checkpoint.cid_counts) {
    count = LoadLittleEndianU64(bytes.data() + cursor);
    cursor += 8U;
    if (count > std::numeric_limits<std::uint64_t>::max() - total) {
      return Status::DataLoss("base-basic checkpoint row count overflows");
    }
    total += count;
  }
  for (std::uint64_t& size : checkpoint.bucket_sizes) {
    size = LoadLittleEndianU64(bytes.data() + cursor);
    cursor += 8U;
  }
  if (version == 1U && total != checkpoint.next_id) {
    return Status::DataLoss(
        "base-basic checkpoint row totals disagree");
  }
  const std::uint64_t record_bytes = RecordBytes(expected_shape, expected_m);
  std::uint64_t bucket_rows = 0;
  for (const std::uint64_t size : checkpoint.bucket_sizes) {
    if (size % record_bytes != 0U ||
        size / record_bytes > checkpoint.next_id - bucket_rows) {
      return Status::DataLoss("base-basic checkpoint bucket extents disagree");
    }
    bucket_rows += size / record_bytes;
  }
  if (bucket_rows != checkpoint.next_id) {
    return Status::DataLoss("base-basic checkpoint bucket row totals disagree");
  }
  return checkpoint;
}

}  // namespace

Status BaseBuildShape::Validate(std::uint32_t m) const {
  if ((m < 2U || m > stlqpp::kMaximumModelLayers) || dimension == 0U || nlist == 0U ||
      cids_per_bucket == 0U) {
    return Status::InvalidArgument("invalid basic bucket build shape");
  }
  const std::uint32_t scalar_bytes =
      static_cast<std::uint32_t>(vector_scalar);
  if (scalar_bytes != 1U && scalar_bytes != 4U) {
    return Status::InvalidArgument("unsupported base vector scalar");
  }
  if (dimension > std::numeric_limits<std::uint32_t>::max() / scalar_bytes) {
    return Status::ResourceExhausted(
        "base vector byte width exceeds u32");
  }
  return Status::Ok();
}

std::uint32_t BaseBuildShape::bucket_count() const noexcept {
  return nlist == 0U || cids_per_bucket == 0U
      ? 0U
      : 1U + (nlist - 1U) / cids_per_bucket;
}

std::uint32_t BaseBuildShape::raw_bytes_per_vector() const noexcept {
  return dimension * static_cast<std::uint32_t>(vector_scalar);
}

StatusOr<detail::BasicBucketStoreMetadata>
detail::BasicBucketStoreMetadata::Open(
    const std::filesystem::path& directory, std::uint32_t expected_m) {
  auto metadata = ReadOnlyMappedFile::Open(directory / "meta.stlqpp");
  if (!metadata.ok()) {
    return metadata.status();
  }
  if (metadata.value().size() < kBucketStoreHeaderBytes) {
    return Status::DataLoss("basic bucket metadata is truncated");
  }
  const std::uint8_t* bytes = metadata.value().data();
  if (LoadLittleEndianU32(bytes) != kBucketStoreMagic ||
      LoadLittleEndianU32(bytes + 4U) != kBucketStoreVersion ||
      (expected_m != 0U && LoadLittleEndianU32(bytes + 8U) != expected_m)) {
    return Status::DataLoss("unsupported basic bucket metadata");
  }
  BasicBucketStoreMetadata store;
  store.directory = directory;
  store.m = LoadLittleEndianU32(bytes + 8U);
  store.shape.dimension = LoadLittleEndianU32(bytes + 12U);
  store.shape.nlist = LoadLittleEndianU32(bytes + 16U);
  store.shape.cids_per_bucket = LoadLittleEndianU32(bytes + 20U);
  store.shape.vector_scalar = static_cast<BaseVectorScalar>(
      LoadLittleEndianU32(bytes + 24U));
  const std::uint32_t stored_bucket_count =
      LoadLittleEndianU32(bytes + 28U);
  store.total_vectors = LoadLittleEndianU64(bytes + 32U);
  const std::uint64_t count_bytes = LoadLittleEndianU64(bytes + 40U);
  if (const Status status = store.shape.Validate(store.m); !status.ok()) {
    return Status::DataLoss(status.message());
  }
  const std::uint64_t expected_count_bytes =
      static_cast<std::uint64_t>(store.shape.nlist) * sizeof(std::uint64_t);
  if (stored_bucket_count != store.shape.bucket_count() ||
      count_bytes != expected_count_bytes ||
      count_bytes > std::numeric_limits<std::size_t>::max() -
                        kBucketStoreHeaderBytes ||
      metadata.value().size() !=
          kBucketStoreHeaderBytes + static_cast<std::size_t>(count_bytes)) {
    return Status::DataLoss("basic bucket metadata extents disagree");
  }
  store.cid_counts.resize(store.shape.nlist);
  std::uint64_t count_sum = 0U;
  for (std::size_t cid = 0U; cid < store.cid_counts.size(); ++cid) {
    const std::uint64_t count = LoadLittleEndianU64(
        bytes + kBucketStoreHeaderBytes + cid * sizeof(std::uint64_t));
    if (count > std::numeric_limits<std::uint64_t>::max() - count_sum) {
      return Status::DataLoss("basic bucket row count overflows u64");
    }
    store.cid_counts[cid] = count;
    count_sum += count;
  }
  if (count_sum != store.total_vectors) {
    return Status::DataLoss("basic bucket CID counts disagree with total");
  }
  const std::uint64_t record_bytes = RecordBytes(store.shape, store.m);
  for (std::uint32_t bucket = 0U; bucket < stored_bucket_count; ++bucket) {
    const std::uint64_t rows = store.bucket_rows(bucket);
    if (rows == 0U) {
      continue;
    }
    if (rows > std::numeric_limits<std::uint64_t>::max() / record_bytes) {
      return Status::DataLoss("basic bucket file extent overflows u64");
    }
    std::error_code error;
    const std::uint64_t actual =
        std::filesystem::file_size(store.bucket_path(bucket), error);
    if (error || actual != rows * record_bytes) {
      return Status::DataLoss(
          "basic bucket file length disagrees with CID counts");
    }
  }
  return store;
}

std::filesystem::path detail::BasicBucketStoreMetadata::bucket_path(
    std::uint32_t bucket) const {
  return directory / BucketFilename(bucket);
}

std::uint64_t detail::BasicBucketStoreMetadata::bucket_rows(
    std::uint32_t bucket) const noexcept {
  const std::uint32_t first = bucket * shape.cids_per_bucket;
  const std::uint32_t end =
      std::min(shape.nlist, first + shape.cids_per_bucket);
  std::uint64_t rows = 0U;
  for (std::uint32_t cid = first; cid < end; ++cid) {
    rows += cid_counts[cid];
  }
  return rows;
}

template <int M>
BasicBucketStore<M>::BasicBucketStore(
    detail::BasicBucketStoreMetadata&& metadata) noexcept
    : metadata_(std::move(metadata)) {}

template <int M>
StatusOr<BasicBucketStore<M>> BasicBucketStore<M>::Open(
    const std::filesystem::path& directory) {
  auto metadata = detail::BasicBucketStoreMetadata::Open(directory, M);
  if (!metadata.ok()) return metadata.status();
  return BasicBucketStore<M>(std::move(metadata).value());
}

template <int M>
std::filesystem::path BasicBucketStore<M>::bucket_path(
    std::uint32_t bucket) const {
  return metadata_.bucket_path(bucket);
}

template <int M>
std::uint64_t BasicBucketStore<M>::bucket_rows(
    std::uint32_t bucket) const noexcept {
  return metadata_.bucket_rows(bucket);
}

// One owner for publication, checkpoint slots and bucket buffers. AppendTyped
// retains its existing fixed/runtime strides and writes directly into this owner.
struct detail::BasicBucketStoreWriterState final {
  [[nodiscard]] static StatusOr<std::unique_ptr<BasicBucketStoreWriterState>>
  Create(const std::filesystem::path& destination, BaseBuildShape shape,
         bool durable, std::size_t bucket_flush_bytes, std::uint32_t m,
         std::uint32_t expected_m);
  [[nodiscard]] static StatusOr<std::unique_ptr<BasicBucketStoreWriterState>>
  CreateCheckpointed(const std::filesystem::path& destination,
                     BaseBuildShape shape, ArtifactHash64 identity,
                     std::size_t bucket_flush_bytes, std::uint32_t m,
                     std::uint32_t expected_m);
  [[nodiscard]] static StatusOr<std::unique_ptr<BasicBucketStoreWriterState>>
  OpenCheckpointed(const std::filesystem::path& destination,
                   BaseBuildShape shape, ArtifactHash64 expected_identity,
                   std::size_t bucket_flush_bytes, std::uint32_t m,
                   std::uint32_t expected_m);
  [[nodiscard]] Status Commit();

  struct Bucket final {
    std::ofstream stream;
    UninitializedByteBuffer buffer;
    std::uint64_t written_bytes = 0U;
  };

  StagingDirectory publication;
  std::filesystem::path checkpoint_directory;
  BaseBuildShape shape;
  std::uint32_t m = 0U;
  ArtifactHash64 checkpoint_identity;
  std::uint64_t next_id = 0U;
  std::uint64_t checkpoint_generation = 0U;
  std::size_t bucket_flush_bytes = 0U;
  bool checkpointed = false;
  std::vector<std::uint64_t> cid_counts;
  std::vector<std::size_t> block_bucket_rows;
  std::vector<Bucket> buckets;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return checkpointed ? checkpoint_directory : publication.path();
  }

  Status Flush(std::uint32_t bucket) {
    Bucket& target = buckets[bucket];
    if (target.buffer.empty()) {
      return Status::Ok();
    }
    if (!target.stream.is_open()) {
      target.stream.open(
          path() / BucketFilename(bucket),
          std::ios::binary |
              (checkpointed ? std::ios::app : std::ios::trunc));
      if (!target.stream.is_open()) {
        return Status::Internal("failed to create basic bucket file");
      }
    }
    Status status = WriteAll(target.stream, target.buffer.data(),
                             target.buffer.size(), "basic bucket rows");
    if (!status.ok()) {
      return status;
    }
    target.written_bytes += target.buffer.size();
    target.buffer.clear();
    return Status::Ok();
  }

  Status SaveCheckpoint() {
    if (!checkpointed) return Status::Ok();
    std::vector<std::uint64_t> bucket_sizes(buckets.size(), 0U);
    for (std::uint32_t bucket = 0U; bucket < buckets.size(); ++bucket) {
      if (const Status status = Flush(bucket); !status.ok()) return status;
      if (buckets[bucket].stream.is_open()) {
        buckets[bucket].stream.flush();
        if (buckets[bucket].stream.fail()) {
          return Status::Internal(
              "failed to flush base-basic checkpoint bucket");
        }
      }
      bucket_sizes[bucket] = buckets[bucket].written_bytes;
    }
    BasicCheckpointState checkpoint = {
        .shape = shape,
        .identity = checkpoint_identity,
        .generation = checkpoint_generation,
        .next_id = next_id,
        .cid_counts = {},  // v1 read compatibility only; v2 stores extents.
        .bucket_sizes = std::move(bucket_sizes),
    };
    std::vector<std::uint8_t> bytes;
    if (const Status status = EncodeCheckpoint(m, checkpoint, &bytes);
        !status.ok()) {
      return status;
    }
    std::ofstream output(
        CheckpointPath(path(), checkpoint_generation),
        std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
      return Status::Internal(
          "failed to create base-basic checkpoint slot");
    }
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    output.close();
    if (output.fail()) {
      return Status::Internal(
          "failed to publish base-basic checkpoint slot");
    }
    ++checkpoint_generation;
    return Status::Ok();
  }
};

template <int M>
BasicBucketStoreWriter<M>::BasicBucketStoreWriter(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
BasicBucketStoreWriter<M>::~BasicBucketStoreWriter() = default;

template <int M>
BasicBucketStoreWriter<M>::BasicBucketStoreWriter(
    BasicBucketStoreWriter&& other) noexcept
    : state_(std::move(other.state_)) {}

template <int M>
BasicBucketStoreWriter<M>& BasicBucketStoreWriter<M>::operator=(
    BasicBucketStoreWriter&& other) noexcept {
  state_ = std::move(other.state_);
  return *this;
}

template <int M>
StatusOr<BasicBucketStoreWriter<M>> BasicBucketStoreWriter<M>::Create(
    const std::filesystem::path& destination,
    BaseBuildShape shape,
    std::size_t bucket_flush_bytes,
    std::uint32_t m) {
  return CreateImpl(destination, shape, true, bucket_flush_bytes, m);
}

template <int M>
StatusOr<BasicBucketStoreWriter<M>>
BasicBucketStoreWriter<M>::CreateTransient(
    const std::filesystem::path& destination,
    BaseBuildShape shape,
    std::size_t bucket_flush_bytes,
    std::uint32_t m) {
  return CreateImpl(destination, shape, false, bucket_flush_bytes, m);
}

StatusOr<std::unique_ptr<detail::BasicBucketStoreWriterState>>
detail::BasicBucketStoreWriterState::CreateCheckpointed(
    const std::filesystem::path& destination,
    BaseBuildShape shape,
    ArtifactHash64 identity,
    std::size_t bucket_flush_bytes,
    std::uint32_t m, std::uint32_t expected_m) {
  if (destination.empty() || identity.empty() || bucket_flush_bytes == 0U) {
    return Status::InvalidArgument(
        "base-basic checkpoint request is incomplete");
  }
  if (expected_m != 0U && m != expected_m) {
    return Status::InvalidArgument("bucket writer M disagrees with fixed-M type");
  }
  if (const Status status = shape.Validate(m); !status.ok()) return status;
  std::error_code error;
  if (std::filesystem::exists(destination, error) || error) {
    return Status::FailedPrecondition(
        "base-basic checkpoint destination already exists");
  }
  std::filesystem::create_directories(destination, error);
  if (error) {
    return Status::Internal(
        "failed to create base-basic checkpoint directory");
  }
  auto state = std::make_unique<BasicBucketStoreWriterState>();
  state->checkpoint_directory = destination;
  state->shape = shape;
  state->m = m;
  state->checkpoint_identity = identity;
  state->bucket_flush_bytes = bucket_flush_bytes;
  state->checkpointed = true;
  state->cid_counts.assign(shape.nlist, 0U);
  state->block_bucket_rows.assign(shape.bucket_count(), 0U);
  state->buckets.resize(shape.bucket_count());
  if (const Status status = state->SaveCheckpoint(); !status.ok()) {
    std::filesystem::remove_all(destination, error);
    return status;
  }
  return state;
}

StatusOr<std::unique_ptr<detail::BasicBucketStoreWriterState>>
detail::BasicBucketStoreWriterState::OpenCheckpointed(
    const std::filesystem::path& destination,
    BaseBuildShape shape,
    ArtifactHash64 expected_identity,
    std::size_t bucket_flush_bytes,
    std::uint32_t m, std::uint32_t expected_m) {
  if (destination.empty() || expected_identity.empty() ||
      bucket_flush_bytes == 0U) {
    return Status::InvalidArgument(
        "base-basic checkpoint reopen request is incomplete");
  }
  if (expected_m != 0U && m != expected_m) {
    return Status::InvalidArgument("bucket writer M disagrees with fixed-M type");
  }
  if (const Status status = shape.Validate(m); !status.ok()) return status;
  std::optional<BasicCheckpointState> selected;
  for (std::uint64_t slot = 0U; slot < 2U; ++slot) {
    const std::filesystem::path path = CheckpointPath(destination, slot);
    std::error_code error;
    if (!std::filesystem::exists(path, error)) {
      if (error) {
        return Status::Internal(
            "failed to inspect base-basic checkpoint slot");
      }
      continue;
    }
    auto checkpoint = ReadCheckpoint(
        path, m, shape, expected_identity);
    if (!checkpoint.ok()) continue;
    if (!selected.has_value() ||
        checkpoint.value().generation > selected->generation) {
      selected = std::move(checkpoint).value();
    }
  }
  if (!selected.has_value()) {
    return Status::DataLoss(
        "base-basic checkpoint has no valid committed slot");
  }
  auto state = std::make_unique<BasicBucketStoreWriterState>();
  state->checkpoint_directory = destination;
  state->shape = shape;
  state->m = m;
  state->checkpoint_identity = expected_identity;
  state->next_id = selected->next_id;
  state->checkpoint_generation = selected->generation + 1U;
  state->bucket_flush_bytes = bucket_flush_bytes;
  state->checkpointed = true;
  state->cid_counts = std::move(selected->cid_counts);
  const bool rebuild_counts = state->cid_counts.empty();
  if (rebuild_counts) state->cid_counts.assign(shape.nlist, 0U);
  state->block_bucket_rows.assign(shape.bucket_count(), 0U);
  state->buckets.resize(shape.bucket_count());
  for (std::uint32_t bucket = 0U; bucket < shape.bucket_count(); ++bucket) {
    state->buckets[bucket].written_bytes = selected->bucket_sizes[bucket];
    const std::filesystem::path file =
        destination / BucketFilename(bucket);
    std::error_code error;
    const bool exists = std::filesystem::exists(file, error);
    if (error || (!exists && selected->bucket_sizes[bucket] != 0U)) {
      return Status::DataLoss(
          "base-basic checkpoint bucket is missing");
    }
    if (exists) {
      const std::uint64_t actual = std::filesystem::file_size(file, error);
      if (!error && actual < selected->bucket_sizes[bucket]) {
        return Status::DataLoss(
            "base-basic checkpoint bucket is shorter than its commit");
      }
      if (!error && actual > selected->bucket_sizes[bucket]) {
        std::filesystem::resize_file(
            file, selected->bucket_sizes[bucket], error);
      }
      if (error) {
        return Status::Internal(
            "failed to truncate base-basic checkpoint bucket");
      }
    }
    // Counts are only needed again after restart. Scan the committed prefix
    // once instead of copying/checksumming/writing every CID on every block.
    // Legacy v1 slots already carry counts and avoid this scan.
    if (rebuild_counts && selected->bucket_sizes[bucket] != 0U) {
      auto mapped = ReadOnlyMappedFile::Open(file);
      if (!mapped.ok()) return mapped.status();
      const std::size_t record_bytes = static_cast<std::size_t>(RecordBytes(shape, m));
      for (std::size_t offset = 0; offset < mapped.value().size(); offset += record_bytes) {
        const std::uint8_t* row = mapped.value().data() + offset;
        const std::uint32_t id = LoadLittleEndianU32(row);
        const std::uint32_t cid = LoadLittleEndianU32(row + 4U);
        if (id >= state->next_id || cid >= shape.nlist ||
            cid / shape.cids_per_bucket != bucket) {
          return Status::DataLoss("base-basic checkpoint row is outside its committed domain");
        }
        ++state->cid_counts[cid];
      }
    }
  }
  return state;
}

StatusOr<std::unique_ptr<detail::BasicBucketStoreWriterState>>
detail::BasicBucketStoreWriterState::Create(
    const std::filesystem::path& destination,
    BaseBuildShape shape,
    bool durable,
    std::size_t bucket_flush_bytes,
    std::uint32_t m, std::uint32_t expected_m) {
  if (bucket_flush_bytes == 0U) {
    return Status::InvalidArgument(
        "basic bucket flush threshold must be positive");
  }
  if (expected_m != 0U && m != expected_m) {
    return Status::InvalidArgument("bucket writer M disagrees with fixed-M type");
  }
  if (const Status status = shape.Validate(m); !status.ok()) {
    return status;
  }
  auto publication = durable
      ? StagingDirectory::Create(destination)
      : StagingDirectory::CreateTransient(destination);
  if (!publication.ok()) {
    return publication.status();
  }
  auto state = std::make_unique<BasicBucketStoreWriterState>();
  state->publication = std::move(publication).value();
  state->shape = shape;
  state->m = m;
  state->bucket_flush_bytes = bucket_flush_bytes;
  state->cid_counts.assign(shape.nlist, 0U);
  state->block_bucket_rows.assign(shape.bucket_count(), 0U);
  state->buckets.resize(shape.bucket_count());
  return state;
}

template <int M>
StatusOr<BasicBucketStoreWriter<M>>
BasicBucketStoreWriter<M>::CreateCheckpointed(
    const std::filesystem::path& destination, BaseBuildShape shape,
    ArtifactHash64 identity, std::size_t bucket_flush_bytes, std::uint32_t m) {
  auto state = State::CreateCheckpointed(
      destination, shape, identity, bucket_flush_bytes, m, M);
  if (!state.ok()) return state.status();
  return BasicBucketStoreWriter<M>(std::move(state).value());
}

template <int M>
StatusOr<BasicBucketStoreWriter<M>>
BasicBucketStoreWriter<M>::OpenCheckpointed(
    const std::filesystem::path& destination, BaseBuildShape shape,
    ArtifactHash64 identity, std::size_t bucket_flush_bytes, std::uint32_t m) {
  auto state = State::OpenCheckpointed(
      destination, shape, identity, bucket_flush_bytes, m, M);
  if (!state.ok()) return state.status();
  return BasicBucketStoreWriter<M>(std::move(state).value());
}

template <int M>
StatusOr<BasicBucketStoreWriter<M>> BasicBucketStoreWriter<M>::CreateImpl(
    const std::filesystem::path& destination, BaseBuildShape shape,
    bool durable, std::size_t bucket_flush_bytes, std::uint32_t m) {
  auto state = State::Create(
      destination, shape, durable, bucket_flush_bytes, m, M);
  if (!state.ok()) return state.status();
  return BasicBucketStoreWriter<M>(std::move(state).value());
}

template <int M>
template <bool CidsValidated, typename Raw>
Status BasicBucketStoreWriter<M>::AppendTyped(
    const EncodedBaseBlock<M, Raw>& block) {
  constexpr bool kRawU8 = std::is_same_v<Raw, std::uint8_t>;
  constexpr bool kRawF32 = std::is_same_v<Raw, float>;
  static_assert(kRawU8 || kRawF32);
  if (state_ == nullptr) {
    return Status::FailedPrecondition(
        "basic bucket writer is not accepting rows");
  }
  const std::uint32_t m = M == 0 ? state_->m : M;
  const std::size_t rows = block.root_cids.size();
  const BaseVectorScalar expected_scalar =
      kRawU8 ? BaseVectorScalar::kU8 : BaseVectorScalar::kF32;
  if (state_->shape.vector_scalar != expected_scalar ||
      block.first_id != state_->next_id || block.raw_vectors == nullptr ||
      block.raw_leading_dimension < state_->shape.dimension ||
      block.small_codes_row_major.size() != rows * (m - 1U) ||
      block.coefficients_row_major.size() != rows * m ||
      static_cast<std::uint64_t>(block.first_id) + rows >
          static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) +
              1U) {
    return Status::InvalidArgument(
        "encoded base block disagrees with bucket writer shape or order");
  }

  const std::size_t record_bytes =
      static_cast<std::size_t>(RecordBytes(state_->shape, m));
  std::fill(state_->block_bucket_rows.begin(),
            state_->block_bucket_rows.end(), 0U);
  for (std::size_t row = 0U; row < rows; ++row) {
    const std::uint32_t cid = block.root_cids[row];
    if constexpr (CidsValidated) {
      assert(cid < state_->shape.nlist);
    } else {
      if (cid >= state_->shape.nlist) {
        return Status::InvalidArgument(
            "encoded base block contains an out-of-domain root CID");
      }
    }
    ++state_->block_bucket_rows[cid / state_->shape.cids_per_bucket];
  }
  for (std::size_t bucket = 0U; bucket < state_->buckets.size(); ++bucket) {
    if (state_->block_bucket_rows[bucket] == 0U) {
      continue;
    }
    const std::size_t maximum_append =
        state_->bucket_flush_bytes >
                std::numeric_limits<std::size_t>::max() - record_bytes
            ? std::numeric_limits<std::size_t>::max()
            : state_->bucket_flush_bytes + record_bytes;
    const std::size_t rows_before_flush = std::min(
        state_->block_bucket_rows[bucket], maximum_append / record_bytes);
    const std::size_t append_bytes = rows_before_flush * record_bytes;
    const std::size_t current = state_->buckets[bucket].buffer.size();
    const std::size_t requested = current > maximum_append - append_bytes
        ? maximum_append
        : std::min(maximum_append, current + append_bytes);
    if (!state_->buckets[bucket].buffer.Reserve(requested, maximum_append)) {
      return Status::ResourceExhausted(
          "basic bucket writer cannot reserve its serialization buffer");
    }
  }
  for (std::size_t row = 0U; row < rows; ++row) {
    const std::uint32_t cid = block.root_cids[row];
    const std::uint32_t bucket = cid / state_->shape.cids_per_bucket;
    UninitializedByteBuffer& output = state_->buckets[bucket].buffer;
    std::uint8_t* cursor = output.Append(record_bytes);
    if (cursor == nullptr) {
      return Status::ResourceExhausted(
          "basic bucket writer cannot grow its serialization buffer");
    }
    StoreLittleEndianU32(block.first_id + static_cast<std::uint32_t>(row),
                         cursor);
    StoreLittleEndianU32(cid, cursor + 4U);
    cursor += 8U;
    const Raw* raw = block.raw_vectors +
        row * static_cast<std::size_t>(block.raw_leading_dimension);
    if constexpr (kRawU8) {
      std::memcpy(cursor, raw, state_->shape.dimension);
      cursor += state_->shape.dimension;
    } else {
      StoreLittleEndianF32Span(raw, state_->shape.dimension, cursor);
      cursor += static_cast<std::size_t>(state_->shape.dimension) *
          sizeof(float);
    }
    const std::uint8_t* small = block.small_codes_row_major.data() +
        row * (m - 1U);
    std::memcpy(cursor, small, m - 1U);
    cursor += m - 1U;
    const float* coefficients = block.coefficients_row_major.data() + row * m;
    StoreLittleEndianF32Span(coefficients, m, cursor);
    cursor += static_cast<std::size_t>(m) * sizeof(float);
    ++state_->cid_counts[cid];
    if (output.size() >= state_->bucket_flush_bytes) {
      if (const Status status = state_->Flush(bucket); !status.ok()) {
        return status;
      }
    }
  }
  state_->next_id += rows;
  return state_->SaveCheckpoint();
}

template <int M>
Status BasicBucketStoreWriter<M>::Append(
    const EncodedBaseBlock<M, std::uint8_t>& block) {
  return AppendTyped<false>(block);
}

template <int M>
Status BasicBucketStoreWriter<M>::Append(
    const EncodedBaseBlock<M, float>& block) {
  return AppendTyped<false>(block);
}

template <int M>
Status BasicBucketStoreWriter<M>::AppendAfterBasicMoments(
    const EncodedBaseBlock<M, std::uint8_t>& block) {
  return AppendTyped<true>(block);
}

template <int M>
Status BasicBucketStoreWriter<M>::AppendAfterBasicMoments(
    const EncodedBaseBlock<M, float>& block) {
  return AppendTyped<true>(block);
}

Status detail::BasicBucketStoreWriterState::Commit() {
  for (std::uint32_t bucket = 0U;
       bucket < shape.bucket_count(); ++bucket) {
    if (const Status status = Flush(bucket); !status.ok()) {
      return status;
    }
    if (buckets[bucket].stream.is_open()) {
      buckets[bucket].stream.close();
      if (buckets[bucket].stream.fail()) {
        return Status::Internal("failed to close basic bucket file");
      }
    }
  }
  std::vector<std::uint8_t> metadata;
  if (const Status status = EncodeMetadata(
          shape, m, next_id, cid_counts, metadata);
      !status.ok()) {
    return status;
  }
  std::ofstream output(path() / "meta.stlqpp",
                       std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    return Status::Internal("failed to create basic bucket metadata");
  }
  if (const Status status = WriteAll(output, metadata.data(), metadata.size(),
                                     "basic bucket metadata");
      !status.ok()) {
    return status;
  }
  output.close();
  if (output.fail()) {
    return Status::Internal("failed to close basic bucket metadata");
  }
  if (!checkpointed) {
    if (const Status status = publication.Publish(); !status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

template <int M>
Status BasicBucketStoreWriter<M>::Commit() {
  if (state_ == nullptr) {
    return Status::FailedPrecondition(
        "basic bucket writer cannot be committed twice");
  }
  if (const Status status = state_->Commit(); !status.ok()) return status;
  state_.reset();
  return Status::Ok();
}

template <int M>
std::uint64_t BasicBucketStoreWriter<M>::next_id() const noexcept {
  return state_ == nullptr ? 0U : state_->next_id;
}

template class BasicBucketStore<0>;
template class BasicBucketStoreWriter<0>;
template class BasicBucketStore<5>;
template class BasicBucketStore<12>;
template class BasicBucketStore<6>;
template class BasicBucketStore<10>;
template class BasicBucketStoreWriter<5>;
template class BasicBucketStoreWriter<12>;
template class BasicBucketStoreWriter<6>;
template class BasicBucketStoreWriter<10>;

}  // namespace stlqpp
