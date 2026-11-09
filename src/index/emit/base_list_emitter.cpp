#include "stlqpp/index/emit/base_list_emitter.h"

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/platform/positioned_file.h"
#include "stlqpp/platform/staging_directory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

constexpr std::uint32_t kBaseListMagic = 0x4C425453U;  // "STBL"
constexpr std::uint32_t kLegacyBaseListVersion = 1U;
constexpr std::uint32_t kIdentityBaseListVersion = 2U;
constexpr std::uint32_t kBaseListVersion = 3U;
constexpr std::size_t kLegacyBaseListHeaderBytes = 48U;
constexpr std::size_t kIdentityBaseListHeaderBytes = 56U;
constexpr std::size_t kBaseListHeaderBytes = 64U;
constexpr const char* kRawRemovedMarker = "raw.removed";

std::uint64_t BucketRecordBytes(const BaseBuildShape& shape, std::uint32_t m) noexcept {
  return 2U * sizeof(std::uint32_t) + shape.raw_bytes_per_vector() +
      (m - 1U) + m * sizeof(float);
}

StatusOr<std::uint64_t> ByteExtent(std::uint64_t rows,
                                   std::uint64_t row_bytes,
                                   const char* label) {
  if (row_bytes != 0U &&
      rows > std::numeric_limits<std::uint64_t>::max() / row_bytes) {
    return Status::ResourceExhausted(std::string(label) +
                                     " byte extent overflows u64");
  }
  return rows * row_bytes;
}

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

bool ReadExact(std::ifstream& stream,
               std::uint8_t* bytes,
               std::size_t byte_count) {
  if (byte_count >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    return false;
  }
  stream.read(reinterpret_cast<char*>(bytes),
              static_cast<std::streamsize>(byte_count));
  return static_cast<std::size_t>(stream.gcount()) == byte_count;
}

Status EncodeMetadata(const BaseBuildShape& shape,
                      std::uint32_t m,
                      ArtifactHash64 identity,
                      ArtifactHash64 source_model_identity,
                      const std::vector<std::uint64_t>& offsets,
                      std::vector<std::uint8_t>& output) {
  const std::uint64_t offset_bytes =
      static_cast<std::uint64_t>(offsets.size()) * sizeof(std::uint64_t);
  if (offset_bytes > std::numeric_limits<std::size_t>::max() -
                         kBaseListHeaderBytes) {
    return Status::ResourceExhausted(
        "base list metadata exceeds addressable memory");
  }
  output.assign(kBaseListHeaderBytes +
                    static_cast<std::size_t>(offset_bytes),
                0U);
  StoreLittleEndianU32(kBaseListMagic, output.data());
  StoreLittleEndianU32(kBaseListVersion, output.data() + 4U);
  StoreLittleEndianU32(m, output.data() + 8U);
  StoreLittleEndianU32(shape.dimension, output.data() + 12U);
  StoreLittleEndianU32(shape.nlist, output.data() + 16U);
  StoreLittleEndianU32(shape.cids_per_bucket, output.data() + 20U);
  StoreLittleEndianU32(static_cast<std::uint32_t>(shape.vector_scalar),
                       output.data() + 24U);
  StoreLittleEndianU32(shape.bucket_count(), output.data() + 28U);
  StoreLittleEndianU64(offsets.empty() ? 0U : offsets.back(),
                       output.data() + 32U);
  StoreLittleEndianU64(offset_bytes, output.data() + 40U);
  StoreLittleEndianU64(identity.value(), output.data() + 48U);
  StoreLittleEndianU64(
      source_model_identity.value(), output.data() + 56U);
  for (std::size_t index = 0U; index < offsets.size(); ++index) {
    StoreLittleEndianU64(offsets[index],
                         output.data() + kBaseListHeaderBytes +
                             index * sizeof(std::uint64_t));
  }
  return Status::Ok();
}

constexpr std::size_t kCidFlushBytes = 4U * 1024U * 1024U;
constexpr std::size_t kBucketReadBytes = 128U * 1024U;

struct CidBuffers final {
  std::uint64_t written_rows = 0U;
  std::vector<std::uint8_t> ids;
  std::vector<std::uint8_t> codes;
  std::vector<std::uint8_t> coefficients;
  std::vector<std::uint8_t> raw;
};

void AppendBytes(std::vector<std::uint8_t>& destination,
                 const std::uint8_t* source,
                 std::size_t byte_count) {
  destination.insert(destination.end(), source, source + byte_count);
}

constexpr std::size_t kPackedWriteBytes = 4U << 20U;
using EmitClock = std::chrono::steady_clock;

double Elapsed(EmitClock::time_point begin) {
  return std::chrono::duration<double>(EmitClock::now() - begin).count();
}

struct EmitContext final {
  const detail::BasicBucketStoreMetadata* buckets = nullptr;
  const std::vector<std::uint64_t>* offsets = nullptr;
  std::array<PositionedFile*, 4> files{};
  std::vector<Status>* results = nullptr;
  std::atomic<std::uint32_t> next_bucket{0U};
  std::atomic<bool> failed{false};
  BaseListEmitStatistics* statistics = nullptr;
  EmitClock::time_point workers_begin{};
};

bool ReadRecords(std::ifstream& input, std::uint8_t* data, std::size_t bytes,
                 BaseListBucketStatistics* statistics) {
  const auto begin = statistics ? EmitClock::now() : EmitClock::time_point{};
  const auto success = ReadExact(input, data, bytes);
  if (statistics) {
    statistics->read_sec += Elapsed(begin);
    if (success) statistics->read_bytes += bytes;
  }
  return success;
}

Status WriteRanges(
    EmitContext& context, std::uint64_t first_row,
    const std::array<std::size_t, 4>& widths,
    const std::array<ArrayView<const std::uint8_t>, 4>& ranges,
    BaseListBucketStatistics* statistics) {
  // Buckets own disjoint CID intervals and output byte ranges. Positioned
  // writes preserve those offsets without serializing independent buckets.
  for (std::size_t field = 0U; field < ranges.size(); ++field) {
    const auto bytes = ranges[field].size();
    for (std::size_t first = 0U; first < bytes;) {
      const auto count = std::min(bytes - first, kPackedWriteBytes);
      const auto begin = statistics ? EmitClock::now() : EmitClock::time_point{};
      const auto status = context.files[field]->WriteAt(
              first_row * widths[field] + first,
              {ranges[field].data() + first, count});
      if (statistics) {
        statistics->write_sec += Elapsed(begin);
        ++statistics->write_calls;
        if (status.ok()) statistics->write_bytes += count;
      }
      if (!status.ok()) {
        return status;
      }
      first += count;
    }
  }
  return Status::Ok();
}

// Pack a bucket into four buffers, then write each CID's ranges independently.
// The cap retains the bounded-CID fallback for unusually large/dense buckets.
constexpr std::uint64_t kPackedBucketBytes = 1ULL << 30U;

Status EmitPackedBucket(EmitContext& context, std::uint32_t bucket,
                        BaseListBucketStatistics* statistics) {
  const auto& source = *context.buckets;
  const auto& shape = source.shape;
  const auto first_cid = bucket * shape.cids_per_bucket;
  const auto end_cid = std::min(shape.nlist, first_cid + shape.cids_per_bucket);
  const auto first_row = (*context.offsets)[first_cid];
  const auto rows = (*context.offsets)[end_cid] - first_row;
  const std::size_t widths[4] = {sizeof(std::uint32_t), source.m - 1U,
                               source.m * sizeof(float),
                               shape.raw_bytes_per_vector()};
  std::unique_ptr<std::uint8_t[]> payload[4];
  for (std::size_t field = 0U; field < 4U; ++field) {
    // Every byte is filled from a validated input record before writing.
    payload[field].reset(new std::uint8_t[rows * widths[field]]);
  }
  std::vector<std::uint64_t> counts(end_cid - first_cid, 0U);
  std::ifstream input(source.bucket_path(bucket), std::ios::binary);
  if (!input.is_open()) {
    return Status::DataLoss("failed to open a non-empty basic bucket");
  }
  const auto record_bytes =
      static_cast<std::size_t>(BucketRecordBytes(shape, source.m));
  const auto read_rows = std::max<std::size_t>(1U, kBucketReadBytes / record_bytes);
  std::vector<std::uint8_t> records(
      std::min<std::uint64_t>(rows, read_rows) * record_bytes);
  for (std::uint64_t first = 0U; first < rows;) {
    const auto count = static_cast<std::size_t>(
        std::min<std::uint64_t>(rows - first, read_rows));
    if (!ReadRecords(input, records.data(), count * record_bytes, statistics)) {
      return Status::DataLoss("basic bucket row is truncated");
    }
    for (std::size_t row = 0U; row < count; ++row) {
      const auto* record = records.data() + row * record_bytes;
      const auto cid = LoadLittleEndianU32(record + 4U);
      if (cid < first_cid || cid >= end_cid) {
        return Status::DataLoss(
            "basic bucket row belongs to a different CID range");
      }
      auto& cid_rows = counts[cid - first_cid];
      if (cid_rows >= source.cid_counts[cid]) {
        return Status::DataLoss(
            "basic bucket CID payload exceeds metadata counts");
      }
      const auto position = (*context.offsets)[cid] - first_row + cid_rows++;
      std::memcpy(payload[0].get() + position * widths[0], record, widths[0]);
      const auto* cursor = record + 2U * sizeof(std::uint32_t);
      std::memcpy(payload[3].get() + position * widths[3], cursor, widths[3]);
      cursor += widths[3];
      std::memcpy(payload[1].get() + position * widths[1], cursor, widths[1]);
      cursor += widths[1];
      std::memcpy(payload[2].get() + position * widths[2], cursor, widths[2]);
    }
    first += count;
  }
  for (std::uint32_t cid = first_cid; cid < end_cid; ++cid) {
    if (counts[cid - first_cid] != source.cid_counts[cid]) {
      return Status::DataLoss(
          "base list CID payload disagrees with metadata counts");
    }
  }
  for (std::uint32_t cid = first_cid; cid < end_cid; ++cid) {
    const auto position = (*context.offsets)[cid];
    const auto cid_rows = source.cid_counts[cid];
    std::array<ArrayView<const std::uint8_t>, 4> ranges;
    for (std::size_t field = 0U; field < ranges.size(); ++field) {
      ranges[field] = {
          payload[field].get() + (position - first_row) * widths[field],
          static_cast<std::size_t>(cid_rows * widths[field])};
    }
    if (const auto status = WriteRanges(
            context, position, {widths[0], widths[1], widths[2], widths[3]},
            ranges, statistics);
        !status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status EmitBucket(EmitContext& context, std::uint32_t bucket,
                  BaseListBucketStatistics* statistics) {
  const detail::BasicBucketStoreMetadata& source = *context.buckets;
  const std::uint32_t m = source.m;
  const BaseBuildShape& shape = source.shape;
  const std::uint64_t expected_rows = source.bucket_rows(bucket);
  if (expected_rows == 0U) {
    return Status::Ok();
  }
  const std::uint64_t payload_row_bytes = sizeof(std::uint32_t) + (m - 1U) +
      m * sizeof(float) + shape.raw_bytes_per_vector();
  if (expected_rows <= kPackedBucketBytes / payload_row_bytes) {
    return EmitPackedBucket(context, bucket, statistics);
  }
  std::ifstream input(source.bucket_path(bucket), std::ios::binary);
  if (!input.is_open()) {
    return Status::DataLoss("failed to open a non-empty basic bucket");
  }
  const std::uint32_t first_cid = bucket * shape.cids_per_bucket;
  const std::uint32_t end_cid =
      std::min(shape.nlist, first_cid + shape.cids_per_bucket);
  const std::size_t raw_row_bytes = shape.raw_bytes_per_vector();
  const std::size_t flush_rows =
      std::max<std::size_t>(1U, kCidFlushBytes / payload_row_bytes);
  std::vector<CidBuffers> buffers(end_cid - first_cid);
  for (std::uint32_t cid = first_cid; cid < end_cid; ++cid) {
    const std::uint64_t rows64 = source.cid_counts[cid];
    const std::uint64_t largest_row_bytes = std::max<std::uint64_t>(
        raw_row_bytes, m * sizeof(float));
    if (rows64 > std::numeric_limits<std::size_t>::max() ||
        (largest_row_bytes != 0U &&
         rows64 > std::numeric_limits<std::size_t>::max() /
                      largest_row_bytes)) {
      return Status::ResourceExhausted(
          "one base list exceeds the host address domain");
    }
    const std::size_t rows = std::min(
        static_cast<std::size_t>(rows64), flush_rows);
    CidBuffers& target = buffers[cid - first_cid];
    target.ids.reserve(rows * sizeof(std::uint32_t));
    target.codes.reserve(rows * (m - 1U));
    target.coefficients.reserve(rows * m * sizeof(float));
    target.raw.reserve(rows * raw_row_bytes);
  }
  const auto flush = [&](std::uint32_t cid) -> Status {
    CidBuffers& payload = buffers[cid - first_cid];
    const std::uint64_t rows = payload.ids.size() / sizeof(std::uint32_t);
    if (rows == 0U) return Status::Ok();
    const std::uint64_t position = (*context.offsets)[cid] + payload.written_rows;
    if (const auto status = WriteRanges(
            context, position,
            {sizeof(std::uint32_t), m - 1U, m * sizeof(float), raw_row_bytes},
            {payload.ids, payload.codes, payload.coefficients, payload.raw},
            statistics);
        !status.ok()) {
      return status;
    }
    payload.written_rows += rows;
    payload.ids.clear();
    payload.codes.clear();
    payload.coefficients.clear();
    payload.raw.clear();
    return Status::Ok();
  };
  const std::size_t record_bytes =
      static_cast<std::size_t>(BucketRecordBytes(shape, m));
  const std::size_t read_rows =
      std::max<std::size_t>(1U, kBucketReadBytes / record_bytes);
  std::vector<std::uint8_t> records(
      std::min<std::uint64_t>(expected_rows, read_rows) * record_bytes);
  // Read complete record blocks. This removes one stream operation per vector
  // while retaining the original order within each CID and bounded buffers.
  for (std::uint64_t first = 0U; first < expected_rows;) {
    const std::size_t count = static_cast<std::size_t>(
        std::min<std::uint64_t>(expected_rows - first, read_rows));
    if (!ReadRecords(input, records.data(), count * record_bytes, statistics)) {
      return Status::DataLoss("basic bucket row is truncated");
    }
    for (std::size_t row = 0U; row < count; ++row) {
      const std::uint8_t* record = records.data() + row * record_bytes;
      const std::uint32_t cid = LoadLittleEndianU32(record + 4U);
      if (cid < first_cid || cid >= end_cid) {
        return Status::DataLoss(
            "basic bucket row belongs to a different CID range");
      }
      CidBuffers& target = buffers[cid - first_cid];
      const std::size_t queued_rows = target.ids.size() / sizeof(std::uint32_t);
      if (target.written_rows + queued_rows >= source.cid_counts[cid]) {
        return Status::DataLoss(
            "basic bucket CID payload exceeds metadata counts");
      }
      const std::uint8_t* cursor = record;
      AppendBytes(target.ids, cursor, sizeof(std::uint32_t));
      cursor += 2U * sizeof(std::uint32_t);
      AppendBytes(target.raw, cursor, raw_row_bytes);
      cursor += raw_row_bytes;
      AppendBytes(target.codes, cursor, m - 1U);
      cursor += m - 1U;
      AppendBytes(target.coefficients, cursor, m * sizeof(float));
      if (queued_rows + 1U == flush_rows) {
        if (const Status status = flush(cid); !status.ok()) return status;
      }
    }
    first += count;
  }
  for (std::uint32_t cid = first_cid; cid < end_cid; ++cid) {
    if (const Status status = flush(cid); !status.ok()) return status;
    if (buffers[cid - first_cid].written_rows != source.cid_counts[cid]) {
      return Status::DataLoss(
          "base list CID payload disagrees with metadata counts");
    }
  }
  return Status::Ok();
}

void EmitBucketsTask(void* opaque,
                     std::uint32_t worker,
                     std::uint32_t) noexcept {
  auto& context = *static_cast<EmitContext*>(opaque);
  // Keep the generic executor's static partitioning for numerical users.
  // Here each executor item is a worker loop; only IO buckets are claimed
  // dynamically, once per complete bucket, with no per-record scheduling.
  while (!context.failed.load(std::memory_order_relaxed)) {
    const auto bucket = context.next_bucket.fetch_add(1U, std::memory_order_relaxed);
    if (bucket >= context.buckets->shape.bucket_count()) break;
    // A failure can race with the claim. Do not start another bucket once it
    // is observed; already executing buckets finish before Run joins them.
    if (context.failed.load(std::memory_order_relaxed)) break;
    auto* statistics = context.statistics
        ? &context.statistics->buckets[bucket] : nullptr;
    if (statistics) {
      statistics->worker_index = worker;
      statistics->rows = context.buckets->bucket_rows(bucket);
      statistics->start_sec = Elapsed(context.workers_begin);
    }
    (*context.results)[bucket] = EmitBucket(context, bucket, statistics);
    if (!(*context.results)[bucket].ok()) {
      context.failed.store(true, std::memory_order_relaxed);
    }
    if (statistics) {
      statistics->end_sec = Elapsed(context.workers_begin);
      statistics->pack_other_sec = std::max(0.0,
          statistics->end_sec - statistics->start_sec - statistics->read_sec -
          statistics->write_sec);
    }
  }
}

}  // namespace

StatusOr<detail::BaseListMetadata> detail::BaseListMetadata::Open(
    const std::filesystem::path& directory, std::uint32_t expected_m) {
  auto metadata = ReadOnlyMappedFile::Open(directory / "meta.stlqpp");
  if (!metadata.ok()) {
    return metadata.status();
  }
  if (metadata.value().size() < kLegacyBaseListHeaderBytes) {
    return Status::DataLoss("base list metadata is truncated");
  }
  const std::uint8_t* bytes = metadata.value().data();
  const std::uint32_t version = LoadLittleEndianU32(bytes + 4U);
  if (LoadLittleEndianU32(bytes) != kBaseListMagic ||
      (version != kLegacyBaseListVersion &&
       version != kIdentityBaseListVersion &&
       version != kBaseListVersion) ||
      (expected_m != 0U && LoadLittleEndianU32(bytes + 8U) != expected_m)) {
    return Status::DataLoss("unsupported base list metadata");
  }
  BaseListMetadata store;
  store.directory = directory;
  store.m = LoadLittleEndianU32(bytes + 8U);
  const std::uint32_t m = store.m;
  const std::size_t header_bytes = version == kBaseListVersion
      ? kBaseListHeaderBytes
      : version == kIdentityBaseListVersion
          ? kIdentityBaseListHeaderBytes : kLegacyBaseListHeaderBytes;
  if (metadata.value().size() < header_bytes) {
    return Status::DataLoss("base list metadata is truncated");
  }
  store.identity = version >= kIdentityBaseListVersion
      ? ArtifactHash64::FromValue(LoadLittleEndianU64(bytes + 48U))
      : ArtifactHash64{};
  store.source_model_identity = version == kBaseListVersion
      ? ArtifactHash64::FromValue(LoadLittleEndianU64(bytes + 56U))
      : ArtifactHash64{};
  store.shape.dimension = LoadLittleEndianU32(bytes + 12U);
  store.shape.nlist = LoadLittleEndianU32(bytes + 16U);
  store.shape.cids_per_bucket = LoadLittleEndianU32(bytes + 20U);
  store.shape.vector_scalar = static_cast<BaseVectorScalar>(
      LoadLittleEndianU32(bytes + 24U));
  const std::uint32_t stored_buckets = LoadLittleEndianU32(bytes + 28U);
  const std::uint64_t total_vectors = LoadLittleEndianU64(bytes + 32U);
  const std::uint64_t offset_bytes = LoadLittleEndianU64(bytes + 40U);
  if (const Status status = store.shape.Validate(m); !status.ok()) {
    return Status::DataLoss(status.message());
  }
  const std::uint64_t expected_offset_bytes =
      (static_cast<std::uint64_t>(store.shape.nlist) + 1U) *
      sizeof(std::uint64_t);
  if (stored_buckets != store.shape.bucket_count() ||
      offset_bytes != expected_offset_bytes ||
      offset_bytes > std::numeric_limits<std::size_t>::max() -
                         header_bytes ||
      metadata.value().size() !=
          header_bytes + static_cast<std::size_t>(offset_bytes)) {
    return Status::DataLoss("base list metadata extents disagree");
  }
  store.offsets.resize(static_cast<std::size_t>(store.shape.nlist) + 1U);
  for (std::size_t index = 0U; index < store.offsets.size(); ++index) {
    store.offsets[index] = LoadLittleEndianU64(
        bytes + header_bytes + index * sizeof(std::uint64_t));
    if (index != 0U && store.offsets[index] < store.offsets[index - 1U]) {
      return Status::DataLoss("base list offsets are not monotonic");
    }
  }
  if (store.offsets.front() != 0U ||
      store.offsets.back() != total_vectors) {
    return Status::DataLoss("base list offsets disagree with total vectors");
  }
  std::error_code marker_error;
  store.has_raw_payload = !std::filesystem::exists(
      directory / kRawRemovedMarker, marker_error);
  if (marker_error) {
    return Status::Internal(
        "cannot inspect base-list raw-retention marker");
  }
  const std::uint64_t raw_row_bytes = store.shape.raw_bytes_per_vector();
  const struct FileExpectation {
    std::filesystem::path path;
    std::uint64_t row_bytes;
  } expectations[] = {
      {store.ids_path(), sizeof(std::uint32_t)},
      {store.codes_path(), m - 1U},
      {store.coefficients_path(), m * sizeof(float)},
  };
  for (const FileExpectation& expectation : expectations) {
    auto expected = ByteExtent(total_vectors, expectation.row_bytes,
                               "base list shard");
    if (!expected.ok()) {
      return expected.status();
    }
    std::error_code error;
    const std::uint64_t actual =
        std::filesystem::file_size(expectation.path, error);
    if (error || actual != expected.value()) {
      return Status::DataLoss(
          "base list shard length disagrees with metadata");
    }
  }
  if (store.has_raw_payload) {
    auto expected = ByteExtent(
        total_vectors, raw_row_bytes, "base list raw shard");
    if (!expected.ok()) return expected.status();
    std::error_code error;
    const std::uint64_t actual =
        std::filesystem::file_size(store.raw_path(), error);
    if (error || actual != expected.value()) {
      return Status::DataLoss(
          "base list raw shard length disagrees with metadata");
    }
  }
  return store;
}

std::filesystem::path detail::BaseListMetadata::ids_path() const {
  return directory / "ids.u32";
}

std::filesystem::path detail::BaseListMetadata::codes_path() const {
  return directory / "codes.u8";
}

std::filesystem::path detail::BaseListMetadata::coefficients_path() const {
  return directory / "coefficients.f32";
}

std::filesystem::path detail::BaseListMetadata::raw_path() const {
  return directory /
      (shape.vector_scalar == BaseVectorScalar::kU8 ? "raw.u8" : "raw.f32");
}

Status detail::BaseListMetadata::DiscardRawPayload(
    const std::filesystem::path& directory, std::uint32_t expected_m) {
  auto opened = Open(directory, expected_m);
  if (!opened.ok()) return opened.status();
  if (!opened.value().has_raw_payload) return Status::Ok();
  const std::filesystem::path marker = directory / kRawRemovedMarker;
  const std::filesystem::path temporary = directory / "raw.removed.tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
      return Status::Internal(
          "cannot create base-list raw-retention marker");
    }
    output << "STLQpp base-list raw payload intentionally removed\n";
    output.close();
    if (output.fail()) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
      return Status::Internal(
          "cannot write base-list raw-retention marker");
    }
  }
  std::error_code error;
  std::filesystem::rename(temporary, marker, error);
  if (error) {
    std::filesystem::remove(temporary, error);
    return Status::Internal(
        "cannot publish base-list raw-retention marker");
  }
  if (!std::filesystem::remove(opened.value().raw_path(), error) || error) {
    std::error_code ignored;
    std::filesystem::remove(marker, ignored);
    return Status::Internal("cannot remove base-list raw payload");
  }
  return Status::Ok();
}

template <int M>
StatusOr<BaseListStore<M>> BaseListStore<M>::Open(
    const std::filesystem::path& directory) {
  auto metadata = detail::BaseListMetadata::Open(directory, M);
  if (!metadata.ok()) return metadata.status();
  return BaseListStore<M>(std::move(metadata).value());
}

template <int M>
std::filesystem::path BaseListStore<M>::ids_path() const {
  return metadata_.ids_path();
}

template <int M>
std::filesystem::path BaseListStore<M>::codes_path() const {
  return metadata_.codes_path();
}

template <int M>
std::filesystem::path BaseListStore<M>::coefficients_path() const {
  return metadata_.coefficients_path();
}

template <int M>
std::filesystem::path BaseListStore<M>::raw_path() const {
  return metadata_.raw_path();
}

template <int M>
Status BaseListStore<M>::DiscardRawPayload(
    const std::filesystem::path& directory) {
  return detail::BaseListMetadata::DiscardRawPayload(directory, M);
}

template <int M>
Status ParallelBaseListEmitter<M>::Emit(
    const BasicBucketStore<M>& buckets,
    const std::filesystem::path& destination,
    const ParallelForExecutor& parallel,
    ArtifactHash64 identity,
    ArtifactHash64 source_model_identity) {
  return EmitImpl(
      buckets, destination, parallel, identity, source_model_identity, true, nullptr);
}

template <int M>
Status ParallelBaseListEmitter<M>::EmitTransient(
    const BasicBucketStore<M>& buckets,
    const std::filesystem::path& destination,
    const ParallelForExecutor& parallel,
    ArtifactHash64 identity,
    ArtifactHash64 source_model_identity,
    BaseListEmitStatistics* statistics) {
  return EmitImpl(
      buckets, destination, parallel, identity, source_model_identity, false, statistics);
}

namespace {

Status EmitBaseLists(
    const detail::BasicBucketStoreMetadata& buckets,
    const std::filesystem::path& destination,
    const ParallelForExecutor& parallel,
    ArtifactHash64 identity,
    ArtifactHash64 source_model_identity,
    bool durable,
    BaseListEmitStatistics* statistics) {
  const auto begin = statistics ? EmitClock::now() : EmitClock::time_point{};
  if (statistics) *statistics = {};
  const std::uint32_t m = buckets.m;
  const BaseBuildShape& shape = buckets.shape;
  std::vector<std::uint64_t> offsets(
      static_cast<std::size_t>(shape.nlist) + 1U, 0U);
  for (std::uint32_t cid = 0U; cid < shape.nlist; ++cid) {
    if (buckets.cid_counts[cid] >
        std::numeric_limits<std::uint64_t>::max() - offsets[cid]) {
      return Status::ResourceExhausted("base list prefix sum overflows u64");
    }
    offsets[cid + 1U] = offsets[cid] + buckets.cid_counts[cid];
  }
  if (offsets.back() != buckets.total_vectors) {
    return Status::DataLoss(
        "base list prefix sum disagrees with bucket total");
  }
  auto publication = durable
      ? StagingDirectory::Create(destination)
      : StagingDirectory::CreateTransient(destination);
  if (!publication.ok()) {
    return publication.status();
  }
  const std::filesystem::path staging = publication.value().path();
  auto ids_bytes = ByteExtent(offsets.back(), sizeof(std::uint32_t),
                              "base IDs");
  auto codes_bytes = ByteExtent(offsets.back(), m - 1U, "base codes");
  auto coefficient_bytes = ByteExtent(
      offsets.back(), m * sizeof(float), "base coefficients");
  auto raw_bytes = ByteExtent(offsets.back(), shape.raw_bytes_per_vector(),
                              "base raw vectors");
  if (!ids_bytes.ok()) return ids_bytes.status();
  if (!codes_bytes.ok()) return codes_bytes.status();
  if (!coefficient_bytes.ok()) return coefficient_bytes.status();
  if (!raw_bytes.ok()) return raw_bytes.status();

  auto ids_created = PositionedFile::Create(staging / "ids.u32",
                                             ids_bytes.value());
  if (!ids_created.ok()) return ids_created.status();
  auto codes_created = PositionedFile::Create(staging / "codes.u8",
                                               codes_bytes.value());
  if (!codes_created.ok()) return codes_created.status();
  auto coefficients_created = PositionedFile::Create(
      staging / "coefficients.f32", coefficient_bytes.value());
  if (!coefficients_created.ok()) return coefficients_created.status();
  auto raw_created = PositionedFile::Create(
      staging / (shape.vector_scalar == BaseVectorScalar::kU8
                     ? "raw.u8"
                     : "raw.f32"),
      raw_bytes.value());
  if (!raw_created.ok()) return raw_created.status();
  auto ids = std::make_unique<PositionedFile>(
      std::move(ids_created).value());
  auto codes = std::make_unique<PositionedFile>(
      std::move(codes_created).value());
  auto coefficients = std::make_unique<PositionedFile>(
      std::move(coefficients_created).value());
  auto raw = std::make_unique<PositionedFile>(
      std::move(raw_created).value());

  std::vector<Status> results(shape.bucket_count());
  if (statistics) statistics->buckets.resize(shape.bucket_count());
  EmitContext context{
      .buckets = &buckets,
      .offsets = &offsets,
      .files = {ids.get(), codes.get(), coefficients.get(), raw.get()},
      .results = &results,
      .statistics = statistics,
  };
  if (statistics) {
    context.workers_begin = EmitClock::now();
    statistics->setup_wall_sec = std::chrono::duration<double>(
        context.workers_begin - begin).count();
  }
  parallel.Run(std::min(shape.bucket_count(), parallel.worker_count()),
               &context, EmitBucketsTask);
  const auto publication_begin = statistics ? EmitClock::now() : EmitClock::time_point{};
  if (statistics) statistics->workers_wall_sec = std::chrono::duration<double>(
      publication_begin - context.workers_begin).count();
  for (const Status& status : results) {
    if (!status.ok()) {
      return status;
    }
  }
  // Durable publication performs one manifest-last fsync pass over every
  // completed shard. Flushing the open positioned files here would fsync the
  // same payload twice without strengthening the transaction.
  ids.reset();
  codes.reset();
  coefficients.reset();
  raw.reset();

  std::vector<std::uint8_t> metadata;
  if (const Status status = EncodeMetadata(
          shape, m, identity, source_model_identity, offsets, metadata);
      !status.ok()) {
    return status;
  }
  std::ofstream output(staging / "meta.stlqpp",
                       std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    return Status::Internal("failed to create base list metadata");
  }
  if (const Status status = WriteAll(output, metadata.data(), metadata.size(),
                                     "base list metadata");
      !status.ok()) {
    return status;
  }
  output.close();
  if (output.fail()) {
    return Status::Internal("failed to close base list metadata");
  }
  const auto status = publication.value().Publish();
  if (statistics) statistics->publication_wall_sec = Elapsed(publication_begin);
  return status;
}

}  // namespace

template <int M>
Status ParallelBaseListEmitter<M>::EmitImpl(
    const BasicBucketStore<M>& buckets,
    const std::filesystem::path& destination,
    const ParallelForExecutor& parallel,
    ArtifactHash64 identity,
    ArtifactHash64 source_model_identity,
    bool durable,
    BaseListEmitStatistics* statistics) {
  return EmitBaseLists(buckets.metadata_, destination, parallel,
                       identity, source_model_identity, durable, statistics);
}

template class BaseListStore<0>;
template class ParallelBaseListEmitter<0>;
template class BaseListStore<5>;
template class BaseListStore<12>;
template class BaseListStore<6>;
template class BaseListStore<10>;
template class ParallelBaseListEmitter<5>;
template class ParallelBaseListEmitter<12>;
template class ParallelBaseListEmitter<6>;
template class ParallelBaseListEmitter<10>;

}  // namespace stlqpp
