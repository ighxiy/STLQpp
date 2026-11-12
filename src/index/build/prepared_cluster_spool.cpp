#include "stlqpp/index/build/prepared_cluster_spool.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/model/topology/parent_louds.h"
#include "stlqpp/platform/sequential_file_writer.h"
#include "stlqpp/platform/staging_directory.h"
#include "stlqpp/storage/canonical_forest_format.h"
#include "stlqpp/storage/canonical_shard_codec.h"

namespace stlqpp {
namespace {

constexpr std::array<std::uint8_t, 8> kSpoolMagic = {'S', 'T', 'L', 'Q',
                                                     'S', 'P', 'R', '3'};
constexpr std::uint32_t kSpoolVersion = 3U;
constexpr std::uint32_t kDirectSpoolVersion = 4U;
// Same record layout; versions 5/6 omit the optional native-float norm stream.
constexpr std::uint32_t kNoFloatNormSpoolVersion = 5U;
constexpr std::uint32_t kDirectNoFloatNormSpoolVersion = 6U;
constexpr std::uint32_t kDirectSpoolRecordHeaderBytes = 80U;
constexpr std::uint32_t kDirectCheckpointBytes = 200U;
constexpr std::uint32_t kSpoolManifestBytes = 152U;
constexpr std::uint32_t kSpoolRecordHeaderBytes = 48U;
constexpr std::array<std::uint8_t, 8> kCheckpointMagic = {'S', 'T', 'L', 'Q',
                                                          'S', 'C', 'K', '3'};
constexpr std::uint32_t kCheckpointVersion = 3U;
constexpr std::uint32_t kCheckpointBytes = 168U;

bool RetainFloatNorms(const PreparedClusterSpoolRequest& request) {
  return request.retain_float_forest && request.retain_float_norm2;
}

std::uint32_t SpoolVersion(const PreparedClusterSpoolRequest& request) {
  if (request.retain_float_forest && !request.retain_float_norm2)
    return request.direct_payloads ? kDirectNoFloatNormSpoolVersion
                                   : kNoFloatNormSpoolVersion;
  return request.direct_payloads ? kDirectSpoolVersion : kSpoolVersion;
}

bool IsDirectVersion(std::uint32_t version) {
  return version == kDirectSpoolVersion ||
         version == kDirectNoFloatNormSpoolVersion;
}

bool IsLegacyNormVersion(std::uint32_t version) {
  return version == kSpoolVersion || version == kDirectSpoolVersion;
}

Status ValidateRequest(const PreparedClusterSpoolRequest& request,
                       std::uint32_t expected_m) {
  if (request.destination.empty()) {
    return Status::InvalidArgument(
        "prepared cluster spool destination must not be empty");
  }
  if (const Status status = request.shape.Validate(); !status.ok()) {
    return status;
  }
  if ((expected_m != 0 && request.shape.m != expected_m) ||
      request.hashes.linkage_list_directory.empty() ||
      (request.publish_quantized_forest &&
       (request.hashes.coefficient_codec.empty() ||
        request.hashes.norm8_cache.empty())) ||
      (!request.publish_quantized_forest &&
       (!request.hashes.coefficient_codec.empty() ||
        !request.hashes.norm8_cache.empty() || !request.retain_float_forest))) {
    return Status::InvalidArgument(
        "prepared cluster spool identity is invalid");
  }
  return Status::Ok();
}

Status AppendFloats(SequentialFileWriter& writer, ArrayView<const float> values,
                    bool nonnegative, const char* invalid_message) {
  if constexpr (std::endian::native == std::endian::little) {
    for (float value : values) {
      if (!std::isfinite(value) || (nonnegative && value < 0.0F)) {
        return Status::DataLoss(invalid_message);
      }
    }
    // Append consumes the span synchronously; its existing FP32 owner remains
    // live throughout the write, including partial-write retries.
    return writer.Append({reinterpret_cast<const std::uint8_t*>(values.data()),
                          values.size() * sizeof(float)});
  } else {
    std::vector<std::uint8_t> bytes(values.size() * sizeof(float));
    for (std::size_t index = 0U; index < values.size(); ++index) {
      const float value = values[index];
      if (!std::isfinite(value) || (nonnegative && value < 0.0F)) {
        return Status::DataLoss(invalid_message);
      }
      StoreLittleEndianF32(value, bytes.data() + index * sizeof(float));
    }
    return writer.Append(bytes);
  }
}

Status WriteManifest(const std::filesystem::path& path,
                     const PreparedClusterSpoolRequest& request,
                     const IndexCounts& counts, std::uint64_t records_bytes,
                     std::uint64_t norm2_bytes,
                     std::uint64_t float_coefficients_bytes,
                     std::uint64_t float_norm2_bytes,
                     std::uint64_t source_norm2_bytes) {
  std::array<std::uint8_t, kSpoolManifestBytes> bytes{};
  std::copy(kSpoolMagic.begin(), kSpoolMagic.end(), bytes.begin());
  StoreLittleEndianU32(
      SpoolVersion(request),
      bytes.data() + 8U);
  StoreLittleEndianU32(kSpoolManifestBytes, bytes.data() + 12U);
  StoreLittleEndianU32(request.shape.dimension, bytes.data() + 16U);
  StoreLittleEndianU32(request.shape.m, bytes.data() + 20U);
  StoreLittleEndianU32(request.shape.nlist, bytes.data() + 24U);
  StoreLittleEndianU64(counts.real, bytes.data() + 32U);
  StoreLittleEndianU64(counts.virtual_roots, bytes.data() + 40U);
  StoreLittleEndianU64(counts.real_roots, bytes.data() + 48U);
  StoreLittleEndianU64(counts.linked_real, bytes.data() + 56U);
  StoreLittleEndianU64(request.hashes.linkage_list_directory.value(),
                       bytes.data() + 64U);
  StoreLittleEndianU64(request.hashes.coefficient_codec.value(),
                       bytes.data() + 72U);
  StoreLittleEndianU64(request.hashes.norm8_cache.value(), bytes.data() + 80U);
  StoreLittleEndianU64(records_bytes, bytes.data() + 88U);
  StoreLittleEndianU64(norm2_bytes, bytes.data() + 96U);
  StoreLittleEndianU64(request.shape.nlist, bytes.data() + 104U);
  StoreLittleEndianU32(request.retain_float_forest ? 1U : 0U,
                       bytes.data() + 112U);
  StoreLittleEndianU32(request.publish_quantized_forest ? 0U : 1U,
                       bytes.data() + 116U);
  StoreLittleEndianU64(float_coefficients_bytes, bytes.data() + 120U);
  StoreLittleEndianU64(float_norm2_bytes, bytes.data() + 128U);
  StoreLittleEndianU32(request.retain_source_norm2 ? 1U : 0U,
                       bytes.data() + 136U);
  StoreLittleEndianU64(source_norm2_bytes, bytes.data() + 144U);
  auto writer_result = SequentialFileWriter::Create(path);
  if (!writer_result.ok()) {
    return writer_result.status();
  }
  auto writer = std::move(writer_result).value();
  if (const Status status = writer.Append(bytes); !status.ok()) {
    return status;
  }
  return writer.Close();
}

StatusOr<PreparedClusterSpoolRequest> DecodeManifest(
    const std::filesystem::path& directory, const ReadOnlyMappedFile& manifest,
    std::uint32_t expected_m, IndexCounts& counts, std::uint64_t& records_bytes,
    std::uint64_t& norm2_bytes, std::uint64_t& float_coefficients_bytes,
    std::uint64_t& float_norm2_bytes, std::uint64_t& source_norm2_bytes) {
  if (manifest.size() != kSpoolManifestBytes ||
      !std::equal(kSpoolMagic.begin(), kSpoolMagic.end(), manifest.data()) ||
      (LoadLittleEndianU32(manifest.data() + 8U) != kSpoolVersion &&
       LoadLittleEndianU32(manifest.data() + 8U) != kDirectSpoolVersion &&
       LoadLittleEndianU32(manifest.data() + 8U) != kNoFloatNormSpoolVersion &&
       LoadLittleEndianU32(manifest.data() + 8U) != kDirectNoFloatNormSpoolVersion) ||
      LoadLittleEndianU32(manifest.data() + 12U) != kSpoolManifestBytes) {
    return Status::DataLoss(
        "prepared cluster spool manifest header is invalid");
  }
  PreparedClusterSpoolRequest request;
  request.destination = directory;
  const auto version = LoadLittleEndianU32(manifest.data() + 8U);
  request.direct_payloads = IsDirectVersion(version);
  request.retain_float_norm2 = IsLegacyNormVersion(version);
  request.shape = {
      .dimension = LoadLittleEndianU32(manifest.data() + 16U),
      .m = LoadLittleEndianU32(manifest.data() + 20U),
      .nlist = LoadLittleEndianU32(manifest.data() + 24U),
  };
  counts = {
      .real = LoadLittleEndianU64(manifest.data() + 32U),
      .virtual_roots = LoadLittleEndianU64(manifest.data() + 40U),
      .real_roots = LoadLittleEndianU64(manifest.data() + 48U),
      .linked_real = LoadLittleEndianU64(manifest.data() + 56U),
  };
  request.hashes = {
      .linkage_list_directory =
          ArtifactHash64::FromValue(LoadLittleEndianU64(manifest.data() + 64U)),
      .coefficient_codec =
          ArtifactHash64::FromValue(LoadLittleEndianU64(manifest.data() + 72U)),
      .norm8_cache =
          ArtifactHash64::FromValue(LoadLittleEndianU64(manifest.data() + 80U)),
  };
  records_bytes = LoadLittleEndianU64(manifest.data() + 88U);
  norm2_bytes = LoadLittleEndianU64(manifest.data() + 96U);
  if (LoadLittleEndianU64(manifest.data() + 104U) != request.shape.nlist) {
    return Status::DataLoss("prepared cluster spool CID count is invalid");
  }
  const std::uint32_t retain_float =
      LoadLittleEndianU32(manifest.data() + 112U);
  if (retain_float > 1U) {
    return Status::DataLoss(
        "prepared cluster spool float-retention flag is invalid");
  }
  request.retain_float_forest = retain_float != 0U;
  const std::uint32_t codec_disabled =
      LoadLittleEndianU32(manifest.data() + 116U);
  if (codec_disabled > 1U) {
    return Status::DataLoss(
        "prepared cluster spool codec capability flag is invalid");
  }
  request.publish_quantized_forest = codec_disabled == 0U;
  float_coefficients_bytes = LoadLittleEndianU64(manifest.data() + 120U);
  float_norm2_bytes = LoadLittleEndianU64(manifest.data() + 128U);
  const std::uint32_t retain_source_norm2 =
      LoadLittleEndianU32(manifest.data() + 136U);
  if (retain_source_norm2 > 1U) {
    return Status::DataLoss(
        "prepared cluster spool source-norm flag is invalid");
  }
  request.retain_source_norm2 = retain_source_norm2 != 0U;
  source_norm2_bytes = LoadLittleEndianU64(manifest.data() + 144U);
  if (const Status status = ValidateRequest(request, expected_m);
      !status.ok()) {
    return status;
  }
  if (counts.real == 0U ||
      counts.real_roots + counts.linked_real != counts.real) {
    return Status::DataLoss("prepared cluster spool count totals are invalid");
  }
  return request;
}

Status ValidateDirectExtent(const ReadOnlyMappedFile& file,
                            std::uint64_t offset, std::uint64_t bytes,
                            std::uint64_t& cursor) {
  const auto aligned = AlignCanonicalRecord(cursor);
  if (offset != aligned || offset > file.size() ||
      bytes > file.size() - offset) {
    return Status::DataLoss("direct spool record does not partition its shard");
  }
  for (auto at = cursor; at < offset; ++at) {
    if (file.data()[at] != 0U)
      return Status::DataLoss("direct spool alignment padding is nonzero");
  }
  cursor = offset + bytes;
  return Status::Ok();
}

Status ValidateCoefficientExtent(ArrayView<const std::uint8_t> bytes,
                                 std::uint32_t m) {
  if (bytes.size() < 8U + 8U * m) {
    return Status::DataLoss("prepared spool coefficient record is truncated");
  }
  const std::uint64_t root = LoadLittleEndianU32(bytes.data());
  const std::uint64_t linkage = LoadLittleEndianU32(bytes.data() + 4U);
  if (CanonicalCoefficientAlphabetFromRecord(m, bytes.size(), root, linkage) ==
      0U) {
    return Status::DataLoss(
        "prepared spool coefficient payload extents are invalid");
  }
  return Status::Ok();
}

std::filesystem::path CheckpointPath(const std::filesystem::path& directory,
                                     std::uint64_t generation) {
  return directory / (generation % 2U == 0U ? "checkpoint.0.stlqpp"
                                            : "checkpoint.1.stlqpp");
}

std::uint64_t Checksum(ArrayView<const std::uint8_t> bytes,
                       std::uint64_t value = 1469598103934665603ULL) noexcept {
  for (const std::uint8_t byte : bytes) {
    value ^= byte;
    value *= 1099511628211ULL;
  }
  return value;
}

struct SpoolCheckpoint final {
  std::uint64_t generation = 0U;
  std::uint32_t next_cid = 0U;
  std::uint64_t records_bytes = 0U;
  std::uint64_t norm2_bytes = 0U;
  std::uint64_t float_coefficients_bytes = 0U;
  std::uint64_t float_norm2_bytes = 0U;
  std::uint64_t source_norm2_bytes = 0U;
  IndexCounts seen;
  bool direct_payloads = false;
  std::uint64_t codes_bytes = 0, parents_bytes = 0, coefficients_bytes = 0;
  std::uint64_t record_log_checksum = 1469598103934665603ULL;
};

std::array<std::uint8_t, kDirectCheckpointBytes> EncodeCheckpoint(
    const PreparedClusterSpoolRequest& request, std::uint32_t m,
    const SpoolCheckpoint& checkpoint) {
  const auto extent =
      request.direct_payloads ? kDirectCheckpointBytes : kCheckpointBytes;
  std::array<std::uint8_t, kDirectCheckpointBytes> bytes{};
  std::copy(kCheckpointMagic.begin(), kCheckpointMagic.end(), bytes.begin());
  StoreLittleEndianU32(
      SpoolVersion(request),
      bytes.data() + 8U);
  StoreLittleEndianU32(m, bytes.data() + 12U);
  StoreLittleEndianU32(request.shape.dimension, bytes.data() + 16U);
  StoreLittleEndianU32(request.shape.nlist, bytes.data() + 20U);
  StoreLittleEndianU64(request.hashes.linkage_list_directory.value(),
                       bytes.data() + 24U);
  StoreLittleEndianU64(request.hashes.coefficient_codec.value(),
                       bytes.data() + 32U);
  StoreLittleEndianU64(request.hashes.norm8_cache.value(), bytes.data() + 40U);
  StoreLittleEndianU64(checkpoint.generation, bytes.data() + 48U);
  StoreLittleEndianU32(checkpoint.next_cid, bytes.data() + 56U);
  StoreLittleEndianU64(checkpoint.records_bytes, bytes.data() + 64U);
  StoreLittleEndianU64(checkpoint.norm2_bytes, bytes.data() + 72U);
  StoreLittleEndianU64(checkpoint.seen.real, bytes.data() + 80U);
  StoreLittleEndianU64(checkpoint.seen.virtual_roots, bytes.data() + 88U);
  StoreLittleEndianU64(checkpoint.seen.real_roots, bytes.data() + 96U);
  StoreLittleEndianU64(checkpoint.seen.linked_real, bytes.data() + 104U);
  StoreLittleEndianU64(checkpoint.float_coefficients_bytes,
                       bytes.data() + 112U);
  StoreLittleEndianU64(checkpoint.float_norm2_bytes, bytes.data() + 120U);
  StoreLittleEndianU32(request.retain_float_forest ? 1U : 0U,
                       bytes.data() + 128U);
  StoreLittleEndianU64(checkpoint.source_norm2_bytes, bytes.data() + 136U);
  StoreLittleEndianU32(request.retain_source_norm2 ? 1U : 0U,
                       bytes.data() + 144U);
  StoreLittleEndianU32(request.publish_quantized_forest ? 0U : 1U,
                       bytes.data() + 148U);
  if (request.direct_payloads) {
    StoreLittleEndianU64(checkpoint.codes_bytes, bytes.data() + 160U);
    StoreLittleEndianU64(checkpoint.parents_bytes, bytes.data() + 168U);
    StoreLittleEndianU64(checkpoint.coefficients_bytes, bytes.data() + 176U);
    StoreLittleEndianU64(checkpoint.record_log_checksum, bytes.data() + 184U);
  }
  StoreLittleEndianU64(Checksum({bytes.data(), extent - 8U}),
                       bytes.data() + extent - 8U);
  return bytes;
}

StatusOr<SpoolCheckpoint> ReadCheckpoint(
    const std::filesystem::path& path,
    const PreparedClusterSpoolRequest& request, std::uint32_t expected_m) {
  std::error_code error;
  const auto extent = std::filesystem::file_size(path, error);
  if (error)
    return Status::Internal("cannot read spool checkpoint size: " +
                            error.message());
  if (extent != kCheckpointBytes && extent != kDirectCheckpointBytes)
    return Status::DataLoss("invalid spool checkpoint size");
  std::vector<std::uint8_t> bytes(extent);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(bytes.data()), bytes.size()) ||
      input.peek() != std::ifstream::traits_type::eof() ||
      !std::equal(kCheckpointMagic.begin(), kCheckpointMagic.end(),
                  bytes.begin()) ||
      (LoadLittleEndianU32(bytes.data() + 8U) !=
          (extent == kDirectCheckpointBytes ? kDirectSpoolVersion
                                            : kCheckpointVersion) &&
       LoadLittleEndianU32(bytes.data() + 8U) !=
          (extent == kDirectCheckpointBytes ? kDirectNoFloatNormSpoolVersion
                                            : kNoFloatNormSpoolVersion)) ||
      LoadLittleEndianU32(bytes.data() + 12U) != expected_m ||
      LoadLittleEndianU32(bytes.data() + 16U) != request.shape.dimension ||
      LoadLittleEndianU32(bytes.data() + 20U) != request.shape.nlist ||
      LoadLittleEndianU64(bytes.data() + 24U) !=
          request.hashes.linkage_list_directory.value() ||
      LoadLittleEndianU64(bytes.data() + 32U) !=
          request.hashes.coefficient_codec.value() ||
      LoadLittleEndianU64(bytes.data() + 40U) !=
          request.hashes.norm8_cache.value() ||
      LoadLittleEndianU32(bytes.data() + 128U) !=
          (request.retain_float_forest ? 1U : 0U) ||
      LoadLittleEndianU32(bytes.data() + 144U) !=
          (request.retain_source_norm2 ? 1U : 0U) ||
      LoadLittleEndianU32(bytes.data() + 148U) !=
          (request.publish_quantized_forest ? 0U : 1U) ||
      LoadLittleEndianU64(bytes.data() + bytes.size() - 8U) !=
          Checksum({bytes.data(), bytes.size() - 8U})) {
    return Status::DataLoss(
        "prepared spool checkpoint identity, shape, or checksum changed");
  }
  SpoolCheckpoint checkpoint;
  const bool stored_float_norms = request.retain_float_forest &&
      IsLegacyNormVersion(LoadLittleEndianU32(bytes.data() + 8U));
  if (RetainFloatNorms(request) && !stored_float_norms) {
    return Status::FailedPrecondition(
        "checkpoint has no float norms required by this publication");
  }
  checkpoint.direct_payloads = extent == kDirectCheckpointBytes;
  if (checkpoint.direct_payloads) {
    checkpoint.codes_bytes = LoadLittleEndianU64(bytes.data() + 160U);
    checkpoint.parents_bytes = LoadLittleEndianU64(bytes.data() + 168U);
    checkpoint.coefficients_bytes = LoadLittleEndianU64(bytes.data() + 176U);
    checkpoint.record_log_checksum = LoadLittleEndianU64(bytes.data() + 184U);
  }
  checkpoint.generation = LoadLittleEndianU64(bytes.data() + 48U);
  checkpoint.next_cid = LoadLittleEndianU32(bytes.data() + 56U);
  checkpoint.records_bytes = LoadLittleEndianU64(bytes.data() + 64U);
  checkpoint.norm2_bytes = LoadLittleEndianU64(bytes.data() + 72U);
  checkpoint.seen = {
      .real = LoadLittleEndianU64(bytes.data() + 80U),
      .virtual_roots = LoadLittleEndianU64(bytes.data() + 88U),
      .real_roots = LoadLittleEndianU64(bytes.data() + 96U),
      .linked_real = LoadLittleEndianU64(bytes.data() + 104U),
  };
  checkpoint.float_coefficients_bytes =
      LoadLittleEndianU64(bytes.data() + 112U);
  checkpoint.float_norm2_bytes = LoadLittleEndianU64(bytes.data() + 120U);
  checkpoint.source_norm2_bytes = LoadLittleEndianU64(bytes.data() + 136U);
  const std::uint64_t nodes =
      checkpoint.seen.real + checkpoint.seen.virtual_roots;
  if (checkpoint.next_cid > request.shape.nlist ||
      checkpoint.seen.real_roots + checkpoint.seen.linked_real !=
          checkpoint.seen.real ||
      checkpoint.norm2_bytes != (request.publish_quantized_forest
                                     ? checkpoint.seen.real * sizeof(float)
                                     : 0U) ||
      (request.retain_float_forest &&
       ((!checkpoint.direct_payloads &&
         checkpoint.float_coefficients_bytes !=
             nodes * expected_m * sizeof(float)) ||
        checkpoint.float_norm2_bytes !=
            (stored_float_norms ? checkpoint.seen.real * sizeof(float) : 0U))) ||
      (!request.retain_float_forest &&
       (checkpoint.float_coefficients_bytes != 0U ||
        checkpoint.float_norm2_bytes != 0U)) ||
      (request.retain_source_norm2 ? checkpoint.source_norm2_bytes !=
                                         checkpoint.seen.real * sizeof(float)
                                   : checkpoint.source_norm2_bytes != 0U)) {
    return Status::DataLoss("prepared spool checkpoint extents are invalid");
  }
  return checkpoint;
}

Status WriteCheckpointSlot(const std::filesystem::path& directory,
                           const PreparedClusterSpoolRequest& request,
                           std::uint32_t m, const SpoolCheckpoint& checkpoint) {
  const auto bytes = EncodeCheckpoint(request, m, checkpoint);
  auto writer = SequentialFileWriter::Create(
      CheckpointPath(directory, checkpoint.generation));
  if (!writer.ok()) return writer.status();
  const auto extent =
      request.direct_payloads ? kDirectCheckpointBytes : kCheckpointBytes;
  if (const Status status = writer.value().Append({bytes.data(), extent});
      !status.ok()) {
    return status;
  }
  // Like DEV's per-CID marker, this is a process-restart checkpoint, not
  // a power-loss durability barrier. Append writes directly to the OS.
  return writer.value().Close();
}

// Own only the brand-new directory until its first valid slot is committed.
// Known-file unlinking needs no directory descriptors, so EMFILE cleanup can
// succeed even when a recursive remove_all cannot open another directory.
class InitialCheckpointGuard final {
 public:
  explicit InitialCheckpointGuard(const std::filesystem::path& path) : path_(path) {}
  ~InitialCheckpointGuard() {
    if (committed_) return;
    std::error_code error;
    for (const char* name : {"records.bin", "norm2.f32", "float_coefficients.f32",
                            "float_norm2.f32", "source_norm2.f32", "codes.bin",
                            "parents.bin", "coefficients.bin", "checkpoint.0.stlqpp",
                            "checkpoint.1.stlqpp"})
      std::filesystem::remove(path_ / name, error);
    std::filesystem::remove(path_, error);
  }
  void Commit() noexcept { committed_ = true; }
 private:
  std::filesystem::path path_;
  bool committed_ = false;
};

}  // namespace

struct PreparedClusterSpoolWriterStorage::State final {
  StagingDirectory publication;
  std::filesystem::path checkpoint_directory;
  PreparedClusterSpoolRequest request;
  SequentialFileWriter records;
  SequentialFileWriter norm2;
  SequentialFileWriter float_coefficients;
  SequentialFileWriter float_norm2;
  SequentialFileWriter source_norm2;
  SequentialFileWriter codes, parents, coefficients;
  std::vector<std::uint8_t> completed;
  std::uint32_t next_cid = 0U;
  std::uint64_t checkpoint_generation = 0U;
  std::uint64_t record_log_checksum = 1469598103934665603ULL;
  IndexCounts seen;
  bool checkpointed = false;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return checkpointed ? checkpoint_directory : publication.path();
  }

  Status CreatePayloadStreams() {
    completed.assign(request.shape.nlist, 0U);
    if (!request.direct_payloads) return Status::Ok();
    for (auto item : {std::pair{&codes, ShardKind::kCodes},
                      std::pair{&parents, ShardKind::kParents},
                      std::pair{&coefficients, ShardKind::kCoefficients}}) {
      if (item.second == ShardKind::kCoefficients &&
          !request.publish_quantized_forest)
        continue;
      const char* name = item.second == ShardKind::kCodes ? "codes.bin"
                         : item.second == ShardKind::kParents
                             ? "parents.bin"
                             : "coefficients.bin";
      auto file = SequentialFileWriter::Create(path() / name);
      if (!file.ok()) return file.status();
      *item.first = std::move(file).value();
      std::vector<std::uint8_t> header;
      if (auto status = CanonicalPayloadShardCodec::EncodeHeader(
              item.second, request.shape, header);
          !status.ok())
        return status;
      if (auto status = item.first->Append(header); !status.ok()) return status;
    }
    if (request.retain_float_forest) {
      std::vector<std::uint8_t> header;
      if (auto status = CanonicalPayloadShardCodec::EncodeHeader(
              ShardKind::kFloatCoefficients, request.shape, header);
          !status.ok())
        return status;
      if (auto status = float_coefficients.Append(header); !status.ok())
        return status;
    }
    return Status::Ok();
  }
  Status SaveCheckpoint() {
    if (!checkpointed) return Status::Ok();
    // Payload Append is unbuffered. Ordered writes plus the alternating,
    // checksummed slots suffice for process restart. Do not fsync every
    // shard for every CID; this does not change other publication writers.
    const SpoolCheckpoint checkpoint = {
        .generation = checkpoint_generation,
        .next_cid = next_cid,
        .records_bytes = records.size(),
        .norm2_bytes = norm2.size(),
        .float_coefficients_bytes =
            request.retain_float_forest ? float_coefficients.size() : 0U,
        .float_norm2_bytes =
            RetainFloatNorms(request) ? float_norm2.size() : 0U,
        .source_norm2_bytes =
            request.retain_source_norm2 ? source_norm2.size() : 0U,
        .seen = seen,
        .direct_payloads = request.direct_payloads,
        .codes_bytes = request.direct_payloads ? codes.size() : 0U,
        .parents_bytes = request.direct_payloads ? parents.size() : 0U,
        .coefficients_bytes =
            request.direct_payloads && request.publish_quantized_forest
                ? coefficients.size()
                : 0U,
        .record_log_checksum = record_log_checksum,
    };
    if (const Status status = WriteCheckpointSlot(checkpoint_directory, request,
                                                  request.shape.m, checkpoint);
        !status.ok()) {
      return status;
    }
    ++checkpoint_generation;
    return Status::Ok();
  }
};

PreparedClusterSpoolWriterStorage::PreparedClusterSpoolWriterStorage(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

PreparedClusterSpoolWriterStorage::~PreparedClusterSpoolWriterStorage() =
    default;

PreparedClusterSpoolWriterStorage::PreparedClusterSpoolWriterStorage(
    PreparedClusterSpoolWriterStorage&&) noexcept = default;

PreparedClusterSpoolWriterStorage& PreparedClusterSpoolWriterStorage::operator=(
    PreparedClusterSpoolWriterStorage&&) noexcept = default;

StatusOr<PreparedClusterSpoolWriterStorage>
PreparedClusterSpoolWriterStorage::Create(
    const PreparedClusterSpoolRequest& request, int expected_m) {
  if (const Status status = ValidateRequest(request, expected_m);
      !status.ok()) {
    return status;
  }
  // Completion-order payloads are adopted by the canonical writer. This
  // non-checkpoint spool is an intermediate workset; its atomic rename need
  // not force the same files to storage before final durable publication.
  // Checkpoint creation/reopen uses its separate restartable directory path.
  auto publication_result = request.direct_payloads
      ? StagingDirectory::CreateTransient(request.destination)
      : StagingDirectory::Create(request.destination);
  if (!publication_result.ok()) {
    return publication_result.status();
  }
  auto state = std::make_unique<State>();
  state->publication = std::move(publication_result).value();
  state->request = request;
  auto records_result =
      SequentialFileWriter::Create(state->publication.path() / "records.bin");
  if (!records_result.ok()) {
    return records_result.status();
  }
  state->records = std::move(records_result).value();
  auto norm_result =
      SequentialFileWriter::Create(state->publication.path() / "norm2.f32");
  if (!norm_result.ok()) {
    return norm_result.status();
  }
  state->norm2 = std::move(norm_result).value();
  if (request.retain_float_forest) {
    auto coefficients = SequentialFileWriter::Create(state->publication.path() /
                                                     "float_coefficients.f32");
    if (!coefficients.ok()) return coefficients.status();
    state->float_coefficients = std::move(coefficients).value();
  }
  if (RetainFloatNorms(request)) {
    auto float_norm2 = SequentialFileWriter::Create(state->publication.path() /
                                                    "float_norm2.f32");
    if (!float_norm2.ok()) return float_norm2.status();
    state->float_norm2 = std::move(float_norm2).value();
  }
  if (request.retain_source_norm2) {
    auto source_norm2 = SequentialFileWriter::Create(state->publication.path() /
                                                     "source_norm2.f32");
    if (!source_norm2.ok()) return source_norm2.status();
    state->source_norm2 = std::move(source_norm2).value();
  }
  if (auto status = state->CreatePayloadStreams(); !status.ok()) return status;
  return PreparedClusterSpoolWriterStorage(std::move(state));
}

StatusOr<PreparedClusterSpoolWriterStorage>
PreparedClusterSpoolWriterStorage::CreateCheckpointed(
    const PreparedClusterSpoolRequest& request, int expected_m) {
  if (const Status status = ValidateRequest(request, expected_m);
      !status.ok()) {
    return status;
  }
  std::error_code error;
  if (std::filesystem::exists(request.destination, error) || error) {
    return Status::FailedPrecondition(
        "prepared spool checkpoint destination already exists");
  }
  if (!request.destination.parent_path().empty())
    std::filesystem::create_directories(request.destination.parent_path(), error);
  if (error || !std::filesystem::create_directory(request.destination, error))
    return Status::FailedPrecondition("cannot exclusively create checkpoint directory");
  InitialCheckpointGuard creation(request.destination);
  auto state = std::make_unique<State>();
  state->checkpoint_directory = request.destination;
  state->request = request;
  state->checkpointed = true;
  auto records =
      SequentialFileWriter::Create(request.destination / "records.bin");
  auto norm2 = SequentialFileWriter::Create(request.destination / "norm2.f32");
  if (!records.ok() || !norm2.ok()) {
    return records.ok() ? norm2.status() : records.status();
  }
  state->records = std::move(records).value();
  state->norm2 = std::move(norm2).value();
  if (request.retain_float_forest) {
    auto coefficients = SequentialFileWriter::Create(request.destination /
                                                     "float_coefficients.f32");
    if (!coefficients.ok()) return coefficients.status();
    state->float_coefficients = std::move(coefficients).value();
  }
  if (RetainFloatNorms(request)) {
    auto float_norm2 =
        SequentialFileWriter::Create(request.destination / "float_norm2.f32");
    if (!float_norm2.ok()) return float_norm2.status();
    state->float_norm2 = std::move(float_norm2).value();
  }
  if (request.retain_source_norm2) {
    auto source_norm2 =
        SequentialFileWriter::Create(request.destination / "source_norm2.f32");
    if (!source_norm2.ok()) return source_norm2.status();
    state->source_norm2 = std::move(source_norm2).value();
  }
  if (auto status = state->CreatePayloadStreams(); !status.ok()) return status;
  if (const Status status = state->SaveCheckpoint(); !status.ok()) {
    return status;
  }
  creation.Commit();
  return PreparedClusterSpoolWriterStorage(std::move(state));
}

StatusOr<PreparedClusterSpoolWriterStorage>
PreparedClusterSpoolWriterStorage::OpenCheckpointed(
    const PreparedClusterSpoolRequest& request, int expected_m) {
  if (const Status status = ValidateRequest(request, expected_m);
      !status.ok()) {
    return status;
  }
  std::optional<SpoolCheckpoint> selected;
  for (std::uint64_t slot = 0U; slot < 2U; ++slot) {
    const std::filesystem::path path =
        CheckpointPath(request.destination, slot);
    std::error_code error;
    if (!std::filesystem::exists(path, error)) {
      if (error) {
        return Status::Internal(
            "failed to inspect prepared spool checkpoint slot");
      }
      continue;
    }
    auto checkpoint = ReadCheckpoint(path, request, request.shape.m);
    if (!checkpoint.ok()) continue;
    if (!selected.has_value() ||
        checkpoint.value().generation > selected->generation) {
      selected = std::move(checkpoint).value();
    }
  }
  if (!selected.has_value()) {
    return Status::DataLoss(
        "prepared spool has no valid committed checkpoint slot");
  }
  auto records = SequentialFileWriter::OpenAppend(
      request.destination / "records.bin", selected->records_bytes);
  if (!records.ok()) return records.status();
  auto norm2 = SequentialFileWriter::OpenAppend(
      request.destination / "norm2.f32", selected->norm2_bytes);
  if (!norm2.ok()) return norm2.status();
  auto state = std::make_unique<State>();
  state->checkpoint_directory = request.destination;
  state->request = request;
  state->records = std::move(records).value();
  state->norm2 = std::move(norm2).value();
  if (request.retain_float_forest) {
    auto coefficients = SequentialFileWriter::OpenAppend(
        request.destination / "float_coefficients.f32",
        selected->float_coefficients_bytes);
    if (!coefficients.ok()) return coefficients.status();
    state->float_coefficients = std::move(coefficients).value();
  }
  // Legacy checkpoints may contain this stream. When the new publication
  // does not consume it, leave it untouched and stop producing more norms.
  if (RetainFloatNorms(request)) {
    auto float_norm2 = SequentialFileWriter::OpenAppend(
        request.destination / "float_norm2.f32", selected->float_norm2_bytes);
    if (!float_norm2.ok()) return float_norm2.status();
    state->float_norm2 = std::move(float_norm2).value();
  }
  if (request.retain_source_norm2) {
    auto source_norm2 = SequentialFileWriter::OpenAppend(
        request.destination / "source_norm2.f32", selected->source_norm2_bytes);
    if (!source_norm2.ok()) return source_norm2.status();
    state->source_norm2 = std::move(source_norm2).value();
  }
  state->request.direct_payloads = selected->direct_payloads;
  state->completed.assign(request.shape.nlist, 0U);
  if (selected->direct_payloads) {
    for (auto item :
         {std::pair{&state->codes,
                    std::pair{"codes.bin", selected->codes_bytes}},
          std::pair{&state->parents,
                    std::pair{"parents.bin", selected->parents_bytes}},
          std::pair{
              &state->coefficients,
              std::pair{"coefficients.bin", selected->coefficients_bytes}}}) {
      if (item.second.second == 0U) continue;
      auto file = SequentialFileWriter::OpenAppend(
          request.destination / item.second.first, item.second.second);
      if (!file.ok()) return file.status();
      *item.first = std::move(file).value();
    }
    auto record_log =
        ReadOnlyMappedFile::Open(request.destination / "records.bin");
    if (!record_log.ok()) return record_log.status();
    if (Checksum({record_log.value().data(), record_log.value().size()}) !=
        selected->record_log_checksum)
      return Status::DataLoss(
          "completion log checksum disagrees with checkpoint");
    state->record_log_checksum = selected->record_log_checksum;
    if (record_log.value().size() % kDirectSpoolRecordHeaderBytes != 0U)
      return Status::DataLoss("invalid completion log extent");
    for (std::size_t offset = 0; offset < record_log.value().size();
         offset += kDirectSpoolRecordHeaderBytes) {
      auto cid = LoadLittleEndianU32(record_log.value().data() + offset);
      if (cid >= request.shape.nlist || state->completed[cid]++)
        return Status::DataLoss("duplicate completion log CID");
    }
    std::uint32_t cursor = 0;
    while (cursor < request.shape.nlist && state->completed[cursor]) ++cursor;
    if (cursor != selected->next_cid)
      return Status::DataLoss(
          "completion log cursor disagrees with checkpoint");
  } else {
    std::fill_n(state->completed.begin(), selected->next_cid, 1U);
  }
  state->next_cid = selected->next_cid;
  state->checkpoint_generation = selected->generation + 1U;
  state->seen = selected->seen;
  state->checkpointed = true;
  return PreparedClusterSpoolWriterStorage(std::move(state));
}

Status PreparedClusterSpoolWriterStorage::CheckAppend(std::uint32_t m,
                                                      std::uint32_t cid,
                                                      bool quantized,
                                                      bool complete) const {
  if (state_ == nullptr) {
    return Status::FailedPrecondition(
        "prepared cluster spool is not accepting CIDs");
  }
  if (m != state_->request.shape.m) {
    return Status::InvalidArgument("spool cluster M differs from model");
  }
  if (cid >= state_->request.shape.nlist || state_->completed[cid] ||
      (!state_->request.direct_payloads && cid != state_->next_cid)) {
    return Status::InvalidArgument(
        "prepared spool CID is duplicated or violates its ordering policy");
  }
  if (quantized != state_->request.publish_quantized_forest) {
    return Status::DataLoss(
        "prepared cluster representation disagrees with its spool");
  }
  if (state_->request.publish_quantized_forest && !complete) {
    return Status::FailedPrecondition(
        "prepared cluster coefficient/norm stages are incomplete");
  }
  return Status::Ok();
}

Status PreparedClusterSpoolWriterStorage::AppendSerialized(
    const Payload& cluster) {
  const auto m = state_->request.shape.m;
  const auto& records = cluster.records;
  if ((state_->request.publish_quantized_forest &&
       cluster.exact_real_norm2.size() != cluster.real_count) ||
      (!state_->request.publish_quantized_forest &&
       !cluster.exact_real_norm2.empty())) {
    return Status::DataLoss("prepared cluster exact norm count is invalid");
  }
  const std::uint64_t node_count =
      static_cast<std::uint64_t>(cluster.real_count) + cluster.virtual_count;
  if (state_->request.retain_float_forest &&
      (cluster.float_coefficients_layer_major.size() != node_count * m ||
       (RetainFloatNorms(state_->request) &&
        cluster.float_real_norm2.size() != cluster.real_count))) {
    return Status::DataLoss(
        "prepared cluster is missing its native-float companion");
  }
  if (state_->request.retain_source_norm2 &&
      cluster.source_real_norm2.size() != cluster.real_count) {
    return Status::DataLoss(
        "prepared cluster is missing retained original-vector norms");
  }
  const std::uint64_t linked = cluster.real_count - cluster.real_root_count;
  std::array<std::uint8_t, kDirectSpoolRecordHeaderBytes> header{};
  const auto header_bytes = state_->request.direct_payloads
                                ? kDirectSpoolRecordHeaderBytes
                                : kSpoolRecordHeaderBytes;
  StoreLittleEndianU32(cluster.cid, header.data());
  StoreLittleEndianU32(cluster.real_count, header.data() + 4U);
  StoreLittleEndianU32(cluster.virtual_count, header.data() + 8U);
  StoreLittleEndianU32(cluster.real_root_count, header.data() + 12U);
  StoreLittleEndianU64(records.codes.size(), header.data() + 16U);
  StoreLittleEndianU64(records.parents.size(), header.data() + 24U);
  StoreLittleEndianU64(records.coefficients.size(), header.data() + 32U);
  StoreLittleEndianU64(state_->seen.real, header.data() + 40U);
  if (state_->request.direct_payloads) {
    const auto append = [&](SequentialFileWriter& file,
                            ArrayView<const std::uint8_t> bytes,
                            std::size_t at) -> Status {
      const auto offset = AlignCanonicalRecord(file.size());
      const std::array<std::uint8_t, 8> zero{};
      if (auto status = file.Append(
              {zero.data(), static_cast<std::size_t>(offset - file.size())});
          !status.ok())
        return status;
      StoreLittleEndianU64(offset, header.data() + at);
      return file.Append(bytes);
    };
    if (auto status = append(state_->codes, records.codes, 48U); !status.ok())
      return status;
    if (auto status = append(state_->parents, records.parents, 56U);
        !status.ok())
      return status;
    if (state_->request.publish_quantized_forest)
      if (auto status = append(state_->coefficients, records.coefficients, 64U);
          !status.ok())
        return status;
    if (state_->request.retain_float_forest) {
      const auto offset =
          AlignCanonicalRecord(state_->float_coefficients.size());
      const std::array<std::uint8_t, 8> zero{};
      if (auto status = state_->float_coefficients.Append(
              {zero.data(), static_cast<std::size_t>(
                                offset - state_->float_coefficients.size())});
          !status.ok())
        return status;
      StoreLittleEndianU64(offset, header.data() + 72U);
    }
  } else {
    if (auto status = state_->records.Append({header.data(), header_bytes});
        !status.ok())
      return status;
    for (auto bytes : {records.codes, records.parents, records.coefficients})
      if (auto status = state_->records.Append(bytes); !status.ok())
        return status;
  }
  if (const Status status = AppendFloats(
          state_->norm2, cluster.exact_real_norm2, true,
          "prepared cluster contains an invalid reconstruction norm");
      !status.ok()) {
    return status;
  }
  if (state_->request.retain_source_norm2) {
    if (const Status status = AppendFloats(
            state_->source_norm2, cluster.source_real_norm2, true,
            "prepared cluster contains an invalid original-vector norm");
        !status.ok()) {
      return status;
    }
  }
  if (state_->request.retain_float_forest) {
    if (const Status status = AppendFloats(
            state_->float_coefficients,
            cluster.float_coefficients_layer_major, false,
            "prepared cluster contains an invalid float coefficient");
        !status.ok()) {
      return status;
    }
  }
  if (RetainFloatNorms(state_->request)) {
    if (const Status status = AppendFloats(
            state_->float_norm2, cluster.float_real_norm2, true,
            "prepared cluster contains an invalid float reconstruction norm");
        !status.ok()) {
      return status;
    }
  }
  state_->seen.real += cluster.real_count;
  state_->seen.virtual_roots += cluster.virtual_count;
  state_->seen.real_roots += cluster.real_root_count;
  state_->seen.linked_real += linked;
  if (state_->request.direct_payloads) {
    if (auto status = state_->records.Append({header.data(), header_bytes});
        !status.ok())
      return status;
    state_->record_log_checksum =
        Checksum({header.data(), header_bytes}, state_->record_log_checksum);
  }
  state_->completed[cluster.cid] = 1U;
  while (state_->next_cid < state_->completed.size() &&
         state_->completed[state_->next_cid])
    ++state_->next_cid;
  return state_->SaveCheckpoint();
}

template <int M>
Status PreparedClusterSpoolWriter<M>::Append(
    const PreparedCanonicalCluster<M>& cluster) {
  const Status ready = storage_.CheckAppend(
      cluster.m(), cluster.cid(), cluster.has_quantized_forest(),
      cluster.completion_.stage ==
          PreparedCanonicalCluster<M>::Completion::Stage::kReady);
  if (!ready.ok()) return ready;
  CanonicalClusterRecords records;
  const Status encoded = cluster.has_quantized_forest()
                             ? CanonicalClusterCodec<M>::EncodePreparedPayload(
                                   cluster.payload_source(), records)
                             : CanonicalClusterCodec<M>::EncodeTopology(
                                   cluster.topology_source(), records);
  if (!encoded.ok()) return encoded;
  return storage_.AppendSerialized(
      {cluster.cid(), cluster.real_count(), cluster.virtual_count(),
       cluster.real_root_count(), records, cluster.exact_real_norm2(),
       cluster.float_coefficients_layer_major(), cluster.float_real_norm2(),
       cluster.source_real_norm2()});
}

Status PreparedClusterSpoolWriterStorage::Commit() {
  if (state_ == nullptr) {
    return Status::FailedPrecondition(
        "prepared cluster spool cannot be committed twice");
  }
  if (state_->next_cid != state_->request.shape.nlist ||
      state_->seen.real == 0U ||
      state_->seen.real_roots + state_->seen.linked_real != state_->seen.real) {
    return Status::FailedPrecondition(
        "prepared cluster spool has not received the registered index");
  }
  if (state_->request.direct_payloads) {
    if (auto status = state_->codes.Close(); !status.ok()) return status;
    if (auto status = state_->parents.Close(); !status.ok()) return status;
    if (state_->request.publish_quantized_forest)
      if (auto status = state_->coefficients.Close(); !status.ok())
        return status;
  }
  const std::uint64_t records_bytes = state_->records.size();
  const std::uint64_t norm2_bytes = state_->norm2.size();
  const std::uint64_t float_coefficients_bytes =
      state_->request.retain_float_forest ? state_->float_coefficients.size()
                                          : 0U;
  const std::uint64_t float_norm2_bytes =
      RetainFloatNorms(state_->request) ? state_->float_norm2.size() : 0U;
  const std::uint64_t source_norm2_bytes =
      state_->request.retain_source_norm2 ? state_->source_norm2.size() : 0U;
  if (const Status status = state_->records.Close(); !status.ok()) {
    return status;
  }
  if (const Status status = state_->norm2.Close(); !status.ok()) {
    return status;
  }
  if (state_->request.retain_float_forest) {
    if (const Status status = state_->float_coefficients.Close();
        !status.ok()) {
      return status;
    }
  }
  if (RetainFloatNorms(state_->request)) {
    if (const Status status = state_->float_norm2.Close(); !status.ok()) {
      return status;
    }
  }
  if (state_->request.retain_source_norm2) {
    if (const Status status = state_->source_norm2.Close(); !status.ok()) {
      return status;
    }
  }
  if (const Status status = WriteManifest(
          state_->path() / "spool.manifest", state_->request, state_->seen,
          records_bytes, norm2_bytes, float_coefficients_bytes,
          float_norm2_bytes, source_norm2_bytes);
      !status.ok()) {
    return status;
  }
  if (!state_->checkpointed) {
    if (const Status status = state_->publication.Publish(); !status.ok()) {
      return status;
    }
  }
  state_.reset();
  return Status::Ok();
}

std::uint32_t PreparedClusterSpoolWriterStorage::next_cid() const noexcept {
  return state_ == nullptr ? 0U : state_->next_cid;
}

const IndexCounts& PreparedClusterSpoolWriterStorage::committed_counts()
    const noexcept {
  static const IndexCounts empty;
  return state_ == nullptr ? empty : state_->seen;
}

bool PreparedClusterSpoolWriterStorage::direct_payloads() const noexcept {
  return state_->request.direct_payloads;
}
const std::vector<std::uint8_t>&
PreparedClusterSpoolWriterStorage::completed_cids() const noexcept {
  return state_->completed;
}

StatusOr<PreparedClusterSpoolStorage> PreparedClusterSpoolStorage::Open(
    const std::filesystem::path& directory, int expected_m) {
  auto manifest_result = ReadOnlyMappedFile::Open(directory / "spool.manifest");
  if (!manifest_result.ok()) {
    return manifest_result.status();
  }
  auto manifest = std::move(manifest_result).value();
  std::uint64_t expected_records = 0U;
  std::uint64_t expected_norm2 = 0U;
  std::uint64_t expected_float_coefficients = 0U;
  std::uint64_t expected_float_norm2 = 0U;
  std::uint64_t expected_source_norm2 = 0U;
  IndexCounts counts;
  auto request_result = DecodeManifest(
      directory, manifest, expected_m, counts, expected_records, expected_norm2,
      expected_float_coefficients, expected_float_norm2, expected_source_norm2);
  if (!request_result.ok()) {
    return request_result.status();
  }
  auto records_result = ReadOnlyMappedFile::Open(directory / "records.bin");
  if (!records_result.ok()) {
    return records_result.status();
  }
  auto norm_result = ReadOnlyMappedFile::Open(directory / "norm2.f32");
  if (!norm_result.ok()) {
    return norm_result.status();
  }
  PreparedClusterSpoolStorage output;
  output.directory_ = directory;
  output.shape_ = request_result.value().shape;
  const auto m = output.shape_.m;
  output.counts_ = counts;
  output.hashes_ = request_result.value().hashes;
  output.records_ = std::move(records_result).value();
  output.norm2_ = std::move(norm_result).value();
  output.has_float_forest_ = request_result.value().retain_float_forest;
  output.has_float_norm2_ = RetainFloatNorms(request_result.value());
  output.has_source_norm2_ = request_result.value().retain_source_norm2;
  output.has_quantized_forest_ =
      request_result.value().publish_quantized_forest;
  output.direct_payloads_ = request_result.value().direct_payloads;
  if (output.direct_payloads_) {
    for (auto item :
         {std::pair{&output.codes_, ShardKind::kCodes},
          std::pair{&output.parents_, ShardKind::kParents},
          std::pair{&output.coefficients_, ShardKind::kCoefficients}}) {
      if (item.second == ShardKind::kCoefficients &&
          !output.has_quantized_forest_)
        continue;
      auto file = ReadOnlyMappedFile::Open(
          directory / (item.second == ShardKind::kCodes ? "codes.bin"
                       : item.second == ShardKind::kParents
                           ? "parents.bin"
                           : "coefficients.bin"));
      if (!file.ok()) return file.status();
      *item.first = std::move(file).value();
      if (auto status = CanonicalPayloadShardCodec::ValidateHeader(
              item.first->data(), item.first->size(), item.second,
              output.shape_);
          !status.ok())
        return status;
    }
  }
  if (output.has_float_forest_) {
    auto coefficients =
        ReadOnlyMappedFile::Open(directory / "float_coefficients.f32");
    if (!coefficients.ok()) return coefficients.status();
    output.float_coefficients_ = std::move(coefficients).value();
    if (output.direct_payloads_) {
      const Status status = CanonicalPayloadShardCodec::ValidateHeader(
          output.float_coefficients_.data(), output.float_coefficients_.size(),
          ShardKind::kFloatCoefficients, output.shape_);
      if (!status.ok()) return status;
    }
  }
  if (output.has_float_norm2_) {
    auto float_norm2 = ReadOnlyMappedFile::Open(directory / "float_norm2.f32");
    if (!float_norm2.ok()) return float_norm2.status();
    output.float_norm2_ = std::move(float_norm2).value();
  }
  if (output.has_source_norm2_) {
    auto source_norm2 =
        ReadOnlyMappedFile::Open(directory / "source_norm2.f32");
    if (!source_norm2.ok()) return source_norm2.status();
    output.source_norm2_ = std::move(source_norm2).value();
  }
  if (output.counts_.real >
          std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
      output.records_.size() != expected_records ||
      output.norm2_.size() != expected_norm2 ||
      expected_norm2 != (output.has_quantized_forest_
                             ? output.counts_.real * sizeof(float)
                             : 0U)) {
    return Status::DataLoss(
        "prepared cluster spool file sizes disagree with its manifest");
  }
  const std::uint64_t total_nodes =
      output.counts_.real + output.counts_.virtual_roots;
  if ((output.has_float_forest_ &&
       (output.float_coefficients_.size() != expected_float_coefficients ||
        output.float_norm2_.size() != expected_float_norm2 ||
        (!output.direct_payloads_ &&
         expected_float_coefficients != total_nodes * m * sizeof(float)) ||
        expected_float_norm2 !=
            (output.has_float_norm2_ ? output.counts_.real * sizeof(float) : 0U))) ||
      (!output.has_float_forest_ &&
       (expected_float_coefficients != 0U || expected_float_norm2 != 0U))) {
    return Status::DataLoss(
        "prepared float companion sizes disagree with its manifest");
  }
  if ((output.has_source_norm2_ &&
       (output.source_norm2_.size() != expected_source_norm2 ||
        expected_source_norm2 != output.counts_.real * sizeof(float))) ||
      (!output.has_source_norm2_ && expected_source_norm2 != 0U)) {
    return Status::DataLoss(
        "prepared original-vector norm size disagrees with its manifest");
  }

  output.entries_.resize(output.shape_.nlist);
  std::vector<std::uint8_t> present(output.shape_.nlist, 0U);
  std::uint64_t cursor = 0U;
  std::uint64_t norm_row = 0U;
  IndexCounts seen;
  std::uint64_t codes_end = kCanonicalPayloadHeaderBytes;
  std::uint64_t parents_end = kCanonicalPayloadHeaderBytes;
  std::uint64_t coefficients_end = kCanonicalPayloadHeaderBytes;
  std::uint64_t float_coefficients_end = kCanonicalPayloadHeaderBytes;
  for (std::uint32_t physical = 0U; physical < output.shape_.nlist;
       ++physical) {
    const auto header_bytes = output.direct_payloads_
                                  ? kDirectSpoolRecordHeaderBytes
                                  : kSpoolRecordHeaderBytes;
    if (cursor > output.records_.size() ||
        output.records_.size() - cursor < header_bytes) {
      return Status::DataLoss(
          "prepared cluster spool record header is truncated");
    }
    const std::uint8_t* header = output.records_.data() + cursor;
    PreparedClusterSpoolEntry entry;
    entry.cid = LoadLittleEndianU32(header);
    entry.real_count = LoadLittleEndianU32(header + 4U);
    entry.virtual_count = LoadLittleEndianU32(header + 8U);
    entry.real_root_count = LoadLittleEndianU32(header + 12U);
    entry.codes_bytes = LoadLittleEndianU64(header + 16U);
    entry.parents_bytes = LoadLittleEndianU64(header + 24U);
    entry.coefficients_bytes = LoadLittleEndianU64(header + 32U);
    entry.norm2_row_offset = LoadLittleEndianU64(header + 40U);
    entry.float_coefficient_row_offset = seen.real + seen.virtual_roots;
    entry.physical_norm2_row_offset = norm_row;
    if (entry.cid >= output.shape_.nlist || present[entry.cid]++ ||
        (!output.direct_payloads_ && entry.cid != physical) ||
        entry.real_root_count > entry.real_count ||
        entry.norm2_row_offset != norm_row) {
      return Status::DataLoss("prepared cluster spool CID metadata is invalid");
    }
    cursor += header_bytes;
    if (entry.parents_bytes >
            std::numeric_limits<std::uint64_t>::max() - entry.codes_bytes ||
        entry.coefficients_bytes > std::numeric_limits<std::uint64_t>::max() -
                                       entry.codes_bytes -
                                       entry.parents_bytes) {
      return Status::DataLoss(
          "prepared cluster spool payload extent overflows u64");
    }
    const std::uint64_t payload_bytes =
        entry.codes_bytes + entry.parents_bytes + entry.coefficients_bytes;
    if (!output.direct_payloads_ &&
        (cursor > output.records_.size() ||
         payload_bytes > output.records_.size() - cursor)) {
      return Status::DataLoss("prepared cluster spool payload is truncated");
    }
    const std::uint64_t expected_codes =
        static_cast<std::uint64_t>(entry.real_count) * (m + 4U) +
        static_cast<std::uint64_t>(entry.virtual_count) * (m - 1U);
    const std::uint64_t expected_parents =
        ParentLoudsWordCount(entry.real_count + entry.virtual_count) * 8U;
    if (entry.codes_bytes != expected_codes ||
        entry.parents_bytes != expected_parents) {
      return Status::DataLoss(
          "prepared cluster spool canonical extents are invalid");
    }
    entry.codes_offset = cursor;
    entry.parents_offset = entry.codes_offset + entry.codes_bytes;
    entry.coefficients_offset = entry.parents_offset + entry.parents_bytes;
    if (output.direct_payloads_) {
      entry.codes_offset = LoadLittleEndianU64(header + 48U);
      entry.parents_offset = LoadLittleEndianU64(header + 56U);
      entry.coefficients_offset = LoadLittleEndianU64(header + 64U);
      entry.float_coefficient_byte_offset = LoadLittleEndianU64(header + 72U);
      if (const Status status = ValidateDirectExtent(
              output.codes_, entry.codes_offset, entry.codes_bytes, codes_end);
          !status.ok())
        return status;
      if (const Status status =
              ValidateDirectExtent(output.parents_, entry.parents_offset,
                                   entry.parents_bytes, parents_end);
          !status.ok())
        return status;
      if (output.has_quantized_forest_) {
        if (const Status status = ValidateDirectExtent(
                output.coefficients_, entry.coefficients_offset,
                entry.coefficients_bytes, coefficients_end);
            !status.ok())
          return status;
      }
      if (output.has_float_forest_) {
        const auto bytes = (static_cast<std::uint64_t>(entry.real_count) +
                            entry.virtual_count) *
                           m * sizeof(float);
        if (const Status status = ValidateDirectExtent(
                output.float_coefficients_, entry.float_coefficient_byte_offset,
                bytes, float_coefficients_end);
            !status.ok())
          return status;
      }
    }
    if (output.has_quantized_forest_) {
      if (const Status status = ValidateCoefficientExtent(
              {(output.direct_payloads_ ? output.coefficients_.data()
                                        : output.records_.data()) +
                   entry.coefficients_offset,
               static_cast<std::size_t>(entry.coefficients_bytes)},
              m);
          !status.ok()) {
        return status;
      }
    } else if (entry.coefficients_bytes != 0U) {
      return Status::DataLoss(
          "float-only prepared spool contains coefficient payload bytes");
    }
    if (!output.direct_payloads_) cursor += payload_bytes;
    norm_row += entry.real_count;
    seen.real += entry.real_count;
    seen.virtual_roots += entry.virtual_count;
    seen.real_roots += entry.real_root_count;
    seen.linked_real += entry.real_count - entry.real_root_count;
    output.entries_[entry.cid] = entry;
  }
  if (cursor != output.records_.size() || norm_row != output.counts_.real ||
      seen.real != output.counts_.real ||
      seen.virtual_roots != output.counts_.virtual_roots ||
      seen.real_roots != output.counts_.real_roots ||
      seen.linked_real != output.counts_.linked_real) {
    return Status::DataLoss(
        "prepared cluster spool totals disagree with its manifest");
  }
  if (output.direct_payloads_ &&
      (codes_end != output.codes_.size() ||
       parents_end != output.parents_.size() ||
       (output.has_quantized_forest_ &&
        coefficients_end != output.coefficients_.size()) ||
       (output.has_float_forest_ &&
        float_coefficients_end != output.float_coefficients_.size()))) {
    return Status::DataLoss(
        "direct spool payload contains uncommitted trailing bytes");
  }
  output.logical_norm_starts_.push_back(0U);
  for (auto& entry : output.entries_) {
    entry.norm2_row_offset = output.logical_norm_starts_.back();
    output.logical_norm_starts_.push_back(entry.norm2_row_offset +
                                          entry.real_count);
  }
  return output;
}

CanonicalClusterRecordViews PreparedClusterSpoolStorage::payload_records(
    std::uint32_t cid) const noexcept {
  const PreparedClusterSpoolEntry& value = entries_[cid];
  return {
      .codes = {(direct_payloads_ ? codes_.data() : records_.data()) +
                    value.codes_offset,
                static_cast<std::size_t>(value.codes_bytes)},
      .parents = {(direct_payloads_ ? parents_.data() : records_.data()) +
                      value.parents_offset,
                  static_cast<std::size_t>(value.parents_bytes)},
      .coefficients = {has_quantized_forest_
                           ? (direct_payloads_ ? coefficients_.data()
                                               : records_.data()) +
                                 value.coefficients_offset
                           : nullptr,
                       static_cast<std::size_t>(value.coefficients_bytes)},
      .norm8 = {},
  };
}

Status PreparedClusterSpoolStorage::ReadNormStream(
    const ReadOnlyMappedFile& file, std::uint64_t row_begin,
    ArrayView<float> output) const {
  std::size_t written = 0;
  while (written < output.size()) {
    const auto logical = row_begin + written;
    const auto it = std::upper_bound(logical_norm_starts_.begin(),
                                     logical_norm_starts_.end(), logical);
    const auto cid =
        static_cast<std::size_t>(it - logical_norm_starts_.begin() - 1);
    const auto& entry = entries_[cid];
    const auto within = logical - entry.norm2_row_offset;
    const auto take = std::min<std::uint64_t>(output.size() - written,
                                              entry.real_count - within);
    const auto* source =
        file.data() +
        (entry.physical_norm2_row_offset + within) * sizeof(float);
    for (std::size_t i = 0; i < take; ++i) {
      const float value = LoadLittleEndianF32(source + i * sizeof(float));
      if (!std::isfinite(value) || value < 0)
        return Status::DataLoss("prepared spool contains an invalid norm2");
      output[written + i] = value;
    }
    written += take;
  }
  return Status::Ok();
}

Status PreparedClusterSpoolStorage::ReadNorm2(std::uint64_t row_begin,
                                              ArrayView<float> output) const {
  if (!has_quantized_forest_) {
    return Status::FailedPrecondition(
        "prepared spool has no quantized reconstruction norms");
  }
  if (row_begin > counts_.real || output.size() > counts_.real - row_begin) {
    return Status::InvalidArgument(
        "prepared spool norm range exceeds the real-row domain");
  }
  return ReadNormStream(norm2_, row_begin, output);
}

Status PreparedClusterSpoolStorage::ReadFloatCoefficients(
    std::uint32_t cid, ArrayView<float> output) const {
  if (!has_float_forest_) {
    return Status::FailedPrecondition(
        "prepared spool has no native-float companion");
  }
  if (cid >= entries_.size()) {
    return Status::InvalidArgument("prepared float CID exceeds nlist");
  }
  const PreparedClusterSpoolEntry& item = entries_[cid];
  const std::uint64_t nodes =
      static_cast<std::uint64_t>(item.real_count) + item.virtual_count;
  if (output.size() != nodes * shape_.m) {
    return Status::InvalidArgument(
        "prepared float coefficient output has the wrong extent");
  }
  const std::uint64_t begin = item.float_coefficient_row_offset * shape_.m;
  const std::uint8_t* source =
      float_coefficients_.data() + (direct_payloads_
                                        ? item.float_coefficient_byte_offset
                                        : begin * sizeof(float));
  for (std::size_t offset = 0U; offset < output.size(); ++offset) {
    const float value = LoadLittleEndianF32(source + offset * sizeof(float));
    if (!std::isfinite(value)) {
      return Status::DataLoss(
          "prepared spool contains an invalid float coefficient");
    }
    output[offset] = value;
  }
  return Status::Ok();
}

Status PreparedClusterSpoolStorage::ReadFloatNorm2(
    std::uint64_t row_begin, ArrayView<float> output) const {
  if (!has_float_norm2_) {
    return Status::FailedPrecondition(
        "prepared spool has no native-float norm stream");
  }
  if (row_begin > counts_.real || output.size() > counts_.real - row_begin) {
    return Status::InvalidArgument(
        "prepared float norm range exceeds the real-row domain");
  }
  return ReadNormStream(float_norm2_, row_begin, output);
}

Status PreparedClusterSpoolStorage::ReadSourceNorm2(
    std::uint64_t row_begin, ArrayView<float> output) const {
  if (!has_source_norm2_) {
    return Status::FailedPrecondition(
        "prepared spool has no original-vector norm stream");
  }
  if (row_begin > counts_.real || output.size() > counts_.real - row_begin) {
    return Status::InvalidArgument(
        "prepared source norm range exceeds the real-row domain");
  }
  return ReadNormStream(source_norm2_, row_begin, output);
}

template class PreparedClusterSpoolWriter<0>;
template class PreparedClusterSpoolWriter<5>;
template class PreparedClusterSpoolWriter<12>;
template class PreparedClusterSpoolWriter<6>;
template class PreparedClusterSpoolWriter<10>;

}  // namespace stlqpp
