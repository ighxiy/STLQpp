#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/checkpoint/training_forest_codec.h"

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/model/topology/canonical_bfs_forest_order.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace stlqpp {
namespace {

constexpr std::array<std::uint8_t, 8> kDirectoryMagic = {
    'S', 'T', 'L', 'Q', 'T', 'F', 'D', '4'};

Status ExpectedRecordBytes(std::uint32_t dimension,
                           std::uint32_t m,
                           std::uint32_t real_count,
                           std::uint32_t virtual_count,
                           std::uint64_t* output) {
  static_cast<void>(dimension);
  const std::uint64_t virtual_stride = sizeof(std::uint32_t) +
      (m - 1U) + static_cast<std::uint64_t>(m) * sizeof(float);
  const std::uint64_t real_stride = 2U * sizeof(std::uint32_t);
  if (virtual_count != 0U &&
      virtual_stride > std::numeric_limits<std::uint64_t>::max() /
                           virtual_count) {
    return Status::ResourceExhausted(
        "training virtual-root record extent overflows u64");
  }
  const std::uint64_t virtual_bytes = virtual_stride * virtual_count;
  const std::uint64_t real_bytes = real_stride * real_count;
  if (real_bytes > std::numeric_limits<std::uint64_t>::max() -
                       virtual_bytes) {
    return Status::ResourceExhausted(
        "training forest record extent overflows u64");
  }
  *output = virtual_bytes + real_bytes;
  return Status::Ok();
}

Status ValidateSource(const TrainingForestClusterSource& source,
                      std::uint32_t m) {
  const std::size_t real_count = source.real_roles.size();
  const std::size_t virtual_count = source.virtual_sample_ids.size();
  if ((m < 2U || m > stlqpp::kMaximumModelLayers) || source.dimension == 0U ||
      source.virtual_small_codes.size() != virtual_count * (m - 1U) ||
      source.virtual_coefficients.size() != virtual_count * m ||
      source.real_sample_ids.size() != real_count ||
      source.real_parents.size() != real_count ||
      source.real_depths.size() != real_count ||
      real_count > std::numeric_limits<std::uint32_t>::max() ||
      virtual_count > std::numeric_limits<std::uint32_t>::max()) {
    return Status::InvalidArgument(
        "training forest cluster streams have incompatible shapes");
  }
  for (const std::uint64_t id : source.virtual_sample_ids) {
    if (id >> 32U != source.cid) {
      return Status::InvalidArgument(
          "training virtual ID does not belong to its CID");
    }
  }
  for (const std::uint64_t id : source.real_sample_ids) {
    if (id > std::numeric_limits<std::uint32_t>::max()) {
      return Status::ResourceExhausted("training real ID exceeds u32");
    }
  }
  for (const float value : source.virtual_coefficients) {
    if (!std::isfinite(value)) {
      return Status::InvalidArgument(
          "training virtual root coefficient is non-finite");
    }
  }
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    const LinkageNodeRole role = source.real_roles[real];
    const auto& parent = source.real_parents[real];
    const std::uint32_t depth = source.real_depths[real];
    if (role == LinkageNodeRole::kRoot) {
      if (parent.has_value() || depth != 0U) {
        return Status::InvalidArgument(
            "training real root has a parent or nonzero depth");
      }
      continue;
    }
    if (role != LinkageNodeRole::kLinked || !parent.has_value() ||
        depth == 0U) {
      return Status::InvalidArgument(
          "training linked row has no causal parent/depth");
    }
    if (parent->kind == LinkageParentKind::kVirtual) {
      if (parent->local_id >= virtual_count || depth != 1U) {
        return Status::InvalidArgument(
            "training virtual-parent edge is invalid");
      }
    } else if (parent->local_id >= real_count ||
               source.real_depths[parent->local_id] + 1U != depth) {
      return Status::InvalidArgument(
          "training real-parent edge violates depth causality");
    }
  }
  return Status::Ok();
}

#ifndef NDEBUG
Status ValidateDecoded(TrainingForestCluster& cluster, std::uint32_t m) {
  const TrainingForestClusterSource source = {
      .cid = cluster.cid,
      .dimension = cluster.dimension,
      .virtual_sample_ids = cluster.virtual_sample_ids,
      .virtual_small_codes = cluster.virtual_small_codes,
      .virtual_coefficients = cluster.virtual_coefficients,
      .real_sample_ids = cluster.real_sample_ids,
      .real_roles = cluster.real_roles,
      .real_parents = cluster.real_parents,
      .real_depths = cluster.real_depths,
  };
  return ValidateSource(source, m);
}
#endif

}  // namespace

Status TrainingForestCodec::EncodeRecord(
    const TrainingForestClusterSource& source,
    std::uint32_t m,
    std::vector<std::uint8_t>& output) {
  auto bytes = EncodeRecordAt(source, m, output, 0U);
  return bytes.ok() ? Status::Ok() : bytes.status();
}

StatusOr<std::uint64_t> TrainingForestCodec::EncodeRecordAt(
    const TrainingForestClusterSource& source,
    std::uint32_t m,
    std::vector<std::uint8_t>& output,
    std::uint64_t offset,
    CanonicalBfsForestOrder* order) {
  if (offset != 0U && offset < output.size()) {
    return Status::InvalidArgument(
        "training forest record offset overlaps existing bytes");
  }
  if (const Status status = ValidateSource(source, m); !status.ok()) {
    return status;
  }
  const std::uint32_t dimension = source.dimension;
  const std::uint32_t real_count =
      static_cast<std::uint32_t>(source.real_roles.size());
  const std::uint32_t virtual_count =
      static_cast<std::uint32_t>(source.virtual_sample_ids.size());
  auto bfs = BuildCanonicalBfsForestOrder({
      .virtual_count = virtual_count,
      .real_roles = source.real_roles,
      .real_parents = source.real_parents,
  });
  if (!bfs.ok()) return bfs.status();
  std::uint64_t expected = 0U;
  if (const Status status = ExpectedRecordBytes(
          dimension, m, real_count, virtual_count, &expected);
      !status.ok()) {
    return status;
  }
  if (offset > std::numeric_limits<std::size_t>::max() ||
      expected > std::numeric_limits<std::size_t>::max() - offset) {
    return Status::ResourceExhausted(
        "training forest record exceeds addressable memory");
  }
  output.resize(static_cast<std::size_t>(offset + expected), 0U);
  std::uint8_t* record = expected == 0U
      ? nullptr : output.data() + offset;
  std::size_t cursor = 0U;
  for (std::uint64_t sample_id : source.virtual_sample_ids) {
    StoreLittleEndianU32(static_cast<std::uint32_t>(sample_id),
                         record + cursor);
    cursor += sizeof(std::uint32_t);
  }
  if (!source.virtual_small_codes.empty()) {
    std::memcpy(record + cursor, source.virtual_small_codes.data(),
                source.virtual_small_codes.size());
    cursor += source.virtual_small_codes.size();
  }
  for (const float value : source.virtual_coefficients) {
    StoreLittleEndianF32(value, record + cursor);
    cursor += sizeof(float);
  }
  for (const std::uint32_t old_real : bfs.value().new_to_old_real) {
    StoreLittleEndianU32(
        static_cast<std::uint32_t>(source.real_sample_ids[old_real]),
        record + cursor);
    cursor += sizeof(std::uint32_t);
  }
  for (const std::uint32_t parent : bfs.value().parent_local_1based) {
    StoreLittleEndianU32(parent, record + cursor);
    cursor += sizeof(std::uint32_t);
  }
  if (order != nullptr) {
    *order = std::move(bfs).value();
  }
  return expected;
}

Status TrainingForestCodec::DecodeRecord(
    std::uint32_t dimension,
    std::uint32_t m,
    const TrainingForestEntry& entry,
    ArrayView<const std::uint8_t> bytes,
    TrainingForestCluster& output) {
  std::uint64_t expected = 0U;
  if (dimension == 0U || (m < 2U || m > stlqpp::kMaximumModelLayers) ||
      !ExpectedRecordBytes(dimension, m, entry.real_count,
                           entry.virtual_count, &expected).ok() ||
      expected != bytes.size() ||
      (bytes.data() == nullptr && !bytes.empty())) {
    return Status::DataLoss(
        "training forest record length disagrees with its directory entry");
  }
  TrainingForestCluster decoded;
  decoded.cid = entry.cid;
  decoded.dimension = dimension;
  decoded.virtual_sample_ids.resize(entry.virtual_count);
  decoded.virtual_small_codes.resize(
      static_cast<std::size_t>(entry.virtual_count) * (m - 1U));
  decoded.virtual_coefficients.resize(
      static_cast<std::size_t>(entry.virtual_count) * m);
  decoded.real_sample_ids.resize(entry.real_count);
  decoded.real_roles.resize(entry.real_count);
  decoded.real_parents.resize(entry.real_count);
  decoded.real_depths.resize(entry.real_count);
  std::size_t cursor = 0U;
  for (std::uint64_t& sample_id : decoded.virtual_sample_ids) {
    sample_id = (static_cast<std::uint64_t>(entry.cid) << 32U) |
        LoadLittleEndianU32(bytes.data() + cursor);
    cursor += sizeof(std::uint32_t);
  }
  if (!decoded.virtual_small_codes.empty()) {
    std::memcpy(decoded.virtual_small_codes.data(), bytes.data() + cursor,
                decoded.virtual_small_codes.size());
    cursor += decoded.virtual_small_codes.size();
  }
  for (float& value : decoded.virtual_coefficients) {
    value = LoadLittleEndianF32(bytes.data() + cursor);
    cursor += sizeof(float);
    if (!std::isfinite(value)) {
      return Status::DataLoss(
          "training virtual root coefficient is non-finite");
    }
  }
  for (std::uint64_t& sample_id : decoded.real_sample_ids) {
    sample_id = LoadLittleEndianU32(bytes.data() + cursor);
    cursor += sizeof(std::uint32_t);
  }
  bool saw_linked = false;
  std::uint32_t previous_parent = 0U;
  for (std::uint32_t real = 0U; real < entry.real_count; ++real) {
    const std::uint32_t encoded =
        LoadLittleEndianU32(bytes.data() + cursor);
    cursor += sizeof(std::uint32_t);
    if (encoded == 0U) {
      if (saw_linked) {
        return Status::DataLoss(
            "training BFS forest contains a root after linked rows");
      }
      decoded.real_roles[real] = LinkageNodeRole::kRoot;
      decoded.real_parents[real] = std::nullopt;
    } else if (encoded < previous_parent) {
      return Status::DataLoss(
          "training BFS forest parent order decreases");
    } else if (encoded <= entry.virtual_count) {
      decoded.real_roles[real] = LinkageNodeRole::kLinked;
      decoded.real_parents[real] = LinkageParentReference{
          LinkageParentKind::kVirtual, encoded - 1U};
    } else {
      const std::uint32_t parent = encoded - entry.virtual_count - 1U;
      if (parent >= real) {
        return Status::DataLoss(
            "training BFS forest parent does not precede its child");
      }
      decoded.real_roles[real] = LinkageNodeRole::kLinked;
      decoded.real_parents[real] = LinkageParentReference{
          LinkageParentKind::kReal, parent};
    }
    if (encoded != 0U) {
      saw_linked = true;
      previous_parent = encoded;
      decoded.real_depths[real] = encoded <= entry.virtual_count
          ? 1U
          : decoded.real_depths[encoded - entry.virtual_count - 1U] + 1U;
      if (real != 0U &&
          decoded.real_depths[real] < decoded.real_depths[real - 1U]) {
        return Status::DataLoss(
            "training forest rows are not BFS depth ordered");
      }
    }
  }
#ifndef NDEBUG
  assert(ValidateDecoded(decoded, m).ok());
#endif
  output = std::move(decoded);
  return Status::Ok();
}

Status TrainingForestCodec::EncodeDirectory(
    std::uint32_t dimension,
    std::uint32_t m,
    std::uint32_t nlist,
    ArrayView<const TrainingForestEntry> entries,
    std::vector<std::uint8_t>& output) {
  if (dimension == 0U || (m < 2U || m > stlqpp::kMaximumModelLayers) || nlist == 0U ||
      entries.size() != nlist) {
    return Status::InvalidArgument(
        "training forest directory shape is invalid");
  }
  std::uint64_t real_total = 0U;
  std::uint64_t virtual_total = 0U;
  std::uint64_t cursor = kTrainingForestHeaderBytes;
  for (std::uint32_t cid = 0U; cid < nlist; ++cid) {
    const TrainingForestEntry& entry = entries[cid];
    std::uint64_t expected = 0U;
    if (entry.cid != cid ||
        !ExpectedRecordBytes(
             dimension, m, entry.real_count, entry.virtual_count, &expected)
             .ok() ||
        entry.bytes != expected ||
        entry.offset != AlignTrainingForestRecord(cursor) ||
        entry.bytes > std::numeric_limits<std::uint64_t>::max() -
                          entry.offset) {
      return Status::InvalidArgument(
          "training forest directory entry is inconsistent");
    }
    cursor = entry.offset + entry.bytes;
    real_total += entry.real_count;
    virtual_total += entry.virtual_count;
  }
  const std::uint64_t byte_count =
      kTrainingForestDirectoryHeaderBytes +
      static_cast<std::uint64_t>(nlist) *
          kTrainingForestDirectoryEntryBytes;
  if (byte_count > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted(
        "training forest directory exceeds addressable memory");
  }
  output.assign(static_cast<std::size_t>(byte_count), 0U);
  std::memcpy(output.data(), kDirectoryMagic.data(), kDirectoryMagic.size());
  StoreLittleEndianU32(kTrainingForestVersion, output.data() + 8U);
  StoreLittleEndianU32(
      kTrainingForestDirectoryHeaderBytes, output.data() + 12U);
  StoreLittleEndianU32(
      kTrainingForestDirectoryEntryBytes, output.data() + 16U);
  StoreLittleEndianU32(dimension, output.data() + 20U);
  StoreLittleEndianU32(nlist, output.data() + 24U);
  StoreLittleEndianU64(real_total, output.data() + 32U);
  StoreLittleEndianU64(virtual_total, output.data() + 40U);
  for (std::uint32_t cid = 0U; cid < nlist; ++cid) {
    const TrainingForestEntry& entry = entries[cid];
    std::uint8_t* destination = output.data() +
        kTrainingForestDirectoryHeaderBytes +
        static_cast<std::size_t>(cid) *
            kTrainingForestDirectoryEntryBytes;
    StoreLittleEndianU32(entry.cid, destination);
    StoreLittleEndianU32(entry.real_count, destination + 4U);
    StoreLittleEndianU32(entry.virtual_count, destination + 8U);
    StoreLittleEndianU64(entry.offset, destination + 16U);
    StoreLittleEndianU64(entry.bytes, destination + 24U);
  }
  return Status::Ok();
}

Status TrainingForestCodec::DecodeDirectory(
    const std::uint8_t* bytes,
    std::size_t byte_count,
    std::uint32_t expected_dimension,
    std::uint32_t expected_m,
    std::uint32_t expected_nlist,
    std::uint64_t expected_forest_bytes,
    std::vector<TrainingForestEntry>& entries) {
  const std::uint64_t expected_directory_bytes =
      kTrainingForestDirectoryHeaderBytes +
      static_cast<std::uint64_t>(expected_nlist) *
          kTrainingForestDirectoryEntryBytes;
  if ((expected_m < 2U || expected_m > stlqpp::kMaximumModelLayers) || bytes == nullptr ||
      byte_count != expected_directory_bytes ||
      std::memcmp(bytes, kDirectoryMagic.data(), kDirectoryMagic.size()) != 0 ||
      LoadLittleEndianU32(bytes + 8U) != kTrainingForestVersion ||
      LoadLittleEndianU32(bytes + 12U) !=
          kTrainingForestDirectoryHeaderBytes ||
      LoadLittleEndianU32(bytes + 16U) !=
          kTrainingForestDirectoryEntryBytes ||
      LoadLittleEndianU32(bytes + 20U) != expected_dimension ||
      LoadLittleEndianU32(bytes + 24U) != expected_nlist ||
      LoadLittleEndianU32(bytes + 28U) != 0U) {
    return Status::DataLoss(
        "training forest directory header is invalid");
  }
  entries.resize(expected_nlist);
  std::uint64_t real_total = 0U;
  std::uint64_t virtual_total = 0U;
  std::uint64_t cursor = kTrainingForestHeaderBytes;
  for (std::uint32_t cid = 0U; cid < expected_nlist; ++cid) {
    const std::uint8_t* source = bytes +
        kTrainingForestDirectoryHeaderBytes +
        static_cast<std::size_t>(cid) *
            kTrainingForestDirectoryEntryBytes;
    TrainingForestEntry entry = {
        .cid = LoadLittleEndianU32(source),
        .real_count = LoadLittleEndianU32(source + 4U),
        .virtual_count = LoadLittleEndianU32(source + 8U),
        .offset = LoadLittleEndianU64(source + 16U),
        .bytes = LoadLittleEndianU64(source + 24U),
    };
    std::uint64_t expected_record = 0U;
    if (LoadLittleEndianU32(source + 12U) != 0U || entry.cid != cid ||
        !ExpectedRecordBytes(expected_dimension, expected_m,
                             entry.real_count, entry.virtual_count,
                             &expected_record).ok() ||
        entry.bytes != expected_record ||
        entry.offset != AlignTrainingForestRecord(cursor) ||
        entry.offset > expected_forest_bytes ||
        entry.bytes > expected_forest_bytes - entry.offset) {
      return Status::DataLoss(
          "training forest directory contains an invalid record range");
    }
    cursor = entry.offset + entry.bytes;
    real_total += entry.real_count;
    virtual_total += entry.virtual_count;
    entries[cid] = entry;
  }
  if (cursor != expected_forest_bytes ||
      real_total != LoadLittleEndianU64(bytes + 32U) ||
      virtual_total != LoadLittleEndianU64(bytes + 40U)) {
    return Status::DataLoss(
        "training forest directory totals or final extent are invalid");
  }
  return Status::Ok();
}

}  // namespace stlqpp
