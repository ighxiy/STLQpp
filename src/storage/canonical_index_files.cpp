#include "stlqpp/storage/canonical_index_files.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>

#include "stlqpp/application/sha256.h"
#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/storage/canonical_directory_codec.h"
#include "stlqpp/storage/canonical_shard_codec.h"
#include "stlqpp/storage/index_manifest_codec.h"

namespace stlqpp {

std::filesystem::path CanonicalShardPath(const std::filesystem::path& index,
                                         const ShardDescriptor& shard) {
  return (index / (shard.reference_path.empty() ? shard.relative_path
                                                : shard.reference_path))
      .lexically_normal();
}

StatusOr<ShardDescriptor> ReferenceCanonicalShard(
    const std::filesystem::path& source,
    const std::filesystem::path& destination, const ShardDescriptor& shard) {
  const auto path = std::filesystem::absolute(CanonicalShardPath(source, shard))
                        .lexically_normal();
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error) return Status::DataLoss("cannot inspect source shard: " + error.message());
  if (shard.payload_offset > size || shard.bytes > size - shard.payload_offset ||
      (shard.payload_offset == 0U && shard.bytes != size))
    return Status::DataLoss("source shard extent disagrees with manifest");
  ShardDescriptor reference = shard;
  reference.reference_path =
      path.lexically_relative(
              std::filesystem::absolute(destination).lexically_normal())
          .generic_string();
  reference.reference_sha256.clear();
  return reference;
}

std::string CanonicalShardIdentity(const std::filesystem::path& index,
                                  const IndexManifest& manifest,
                                  const ShardDescriptor& shard) {
  const auto owner = std::filesystem::absolute(CanonicalSourceIndexDirectory(index, manifest))
                         .lexically_normal();
  const auto path = std::filesystem::absolute(CanonicalShardPath(index, shard))
                        .lexically_normal();
  std::ostringstream text;
  text << "canonical-source-v1\n" << manifest.hashes.linkage_list_directory.ToHex()
       << '\n' << (shard.kind == ShardKind::kCoefficients
                          ? manifest.hashes.coefficient_codec.ToHex() : "") << '\n'
       << static_cast<unsigned>(shard.kind) << '\n' << shard.ordinal << '\n'
       << path.lexically_relative(owner).generic_string() << '\n'
       << shard.payload_offset << '\n' << shard.bytes << '\n' << shard.elements;
  return app::Sha256Bytes(text.str());
}

std::filesystem::path CanonicalSourceIndexDirectory(
    const std::filesystem::path& index, const IndexManifest& manifest) {
  for (const auto& shard : manifest.shards)
    if (shard.kind == ShardKind::kCodes)
      return CanonicalShardPath(index, shard).parent_path();
  return index;
}

namespace {

const ShardDescriptor* FindShard(const IndexManifest& manifest,
                                 ShardKind kind) noexcept {
  for (const ShardDescriptor& shard : manifest.shards) {
    if (shard.kind == kind) {
      return &shard;
    }
  }
  return nullptr;
}

Status MapRegisteredShard(const std::filesystem::path& directory,
                          const ShardDescriptor& descriptor,
                          ReadOnlyMappedFile& output) {
  if (descriptor.bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted(
        "canonical shard exceeds the process address domain");
  }
  const auto path = CanonicalShardPath(directory, descriptor);
  auto mapped = ReadOnlyMappedFile::Open(path);
  if (!mapped.ok()) {
    return mapped.status();
  }
  if (descriptor.payload_offset > mapped.value().size() ||
      descriptor.bytes > mapped.value().size() - descriptor.payload_offset ||
      (descriptor.payload_offset == 0U &&
       mapped.value().size() != descriptor.bytes)) {
    return Status::DataLoss(
        "canonical shard byte count disagrees with the manifest");
  }
  output = std::move(mapped).value();
  return Status::Ok();
}

Status ValidateShardElements(const IndexManifest& manifest) {
  const std::uint64_t nodes =
      manifest.counts.real + manifest.counts.virtual_roots;
  const std::array<std::pair<ShardKind, std::uint64_t>, 4> expected = {{
      {ShardKind::kCodebooks, 1U},
      {ShardKind::kIvfDirectory, manifest.shape.nlist},
      {ShardKind::kCodes, nodes},
      {ShardKind::kParents, nodes},
  }};
  for (const auto& item : expected) {
    const ShardDescriptor* shard = FindShard(manifest, item.first);
    if (shard == nullptr || shard->elements != item.second) {
      return Status::DataLoss(
          "canonical shard element count disagrees with the manifest shape");
    }
  }
  if (manifest.has_quantized_forest()) {
    const ShardDescriptor* coefficients =
        FindShard(manifest, ShardKind::kCoefficients);
    const ShardDescriptor* norm8 = FindShard(manifest, ShardKind::kNorm8);
    if (coefficients == nullptr || norm8 == nullptr ||
        coefficients->elements != nodes ||
        norm8->elements != manifest.counts.real) {
      return Status::DataLoss(
          "quantized shard element counts disagree with the manifest");
    }
  }
  if (manifest.has_float_forest()) {
    const ShardDescriptor* coefficients =
        FindShard(manifest, ShardKind::kFloatCoefficients);
    const ShardDescriptor* norm2 = FindShard(manifest, ShardKind::kFloatNorm2);
    if (coefficients == nullptr ||
        coefficients->elements != nodes * manifest.shape.m ||
        (norm2 != nullptr && norm2->elements != manifest.counts.real)) {
      return Status::DataLoss(
          "native-float shard element counts disagree with the manifest");
    }
  }
  if (manifest.has_scaled_score8()) {
    const ShardDescriptor* scales =
        FindShard(manifest, ShardKind::kScaledScore8);
    if (scales == nullptr || scales->elements != 512U) {
      return Status::DataLoss(
          "scaled score8 element count disagrees with the codec contract");
    }
  }
  if (manifest.has_float_source_norm2()) {
    const ShardDescriptor* source_norm =
        FindShard(manifest, ShardKind::kFloatSourceNorm2);
    if (source_norm == nullptr ||
        source_norm->elements != manifest.counts.real) {
      return Status::DataLoss(
          "float source norm element count disagrees with the manifest");
    }
  }
  return Status::Ok();
}

using EntryOffset = std::uint64_t CanonicalCidEntry::*;

Status ValidateRecordPartition(ArrayView<const CanonicalCidEntry> entries,
                               const ReadOnlyMappedFile& shard,
                               std::uint64_t record_begin,
                               EntryOffset offset_member,
                               EntryOffset bytes_member, const char* label,
                               bool completion_order = false) {
  std::vector<std::uint32_t> order(entries.size());
  std::iota(order.begin(), order.end(), 0U);
  if (completion_order)
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
      return std::pair{entries[a].*offset_member, entries[a].*bytes_member} <
             std::pair{entries[b].*offset_member, entries[b].*bytes_member};
    });
  std::uint64_t cursor = record_begin;
  for (auto cid : order) {
    const auto& entry = entries[cid];
    const std::uint64_t offset = entry.*offset_member;
    const std::uint64_t bytes = entry.*bytes_member;
    const std::uint64_t aligned = AlignCanonicalRecord(cursor);
    if (offset != aligned || offset > shard.size() ||
        bytes > static_cast<std::uint64_t>(shard.size()) - offset) {
      return Status::DataLoss(std::string("canonical ") + label +
                              " records do not form their registered shard");
    }
    for (std::uint64_t padding = cursor; padding < offset; ++padding) {
      if (shard.data()[static_cast<std::size_t>(padding)] != 0U) {
        return Status::DataLoss(std::string("canonical ") + label +
                                " alignment padding is nonzero");
      }
    }
    cursor = offset + bytes;
  }
  if (cursor != shard.size()) {
    return Status::DataLoss(std::string("canonical ") + label +
                            " shard contains trailing or unowned bytes");
  }
  return Status::Ok();
}

ArrayView<const std::uint8_t> Slice(const ReadOnlyMappedFile& file,
                                    std::uint64_t offset,
                                    std::uint64_t bytes) noexcept {
  return {
      file.data() + static_cast<std::size_t>(offset),
      static_cast<std::size_t>(bytes),
  };
}

}  // namespace

StatusOr<CanonicalIndexFiles> CanonicalIndexFiles::Open(
    const std::filesystem::path& directory) {
  return OpenImpl(directory, true, false);
}

StatusOr<CanonicalIndexFiles> CanonicalIndexFiles::OpenWithSourceNorm(
    const std::filesystem::path& directory) {
  return OpenImpl(directory, true, true);
}

StatusOr<CanonicalIndexFiles> CanonicalIndexFiles::OpenNativeFloat(
    const std::filesystem::path& directory) {
  return OpenImpl(directory, false, false);
}

StatusOr<CanonicalIndexFiles> CanonicalIndexFiles::OpenNativeFloatScaled(
    const std::filesystem::path& directory) {
  return OpenImpl(directory, false, true);
}

StatusOr<CanonicalIndexFiles> CanonicalIndexFiles::OpenQuantized(
    const std::filesystem::path& directory, bool include_source_norm2) {
  return OpenImpl(directory, true, include_source_norm2, true, false);
}

StatusOr<CanonicalIndexFiles> CanonicalIndexFiles::OpenCoefficients(
    const std::filesystem::path& directory, bool native_float,
    bool include_source_norm2) {
  return OpenImpl(directory, !native_float, include_source_norm2, false,
                  native_float);
}

StatusOr<CanonicalIndexFiles> CanonicalIndexFiles::OpenImpl(
    const std::filesystem::path& directory,
    bool include_compressed_representation, bool include_float_source_norm2,
    bool include_norms, bool include_native_float) {
  if (directory.empty()) {
    return Status::InvalidArgument(
        "canonical index directory path must not be empty");
  }
  CanonicalIndexFiles output;
  auto manifest_file = ReadOnlyMappedFile::Open(directory / "manifest.stlqpp");
  if (!manifest_file.ok()) {
    return manifest_file.status();
  }
  output.manifest_file_ = std::move(manifest_file).value();
  if (const Status status = IndexManifestCodec::Decode(
          output.manifest_file_.data(), output.manifest_file_.size(),
          output.manifest_);
      !status.ok()) {
    return status;
  }
  if (output.manifest_.has_quantized_forest() &&
      FindShard(output.manifest_, ShardKind::kNorm2Lut) == nullptr) {
    return Status::FailedPrecondition(
        "quantized index requires separate norm2 labels and LUT files");
  }
  if (const Status status = ValidateShardElements(output.manifest_);
      !status.ok()) {
    return status;
  }
  if (include_compressed_representation &&
      !output.manifest_.has_quantized_forest()) {
    return Status::FailedPrecondition(
        "quantized index open requested but coefficient/norm8 shards are "
        "absent");
  }

  const auto map_kind = [&](ShardKind kind, ReadOnlyMappedFile& file) {
    const ShardDescriptor* descriptor = FindShard(output.manifest_, kind);
    if (descriptor == nullptr) {
      return Status::DataLoss("canonical manifest is missing a shard kind");
    }
    return MapRegisteredShard(directory, *descriptor, file);
  };
  if (const Status status = map_kind(ShardKind::kCodebooks, output.model_);
      !status.ok()) {
    return status;
  }
  output.model_payload_offset_ =
      FindShard(output.manifest_, ShardKind::kCodebooks)->payload_offset;
  output.model_payload_bytes_ =
      FindShard(output.manifest_, ShardKind::kCodebooks)->bytes;
  if (const Status status =
          map_kind(ShardKind::kIvfDirectory, output.directory_);
      !status.ok()) {
    return status;
  }
  if (const Status status = map_kind(ShardKind::kCodes, output.codes_);
      !status.ok()) {
    return status;
  }
  if (const Status status = map_kind(ShardKind::kParents, output.parents_);
      !status.ok()) {
    return status;
  }
  if (include_compressed_representation) {
    if (const Status status =
            map_kind(ShardKind::kCoefficients, output.coefficients_);
        !status.ok()) {
      return status;
    }
    if (include_norms) {
      if (const Status status = map_kind(ShardKind::kNorm8, output.norm8_);
          !status.ok()) {
        return status;
      }
      if (include_norms && output.manifest_.has_scaled_score8()) {
        if (const Status status =
                map_kind(ShardKind::kScaledScore8, output.scaled_score8_file_);
            !status.ok()) {
          return status;
        }
      }
    }
  } else if (!output.manifest_.has_float_forest()) {
    return Status::FailedPrecondition(
        "native-float index open requested but float shards are absent");
  }
  if (include_native_float && output.manifest_.has_float_forest()) {
    if (const Status status =
            map_kind(ShardKind::kFloatCoefficients, output.float_coefficients_);
        !status.ok()) {
      return status;
    }
    if (include_norms && output.manifest_.has_float_norm2()) {
      if (const Status status =
              map_kind(ShardKind::kFloatNorm2, output.float_norm2_);
          !status.ok()) {
        return status;
      }
    }
  }
  if (include_float_source_norm2) {
    if (!output.manifest_.has_float_source_norm2()) {
      return Status::FailedPrecondition(
          "float_scaled open requested but original-vector norms are absent");
    }
    if (const Status status =
            map_kind(ShardKind::kFloatSourceNorm2, output.float_source_norm2_);
        !status.ok()) {
      return status;
    }
  }

  if (const Status status = CanonicalDirectoryCodec::Decode(
          output.directory_.data(), output.directory_.size(),
          output.manifest_.shape, output.manifest_.counts, output.entries_,
          output.manifest_.has_quantized_forest(),
          output.manifest_.format_version >=
              IndexManifest::kNormCacheLayoutFormatVersion,
          output.manifest_.format_version ==
              IndexManifest::kCompletionOrderFormatVersion);
      !status.ok()) {
    return status;
  }
  if (const Status status = CanonicalPayloadShardCodec::ValidateHeader(
          output.codes_.data(), output.codes_.size(), ShardKind::kCodes,
          output.manifest_.shape);
      !status.ok()) {
    return status;
  }
  if (const Status status = CanonicalPayloadShardCodec::ValidateHeader(
          output.parents_.data(), output.parents_.size(), ShardKind::kParents,
          output.manifest_.shape);
      !status.ok()) {
    return status;
  }
  if (include_compressed_representation) {
    if (const Status status = CanonicalPayloadShardCodec::ValidateHeader(
            output.coefficients_.data(), output.coefficients_.size(),
            ShardKind::kCoefficients, output.manifest_.shape);
        !status.ok()) {
      return status;
    }
  }
  if (include_native_float && output.manifest_.has_float_forest()) {
    if (const Status status = CanonicalPayloadShardCodec::ValidateHeader(
            output.float_coefficients_.data(),
            output.float_coefficients_.size(), ShardKind::kFloatCoefficients,
            output.manifest_.shape);
        !status.ok()) {
      return status;
    }
    if (output.float_norm2_.size() != 0U) {
      if (const Status status = CanonicalPayloadShardCodec::ValidateHeader(
              output.float_norm2_.data(), output.float_norm2_.size(),
              ShardKind::kFloatNorm2, output.manifest_.shape);
          !status.ok()) {
        return status;
      }
    }
  }
  if (include_float_source_norm2) {
    if (const Status status = CanonicalPayloadShardCodec::ValidateHeader(
            output.float_source_norm2_.data(),
            output.float_source_norm2_.size(), ShardKind::kFloatSourceNorm2,
            output.manifest_.shape);
        !status.ok()) {
      return status;
    }
  }
  if (include_compressed_representation && include_norms) {
    const auto* lut = FindShard(output.manifest_, ShardKind::kNorm2Lut);
    ReadOnlyMappedFile centers;
    if (auto status = MapRegisteredShard(directory, *lut, centers);
        !status.ok())
      return status;
    if (centers.size() != 1024U)
      return Status::DataLoss("invalid norm2 LUT extent");
    output.norm8_centers_.resize(256U);
    for (unsigned l = 0; l < 256; ++l) {
      auto value = LoadLittleEndianF32(centers.data() + l * 4U);
      if (!std::isfinite(value) || value < 0)
        return Status::DataLoss("invalid norm2 LUT center");
      output.norm8_centers_[l] = value;
    }
    if (output.manifest_.has_scaled_score8()) {
      if (const Status status = CanonicalPayloadShardCodec::ValidateHeader(
              output.scaled_score8_file_.data(),
              output.scaled_score8_file_.size(), ShardKind::kScaledScore8,
              output.manifest_.shape);
          !status.ok()) {
        return status;
      }
      constexpr std::size_t count = 256U;
      const std::size_t expected =
          kCanonicalPayloadHeaderBytes + 2U * count * sizeof(float);
      if (output.scaled_score8_file_.size() != expected ||
          output.norm8_centers_.size() != count) {
        return Status::DataLoss("scaled score8 table extent is invalid");
      }
      std::array<float, count> scales{};
      std::array<float, count> biases{};
      for (std::size_t index = 0U; index < count; ++index) {
        scales[index] = LoadLittleEndianF32(
            output.scaled_score8_file_.data() + kCanonicalPayloadHeaderBytes +
            2U * index * sizeof(float));
        biases[index] = LoadLittleEndianF32(
            output.scaled_score8_file_.data() + kCanonicalPayloadHeaderBytes +
            (2U * index + 1U) * sizeof(float));
      }
      auto codebook = ScaledScore8Codebook::Create(scales, biases);
      if (!codebook.ok()) {
        return Status::DataLoss(codebook.status().message());
      }
      output.scaled_score8_ = std::move(codebook).value();
    }
  }

  const ArrayView<const CanonicalCidEntry> entries(output.entries_);
  if (const Status status = ValidateRecordPartition(
          entries, output.codes_, kCanonicalPayloadHeaderBytes,
          &CanonicalCidEntry::codes_offset, &CanonicalCidEntry::codes_bytes,
          "code",
          output.manifest_.format_version ==
              IndexManifest::kCompletionOrderFormatVersion);
      !status.ok()) {
    return status;
  }
  if (const Status status = ValidateRecordPartition(
          entries, output.parents_, kCanonicalPayloadHeaderBytes,
          &CanonicalCidEntry::parents_offset, &CanonicalCidEntry::parents_bytes,
          "parent",
          output.manifest_.format_version ==
              IndexManifest::kCompletionOrderFormatVersion);
      !status.ok()) {
    return status;
  }
  if (include_compressed_representation) {
    if (const Status status = ValidateRecordPartition(
            entries, output.coefficients_, kCanonicalPayloadHeaderBytes,
            &CanonicalCidEntry::coefficients_offset,
            &CanonicalCidEntry::coefficients_bytes, "coefficient",
            output.manifest_.format_version ==
                IndexManifest::kCompletionOrderFormatVersion);
        !status.ok()) {
      return status;
    }
    if (include_norms) {
      if (const Status status = ValidateRecordPartition(
              entries, output.norm8_, 0U,
              &CanonicalCidEntry::norm8_offset, &CanonicalCidEntry::norm8_bytes,
              "norm8",
              output.manifest_.format_version ==
                  IndexManifest::kCompletionOrderFormatVersion);
          !status.ok()) {
        return status;
      }
    }
  }
  if ((include_native_float && output.manifest_.has_float_forest()) ||
      include_float_source_norm2) {
    output.float_extents_.reserve(output.entries_.size());
    std::uint64_t coefficient_cursor = kCanonicalPayloadHeaderBytes;
    std::uint64_t norm_cursor = kCanonicalPayloadHeaderBytes;
    std::uint64_t source_norm_cursor = kCanonicalPayloadHeaderBytes;
    for (const CanonicalCidEntry& entry : output.entries_) {
      CanonicalIndexFiles::FloatCidExtent extent;
      extent.coefficient_offset =
          output.manifest_.format_version ==
                  IndexManifest::kCompletionOrderFormatVersion
              ? entry.float_coefficients_offset
              : AlignCanonicalRecord(coefficient_cursor);
      extent.coefficient_bytes =
          (static_cast<std::uint64_t>(entry.real_count) + entry.virtual_count) *
          output.manifest_.shape.m * sizeof(float);
      extent.norm2_offset = AlignCanonicalRecord(norm_cursor);
      extent.norm2_bytes =
          static_cast<std::uint64_t>(entry.real_count) * sizeof(float);
      if (include_float_source_norm2) {
        extent.source_norm2_offset = AlignCanonicalRecord(source_norm_cursor);
        extent.source_norm2_bytes = extent.norm2_bytes;
      }
      if ((output.float_coefficients_.size() != 0U &&
           (extent.coefficient_offset > output.float_coefficients_.size() ||
            extent.coefficient_bytes > output.float_coefficients_.size() -
                                           extent.coefficient_offset)) ||
          (output.float_norm2_.size() != 0U &&
           (extent.norm2_offset > output.float_norm2_.size() ||
            extent.norm2_bytes >
                output.float_norm2_.size() - extent.norm2_offset)) ||
          (include_float_source_norm2 &&
           (extent.source_norm2_offset > output.float_source_norm2_.size() ||
            extent.source_norm2_bytes > output.float_source_norm2_.size() -
                                            extent.source_norm2_offset))) {
        return Status::DataLoss(
            "native-float CID record exceeds its registered shard");
      }
      for (std::uint64_t byte =
               output.manifest_.format_version ==
                       IndexManifest::kCompletionOrderFormatVersion
                   ? extent.coefficient_offset
                   : coefficient_cursor;
           output.float_coefficients_.size() != 0U &&
           byte < extent.coefficient_offset;
           ++byte) {
        if (output.float_coefficients_.data()[byte] != 0U) {
          return Status::DataLoss(
              "native-float coefficient alignment padding is nonzero");
        }
      }
      for (std::uint64_t byte = norm_cursor;
           output.float_norm2_.size() != 0U && byte < extent.norm2_offset;
           ++byte) {
        if (output.float_norm2_.data()[byte] != 0U) {
          return Status::DataLoss(
              "native-float norm alignment padding is nonzero");
        }
      }
      if (include_float_source_norm2) {
        for (std::uint64_t byte = source_norm_cursor;
             byte < extent.source_norm2_offset; ++byte) {
          if (output.float_source_norm2_.data()[byte] != 0U) {
            return Status::DataLoss(
                "float source norm alignment padding is nonzero");
          }
        }
      }
      coefficient_cursor = extent.coefficient_offset + extent.coefficient_bytes;
      norm_cursor = extent.norm2_offset + extent.norm2_bytes;
      if (include_float_source_norm2) {
        source_norm_cursor =
            extent.source_norm2_offset + extent.source_norm2_bytes;
      }
      output.float_extents_.push_back(extent);
    }
    if ((output.float_coefficients_.size() != 0U &&
         output.manifest_.format_version !=
             IndexManifest::kCompletionOrderFormatVersion &&
         coefficient_cursor != output.float_coefficients_.size()) ||
        (output.float_norm2_.size() != 0U &&
         norm_cursor != output.float_norm2_.size()) ||
        (include_float_source_norm2 &&
         source_norm_cursor != output.float_source_norm2_.size())) {
      return Status::DataLoss(
          "native-float shards contain trailing or unowned bytes");
    }
  }
  output.physical_cid_order_.resize(output.entries_.size());
  std::iota(output.physical_cid_order_.begin(),
            output.physical_cid_order_.end(), 0U);
  if (output.manifest_.format_version ==
      IndexManifest::kCompletionOrderFormatVersion) {
    std::stable_sort(output.physical_cid_order_.begin(),
                     output.physical_cid_order_.end(), [&](auto a, auto b) {
                       return std::pair{output.entries_[a].codes_offset,
                                        output.entries_[a].codes_bytes} <
                              std::pair{output.entries_[b].codes_offset,
                                        output.entries_[b].codes_bytes};
                     });
    if (output.float_coefficients_.size()) {
      auto float_entries = output.entries_;
      for (auto& e : float_entries) {
        e.codes_offset = e.float_coefficients_offset;
        e.codes_bytes =
            (static_cast<std::uint64_t>(e.real_count) + e.virtual_count) *
            output.manifest_.shape.m * sizeof(float);
      }
      if (auto status = ValidateRecordPartition(
              float_entries, output.float_coefficients_,
              kCanonicalPayloadHeaderBytes, &CanonicalCidEntry::codes_offset,
              &CanonicalCidEntry::codes_bytes, "float coefficients", true);
          !status.ok())
        return status;
    }
  }
  return output;
}

CanonicalClusterRecordViews CanonicalIndexFiles::cluster_records(
    std::uint32_t cid) const noexcept {
  const CanonicalCidEntry& item = entries_[cid];
  return {
      .codes = Slice(codes_, item.codes_offset, item.codes_bytes),
      .parents = Slice(parents_, item.parents_offset, item.parents_bytes),
      .coefficients = coefficients_.size() == 0U
                          ? ArrayView<const std::uint8_t>{}
                          : Slice(coefficients_, item.coefficients_offset,
                                  item.coefficients_bytes),
      .norm8 = norm8_.size() == 0U
                   ? ArrayView<const std::uint8_t>{}
                   : Slice(norm8_, item.norm8_offset, item.norm8_bytes),
  };
}

CanonicalTopologyRecordViews CanonicalIndexFiles::topology_records(
    std::uint32_t cid) const noexcept {
  const CanonicalCidEntry& entry = entries_[cid];
  return {
      .codes = Slice(codes_, entry.codes_offset, entry.codes_bytes),
      .parents = Slice(parents_, entry.parents_offset, entry.parents_bytes),
  };
}

CanonicalFloatRecordViews CanonicalIndexFiles::float_cluster_records(
    std::uint32_t cid) const noexcept {
  if (float_coefficients_.size() == 0U) return {};
  const FloatCidExtent& item = float_extents_[cid];
  return {
      .coefficients = Slice(float_coefficients_, item.coefficient_offset,
                            item.coefficient_bytes),
      .norm2 = float_norm2_.size() == 0U
                   ? ArrayView<const std::uint8_t>{}
                   : Slice(float_norm2_, item.norm2_offset, item.norm2_bytes),
  };
}

Status CanonicalIndexFiles::ReadFloatCoefficients(
    std::uint32_t cid, ArrayView<float> output) const {
  if (!manifest_.has_float_forest() || cid >= float_extents_.size()) {
    return Status::FailedPrecondition(
        "canonical index has no native-float coefficient source");
  }
  const auto bytes = float_cluster_records(cid).coefficients;
  if (output.size() > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      bytes.size() != output.size() * sizeof(float)) {
    return Status::InvalidArgument(
        "native-float coefficient output has the wrong extent");
  }
  for (std::size_t index = 0U; index < output.size(); ++index) {
    output[index] = LoadLittleEndianF32(bytes.data() + index * sizeof(float));
  }
  return Status::Ok();
}

Status CanonicalIndexFiles::ReadFloatNorm2(std::uint32_t cid,
                                           ArrayView<float> output) const {
  if (float_norm2_.size() == 0U || cid >= float_extents_.size()) {
    return Status::FailedPrecondition(
        "canonical index has no native-float norm source");
  }
  const auto bytes = float_cluster_records(cid).norm2;
  if (output.size() > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      bytes.size() != output.size() * sizeof(float)) {
    return Status::InvalidArgument(
        "native-float norm output has the wrong extent");
  }
  for (std::size_t index = 0U; index < output.size(); ++index) {
    output[index] = LoadLittleEndianF32(bytes.data() + index * sizeof(float));
  }
  return Status::Ok();
}

Status CanonicalIndexFiles::ReadFloatSourceNorm2(
    std::uint32_t cid, ArrayView<float> output) const {
  if (!manifest_.has_float_source_norm2() || float_source_norm2_.size() == 0U ||
      cid >= float_extents_.size()) {
    return Status::FailedPrecondition(
        "canonical index has no original-vector norm source");
  }
  const FloatCidExtent& extent = float_extents_[cid];
  if (output.size() > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      extent.source_norm2_bytes != output.size() * sizeof(float)) {
    return Status::InvalidArgument(
        "float source norm output has the wrong extent");
  }
  const auto bytes = Slice(float_source_norm2_, extent.source_norm2_offset,
                           extent.source_norm2_bytes);
  for (std::size_t index = 0U; index < output.size(); ++index) {
    output[index] = LoadLittleEndianF32(bytes.data() + index * sizeof(float));
  }
  return Status::Ok();
}

}  // namespace stlqpp
