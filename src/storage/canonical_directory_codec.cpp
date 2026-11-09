#include "stlqpp/storage/canonical_directory_codec.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/model/topology/parent_louds.h"

namespace stlqpp {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {
    'S', 'T', 'L', 'Q', 'D', 'I', 'R', '1',
};

// Registered M is at most 10 and row counts are u32, so this fits in u64.
std::uint64_t CodesBytes(const ModelShape& shape,
                         const CanonicalCidEntry& entry) {
  const std::uint64_t real_width = static_cast<std::uint64_t>(shape.m) + 4U;
  const std::uint64_t virtual_width = static_cast<std::uint64_t>(shape.m) - 1U;
  const std::uint64_t real = entry.real_count;
  const std::uint64_t virtual_rows = entry.virtual_count;
  return real * real_width + virtual_rows * virtual_width;
}

Status ValidateRange(std::uint64_t offset, std::uint64_t bytes,
                     std::uint64_t minimum_offset, std::uint64_t& previous_end,
                     const char* label) {
  if (offset < minimum_offset || offset != AlignCanonicalRecord(offset) ||
      offset < AlignCanonicalRecord(previous_end) ||
      bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
    return Status::DataLoss(std::string("canonical ") + label +
                            " range is invalid or overlaps a prior CID");
  }
  previous_end = offset + bytes;
  return Status::Ok();
}

Status ValidateEntries(const ModelShape& shape, const IndexCounts& counts,
                       ArrayView<const CanonicalCidEntry> entries,
                       bool has_quantized_forest, bool split_norm_lut,
                       bool completion_order) {
  if (entries.size() != shape.nlist) {
    return Status::DataLoss(
        "canonical directory entry count does not match nlist");
  }
  std::uint64_t real_total = 0U;
  std::uint64_t virtual_total = 0U;
  std::uint64_t real_root_total = 0U;
  std::uint64_t linked_total = 0U;
  std::uint64_t codes_end = kCanonicalPayloadHeaderBytes;
  std::uint64_t parents_end = kCanonicalPayloadHeaderBytes;
  std::uint64_t coefficients_end = kCanonicalPayloadHeaderBytes;
  const auto norm_begin = split_norm_lut ? 0U : kCanonicalPayloadHeaderBytes;
  std::uint64_t norm8_end = norm_begin;
  const std::uint64_t coefficient_header =
      CanonicalCoefficientHeaderBytes(shape.m, 3U);

  for (std::uint32_t cid = 0U; cid < shape.nlist; ++cid) {
    const CanonicalCidEntry& entry = entries[cid];
    if (entry.cid != cid || entry.real_root_count > entry.real_count) {
      return Status::DataLoss(
          "canonical directory CID order or root count is invalid");
    }
    const std::uint64_t nodes =
        static_cast<std::uint64_t>(entry.real_count) + entry.virtual_count;
    if (nodes > std::numeric_limits<std::uint32_t>::max()) {
      return Status::ResourceExhausted(
          "canonical CID exceeds the u32 node domain");
    }
    const std::uint64_t expected_codes = CodesBytes(shape, entry);
    if (entry.codes_bytes != expected_codes ||
        entry.parents_bytes !=
            ParentLoudsWordCount(static_cast<std::uint32_t>(nodes)) * 8U ||
        (has_quantized_forest &&
         (entry.coefficients_bytes < coefficient_header ||
          entry.norm8_bytes != entry.real_count)) ||
        (!has_quantized_forest &&
         (entry.coefficients_offset != 0U || entry.coefficients_bytes != 0U ||
          entry.norm8_offset != 0U || entry.norm8_bytes != 0U))) {
      return Status::DataLoss(
          "canonical directory record span disagrees with its CID counts");
    }
    if (!completion_order) {
      if (const Status status =
              ValidateRange(entry.codes_offset, entry.codes_bytes,
                            kCanonicalPayloadHeaderBytes, codes_end, "codes");
          !status.ok()) {
        return status;
      }
      if (const Status status = ValidateRange(
              entry.parents_offset, entry.parents_bytes,
              kCanonicalPayloadHeaderBytes, parents_end, "parents");
          !status.ok()) {
        return status;
      }
      if (has_quantized_forest) {
        if (const Status status = ValidateRange(
                entry.coefficients_offset, entry.coefficients_bytes,
                kCanonicalPayloadHeaderBytes, coefficients_end, "coefficients");
            !status.ok()) {
          return status;
        }
        if (const Status status =
                ValidateRange(entry.norm8_offset, entry.norm8_bytes, norm_begin,
                              norm8_end, "norm8");
            !status.ok()) {
          return status;
        }
      }
    }
    const std::uint64_t linked = entry.real_count - entry.real_root_count;
    real_total += entry.real_count;
    virtual_total += entry.virtual_count;
    real_root_total += entry.real_root_count;
    linked_total += linked;
  }
  if (completion_order) {
    const auto validate = [&](auto offset, auto bytes, std::uint64_t begin,
                              const char* label) -> Status {
      std::vector<std::uint32_t> order(shape.nlist);
      std::iota(order.begin(), order.end(), 0U);
      std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
        return std::pair{entries[a].*offset, entries[a].*bytes} <
               std::pair{entries[b].*offset, entries[b].*bytes};
      });
      std::uint64_t end = begin;
      for (auto cid : order) {
        const auto& e = entries[cid];
        if (auto status = ValidateRange(e.*offset, e.*bytes, begin, end, label);
            !status.ok())
          return status;
      }
      return Status::Ok();
    };
    for (auto fields : {std::pair{&CanonicalCidEntry::codes_offset,
                                  &CanonicalCidEntry::codes_bytes},
                        std::pair{&CanonicalCidEntry::parents_offset,
                                  &CanonicalCidEntry::parents_bytes}})
      if (auto status = validate(fields.first, fields.second,
                                 kCanonicalPayloadHeaderBytes, "topology");
          !status.ok())
        return status;
    if (has_quantized_forest) {
      if (auto status = validate(&CanonicalCidEntry::coefficients_offset,
                                 &CanonicalCidEntry::coefficients_bytes,
                                 kCanonicalPayloadHeaderBytes, "coefficients");
          !status.ok())
        return status;
      if (auto status =
              validate(&CanonicalCidEntry::norm8_offset,
                       &CanonicalCidEntry::norm8_bytes, norm_begin, "norm8");
          !status.ok())
        return status;
    }
  }
  if (real_total != counts.real || virtual_total != counts.virtual_roots ||
      real_root_total != counts.real_roots ||
      linked_total != counts.linked_real) {
    return Status::DataLoss(
        "canonical directory totals disagree with the manifest");
  }
  return Status::Ok();
}

void StoreEntry(const CanonicalCidEntry& entry,
                std::uint8_t* destination) noexcept {
  StoreLittleEndianU32(entry.cid, destination + 0U);
  StoreLittleEndianU32(entry.real_count, destination + 4U);
  StoreLittleEndianU32(entry.virtual_count, destination + 8U);
  StoreLittleEndianU32(entry.real_root_count, destination + 12U);
  StoreLittleEndianU64(entry.codes_offset, destination + 16U);
  StoreLittleEndianU64(entry.codes_bytes, destination + 24U);
  StoreLittleEndianU64(entry.parents_offset, destination + 32U);
  StoreLittleEndianU64(entry.parents_bytes, destination + 40U);
  StoreLittleEndianU64(entry.coefficients_offset, destination + 48U);
  StoreLittleEndianU64(entry.coefficients_bytes, destination + 56U);
  StoreLittleEndianU64(entry.norm8_offset, destination + 64U);
  StoreLittleEndianU64(entry.norm8_bytes, destination + 72U);
}

CanonicalCidEntry LoadEntry(const std::uint8_t* source) noexcept {
  return {
      .cid = LoadLittleEndianU32(source + 0U),
      .real_count = LoadLittleEndianU32(source + 4U),
      .virtual_count = LoadLittleEndianU32(source + 8U),
      .real_root_count = LoadLittleEndianU32(source + 12U),
      .codes_offset = LoadLittleEndianU64(source + 16U),
      .codes_bytes = LoadLittleEndianU64(source + 24U),
      .parents_offset = LoadLittleEndianU64(source + 32U),
      .parents_bytes = LoadLittleEndianU64(source + 40U),
      .coefficients_offset = LoadLittleEndianU64(source + 48U),
      .coefficients_bytes = LoadLittleEndianU64(source + 56U),
      .norm8_offset = LoadLittleEndianU64(source + 64U),
      .norm8_bytes = LoadLittleEndianU64(source + 72U),
  };
}

}  // namespace

Status CanonicalDirectoryCodec::Encode(
    const ModelShape& shape, const IndexCounts& counts,
    ArrayView<const CanonicalCidEntry> entries,
    std::vector<std::uint8_t>& output, bool has_quantized_forest,
    bool split_norm_lut, bool completion_order) {
  if (const Status status = shape.Validate(); !status.ok()) {
    return status;
  }
  if (counts.real_roots + counts.linked_real != counts.real) {
    return Status::DataLoss(
        "canonical directory manifest counts are inconsistent");
  }
  if (const Status status =
          ValidateEntries(shape, counts, entries, has_quantized_forest,
                          split_norm_lut, completion_order);
      !status.ok()) {
    return status;
  }
  const auto entry_bytes = completion_order
                               ? kCompletionOrderDirectoryEntryBytes
                               : kCanonicalDirectoryEntryBytes;
  const std::uint64_t byte_count =
      kCanonicalDirectoryHeaderBytes +
      static_cast<std::uint64_t>(shape.nlist) * entry_bytes;
  if (byte_count > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted(
        "canonical directory exceeds addressable memory");
  }
  output.assign(static_cast<std::size_t>(byte_count), 0U);
  std::memcpy(output.data(), kMagic.data(), kMagic.size());
  StoreLittleEndianU32(completion_order
                           ? kCompletionOrderDirectoryEncodingVersion
                           : kCanonicalForestEncodingVersion,
                       output.data() + 8U);
  StoreLittleEndianU32(kCanonicalDirectoryHeaderBytes, output.data() + 12U);
  StoreLittleEndianU32(entry_bytes, output.data() + 16U);
  StoreLittleEndianU32(shape.m, output.data() + 20U);
  StoreLittleEndianU32(shape.nlist, output.data() + 24U);
  StoreLittleEndianU64(counts.real, output.data() + 32U);
  StoreLittleEndianU64(counts.virtual_roots, output.data() + 40U);
  StoreLittleEndianU64(counts.real_roots, output.data() + 48U);
  StoreLittleEndianU64(counts.linked_real, output.data() + 56U);
  for (std::uint32_t cid = 0U; cid < shape.nlist; ++cid) {
    StoreEntry(entries[cid], output.data() + kCanonicalDirectoryHeaderBytes +
                                 static_cast<std::size_t>(cid) * entry_bytes);
    if (completion_order)
      StoreLittleEndianU64(entries[cid].float_coefficients_offset,
                           output.data() + kCanonicalDirectoryHeaderBytes +
                               static_cast<std::size_t>(cid) * entry_bytes +
                               80U);
  }
  return Status::Ok();
}

Status CanonicalDirectoryCodec::Decode(
    const std::uint8_t* bytes, std::size_t byte_count,
    const ModelShape& expected_shape, const IndexCounts& expected_counts,
    std::vector<CanonicalCidEntry>& entries, bool has_quantized_forest,
    bool split_norm_lut, bool completion_order) {
  if (const Status status = expected_shape.Validate(); !status.ok()) {
    return status;
  }
  const auto entry_bytes = completion_order
                               ? kCompletionOrderDirectoryEntryBytes
                               : kCanonicalDirectoryEntryBytes;
  const std::uint64_t expected_bytes =
      kCanonicalDirectoryHeaderBytes +
      static_cast<std::uint64_t>(expected_shape.nlist) * entry_bytes;
  if (bytes == nullptr || byte_count != expected_bytes ||
      std::memcmp(bytes, kMagic.data(), kMagic.size()) != 0 ||
      LoadLittleEndianU32(bytes + 8U) !=
          (completion_order ? kCompletionOrderDirectoryEncodingVersion
                            : kCanonicalForestEncodingVersion) ||
      LoadLittleEndianU32(bytes + 12U) != kCanonicalDirectoryHeaderBytes ||
      LoadLittleEndianU32(bytes + 16U) != entry_bytes ||
      LoadLittleEndianU32(bytes + 20U) != expected_shape.m ||
      LoadLittleEndianU32(bytes + 24U) != expected_shape.nlist ||
      LoadLittleEndianU32(bytes + 28U) != 0U ||
      LoadLittleEndianU64(bytes + 32U) != expected_counts.real ||
      LoadLittleEndianU64(bytes + 40U) != expected_counts.virtual_roots ||
      LoadLittleEndianU64(bytes + 48U) != expected_counts.real_roots ||
      LoadLittleEndianU64(bytes + 56U) != expected_counts.linked_real) {
    return Status::DataLoss(
        "canonical directory header or extent disagrees with the manifest");
  }
  std::vector<CanonicalCidEntry> decoded(expected_shape.nlist);
  for (std::uint32_t cid = 0U; cid < expected_shape.nlist; ++cid) {
    decoded[cid] = LoadEntry(bytes + kCanonicalDirectoryHeaderBytes +
                             static_cast<std::size_t>(cid) * entry_bytes);
    if (completion_order)
      decoded[cid].float_coefficients_offset = LoadLittleEndianU64(
          bytes + kCanonicalDirectoryHeaderBytes +
          static_cast<std::size_t>(cid) * entry_bytes + 80U);
  }
  if (const Status status = ValidateEntries(expected_shape, expected_counts,
                                            decoded, has_quantized_forest,
                                            split_norm_lut, completion_order);
      !status.ok()) {
    return status;
  }
  entries = std::move(decoded);
  return Status::Ok();
}

}  // namespace stlqpp
