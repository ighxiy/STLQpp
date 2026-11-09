#include "stlqpp/model/model_shape.h"
#include "stlqpp/storage/canonical_cluster_codec.h"

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/model/codec/coefficient_bit_width.h"
#include "stlqpp/model/topology/parent_louds.h"
#include "stlqpp/storage/canonical_forest_format.h"
#include "../model/topology/parent_louds_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

template <typename Source>
Status ValidateTopologySource(const Source& source, std::uint32_t m) {
  if (m < 2U || m > stlqpp::kMaximumModelLayers) return Status::DataLoss("invalid topology M");
  const std::size_t real_count = source.real_ids.size();
  const std::size_t virtual_count = source.virtual_count;
  if (real_count > std::numeric_limits<std::uint32_t>::max() ||
      virtual_count > std::numeric_limits<std::uint32_t>::max() ||
      real_count + virtual_count > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "canonical cluster exceeds the u32 node domain");
  }
  if (source.real_root_count > real_count ||
      source.parent_local_1based.size() != real_count ||
      source.real_codes_small.size() != real_count * (m - 1U) ||
      source.edge_code0.size() != real_count ||
      source.virtual_codes_small.size() != virtual_count * (m - 1U)) {
    return Status::DataLoss(
        "canonical cluster topology streams disagree with its row counts");
  }
  std::uint32_t previous_parent = 0U;
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    const std::uint32_t parent = source.parent_local_1based[real];
    const std::uint32_t node = source.virtual_count + real + 1U;
    if ((real < source.real_root_count && parent != 0U) ||
        (real >= source.real_root_count &&
         (parent == 0U || parent >= node)) ||
        parent < previous_parent) {
      return Status::DataLoss(
          "canonical cluster topology is not in BFS causal order");
    }
    previous_parent = parent;
  }
  return Status::Ok();
}

template <int M>
Status ValidatePayloadShape(
    const CanonicalClusterPayloadSource<M>& source) {
  const auto m = M == 0 ? source.m : M;
  if (m < 2U || m > stlqpp::kMaximumModelLayers || source.m != m || source.coefficients.m != m)
    return Status::DataLoss("canonical payload M mismatch");
  const std::size_t real_count = source.real_ids.size();
  const std::size_t virtual_count = source.virtual_count;
  if (real_count > std::numeric_limits<std::uint32_t>::max() ||
      virtual_count > std::numeric_limits<std::uint32_t>::max() ||
      real_count + virtual_count >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "canonical cluster exceeds the u32 node domain");
  }
  if (source.real_root_count > real_count ||
      source.parent_local_1based.size() != real_count ||
      source.real_codes_small.size() != real_count * (source.m - 1U) ||
      source.edge_code0.size() != real_count ||
      source.virtual_codes_small.size() != virtual_count * (source.m - 1U)) {
    return Status::DataLoss(
        "canonical cluster source streams disagree with its row counts");
  }
  if (source.coefficients.virtual_root_count != source.virtual_count ||
      source.coefficients.real_root_count != source.real_root_count ||
      source.coefficients.linked_real_count !=
          real_count - source.real_root_count) {
    return Status::DataLoss(
        "canonical coefficient shape disagrees with cluster topology");
  }
  return Status::Ok();
}

template <int M, bool kPrepared>
Status ValidatePayloadSource(
    const CanonicalClusterPayloadSource<M>& source) {
  if (const Status status = ValidatePayloadShape(source); !status.ok()) {
    return status;
  }
  if constexpr (!kPrepared) {
    auto decoded = DecodeScalarInt6Cluster(source.coefficients);
    if (!decoded.ok()) {
      return decoded.status();
    }
  }
  std::uint32_t previous_parent = 0U;
  for (std::uint32_t real = 0U; real < source.real_ids.size(); ++real) {
    const std::uint32_t parent = source.parent_local_1based[real];
    const std::uint32_t node = source.virtual_count + real + 1U;
    if ((real < source.real_root_count && parent != 0U) ||
        (real >= source.real_root_count &&
         (parent == 0U || parent >= node)) ||
        parent < previous_parent) {
      return Status::DataLoss(
          "canonical cluster payload is not in BFS causal order");
    }
    previous_parent = parent;
  }
  return Status::Ok();
}

template <typename T>
Status CheckedProduct(T left, T right, T& result, const char* label) {
  if (left != 0U && right > std::numeric_limits<T>::max() / left) {
    return Status::ResourceExhausted(label);
  }
  result = left * right;
  return Status::Ok();
}

template <typename Source>
Status EncodeCodes(const Source& source,
                   std::vector<std::uint8_t>& output, std::uint32_t m) {
  std::size_t real_small_bytes = 0U;
  std::size_t virtual_small_bytes = 0U;
  if (const Status status = CheckedProduct(
          source.real_ids.size(), static_cast<std::size_t>(m - 1U),
          real_small_bytes, "canonical real code extent overflows size_t");
      !status.ok()) {
    return status;
  }
  if (const Status status = CheckedProduct(
          static_cast<std::size_t>(source.virtual_count),
          static_cast<std::size_t>(m - 1U), virtual_small_bytes,
          "canonical virtual code extent overflows size_t");
      !status.ok()) {
    return status;
  }
  const std::size_t real_id_bytes =
      source.real_ids.size() * sizeof(std::uint32_t);
  if (real_id_bytes > std::numeric_limits<std::size_t>::max() -
                          real_small_bytes ||
      real_id_bytes + real_small_bytes >
          std::numeric_limits<std::size_t>::max() - source.edge_code0.size() ||
      real_id_bytes + real_small_bytes + source.edge_code0.size() >
          std::numeric_limits<std::size_t>::max() - virtual_small_bytes) {
    return Status::ResourceExhausted(
        "canonical code record exceeds addressable memory");
  }
  output.resize(real_id_bytes + real_small_bytes + source.edge_code0.size() +
                virtual_small_bytes);
  std::size_t cursor = 0U;
  for (std::uint32_t id : source.real_ids) {
    StoreLittleEndianU32(id, output.data() + cursor);
    cursor += sizeof(std::uint32_t);
  }
  if (real_small_bytes != 0U) {
    std::memcpy(output.data() + cursor, source.real_codes_small.data(),
                real_small_bytes);
    cursor += real_small_bytes;
  }
  if (!source.edge_code0.empty()) {
    std::memcpy(output.data() + cursor, source.edge_code0.data(),
                source.edge_code0.size());
    cursor += source.edge_code0.size();
  }
  if (virtual_small_bytes != 0U) {
    std::memcpy(output.data() + cursor, source.virtual_codes_small.data(),
                virtual_small_bytes);
  }
  return Status::Ok();
}

template <typename Source>
Status EncodeParents(const Source& source,
                     std::vector<std::uint8_t>& output) {
  const std::size_t node_count =
      static_cast<std::size_t>(source.virtual_count) + source.real_ids.size();
  std::vector<std::uint32_t> all_parents(node_count, 0U);
  std::copy(source.parent_local_1based.begin(),
            source.parent_local_1based.end(),
            all_parents.begin() + source.virtual_count);
  std::vector<std::uint64_t> words;
  if (const Status status =
          internal::EncodeValidatedParentLouds(all_parents, words);
      !status.ok()) {
    return status;
  }
  output.resize(words.size() * sizeof(std::uint64_t));
  for (std::size_t index = 0U; index < words.size(); ++index) {
    StoreLittleEndianU64(words[index],
                         output.data() + index * sizeof(std::uint64_t));
  }
  return Status::Ok();
}

Status PackCodeLengths(ArrayView<const std::uint8_t> lengths,
                       std::uint8_t* output) {
  if (lengths.empty() || lengths.size() > 255U ||
      (lengths.size() & 1U) == 0U) {
    return Status::DataLoss(
        "canonical coefficient alphabet is invalid");
  }
  const std::size_t packed_bytes = static_cast<std::size_t>(
      CanonicalHuffmanLengthBytes(
          static_cast<std::uint32_t>(lengths.size())));
  std::fill(output, output + packed_bytes, 0U);
  std::uint64_t pending = 0U;
  std::uint32_t pending_bits = 0U;
  std::size_t cursor = 0U;
  for (const std::uint8_t length : lengths) {
    if (length > 32U) {
      return Status::DataLoss(
          "canonical Huffman code length exceeds the 32-bit format");
    }
    pending = (pending << kCanonicalHuffmanLengthBits) | length;
    pending_bits += kCanonicalHuffmanLengthBits;
    while (pending_bits >= 8U) {
      const std::uint32_t shift = pending_bits - 8U;
      output[cursor++] =
          static_cast<std::uint8_t>((pending >> shift) & 0xFFU);
      pending_bits -= 8U;
      if (pending_bits == 0U) {
        pending = 0U;
      } else {
        pending &= (std::uint64_t{1U} << pending_bits) - 1U;
      }
    }
  }
  if (pending_bits != 0U) {
    output[cursor] =
        static_cast<std::uint8_t>(pending << (8U - pending_bits));
  }
  return Status::Ok();
}

Status UnpackCodeLengths(ArrayView<const std::uint8_t> packed,
                         std::uint32_t alphabet_size,
                         std::vector<std::uint8_t>& lengths) {
  const std::uint64_t expected =
      CanonicalHuffmanLengthBytes(alphabet_size);
  const std::uint32_t padding_bits = static_cast<std::uint32_t>(
      expected * 8U -
      static_cast<std::uint64_t>(alphabet_size) *
          kCanonicalHuffmanLengthBits);
  const std::uint8_t padding_mask = padding_bits == 0U
      ? 0U
      : static_cast<std::uint8_t>((1U << padding_bits) - 1U);
  if (alphabet_size == 0U || alphabet_size > 255U ||
      (alphabet_size & 1U) == 0U || packed.size() != expected ||
      (!packed.empty() && (packed.back() & padding_mask) != 0U)) {
    return Status::DataLoss(
        "canonical packed Huffman lengths have an invalid extent or padding");
  }
  lengths.assign(alphabet_size, 0U);
  std::size_t bit = 0U;
  for (std::uint8_t& length : lengths) {
    std::uint8_t value = 0U;
    for (std::uint32_t consumed = 0U;
         consumed < kCanonicalHuffmanLengthBits; ++consumed, ++bit) {
      value = static_cast<std::uint8_t>(
          (value << 1U) |
          ((packed[bit / 8U] >> (7U - bit % 8U)) & 1U));
    }
    if (value > 32U) {
      return Status::DataLoss(
          "canonical packed Huffman code length exceeds 32 bits");
    }
    length = value;
  }
  return Status::Ok();
}

template <typename Source>
Status EncodeCoefficients(const Source& source,
                          std::vector<std::uint8_t>& output, std::uint32_t m) {
  if (source.root_scales.size() != m ||
      source.linkage_scales.size() != m) {
    return Status::DataLoss(
        "canonical coefficient scale extent disagrees with m");
  }
  for (const float scale : source.root_scales) {
    if (!std::isfinite(scale)) {
      return Status::DataLoss("canonical root coefficient scale is invalid");
    }
  }
  for (const float scale : source.linkage_scales) {
    if (!std::isfinite(scale)) {
      return Status::DataLoss(
          "canonical linkage coefficient scale is invalid");
    }
  }
  if (source.root.code_lengths.size() !=
          source.linkage.code_lengths.size() ||
      source.root.code_lengths.empty() ||
      source.root.code_lengths.size() > 255U ||
      (source.root.code_lengths.size() & 1U) == 0U) {
    return Status::DataLoss(
        "canonical coefficient roles require one shared odd alphabet");
  }
  if (source.root.payload.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      source.linkage.payload.size() >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "canonical coefficient payload exceeds the u32 record extent");
  }
  const std::size_t length_bytes = static_cast<std::size_t>(
      CanonicalHuffmanLengthBytes(static_cast<std::uint32_t>(
          source.root.code_lengths.size())));
  const std::size_t header_bytes = static_cast<std::size_t>(
      CanonicalCoefficientHeaderBytes(m, static_cast<std::uint32_t>(
          source.root.code_lengths.size())));
  if (source.root.payload.size() >
          std::numeric_limits<std::size_t>::max() - header_bytes ||
      source.linkage.payload.size() >
          std::numeric_limits<std::size_t>::max() - header_bytes -
              source.root.payload.size()) {
    return Status::ResourceExhausted(
        "canonical coefficient record exceeds addressable memory");
  }
  output.resize(header_bytes + source.root.payload.size() +
                source.linkage.payload.size());
  StoreLittleEndianU32(static_cast<std::uint32_t>(source.root.payload.size()),
                       output.data());
  StoreLittleEndianU32(
      static_cast<std::uint32_t>(source.linkage.payload.size()),
      output.data() + 4U);
  std::size_t cursor = 8U;
  for (float scale : source.root_scales) {
    StoreLittleEndianF32(scale, output.data() + cursor);
    cursor += sizeof(float);
  }
  for (float scale : source.linkage_scales) {
    StoreLittleEndianF32(scale, output.data() + cursor);
    cursor += sizeof(float);
  }
  if (const Status status = PackCodeLengths(
          source.root.code_lengths, output.data() + cursor);
      !status.ok()) {
    return status;
  }
  cursor += length_bytes;
  if (const Status status = PackCodeLengths(
          source.linkage.code_lengths, output.data() + cursor);
      !status.ok()) {
    return status;
  }
  cursor += length_bytes;
  if (!source.root.payload.empty()) {
    std::memcpy(output.data() + cursor, source.root.payload.data(),
                source.root.payload.size());
    cursor += source.root.payload.size();
  }
  if (!source.linkage.payload.empty()) {
    std::memcpy(output.data() + cursor, source.linkage.payload.data(),
                source.linkage.payload.size());
  }
  return Status::Ok();
}

Status DecodeCodesRuntime(std::uint32_t m,
                   std::uint32_t real_count,
                   std::uint32_t virtual_count,
                   ArrayView<const std::uint8_t> bytes,
                   std::vector<std::uint32_t>& real_ids,
                   std::vector<std::uint8_t>& real_codes_small,
                   std::vector<std::uint8_t>& edge_code0,
                   std::vector<std::uint8_t>& virtual_codes_small) {
  const std::uint64_t expected64 =
      static_cast<std::uint64_t>(real_count) * (m + 4U) +
      static_cast<std::uint64_t>(virtual_count) * (m - 1U);
  if (expected64 > std::numeric_limits<std::size_t>::max() ||
      bytes.size() != static_cast<std::size_t>(expected64)) {
    return Status::DataLoss(
        "canonical code record length disagrees with CID counts");
  }
  real_ids.resize(real_count);
  std::size_t cursor = 0U;
  for (std::uint32_t row = 0U; row < real_count; ++row) {
    real_ids[row] = LoadLittleEndianU32(bytes.data() + cursor);
    cursor += sizeof(std::uint32_t);
  }
  const std::size_t real_small =
      static_cast<std::size_t>(real_count) * (m - 1U);
  if (real_small != 0U) {
    real_codes_small.assign(bytes.data() + cursor,
                            bytes.data() + cursor + real_small);
  }
  cursor += real_small;
  if (real_count != 0U) {
    edge_code0.assign(bytes.data() + cursor,
                      bytes.data() + cursor + real_count);
  }
  cursor += real_count;
  const std::size_t virtual_small =
      static_cast<std::size_t>(virtual_count) * (m - 1U);
  if (virtual_small != 0U) {
    virtual_codes_small.assign(bytes.data() + cursor,
                               bytes.data() + cursor + virtual_small);
  }
  return Status::Ok();
}

template <int M>
Status DecodeCodes(std::uint32_t real_count,
                   std::uint32_t virtual_count,
                   ArrayView<const std::uint8_t> bytes,
                   std::vector<std::uint32_t>& real_ids,
                   std::vector<std::uint8_t>& real_codes_small,
                   std::vector<std::uint8_t>& edge_code0,
                   std::vector<std::uint8_t>& virtual_codes_small) {
  return DecodeCodesRuntime(M, real_count, virtual_count, bytes, real_ids,
                            real_codes_small, edge_code0,
                            virtual_codes_small);
}

Status DecodeParentWords(std::uint32_t node_count,
                         std::uint32_t root_count,
                         ArrayView<const std::uint8_t> bytes,
                         std::vector<std::uint64_t>& words) {
  const std::uint64_t expected = ParentLoudsWordCount(node_count) * 8U;
  if (expected > std::numeric_limits<std::size_t>::max() ||
      bytes.size() != static_cast<std::size_t>(expected)) {
    return Status::DataLoss(
        "canonical parent record length disagrees with CID counts");
  }
  words.resize(static_cast<std::size_t>(ParentLoudsWordCount(node_count)));
  for (std::size_t index = 0U; index < words.size(); ++index) {
    words[index] = LoadLittleEndianU64(
        bytes.data() + index * sizeof(std::uint64_t));
  }
  return ValidateParentLouds(words, node_count, root_count);
}

Status DecodeParents(std::uint32_t real_count,
                     std::uint32_t virtual_count,
                     std::uint32_t real_root_count,
                     ArrayView<const std::uint8_t> bytes,
                     std::vector<std::uint32_t>& real_parents) {
  const std::uint32_t node_count = real_count + virtual_count;
  std::vector<std::uint64_t> words;
  if (const Status status = DecodeParentWords(
          node_count, virtual_count + real_root_count, bytes, words);
      !status.ok()) {
    return status;
  }
  ParentLoudsSequentialDecoder decoder(words);
  decoder.InitializeAfterValidatedRootPrefix(
      virtual_count + real_root_count);
  real_parents.assign(real_count, 0U);
  for (std::uint32_t real = real_root_count; real < real_count; ++real) {
    real_parents[real] = decoder.NextParent1Based();
  }
  return Status::Ok();
}

struct DecodedCoefficientRecordView final {
  std::vector<float> root_scales;
  std::vector<float> linkage_scales;
  std::vector<std::uint8_t> root_lengths;
  std::vector<std::uint8_t> linkage_lengths;
  ArrayView<const std::uint8_t> root_payload;
  ArrayView<const std::uint8_t> linkage_payload;
};

StatusOr<DecodedCoefficientRecordView> DecodeCoefficientRecordView(
    std::uint32_t m, ArrayView<const std::uint8_t> bytes) {
  if (m < 2U || m > stlqpp::kMaximumModelLayers) {
    return Status::InvalidArgument("runtime coefficient record M must be 2..24");
  }
  const std::size_t fixed_header_bytes = 8U + 8U * m;
  if (bytes.size() < fixed_header_bytes) {
    return Status::DataLoss("canonical coefficient record is truncated");
  }
  const std::uint32_t root_payload_bytes = LoadLittleEndianU32(bytes.data());
  const std::uint32_t linkage_payload_bytes =
      LoadLittleEndianU32(bytes.data() + 4U);
  if (static_cast<std::uint64_t>(fixed_header_bytes) + root_payload_bytes +
          linkage_payload_bytes > bytes.size()) {
    return Status::DataLoss(
        "canonical coefficient payload sizes do not span their record");
  }
  const std::uint64_t packed_pair_bytes =
      bytes.size() - fixed_header_bytes - root_payload_bytes -
      linkage_payload_bytes;
  std::uint32_t alphabet_size = 0U;
  for (std::uint32_t bits = kMinimumCoefficientBits;
       bits <= kMaximumCoefficientBits; ++bits) {
    const std::uint32_t candidate = CoefficientAlphabetFromBits(bits);
    if (2U * CanonicalHuffmanLengthBytes(candidate) == packed_pair_bytes) {
      if (alphabet_size != 0U) {
        return Status::DataLoss(
            "canonical coefficient alphabet extent is ambiguous");
      }
      alphabet_size = candidate;
    }
  }
  if (alphabet_size == 0U) {
    return Status::DataLoss(
        "canonical coefficient alphabet extent is unsupported");
  }
  const std::size_t length_bytes = static_cast<std::size_t>(
      CanonicalHuffmanLengthBytes(alphabet_size));
  DecodedCoefficientRecordView output;
  output.root_scales.resize(m);
  output.linkage_scales.resize(m);
  std::size_t cursor = 8U;
  for (float& scale : output.root_scales) {
    scale = LoadLittleEndianF32(bytes.data() + cursor);
    cursor += sizeof(float);
  }
  for (float& scale : output.linkage_scales) {
    scale = LoadLittleEndianF32(bytes.data() + cursor);
    cursor += sizeof(float);
  }
  if (const Status status = UnpackCodeLengths(
          {bytes.data() + cursor, length_bytes}, alphabet_size,
          output.root_lengths);
      !status.ok()) {
    return status;
  }
  cursor += length_bytes;
  if (const Status status = UnpackCodeLengths(
          {bytes.data() + cursor, length_bytes}, alphabet_size,
          output.linkage_lengths);
      !status.ok()) {
    return status;
  }
  cursor += length_bytes;
  output.root_payload = {bytes.data() + cursor, root_payload_bytes};
  cursor += root_payload_bytes;
  output.linkage_payload = {bytes.data() + cursor, linkage_payload_bytes};
  return output;
}

template <int M>
StatusOr<DecodedScalarInt6Cluster<M>> DecodeCoefficients(
    std::uint32_t virtual_count,
    std::uint32_t real_root_count,
    std::uint32_t linked_count,
    ArrayView<const std::uint8_t> bytes) {
  auto record = DecodeCoefficientRecordView(M, bytes);
  if (!record.ok()) return record.status();
  return DecodeScalarInt6Cluster<M>({
      .virtual_root_count = virtual_count,
      .real_root_count = real_root_count,
      .linked_real_count = linked_count,
      .root_scales = record.value().root_scales,
      .linkage_scales = record.value().linkage_scales,
      .root = {.code_lengths = record.value().root_lengths,
               .payload = record.value().root_payload},
      .linkage = {
          .code_lengths = record.value().linkage_lengths,
          .payload = record.value().linkage_payload,
      },
  });
}

Status ValidateNorm8Codebook(Norm8CodebookView norm8) {
  if (norm8.centers == nullptr || norm8.center_count == 0U ||
      norm8.center_count > 256U) {
    return Status::DataLoss(
        "canonical norm8 codebook must contain 1..256 centers");
  }
  for (std::uint32_t index = 0U; index < norm8.center_count; ++index) {
    if (!std::isfinite(norm8.centers[index]) || norm8.centers[index] < 0.0F) {
      return Status::DataLoss(
          "canonical norm8 codebook contains an invalid center");
    }
  }
  return Status::Ok();
}

}  // namespace


Status EncodeCanonicalTopologyRuntime(
    const RuntimeCanonicalClusterTopologySource& source,
    CanonicalClusterRecords& output) {
  if (source.m < 2U || source.m > stlqpp::kMaximumModelLayers)
    return Status::InvalidArgument("runtime canonical topology M must be 2..24");
  if (const auto status = ValidateTopologySource(source, source.m); !status.ok()) return status;
  CanonicalClusterRecords records;
  if (const auto status = EncodeCodes(source, records.codes, source.m); !status.ok()) return status;
  if (const auto status = EncodeParents(source, records.parents); !status.ok()) return status;
  output = std::move(records);
  return Status::Ok();
}

Status EncodeCanonicalClusterRuntime(
    const RuntimeCanonicalClusterTopologySource& source,
    const RuntimeScalarInt6ClusterView& coefficients,
    ArrayView<const std::uint8_t> real_norm8, Norm8CodebookView norm8,
    CanonicalClusterRecords& output) {
  if (coefficients.m != source.m ||
      coefficients.virtual_root_count != source.virtual_count ||
      coefficients.real_root_count != source.real_root_count ||
      source.real_root_count > source.real_ids.size() ||
      coefficients.linked_real_count != source.real_ids.size() - source.real_root_count ||
      real_norm8.size() != source.real_ids.size())
    return Status::DataLoss("runtime coefficients/norms disagree with topology");
  if (const auto status = ValidateNorm8Codebook(norm8); !status.ok()) return status;
  for (const auto label : real_norm8) {
    if (label >= norm8.center_count)
      return Status::DataLoss("runtime norm label exceeds codebook");
  }
  auto decoded = DecodeScalarInt6ClusterRuntime(coefficients);
  if (!decoded.ok()) return decoded.status();
  CanonicalClusterRecords records;
  if (const auto status = EncodeCanonicalTopologyRuntime(source, records); !status.ok()) return status;
  if (const auto status = EncodeCoefficients(coefficients, records.coefficients, source.m);
      !status.ok()) return status;
  records.norm8.assign(real_norm8.begin(), real_norm8.end());
  output = std::move(records);
  return Status::Ok();
}

StatusOr<RuntimeDecodedCanonicalTopology>
DecodeCanonicalNativeBfsTopologyRuntime(
    std::uint32_t m, std::uint32_t cid, std::uint32_t real_count,
    std::uint32_t virtual_count, std::uint32_t real_root_count,
    ArrayView<const std::uint8_t> codes,
    ArrayView<const std::uint8_t> parents) {
  if (m < 2U || m > stlqpp::kMaximumModelLayers) {
    return Status::InvalidArgument("runtime canonical topology M must be 2..24");
  }
  if (real_root_count > real_count ||
      static_cast<std::uint64_t>(real_count) + virtual_count >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::DataLoss("canonical cluster counts are invalid");
  }
  RuntimeDecodedCanonicalTopology output;
  output.m = m;
  output.cid = cid;
  output.virtual_count = virtual_count;
  output.real_root_count = real_root_count;
  if (const Status status = DecodeCodesRuntime(
          m, real_count, virtual_count, codes, output.real_ids,
          output.real_codes_small, output.edge_code0,
          output.virtual_codes_small);
      !status.ok()) {
    return status;
  }
  if (const Status status = DecodeParentWords(
          real_count + virtual_count, virtual_count + real_root_count,
          parents, output.parent_louds_words);
      !status.ok()) {
    return status;
  }
  return output;
}

StatusOr<RuntimeDecodedCanonicalCluster>
DecodeCanonicalNativeBfsClusterRuntime(
    std::uint32_t m, std::uint32_t cid, std::uint32_t real_count,
    std::uint32_t virtual_count, std::uint32_t real_root_count,
    CanonicalClusterRecordViews records, Norm8CodebookView norm8, bool decode_norms) {
  auto topology = DecodeCanonicalNativeBfsTopologyRuntime(
      m, cid, real_count, virtual_count, real_root_count,
      records.codes, records.parents);
  if (!topology.ok()) return topology.status();
  if (decode_norms) {
    if (const Status status = ValidateNorm8Codebook(norm8); !status.ok()) {
      return status;
    }
    if (records.norm8.size() != real_count) {
      return Status::DataLoss(
          "canonical norm8 record length disagrees with real count");
    }
    if (norm8.center_count != 256U) {
      for (const std::uint8_t label : records.norm8) {
        if (label >= norm8.center_count) {
          return Status::DataLoss(
              "cluster norm8 code exceeds its center table");
        }
      }
    }
  }
  auto record = DecodeCoefficientRecordView(m, records.coefficients);
  if (!record.ok()) return record.status();
  auto coefficients = DecodeScalarInt6ClusterRuntime({
      .m = m,
      .virtual_root_count = virtual_count,
      .real_root_count = real_root_count,
      .linked_real_count = real_count - real_root_count,
      .root_scales = record.value().root_scales,
      .linkage_scales = record.value().linkage_scales,
      .root = {.code_lengths = record.value().root_lengths,
               .payload = record.value().root_payload},
      .linkage = {.code_lengths = record.value().linkage_lengths,
                  .payload = record.value().linkage_payload},
  });
  if (!coefficients.ok()) return coefficients.status();
  RuntimeDecodedCanonicalCluster output;
  output.topology = std::move(topology).value();
  output.coefficients = std::move(coefficients).value();
  if (decode_norms) output.real_norm8.assign(records.norm8.begin(), records.norm8.end());
  else output.real_norm8.resize(real_count, 0U);
  return output;
}

template <int M>
Status CanonicalClusterCodec<M>::EncodeTopology(
    const CanonicalClusterTopologySource<M>& source,
    CanonicalClusterRecords& output) {
  if (const Status status = ValidateTopologySource(source, M == 0 ? source.m : M); !status.ok()) {
    return status;
  }
  const CanonicalClusterPayloadSource<M> topology = {
      .m = source.m,
      .cid = source.cid,
      .virtual_count = source.virtual_count,
      .real_root_count = source.real_root_count,
      .real_ids = source.real_ids,
      .parent_local_1based = source.parent_local_1based,
      .real_codes_small = source.real_codes_small,
      .edge_code0 = source.edge_code0,
      .virtual_codes_small = source.virtual_codes_small,
      .coefficients = {},
  };
  output = {};
  if (const Status status = EncodeCodes(topology, output.codes, source.m); !status.ok()) {
    return status;
  }
  return EncodeParents(topology, output.parents);
}

template <int M>
StatusOr<TrustedClusterView<M>> DecodedCanonicalCluster<M>::trusted_view(
    Norm8CodebookView norm8) const {
  return TrustedClusterView<M>::Create({
      .cid = cid_,
      .real_count = static_cast<std::uint32_t>(real_ids_.size()),
      .virtual_count = virtual_count_,
      .real_root_count = real_root_count_,
      .real_ids = real_ids_.data(),
      .parent_local_1based = parent_local_1based_.data(),
      .real_codes_small = real_codes_small_.data(),
      .edge_code0 = edge_code0_.data(),
      .virtual_codes_small = virtual_codes_small_.data(),
      .coefficients_layer_major = coefficients_.coefficients().data(),
      .root_scales = coefficients_.root_scales().data(),
      .linkage_scales = coefficients_.linkage_scales().data(),
      .real_norm8 = real_norm8_.data(),
      .norm8_centers = norm8.centers,
      .norm8_center_count = norm8.center_count,
  });
}

template <int M>
StatusOr<TrustedBfsClusterView<M>>
DecodedCanonicalCluster<M>::trusted_bfs_view(Norm8CodebookView norm8) const {
  const MappedBfsClusterView source = {
      .cid = cid_,
      .real_count = static_cast<std::uint32_t>(real_ids_.size()),
      .virtual_count = virtual_count_,
      .real_root_count = real_root_count_,
      .real_ids = real_ids_.data(),
      .parent_louds_words = parent_louds_words_,
      .real_codes_small = real_codes_small_.data(),
      .edge_code0 = edge_code0_.data(),
      .virtual_codes_small = virtual_codes_small_.data(),
      .coefficients_layer_major = coefficients_.coefficients().data(),
      .root_scales = coefficients_.root_scales().data(),
      .linkage_scales = coefficients_.linkage_scales().data(),
      .real_norm8 = real_norm8_.data(),
      .norm8_centers = norm8.centers,
      .norm8_center_count = norm8.center_count,
  };
  return louds_provenance_.valid()
      ? TrustedBfsClusterView<M>::CreateAfterValidatedLouds(source)
      : TrustedBfsClusterView<M>::Create(source);
}

template <int M>
Status CanonicalClusterCodec<M>::Encode(
    const CanonicalClusterSource<M>& source,
    CanonicalClusterRecords& output) {
  if (source.real_norm8.size() != source.real_ids.size()) {
    return Status::DataLoss(
        "canonical norm8 labels disagree with the real-row count");
  }
  if (const Status status = ValidateNorm8Codebook(source.norm8);
      !status.ok()) {
    return status;
  }
  if (source.norm8.center_count != 256U) {
    for (const std::uint8_t label : source.real_norm8) {
      if (label >= source.norm8.center_count) {
        return Status::DataLoss(
            "canonical norm8 label exceeds the global center table");
      }
    }
  }
  const CanonicalClusterPayloadSource<M> payload = {
      .m = source.m,
      .cid = source.cid,
      .virtual_count = source.virtual_count,
      .real_root_count = source.real_root_count,
      .real_ids = source.real_ids,
      .parent_local_1based = source.parent_local_1based,
      .real_codes_small = source.real_codes_small,
      .edge_code0 = source.edge_code0,
      .virtual_codes_small = source.virtual_codes_small,
      .coefficients = source.coefficients,
  };
  CanonicalClusterRecords encoded;
  if (const Status status = EncodePayload(payload, encoded); !status.ok()) {
    return status;
  }
  encoded.norm8.assign(source.real_norm8.begin(), source.real_norm8.end());
  output = std::move(encoded);
  return Status::Ok();
}

template <int M>
Status CanonicalClusterCodec<M>::EncodePayload(
    const CanonicalClusterPayloadSource<M>& source,
    CanonicalClusterRecords& output) {
  if (const Status status = ValidatePayloadSource<M, false>(source);
      !status.ok()) {
    return status;
  }
  CanonicalClusterRecords encoded;
  if (const Status status = EncodeCodes(source, encoded.codes, source.m); !status.ok()) {
    return status;
  }
  if (const Status status = EncodeParents(source, encoded.parents);
      !status.ok()) {
    return status;
  }
  if (const Status status =
          EncodeCoefficients(source.coefficients, encoded.coefficients, source.m);
      !status.ok()) {
    return status;
  }
  output = std::move(encoded);
  return Status::Ok();
}

template <int M>
Status CanonicalClusterCodec<M>::EncodePreparedPayload(
    const CanonicalClusterPayloadSource<M>& source,
    CanonicalClusterRecords& output) {
  if (const Status status = ValidatePayloadSource<M, true>(source);
      !status.ok()) {
    return status;
  }
  CanonicalClusterRecords encoded;
  if (const Status status = EncodeCodes(source, encoded.codes, source.m); !status.ok()) {
    return status;
  }
  if (const Status status = EncodeParents(source, encoded.parents);
      !status.ok()) {
    return status;
  }
  if (const Status status =
          EncodeCoefficients(source.coefficients, encoded.coefficients, source.m);
      !status.ok()) {
    return status;
  }
  output = std::move(encoded);
  return Status::Ok();
}

template <int M>
Status CanonicalClusterCodec<M>::EncodeCoefficientRecord(
    const ScalarInt6ClusterView<M>& source,
    std::vector<std::uint8_t>& output) {
  return EncodeCoefficients(source, output, source.m);
}

template <int M>
StatusOr<DecodedCanonicalCluster<M>> CanonicalClusterCodec<M>::Decode(
    std::uint32_t cid,
    std::uint32_t real_count,
    std::uint32_t virtual_count,
    std::uint32_t real_root_count,
    CanonicalClusterRecordViews records,
    Norm8CodebookView norm8) {
  if (real_root_count > real_count ||
      static_cast<std::uint64_t>(real_count) + virtual_count >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::DataLoss("canonical cluster counts are invalid");
  }
  if (const Status status = ValidateNorm8Codebook(norm8); !status.ok()) {
    return status;
  }
  if (records.norm8.size() != real_count) {
    return Status::DataLoss(
        "canonical norm8 record length disagrees with real count");
  }

  DecodedCanonicalCluster<M> output;
  output.cid_ = cid;
  output.virtual_count_ = virtual_count;
  output.real_root_count_ = real_root_count;
  if (const Status status = DecodeCodes<M>(
          real_count, virtual_count, records.codes, output.real_ids_,
          output.real_codes_small_, output.edge_code0_,
          output.virtual_codes_small_);
      !status.ok()) {
    return status;
  }
  if (const Status status = DecodeParents(
          real_count, virtual_count, real_root_count, records.parents,
          output.parent_local_1based_);
      !status.ok()) {
    return status;
  }
  auto coefficients = DecodeCoefficients<M>(
      virtual_count, real_root_count, real_count - real_root_count,
      records.coefficients);
  if (!coefficients.ok()) {
    return coefficients.status();
  }
  output.coefficients_ = std::move(coefficients).value();
  output.real_norm8_.assign(records.norm8.begin(), records.norm8.end());
  auto trusted = output.trusted_view(norm8);
  if (!trusted.ok()) {
    return trusted.status();
  }
  return output;
}

template <int M>
StatusOr<DecodedCanonicalTopology<M>>
CanonicalClusterCodec<M>::DecodeTopology(
    std::uint32_t cid,
    std::uint32_t real_count,
    std::uint32_t virtual_count,
    std::uint32_t real_root_count,
    ArrayView<const std::uint8_t> codes,
    ArrayView<const std::uint8_t> parents, std::uint32_t m) {
  if (m < 2U || m > stlqpp::kMaximumModelLayers || (M != 0 && m != M))
    return Status::DataLoss("canonical topology M mismatch");
  if (real_root_count > real_count ||
      static_cast<std::uint64_t>(real_count) + virtual_count >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::DataLoss("canonical cluster counts are invalid");
  }
  DecodedCanonicalTopology<M> output;
  output.cid_ = cid;
  output.virtual_count_ = virtual_count;
  output.real_root_count_ = real_root_count;
  if (const Status status = DecodeCodesRuntime(
          m, real_count, virtual_count, codes, output.real_ids_,
          output.real_codes_small_, output.edge_code0_,
          output.virtual_codes_small_);
      !status.ok()) {
    return status;
  }
  if (const Status status = DecodeParents(
          real_count, virtual_count, real_root_count, parents,
          output.parent_local_1based_);
      !status.ok()) {
    return status;
  }
  return output;
}

template <int M>
StatusOr<DecodedCanonicalTopology<M>>
CanonicalClusterCodec<M>::DecodeNativeBfsTopology(
    std::uint32_t cid,
    std::uint32_t real_count,
    std::uint32_t virtual_count,
    std::uint32_t real_root_count,
    ArrayView<const std::uint8_t> codes,
    ArrayView<const std::uint8_t> parents) {
  if (real_root_count > real_count ||
      static_cast<std::uint64_t>(real_count) + virtual_count >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::DataLoss("canonical cluster counts are invalid");
  }
  DecodedCanonicalTopology<M> output;
  output.cid_ = cid;
  output.virtual_count_ = virtual_count;
  output.real_root_count_ = real_root_count;
  if (const Status status = DecodeCodes<M>(
          real_count, virtual_count, codes, output.real_ids_,
          output.real_codes_small_, output.edge_code0_,
          output.virtual_codes_small_);
      !status.ok()) {
    return status;
  }
  if (const Status status = DecodeParentWords(
          real_count + virtual_count,
          virtual_count + real_root_count,
          parents, output.parent_louds_words_);
      !status.ok()) {
    return status;
  }
  output.louds_provenance_.Mark();
  return output;
}

template <int M>
StatusOr<DecodedCanonicalCluster<M>>
CanonicalClusterCodec<M>::DecodeNativeBfs(
    std::uint32_t cid,
    std::uint32_t real_count,
    std::uint32_t virtual_count,
    std::uint32_t real_root_count,
    CanonicalClusterRecordViews records,
    Norm8CodebookView norm8, bool decode_norms) {
  if (real_root_count > real_count ||
      static_cast<std::uint64_t>(real_count) + virtual_count >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::DataLoss("canonical cluster counts are invalid");
  }
  if (decode_norms) {
    if (const Status status = ValidateNorm8Codebook(norm8); !status.ok()) {
      return status;
    }
    if (records.norm8.size() != real_count) {
      return Status::DataLoss(
          "canonical norm8 record length disagrees with real count");
    }

  }
  DecodedCanonicalCluster<M> output;
  output.cid_ = cid;
  output.virtual_count_ = virtual_count;
  output.real_root_count_ = real_root_count;
  if (const Status status = DecodeCodes<M>(
          real_count, virtual_count, records.codes, output.real_ids_,
          output.real_codes_small_, output.edge_code0_,
          output.virtual_codes_small_);
      !status.ok()) {
    return status;
  }
  if (const Status status = DecodeParentWords(
          real_count + virtual_count,
          virtual_count + real_root_count,
          records.parents, output.parent_louds_words_);
      !status.ok()) {
    return status;
  }
  output.louds_provenance_.Mark();
  auto coefficients = DecodeCoefficients<M>(
      virtual_count, real_root_count, real_count - real_root_count,
      records.coefficients);
  if (!coefficients.ok()) {
    return coefficients.status();
  }
  output.coefficients_ = std::move(coefficients).value();
  if (decode_norms) output.real_norm8_.assign(records.norm8.begin(), records.norm8.end());
  else output.real_norm8_.resize(real_count, 0U);
  auto trusted = output.trusted_bfs_view(norm8);
  if (!trusted.ok()) {
    return trusted.status();
  }
  return output;
}

template class DecodedCanonicalTopology<5>;
template class DecodedCanonicalTopology<12>;
template class DecodedCanonicalTopology<6>;
template class DecodedCanonicalTopology<10>;
template class DecodedCanonicalCluster<5>;
template class DecodedCanonicalCluster<12>;
template class DecodedCanonicalCluster<6>;
template class DecodedCanonicalCluster<10>;
template Status CanonicalClusterCodec<0>::EncodeTopology(const CanonicalClusterTopologySource<0>&, CanonicalClusterRecords&);
template Status CanonicalClusterCodec<0>::EncodePayload(const CanonicalClusterPayloadSource<0>&, CanonicalClusterRecords&);
template Status CanonicalClusterCodec<0>::EncodePreparedPayload(const CanonicalClusterPayloadSource<0>&, CanonicalClusterRecords&);
template Status CanonicalClusterCodec<0>::Encode(const CanonicalClusterSource<0>&, CanonicalClusterRecords&);
template Status CanonicalClusterCodec<0>::EncodeCoefficientRecord(const ScalarInt6ClusterView<0>&, std::vector<std::uint8_t>&);
template class CanonicalClusterCodec<5>;
template class CanonicalClusterCodec<12>;
template class CanonicalClusterCodec<6>;
template class CanonicalClusterCodec<10>;

template StatusOr<DecodedCanonicalTopology<0>> CanonicalClusterCodec<0>::DecodeTopology(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    ArrayView<const std::uint8_t>, ArrayView<const std::uint8_t>, std::uint32_t);

}  // namespace stlqpp
