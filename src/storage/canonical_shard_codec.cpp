#include "stlqpp/storage/canonical_shard_codec.h"

#include "stlqpp/core/numeric/little_endian.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace stlqpp {
namespace {

using Magic = std::array<std::uint8_t, 8>;

constexpr Magic kCodesMagic = {'S', 'T', 'L', 'Q', 'C', 'O', 'D', '1'};
constexpr Magic kParentsMagic = {'S', 'T', 'L', 'Q', 'P', 'A', 'R', '1'};
constexpr Magic kCoefficientsMagic = {
    'S', 'T', 'L', 'Q', 'C', 'O', 'F', '1'};
constexpr Magic kFloatCoefficientsMagic = {
    'S', 'T', 'L', 'Q', 'F', 'C', 'F', '1'};
constexpr Magic kFloatNorm2Magic = {
    'S', 'T', 'L', 'Q', 'F', 'N', '2', '1'};
constexpr Magic kScaledScore8Magic = {
    'S', 'T', 'L', 'Q', 'R', 'S', 'C', '1'};
constexpr Magic kFloatSourceNorm2Magic = {
    'S', 'T', 'L', 'Q', 'F', 'S', 'N', '1'};

const Magic* PayloadMagic(ShardKind kind) noexcept {
  switch (kind) {
    case ShardKind::kCodes:
      return &kCodesMagic;
    case ShardKind::kParents:
      return &kParentsMagic;
    case ShardKind::kCoefficients:
      return &kCoefficientsMagic;
    case ShardKind::kFloatCoefficients:
      return &kFloatCoefficientsMagic;
    case ShardKind::kFloatNorm2:
      return &kFloatNorm2Magic;
    case ShardKind::kScaledScore8:
      return &kScaledScore8Magic;
    case ShardKind::kFloatSourceNorm2:
      return &kFloatSourceNorm2Magic;
    default:
      return nullptr;
  }
}

}  // namespace

Status CanonicalPayloadShardCodec::EncodeHeader(
    ShardKind kind,
    const ModelShape& shape,
    std::vector<std::uint8_t>& output) {
  if (const Status status = shape.Validate(); !status.ok()) {
    return status;
  }
  const Magic* magic = PayloadMagic(kind);
  if (magic == nullptr) {
    return Status::InvalidArgument(
        "canonical payload header kind is unsupported");
  }
  output.assign(kCanonicalPayloadHeaderBytes, 0U);
  std::memcpy(output.data(), magic->data(), magic->size());
  StoreLittleEndianU32(kCanonicalForestEncodingVersion, output.data() + 8U);
  StoreLittleEndianU32(kCanonicalPayloadHeaderBytes, output.data() + 12U);
  StoreLittleEndianU32(shape.m, output.data() + 16U);
  StoreLittleEndianU32(shape.nlist, output.data() + 20U);
  return Status::Ok();
}

Status CanonicalPayloadShardCodec::ValidateHeader(
    const std::uint8_t* bytes,
    std::size_t byte_count,
    ShardKind expected_kind,
    const ModelShape& expected_shape) {
  if (const Status status = expected_shape.Validate(); !status.ok()) {
    return status;
  }
  const Magic* magic = PayloadMagic(expected_kind);
  if (magic == nullptr) {
    return Status::InvalidArgument(
        "canonical payload header kind is unsupported");
  }
  if (bytes == nullptr || byte_count < kCanonicalPayloadHeaderBytes ||
      std::memcmp(bytes, magic->data(), magic->size()) != 0 ||
      LoadLittleEndianU32(bytes + 8U) !=
          kCanonicalForestEncodingVersion ||
      LoadLittleEndianU32(bytes + 12U) !=
          kCanonicalPayloadHeaderBytes ||
      LoadLittleEndianU32(bytes + 16U) != expected_shape.m ||
      LoadLittleEndianU32(bytes + 20U) != expected_shape.nlist ||
      LoadLittleEndianU64(bytes + 24U) != 0U) {
    return Status::DataLoss(
        "canonical payload shard header is truncated or mismatched");
  }
  return Status::Ok();
}

}  // namespace stlqpp
