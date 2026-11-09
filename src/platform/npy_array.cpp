#include "stlqpp/platform/npy_array.h"

#include "stlqpp/core/numeric/little_endian.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>

namespace stlqpp {
namespace {

void SkipSpace(std::string_view text, std::size_t* position) {
  while (*position < text.size() &&
         (text[*position] == ' ' || text[*position] == '\t')) {
    ++*position;
  }
}

StatusOr<std::size_t> FieldValue(std::string_view header,
                                 std::string_view key) {
  const std::size_t found = header.find(key);
  if (found == std::string_view::npos) {
    return Status::DataLoss("npy header is missing a required field");
  }
  std::size_t position = found + key.size();
  SkipSpace(header, &position);
  if (position == header.size() || header[position] != ':') {
    return Status::DataLoss("npy header field is malformed");
  }
  ++position;
  SkipSpace(header, &position);
  return position;
}

StatusOr<std::uint64_t> Decimal(std::string_view header,
                                std::size_t* position) {
  SkipSpace(header, position);
  std::uint64_t value = 0U;
  const char* first = header.data() + *position;
  const auto [last, error] =
      std::from_chars(first, header.data() + header.size(), value);
  if (error != std::errc{} || last == first) {
    return Status::DataLoss("npy shape contains an invalid dimension");
  }
  *position = static_cast<std::size_t>(last - header.data());
  SkipSpace(header, position);
  return value;
}

}  // namespace

StatusOr<NpyArrayLayout> InspectNpyArray(
    const ReadOnlyMappedFile& file) {
  constexpr std::uint8_t kMagic[] = {0x93U, 'N', 'U', 'M', 'P', 'Y'};
  if (file.size() < 10U) {
    return Status::DataLoss("npy file is too short");
  }
  const std::uint8_t* bytes = file.data();
  for (std::size_t index = 0U; index < sizeof(kMagic); ++index) {
    if (bytes[index] != kMagic[index]) {
      return Status::DataLoss("npy magic is invalid");
    }
  }
  const std::uint8_t version = bytes[6U];
  const std::size_t prefix = version == 1U ? 10U : 12U;
  if ((version != 1U && version != 2U && version != 3U) ||
      file.size() < prefix) {
    return Status::DataLoss("npy version is unsupported");
  }
  const std::uint32_t header_bytes = version == 1U
      ? static_cast<std::uint32_t>(bytes[8U]) |
            (static_cast<std::uint32_t>(bytes[9U]) << 8U)
      : LoadLittleEndianU32(bytes + 8U);
  if (header_bytes == 0U || header_bytes > (1U << 20U) ||
      header_bytes > file.size() - prefix) {
    return Status::DataLoss("npy header extent is invalid");
  }
  const std::string_view header(
      reinterpret_cast<const char*>(bytes + prefix), header_bytes);
  auto descr_position = FieldValue(header, "'descr'");
  auto order_position = FieldValue(header, "'fortran_order'");
  auto shape_position = FieldValue(header, "'shape'");
  if (!descr_position.ok() || !order_position.ok() ||
      !shape_position.ok()) {
    return Status::DataLoss("npy header is missing shape or dtype");
  }
  std::size_t position = descr_position.value();
  if (position == header.size() || header[position] != '\'') {
    return Status::DataLoss("npy dtype field is malformed");
  }
  const std::size_t end = header.find('\'', ++position);
  if (end == std::string_view::npos) {
    return Status::DataLoss("npy dtype field is malformed");
  }
  const std::string_view dtype = header.substr(position, end - position);
  NpyScalar scalar;
  std::uint32_t scalar_bytes = 0U;
  if (dtype == "<f2") {
    scalar = NpyScalar::kFloat16;
    scalar_bytes = 2U;
  } else if (dtype == "<i8") {
    scalar = NpyScalar::kInt64;
    scalar_bytes = 8U;
  } else {
    return Status::DataLoss("npy dtype is not little-endian float16/int64");
  }
  if (header.substr(order_position.value(), 5U) != "False") {
    return Status::DataLoss("npy array is not C-contiguous");
  }
  position = shape_position.value();
  if (position == header.size() || header[position++] != '(') {
    return Status::DataLoss("npy shape is not two-dimensional");
  }
  auto rows = Decimal(header, &position);
  if (!rows.ok() || position == header.size() || header[position++] != ',') {
    return Status::DataLoss("npy row count is invalid");
  }
  auto columns = Decimal(header, &position);
  if (!columns.ok() || position == header.size() || header[position] != ')') {
    return Status::DataLoss("npy column count is invalid");
  }
  if (rows.value() == 0U || columns.value() == 0U ||
      columns.value() > std::numeric_limits<std::uint32_t>::max() ||
      rows.value() > std::numeric_limits<std::uint64_t>::max() /
                         columns.value() / scalar_bytes) {
    return Status::DataLoss("npy shape is invalid or overflows");
  }
  const std::uint64_t offset = prefix + header_bytes;
  if (file.size() - offset != rows.value() * columns.value() * scalar_bytes) {
    return Status::DataLoss("npy payload extent disagrees with its shape");
  }
  return NpyArrayLayout{rows.value(),
                        static_cast<std::uint32_t>(columns.value()),
                        offset, scalar};
}

float LoadNpyFloat16(const std::uint8_t* bytes) noexcept {
  const std::uint32_t half = static_cast<std::uint32_t>(bytes[0U]) |
                             (static_cast<std::uint32_t>(bytes[1U]) << 8U);
  const std::uint32_t sign = (half & 0x8000U) << 16U;
  const std::uint32_t exponent = (half >> 10U) & 0x1fU;
  std::uint32_t fraction = half & 0x3ffU;
  std::uint32_t bits = sign;
  if (exponent == 0U && fraction != 0U) {
    std::int32_t power = -14;
    while ((fraction & 0x400U) == 0U) {
      fraction <<= 1U;
      --power;
    }
    bits |= (static_cast<std::uint32_t>(power + 127) << 23U) |
            ((fraction & 0x3ffU) << 13U);
  } else if (exponent == 0x1fU) {
    bits |= 0x7f800000U | (fraction << 13U);
  } else if (exponent != 0U) {
    bits |= ((exponent + 112U) << 23U) | (fraction << 13U);
  }
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

}  // namespace stlqpp
