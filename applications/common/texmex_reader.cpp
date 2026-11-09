#include "texmex_reader.h"

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/platform/npy_array.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>

namespace stlqpp::app {
namespace {

struct TexmexShape final {
  std::uint32_t count = 0U;
  std::uint32_t dimension = 0U;
  std::size_t record_bytes = 0U;
};

StatusOr<TexmexShape> InspectFlatBinary(
    const ReadOnlyMappedFile& file, std::size_t element_bytes,
    const char* format_name) {
  constexpr std::size_t kHeaderBytes = 2U * sizeof(std::uint32_t);
  if (file.size() < kHeaderBytes) {
    return Status::DataLoss(std::string(format_name) +
                            " file is empty or truncated");
  }
  const std::uint32_t count = LoadLittleEndianU32(file.data());
  const std::uint32_t dimension =
      LoadLittleEndianU32(file.data() + sizeof(std::uint32_t));
  if (count == 0U || dimension == 0U ||
      static_cast<std::uint64_t>(count) * dimension >
          (std::numeric_limits<std::size_t>::max() - kHeaderBytes) /
              element_bytes) {
    return Status::DataLoss(std::string(format_name) +
                            " header is invalid");
  }
  const std::size_t payload_bytes = static_cast<std::size_t>(count) *
                                    dimension * element_bytes;
  if (file.size() != kHeaderBytes + payload_bytes) {
    return Status::DataLoss(std::string(format_name) +
                            " file has an invalid extent");
  }
  return TexmexShape{
      .count = count,
      .dimension = dimension,
      .record_bytes = static_cast<std::size_t>(dimension) * element_bytes,
  };
}

StatusOr<TexmexShape> InspectTexmex(
    const ReadOnlyMappedFile& file, std::size_t element_bytes,
    const char* format_name) {
  if (file.size() < sizeof(std::uint32_t)) {
    return Status::DataLoss(std::string(format_name) +
                            " file is empty or truncated");
  }
  const std::uint32_t dimension = LoadLittleEndianU32(file.data());
  if (dimension == 0U ||
      dimension > (std::numeric_limits<std::size_t>::max() - 4U) /
          element_bytes) {
    return Status::DataLoss(std::string(format_name) +
                            " vector dimension is invalid");
  }
  const std::size_t record_bytes =
      sizeof(std::uint32_t) +
      static_cast<std::size_t>(dimension) * element_bytes;
  if (file.size() % record_bytes != 0U ||
      file.size() / record_bytes >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::DataLoss(std::string(format_name) +
                            " file has an invalid extent");
  }
  return TexmexShape{
      .count = static_cast<std::uint32_t>(file.size() / record_bytes),
      .dimension = dimension,
      .record_bytes = record_bytes,
  };
}

StatusOr<TexmexVectorShape> InspectTexmexPath(
    const std::filesystem::path& path, std::size_t element_bytes,
    const char* format_name) {
  auto file_result = ReadOnlyMappedFile::Open(path);
  if (!file_result.ok()) return file_result.status();
  auto shape = InspectTexmex(
      file_result.value(), element_bytes, format_name);
  if (!shape.ok()) return shape.status();
  return TexmexVectorShape{
      .count = shape.value().count,
      .dimension = shape.value().dimension,
  };
}

StatusOr<TexmexVectorShape> InspectFlatBinaryPath(
    const std::filesystem::path& path, std::size_t element_bytes,
    const char* format_name) {
  auto file_result = ReadOnlyMappedFile::Open(path);
  if (!file_result.ok()) return file_result.status();
  auto shape = InspectFlatBinary(
      file_result.value(), element_bytes, format_name);
  if (!shape.ok()) return shape.status();
  return TexmexVectorShape{
      .count = shape.value().count,
      .dimension = shape.value().dimension,
  };
}

}  // namespace

StatusOr<TexmexVectorShape> InspectFvecs(
    const std::filesystem::path& path) {
  return InspectTexmexPath(path, sizeof(float), "fvecs");
}

StatusOr<TexmexVectorShape> InspectIvecs(
    const std::filesystem::path& path) {
  return InspectTexmexPath(path, sizeof(std::uint32_t), "ivecs");
}

StatusOr<TexmexVectorShape> InspectFbin(
    const std::filesystem::path& path) {
  return InspectFlatBinaryPath(path, sizeof(float), "fbin");
}

StatusOr<TexmexVectorShape> InspectIbin(
    const std::filesystem::path& path) {
  return InspectFlatBinaryPath(path, sizeof(std::uint32_t), "ibin");
}

StatusOr<TexmexVectorShape> InspectBvecs(
    const std::filesystem::path& path) {
  return InspectTexmexPath(path, sizeof(std::uint8_t), "bvecs");
}

namespace {

StatusOr<TexmexVectorShape> InspectNpyPath(
    const std::filesystem::path& path, NpyScalar scalar) {
  auto file = ReadOnlyMappedFile::Open(path);
  if (!file.ok()) return file.status();
  auto layout = InspectNpyArray(file.value());
  if (!layout.ok()) return layout.status();
  if (layout.value().scalar != scalar ||
      layout.value().rows > std::numeric_limits<std::uint32_t>::max()) {
    return Status::DataLoss("npy dtype or row count is unsupported");
  }
  return TexmexVectorShape{
      static_cast<std::uint32_t>(layout.value().rows),
      layout.value().columns};
}

}  // namespace

StatusOr<TexmexVectorShape> InspectNpyFloat16(
    const std::filesystem::path& path) {
  return InspectNpyPath(path, NpyScalar::kFloat16);
}

StatusOr<TexmexVectorShape> InspectNpyInt64(
    const std::filesystem::path& path) {
  return InspectNpyPath(path, NpyScalar::kInt64);
}

StatusOr<TexmexFloatVectors> ReadNpyFloat16Prefix(
    const std::filesystem::path& path, std::uint32_t maximum_count) {
  if (maximum_count == 0U) {
    return Status::InvalidArgument("npy prefix count must be positive");
  }
  auto file = ReadOnlyMappedFile::Open(path);
  if (!file.ok()) return file.status();
  auto layout = InspectNpyArray(file.value());
  if (!layout.ok()) return layout.status();
  if (layout.value().scalar != NpyScalar::kFloat16 ||
      layout.value().rows > std::numeric_limits<std::uint32_t>::max()) {
    return Status::DataLoss("npy queries must be float16");
  }
  TexmexFloatVectors output;
  output.count = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(layout.value().rows, maximum_count));
  output.dimension = layout.value().columns;
  output.values.resize(static_cast<std::size_t>(output.count) *
                       output.dimension);
  const std::uint8_t* payload = file.value().data() +
      layout.value().data_offset;
  for (std::size_t index = 0U; index < output.values.size(); ++index) {
    output.values[index] = LoadNpyFloat16(payload + index * 2U);
  }
  return output;
}

StatusOr<TexmexU32Vectors> ReadNpyInt64Prefix(
    const std::filesystem::path& path, std::uint32_t maximum_count) {
  if (maximum_count == 0U) {
    return Status::InvalidArgument("npy prefix count must be positive");
  }
  auto file = ReadOnlyMappedFile::Open(path);
  if (!file.ok()) return file.status();
  auto layout = InspectNpyArray(file.value());
  if (!layout.ok()) return layout.status();
  if (layout.value().scalar != NpyScalar::kInt64 ||
      layout.value().rows > std::numeric_limits<std::uint32_t>::max()) {
    return Status::DataLoss("npy ground truth must be int64");
  }
  TexmexU32Vectors output;
  output.count = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(layout.value().rows, maximum_count));
  output.dimension = layout.value().columns;
  output.values.resize(static_cast<std::size_t>(output.count) *
                       output.dimension);
  const std::uint8_t* payload = file.value().data() +
      layout.value().data_offset;
  for (std::size_t index = 0U; index < output.values.size(); ++index) {
    const std::uint64_t id = LoadLittleEndianU64(payload + index * 8U);
    if (id > std::numeric_limits<std::uint32_t>::max()) {
      return Status::DataLoss("npy ground truth ID exceeds uint32");
    }
    output.values[index] = static_cast<std::uint32_t>(id);
  }
  return output;
}

StatusOr<TexmexFloatVectors> ReadFvecs(
    const std::filesystem::path& path) {
  return ReadFvecsPrefix(path, std::numeric_limits<std::uint32_t>::max());
}

StatusOr<TexmexFloatVectors> ReadFvecsPrefix(
    const std::filesystem::path& path,
    std::uint32_t maximum_count) {
  if (maximum_count == 0U) {
    return Status::InvalidArgument("fvecs prefix count must be positive");
  }
  auto file_result = ReadOnlyMappedFile::Open(path);
  if (!file_result.ok()) {
    return file_result.status();
  }
  ReadOnlyMappedFile file = std::move(file_result).value();
  auto shape_result = InspectTexmex(file, sizeof(float), "fvecs");
  if (!shape_result.ok()) {
    return shape_result.status();
  }
  const TexmexShape shape = shape_result.value();
  TexmexFloatVectors output;
  output.count = std::min(shape.count, maximum_count);
  output.dimension = shape.dimension;
  output.values.resize(static_cast<std::size_t>(output.count) *
                       shape.dimension);
  for (std::uint32_t row = 0U; row < output.count; ++row) {
    const std::uint8_t* record =
        file.data() + static_cast<std::size_t>(row) * shape.record_bytes;
    if (LoadLittleEndianU32(record) != shape.dimension) {
      return Status::DataLoss(
          "Texmex fvecs contains inconsistent record dimensions");
    }
    for (std::uint32_t column = 0U; column < shape.dimension; ++column) {
      output.values[static_cast<std::size_t>(row) * shape.dimension + column] =
          LoadLittleEndianF32(record + 4U +
                             static_cast<std::size_t>(column) * 4U);
    }
  }
  return output;
}

StatusOr<TexmexU32Vectors> ReadIvecs(
    const std::filesystem::path& path) {
  return ReadIvecsPrefix(path, std::numeric_limits<std::uint32_t>::max());
}

StatusOr<TexmexU32Vectors> ReadIvecsPrefix(
    const std::filesystem::path& path,
    std::uint32_t maximum_count) {
  if (maximum_count == 0U) {
    return Status::InvalidArgument("ivecs prefix count must be positive");
  }
  auto file_result = ReadOnlyMappedFile::Open(path);
  if (!file_result.ok()) {
    return file_result.status();
  }
  ReadOnlyMappedFile file = std::move(file_result).value();
  auto shape_result = InspectTexmex(
      file, sizeof(std::uint32_t), "ivecs");
  if (!shape_result.ok()) {
    return shape_result.status();
  }
  const TexmexShape shape = shape_result.value();
  TexmexU32Vectors output;
  output.count = std::min(shape.count, maximum_count);
  output.dimension = shape.dimension;
  output.values.resize(static_cast<std::size_t>(output.count) *
                       shape.dimension);
  for (std::uint32_t row = 0U; row < output.count; ++row) {
    const std::uint8_t* record =
        file.data() + static_cast<std::size_t>(row) * shape.record_bytes;
    if (LoadLittleEndianU32(record) != shape.dimension) {
      return Status::DataLoss(
          "Texmex ivecs contains inconsistent record dimensions");
    }
    for (std::uint32_t column = 0U; column < shape.dimension; ++column) {
      output.values[static_cast<std::size_t>(row) * shape.dimension + column] =
          LoadLittleEndianU32(record + 4U +
                             static_cast<std::size_t>(column) * 4U);
    }
  }
  return output;
}

StatusOr<TexmexFloatVectors> ReadFbin(
    const std::filesystem::path& path) {
  return ReadFbinPrefix(path, std::numeric_limits<std::uint32_t>::max());
}

StatusOr<TexmexFloatVectors> ReadFbinPrefix(
    const std::filesystem::path& path,
    std::uint32_t maximum_count) {
  if (maximum_count == 0U) {
    return Status::InvalidArgument("fbin prefix count must be positive");
  }
  auto file_result = ReadOnlyMappedFile::Open(path);
  if (!file_result.ok()) return file_result.status();
  ReadOnlyMappedFile file = std::move(file_result).value();
  auto shape_result = InspectFlatBinary(file, sizeof(float), "fbin");
  if (!shape_result.ok()) return shape_result.status();
  const TexmexShape shape = shape_result.value();
  TexmexFloatVectors output;
  output.count = std::min(shape.count, maximum_count);
  output.dimension = shape.dimension;
  output.values.resize(static_cast<std::size_t>(output.count) *
                       shape.dimension);
  constexpr std::size_t kHeaderBytes = 2U * sizeof(std::uint32_t);
  for (std::size_t index = 0U; index < output.values.size(); ++index) {
    output.values[index] = LoadLittleEndianF32(
        file.data() + kHeaderBytes + index * sizeof(float));
  }
  return output;
}

StatusOr<TexmexU32Vectors> ReadIbin(
    const std::filesystem::path& path) {
  return ReadIbinPrefix(path, std::numeric_limits<std::uint32_t>::max());
}

StatusOr<TexmexU32Vectors> ReadIbinPrefix(
    const std::filesystem::path& path,
    std::uint32_t maximum_count) {
  if (maximum_count == 0U) {
    return Status::InvalidArgument("ibin prefix count must be positive");
  }
  auto file_result = ReadOnlyMappedFile::Open(path);
  if (!file_result.ok()) return file_result.status();
  ReadOnlyMappedFile file = std::move(file_result).value();
  auto shape_result = InspectFlatBinary(
      file, sizeof(std::uint32_t), "ibin");
  if (!shape_result.ok()) return shape_result.status();
  const TexmexShape shape = shape_result.value();
  TexmexU32Vectors output;
  output.count = std::min(shape.count, maximum_count);
  output.dimension = shape.dimension;
  output.values.resize(static_cast<std::size_t>(output.count) *
                       shape.dimension);
  constexpr std::size_t kHeaderBytes = 2U * sizeof(std::uint32_t);
  for (std::size_t index = 0U; index < output.values.size(); ++index) {
    output.values[index] = LoadLittleEndianU32(
        file.data() + kHeaderBytes + index * sizeof(std::uint32_t));
  }
  return output;
}

StatusOr<TexmexU8Vectors> ReadBvecsPrefix(
    const std::filesystem::path& path,
    std::uint32_t maximum_count) {
  if (maximum_count == 0U) {
    return Status::InvalidArgument("bvecs prefix count must be positive");
  }
  auto file_result = ReadOnlyMappedFile::Open(path);
  if (!file_result.ok()) {
    return file_result.status();
  }
  ReadOnlyMappedFile file = std::move(file_result).value();
  auto shape_result = InspectTexmex(file, sizeof(std::uint8_t), "bvecs");
  if (!shape_result.ok()) return shape_result.status();
  const TexmexShape shape = shape_result.value();
  const std::uint32_t count = std::min(shape.count, maximum_count);
  TexmexU8Vectors output;
  output.count = count;
  output.dimension = shape.dimension;
  output.values.resize(static_cast<std::size_t>(count) * shape.dimension);
  for (std::uint32_t vector = 0U; vector < count; ++vector) {
    const std::uint8_t* record =
        file.data() + static_cast<std::size_t>(vector) * shape.record_bytes;
    if (LoadLittleEndianU32(record) != shape.dimension) {
      return Status::DataLoss(
          "bvecs contains inconsistent record dimensions");
    }
    std::copy(record + sizeof(std::uint32_t),
              record + shape.record_bytes,
              output.values.begin() +
                  static_cast<std::size_t>(vector) * shape.dimension);
  }
  return output;
}

}  // namespace stlqpp::app
