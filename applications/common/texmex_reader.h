#pragma once

#include "stlqpp/core/status_or.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace stlqpp::app {

struct TexmexFloatVectors final {
  std::uint32_t count = 0U;
  std::uint32_t dimension = 0U;
  std::vector<float> values;
};

struct TexmexU32Vectors final {
  std::uint32_t count = 0U;
  std::uint32_t dimension = 0U;
  std::vector<std::uint32_t> values;
};

struct TexmexU8Vectors final {
  std::uint32_t count = 0U;
  std::uint32_t dimension = 0U;
  std::vector<std::uint8_t> values;
};

struct TexmexVectorShape final {
  std::uint32_t count = 0U;
  std::uint32_t dimension = 0U;
};

[[nodiscard]] StatusOr<TexmexVectorShape> InspectFvecs(
    const std::filesystem::path& path);
[[nodiscard]] StatusOr<TexmexVectorShape> InspectIvecs(
    const std::filesystem::path& path);
[[nodiscard]] StatusOr<TexmexVectorShape> InspectFbin(
    const std::filesystem::path& path);
[[nodiscard]] StatusOr<TexmexVectorShape> InspectIbin(
    const std::filesystem::path& path);
[[nodiscard]] StatusOr<TexmexVectorShape> InspectBvecs(
    const std::filesystem::path& path);
[[nodiscard]] StatusOr<TexmexVectorShape> InspectNpyFloat16(
    const std::filesystem::path& path);
[[nodiscard]] StatusOr<TexmexVectorShape> InspectNpyInt64(
    const std::filesystem::path& path);

[[nodiscard]] StatusOr<TexmexFloatVectors> ReadFvecs(
    const std::filesystem::path& path);
[[nodiscard]] StatusOr<TexmexFloatVectors> ReadFvecsPrefix(
    const std::filesystem::path& path,
    std::uint32_t maximum_count);
[[nodiscard]] StatusOr<TexmexU32Vectors> ReadIvecs(
    const std::filesystem::path& path);
[[nodiscard]] StatusOr<TexmexU32Vectors> ReadIvecsPrefix(
    const std::filesystem::path& path,
    std::uint32_t maximum_count);
[[nodiscard]] StatusOr<TexmexFloatVectors> ReadFbin(
    const std::filesystem::path& path);
[[nodiscard]] StatusOr<TexmexFloatVectors> ReadFbinPrefix(
    const std::filesystem::path& path,
    std::uint32_t maximum_count);
[[nodiscard]] StatusOr<TexmexU32Vectors> ReadIbin(
    const std::filesystem::path& path);
[[nodiscard]] StatusOr<TexmexU32Vectors> ReadIbinPrefix(
    const std::filesystem::path& path,
    std::uint32_t maximum_count);
[[nodiscard]] StatusOr<TexmexU8Vectors> ReadBvecsPrefix(
    const std::filesystem::path& path,
    std::uint32_t maximum_count);
[[nodiscard]] StatusOr<TexmexFloatVectors> ReadNpyFloat16Prefix(
    const std::filesystem::path& path,
    std::uint32_t maximum_count);
[[nodiscard]] StatusOr<TexmexU32Vectors> ReadNpyInt64Prefix(
    const std::filesystem::path& path,
    std::uint32_t maximum_count);

}  // namespace stlqpp::app
