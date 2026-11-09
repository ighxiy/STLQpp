#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/platform/mapped_file.h"

#include <cstdint>

namespace stlqpp {

enum class NpyScalar : std::uint8_t {
  kFloat16,
  kInt64,
};

struct NpyArrayLayout final {
  std::uint64_t rows = 0U;
  std::uint32_t columns = 0U;
  std::uint64_t data_offset = 0U;
  NpyScalar scalar = NpyScalar::kFloat16;
};

// Accepts only little-endian, C-contiguous two-dimensional arrays used by
// Contriever. The mapped file must outlive all views into its payload.
[[nodiscard]] StatusOr<NpyArrayLayout> InspectNpyArray(
    const ReadOnlyMappedFile& file);

[[nodiscard]] float LoadNpyFloat16(const std::uint8_t* bytes) noexcept;

}  // namespace stlqpp
