#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/storage/canonical_forest_format.h"
#include "stlqpp/storage/index_manifest.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

// Encodes the CID table only. Payload shard ownership and file extents are
// validated by the shard owner when an index is published or opened.
class CanonicalDirectoryCodec final {
 public:
  [[nodiscard]] static Status Encode(const ModelShape& shape,
                                     const IndexCounts& counts,
                                     ArrayView<const CanonicalCidEntry> entries,
                                     std::vector<std::uint8_t>& output,
                                     bool has_quantized_forest = true,
                                     bool split_norm_lut = false,
                                     bool completion_order = false);

  [[nodiscard]] static Status Decode(
      const std::uint8_t* bytes, std::size_t byte_count,
      const ModelShape& expected_shape, const IndexCounts& expected_counts,
      std::vector<CanonicalCidEntry>& entries, bool has_quantized_forest = true,
      bool split_norm_lut = false, bool completion_order = false);
};

}  // namespace stlqpp
