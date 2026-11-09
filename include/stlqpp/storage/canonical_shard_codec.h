#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/storage/canonical_forest_format.h"
#include "stlqpp/storage/index_manifest.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

// Common fixed header for the code, parent and coefficient payload shards.
// The shard body starts at byte 32 and is indexed exclusively by directory.bin.
class CanonicalPayloadShardCodec final {
 public:
  [[nodiscard]] static Status EncodeHeader(
      ShardKind kind,
      const ModelShape& shape,
      std::vector<std::uint8_t>& output);

  [[nodiscard]] static Status ValidateHeader(
      const std::uint8_t* bytes,
      std::size_t byte_count,
      ShardKind expected_kind,
      const ModelShape& expected_shape);
};

}  // namespace stlqpp
