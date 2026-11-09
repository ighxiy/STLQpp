#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/storage/index_manifest.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

class IndexManifestCodec final {
 public:
  static constexpr std::uint32_t kEncodingVersion = 1U;

  [[nodiscard]] static Status Encode(const IndexManifest& manifest,
                                     std::vector<std::uint8_t>& output);
  [[nodiscard]] static Status Decode(const std::uint8_t* bytes,
                                     std::size_t byte_count,
                                     IndexManifest& output);
};

}  // namespace stlqpp
