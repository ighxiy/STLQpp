#pragma once
#include <filesystem>

#include "stlqpp/storage/index_manifest.h"
namespace stlqpp {
// Relative paths owned by an index publication; no copied topology/model files.
inline std::filesystem::path PublishedNorm2Directory(
    const IndexArtifactHashes& hashes) {
  return std::filesystem::path("norm-cache") / "lut" /
         ("quantized-" + hashes.coefficient_codec.ToHex() + "-" +
          hashes.norm8_cache.ToHex());
}
[[nodiscard]] Status WritePublishedNorm2Metadata(
    const std::filesystem::path& root, IndexManifest& manifest,
    const char* backend = "provided");
[[nodiscard]] Status SharePublishedNorm2Cache(
    const std::filesystem::path& staged,
    const std::filesystem::path& source_owner,
    const std::filesystem::path& destination, IndexManifest& manifest);
}  // namespace stlqpp
