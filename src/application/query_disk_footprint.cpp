#include "stlqpp/application/query_disk_footprint.h"
#include "stlqpp/storage/canonical_index_files.h"

namespace stlqpp {
StatusOr<QueryDiskFootprint> CollectQueryDiskFootprint(
    const std::filesystem::path& index, const IndexManifest& manifest,
    bool native, bool scaled, bool normless,
    const std::vector<std::filesystem::path>& norm_payloads,
    const std::filesystem::path& routing_index) {
  try {
    QueryDiskFootprint disk;
    const auto shard = [&](ShardKind kind, const char* role) {
      bool found = false;
      for (const auto& s : manifest.shards) if (s.kind == kind) {
        found = true;
        const auto path = CanonicalShardPath(index, s);
        if (kind == ShardKind::kCodebooks && s.payload_offset != 0)
          disk.AddRange(role, path, s.payload_offset, s.bytes);
        else disk.AddFile(role, path);
      }
      if (!found) throw std::runtime_error("missing query shard " + std::to_string(unsigned(kind)));
    };
    shard(ShardKind::kCodebooks, "model");
    shard(ShardKind::kIvfDirectory, "directory");
    shard(ShardKind::kCodes, "codes_and_ids");
    shard(ShardKind::kParents, "parents");
    shard(native ? ShardKind::kFloatCoefficients : ShardKind::kCoefficients, "coefficients");
    if (!normless) {
      if (!norm_payloads.empty()) {
        for (const auto& path : norm_payloads) disk.AddFile("norm", path);
      } else if (native) {
        shard(ShardKind::kFloatNorm2, "norm");
        if (scaled)
          shard(ShardKind::kFloatSourceNorm2, "norm");
      } else {
        shard(ShardKind::kNorm8, "norm");
        if (!scaled) shard(ShardKind::kNorm2Lut, "norm");
        if (scaled) shard(ShardKind::kScaledScore8, "norm");
      }
    }
    if (!routing_index.empty()) disk.AddFile("routing_extra", routing_index);
    return disk;
  } catch (const std::exception& e) {
    return Status::Internal(std::string("minimal query disk statistics failed: ") + e.what());
  }
}
}  // namespace stlqpp
