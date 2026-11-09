#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/norm8_codebook.h"
#include "stlqpp/model/scaled_score8_codebook.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/storage/canonical_cluster_codec.h"
#include "stlqpp/storage/canonical_forest_format.h"
#include "stlqpp/storage/index_manifest.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace stlqpp {

[[nodiscard]] std::filesystem::path CanonicalShardPath(
    const std::filesystem::path& index, const ShardDescriptor& shard);
// Recipe/source/extent identity, not a payload integrity checksum. It remains
// stable when the whole run is relocated or unchanged shards are referenced.
[[nodiscard]] std::string CanonicalShardIdentity(
    const std::filesystem::path& index, const IndexManifest& manifest,
    const ShardDescriptor& shard);
[[nodiscard]] StatusOr<ShardDescriptor> ReferenceCanonicalShard(
    const std::filesystem::path& source, const std::filesystem::path& destination,
    const ShardDescriptor& shard);
// The original forest owns shared per-base caches even when opened through
// one or more derived representations.
[[nodiscard]] std::filesystem::path CanonicalSourceIndexDirectory(
    const std::filesystem::path& index, const IndexManifest& manifest);

struct CanonicalFloatRecordViews final {
  ArrayView<const std::uint8_t> coefficients;
  ArrayView<const std::uint8_t> norm2;
};

struct CanonicalTopologyRecordViews final {
  ArrayView<const std::uint8_t> codes;
  ArrayView<const std::uint8_t> parents;
};

// Checked owner of canonical v1/v2 mappings. All cross-file offsets and extents
// are validated at Open; typed cluster decoding may then consume record views
// without repeating range checks.
class CanonicalIndexFiles final {
 public:
  CanonicalIndexFiles() = default;
  CanonicalIndexFiles(const CanonicalIndexFiles&) = delete;
  CanonicalIndexFiles& operator=(const CanonicalIndexFiles&) = delete;
  CanonicalIndexFiles(CanonicalIndexFiles&&) noexcept = default;
  CanonicalIndexFiles& operator=(CanonicalIndexFiles&&) noexcept = default;

  [[nodiscard]] static StatusOr<CanonicalIndexFiles> Open(
      const std::filesystem::path& directory);
  // Opens compressed records together with an existing source-norm shard.
  [[nodiscard]] static StatusOr<CanonicalIndexFiles> OpenWithSourceNorm(
      const std::filesystem::path& directory);
  [[nodiscard]] static StatusOr<CanonicalIndexFiles> OpenQuantized(
      const std::filesystem::path& directory,
      bool include_source_norm2 = false);
  // Opens only the model, shared topology and native-float shards.  The
  // compressed coefficient and norm8 files are deliberately neither mapped
  // nor parsed, so they cannot become a dependency of native-float search.
  [[nodiscard]] static StatusOr<CanonicalIndexFiles> OpenNativeFloat(
      const std::filesystem::path& directory);
  [[nodiscard]] static StatusOr<CanonicalIndexFiles> OpenNativeFloatScaled(
      const std::filesystem::path& directory);
  // Read the reconstruction without requiring any previously generated norm
  // artifacts. Used to rebuild missing evaluation caches, never to repair IO
  // failures in coefficient/topology files.
  [[nodiscard]] static StatusOr<CanonicalIndexFiles> OpenCoefficients(
      const std::filesystem::path& directory, bool native_float,
      bool include_source_norm2 = false);

  [[nodiscard]] const IndexManifest& manifest() const noexcept {
    return manifest_;
  }
  [[nodiscard]] ArrayView<const std::uint8_t> model_bytes() const noexcept {
    return {model_.data() + model_payload_offset_, model_payload_bytes_};
  }
  [[nodiscard]] Norm8CodebookView norm8() const noexcept {
    return {norm8_centers_.data(),
            static_cast<std::uint32_t>(norm8_centers_.size())};
  }
  [[nodiscard]] ScaledScore8CodebookView scaled_score8() const noexcept {
    return scaled_score8_.view();
  }
  [[nodiscard]] const CanonicalCidEntry& entry(
      std::uint32_t cid) const noexcept {
    return entries_[cid];
  }
  [[nodiscard]] ArrayView<const std::uint32_t> physical_cid_order()
      const noexcept {
    return physical_cid_order_;
  }
  [[nodiscard]] CanonicalClusterRecordViews cluster_records(
      std::uint32_t cid) const noexcept;
  [[nodiscard]] CanonicalTopologyRecordViews topology_records(
      std::uint32_t cid) const noexcept;
  [[nodiscard]] bool has_float_forest() const noexcept {
    return manifest_.has_float_forest();
  }
  [[nodiscard]] CanonicalFloatRecordViews float_cluster_records(
      std::uint32_t cid) const noexcept;
  [[nodiscard]] Status ReadFloatCoefficients(
      std::uint32_t cid, ArrayView<float> output) const;
  [[nodiscard]] Status ReadFloatNorm2(
      std::uint32_t cid, ArrayView<float> output) const;
  [[nodiscard]] Status ReadFloatSourceNorm2(
      std::uint32_t cid, ArrayView<float> output) const;

 private:
  [[nodiscard]] static StatusOr<CanonicalIndexFiles> OpenImpl(
      const std::filesystem::path& directory,
      bool include_compressed_representation,
      bool include_float_source_norm2,
      bool include_norms = true,
      bool include_native_float = true);
  IndexManifest manifest_;
  std::vector<CanonicalCidEntry> entries_;
  std::vector<std::uint32_t> physical_cid_order_;
  std::vector<float> norm8_centers_;
  ScaledScore8Codebook scaled_score8_;
  ReadOnlyMappedFile manifest_file_;
  ReadOnlyMappedFile model_;
  std::size_t model_payload_offset_ = 0U;
  std::size_t model_payload_bytes_ = 0U;
  ReadOnlyMappedFile directory_;
  ReadOnlyMappedFile codes_;
  ReadOnlyMappedFile parents_;
  ReadOnlyMappedFile coefficients_;
  ReadOnlyMappedFile norm8_;
  ReadOnlyMappedFile float_coefficients_;
  ReadOnlyMappedFile float_norm2_;
  ReadOnlyMappedFile scaled_score8_file_;
  ReadOnlyMappedFile float_source_norm2_;
  struct FloatCidExtent final {
    std::uint64_t coefficient_offset = 0U;
    std::uint64_t coefficient_bytes = 0U;
    std::uint64_t norm2_offset = 0U;
    std::uint64_t norm2_bytes = 0U;
    std::uint64_t source_norm2_offset = 0U;
    std::uint64_t source_norm2_bytes = 0U;
  };
  std::vector<FloatCidExtent> float_extents_;
};

}  // namespace stlqpp
