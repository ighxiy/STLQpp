#include "stlqpp/storage/index_manifest_codec.h"

#include <array>
#include <cstring>
#include <string>
#include <utility>

#include "stlqpp/model/codec/coefficient_bit_width.h"

namespace stlqpp {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {
    'S', 'T', 'L', 'Q', 'p', 'p', '\r', '\n',
};
constexpr std::uint32_t kMaximumShardCount = 1'000'000U;
constexpr std::uint32_t kMaximumPathBytes = 4096U;
constexpr std::uint32_t kMaximumCodecCandidateCount = 1024U;
constexpr std::uint32_t kMaximumCoefficientLayerCount = 1024U;

class BinaryWriter final {
 public:
  explicit BinaryWriter(std::vector<std::uint8_t>& output) : output_(output) {}

  void AddU16(std::uint16_t value) {
    for (std::uint32_t byte = 0U; byte < 2U; ++byte) {
      output_.push_back(
          static_cast<std::uint8_t>((value >> (byte * 8U)) & 0xffU));
    }
  }

  void AddU32(std::uint32_t value) {
    for (std::uint32_t byte = 0U; byte < 4U; ++byte) {
      output_.push_back(
          static_cast<std::uint8_t>((value >> (byte * 8U)) & 0xffU));
    }
  }

  void AddU64(std::uint64_t value) {
    for (std::uint32_t byte = 0U; byte < 8U; ++byte) {
      output_.push_back(
          static_cast<std::uint8_t>((value >> (byte * 8U)) & 0xffU));
    }
  }

  void AddBytes(const std::uint8_t* bytes, std::size_t count) {
    output_.insert(output_.end(), bytes, bytes + count);
  }

  void AddString(const std::string& value) {
    AddU32(static_cast<std::uint32_t>(value.size()));
    AddBytes(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
  }

 private:
  std::vector<std::uint8_t>& output_;
};

class BinaryReader final {
 public:
  BinaryReader(const std::uint8_t* bytes, std::size_t byte_count)
      : cursor_(bytes), remaining_(byte_count) {}

  [[nodiscard]] bool ReadU16(std::uint16_t& value) {
    std::uint64_t widened = 0U;
    if (!ReadLittleEndian(2U, widened)) {
      return false;
    }
    value = static_cast<std::uint16_t>(widened);
    return true;
  }

  [[nodiscard]] bool ReadU32(std::uint32_t& value) {
    std::uint64_t widened = 0U;
    if (!ReadLittleEndian(4U, widened)) {
      return false;
    }
    value = static_cast<std::uint32_t>(widened);
    return true;
  }

  [[nodiscard]] bool ReadU64(std::uint64_t& value) {
    return ReadLittleEndian(8U, value);
  }

  [[nodiscard]] bool ReadBytes(std::uint8_t* output, std::size_t count) {
    if (remaining_ < count) {
      return false;
    }
    for (std::size_t index = 0U; index < count; ++index) {
      output[index] = cursor_[index];
    }
    cursor_ += count;
    remaining_ -= count;
    return true;
  }

  [[nodiscard]] bool ReadString(std::string& value) {
    std::uint32_t count = 0U;
    if (!ReadU32(count) || count > kMaximumPathBytes || remaining_ < count) {
      return false;
    }
    value.assign(reinterpret_cast<const char*>(cursor_), count);
    cursor_ += count;
    remaining_ -= count;
    return true;
  }

  [[nodiscard]] bool empty() const noexcept { return remaining_ == 0U; }

 private:
  [[nodiscard]] bool ReadLittleEndian(std::size_t count, std::uint64_t& value) {
    if (remaining_ < count) {
      return false;
    }
    value = 0U;
    for (std::size_t byte = 0U; byte < count; ++byte) {
      value |= static_cast<std::uint64_t>(cursor_[byte]) << (byte * 8U);
    }
    cursor_ += count;
    remaining_ -= count;
    return true;
  }

  const std::uint8_t* cursor_;
  std::size_t remaining_;
};

Status TruncatedManifest() {
  return Status::DataLoss("manifest is truncated or contains an invalid field");
}

}  // namespace

Status IndexManifestCodec::Encode(const IndexManifest& input,
                                  std::vector<std::uint8_t>& output) {
  if (const Status status = input.Validate(); !status.ok()) {
    return status;
  }
  IndexManifest manifest = input;
  const bool references =
      (input.format_version == IndexManifest::kReferencedShardFormatVersion ||
       input.format_version >= IndexManifest::kNormCacheLayoutFormatVersion);
  if (references) {
    manifest.format_version = IndexManifest::kIndependentNormFormatVersion;
    if (manifest.producer_recipe &&
        manifest.producer_recipe->coefficient_bits_per_layer.empty())
      manifest.producer_recipe->coefficient_bits_per_layer.assign(
          manifest.shape.m, kDefaultCoefficientBits);
  }
  for (const ShardDescriptor& shard : manifest.shards) {
    if (shard.relative_path.size() > kMaximumPathBytes) {
      return Status::ResourceExhausted("manifest shard path is too long");
    }
  }

  output.clear();
  BinaryWriter writer(output);
  writer.AddBytes(kMagic.data(), kMagic.size());
  writer.AddU32(kEncodingVersion);
  writer.AddU32(input.format_version);
  writer.AddU32(manifest.shape.dimension);
  writer.AddU32(manifest.shape.m);
  writer.AddU32(manifest.shape.nlist);
  writer.AddU32(manifest.shape.residual_codebook_size);
  writer.AddU64(manifest.counts.real);
  writer.AddU64(manifest.counts.virtual_roots);
  writer.AddU64(manifest.counts.real_roots);
  writer.AddU64(manifest.counts.linked_real);
  writer.AddU16(static_cast<std::uint16_t>(manifest.coefficient_codec));
  writer.AddU16(static_cast<std::uint16_t>(manifest.norm_codec));
  writer.AddU64(manifest.hashes.linkage_list_directory.value());
  writer.AddU64(manifest.hashes.coefficient_codec.value());
  writer.AddU64(manifest.hashes.norm8_cache.value());
  if (manifest.format_version == IndexManifest::kScaledScoreFormatVersion ||
      manifest.format_version ==
          IndexManifest::kConfigurableCoefficientFormatVersion ||
      manifest.format_version ==
          IndexManifest::kConfigurableCodecFormatVersion ||
      manifest.format_version == IndexManifest::kFloatScaledFormatVersion ||
      manifest.format_version == IndexManifest::kIndependentNormFormatVersion) {
    writer.AddU64(manifest.scaled_score8_hash.value());
  }
  if (manifest.format_version == IndexManifest::kScalarRecipeFormatVersion) {
    std::uint64_t coefficient_quantile_bits = 0U;
    static_assert(sizeof(coefficient_quantile_bits) == sizeof(double));
    const double coefficient_quantile =
        manifest.producer_recipe->coefficient_first_magnitude_quantiles.front();
    std::memcpy(&coefficient_quantile_bits, &coefficient_quantile,
                sizeof(coefficient_quantile_bits));
    writer.AddU64(coefficient_quantile_bits);
    writer.AddU32(manifest.producer_recipe->norm8_lloyd_iterations);
  } else if (manifest.format_version ==
                 IndexManifest::kMultiCandidateRecipeFormatVersion ||
             manifest.format_version ==
                 IndexManifest::kNormModeRecipeFormatVersion ||
             manifest.format_version == IndexManifest::kCurrentFormatVersion ||
             manifest.format_version ==
                 IndexManifest::kScaledScoreFormatVersion ||
             manifest.format_version ==
                 IndexManifest::kConfigurableCoefficientFormatVersion ||
             manifest.format_version ==
                 IndexManifest::kConfigurableCodecFormatVersion ||
             manifest.format_version ==
                 IndexManifest::kFloatScaledFormatVersion ||
             manifest.format_version ==
                 IndexManifest::kIndependentNormFormatVersion ||
             manifest.format_version ==
                 IndexManifest::kFloatOnlyFormatVersion) {
    if (manifest.format_version == IndexManifest::kScaledScoreFormatVersion ||
        manifest.format_version == IndexManifest::kFloatScaledFormatVersion ||
        manifest.format_version ==
            IndexManifest::kIndependentNormFormatVersion ||
        manifest.format_version == IndexManifest::kFloatOnlyFormatVersion) {
      writer.AddU32(manifest.producer_recipe.has_value() ? 1U : 0U);
    }
    if (manifest.producer_recipe.has_value()) {
      if (manifest.format_version ==
              IndexManifest::kConfigurableCoefficientFormatVersion ||
          manifest.format_version ==
              IndexManifest::kConfigurableCodecFormatVersion ||
          manifest.format_version == IndexManifest::kFloatScaledFormatVersion ||
          manifest.format_version ==
              IndexManifest::kIndependentNormFormatVersion) {
        const auto write_widths =
            [&](const std::vector<std::uint32_t>& widths) {
              writer.AddU32(static_cast<std::uint32_t>(widths.size()));
              for (const std::uint32_t width : widths) writer.AddU32(width);
            };
        write_widths(manifest.producer_recipe->coefficient_bits_per_layer);
        write_widths(manifest.producer_recipe->coefficient_bits_per_layer_root);
        write_widths(
            manifest.producer_recipe->coefficient_bits_per_layer_linkage);
      }
      if (manifest.format_version ==
              IndexManifest::kConfigurableCodecFormatVersion ||
          manifest.format_version == IndexManifest::kFloatScaledFormatVersion ||
          manifest.format_version ==
              IndexManifest::kIndependentNormFormatVersion) {
        writer.AddU32(
            manifest.producer_recipe->coefficient_weighted_quantile ? 1U : 0U);
        writer.AddU32(
            manifest.producer_recipe->coefficient_allow_clipping ? 1U : 0U);
        writer.AddU32(manifest.producer_recipe->coefficient_fit_scale ? 1U
                                                                      : 0U);
        writer.AddU32(manifest.producer_recipe->coefficient_refinement_sweeps);
        writer.AddU32(
            manifest.producer_recipe->coefficient_refinement_max_layer);
        writer.AddU32(
            manifest.producer_recipe->coefficient_refinement_step_limit);
      }
      const auto write_candidates = [&](const std::vector<double>& candidates) {
        writer.AddU32(static_cast<std::uint32_t>(candidates.size()));
        for (const double candidate : candidates) {
          std::uint64_t bits = 0U;
          static_assert(sizeof(bits) == sizeof(candidate));
          std::memcpy(&bits, &candidate, sizeof(bits));
          writer.AddU64(bits);
        }
      };
      write_candidates(
          manifest.producer_recipe->coefficient_first_magnitude_quantiles);
      write_candidates(
          manifest.producer_recipe->coefficient_rest_magnitude_quantiles);
      if (manifest.format_version ==
              IndexManifest::kNormModeRecipeFormatVersion ||
          manifest.format_version == IndexManifest::kCurrentFormatVersion ||
          manifest.format_version == IndexManifest::kScaledScoreFormatVersion ||
          manifest.format_version ==
              IndexManifest::kConfigurableCoefficientFormatVersion ||
          manifest.format_version ==
              IndexManifest::kConfigurableCodecFormatVersion ||
          manifest.format_version == IndexManifest::kFloatScaledFormatVersion ||
          manifest.format_version ==
              IndexManifest::kIndependentNormFormatVersion) {
        writer.AddU32(
            static_cast<std::uint32_t>(manifest.producer_recipe->norm8_mode));
      }
      writer.AddU32(manifest.producer_recipe->norm8_lloyd_iterations);
    }
  }
  writer.AddU32(static_cast<std::uint32_t>(manifest.shards.size()));
  for (const ShardDescriptor& shard : manifest.shards) {
    writer.AddU16(static_cast<std::uint16_t>(shard.kind));
    writer.AddU16(0U);
    writer.AddU32(shard.ordinal);
    writer.AddU64(shard.bytes);
    writer.AddU64(shard.elements);
    writer.AddString(shard.relative_path);
    if (references) {
      writer.AddString(shard.reference_path);
      writer.AddString(shard.reference_sha256);
      writer.AddU64(shard.payload_offset);
    }
  }
  return Status::Ok();
}

Status IndexManifestCodec::Decode(const std::uint8_t* bytes,
                                  std::size_t byte_count,
                                  IndexManifest& output) {
  BinaryReader reader(bytes, byte_count);
  std::array<std::uint8_t, kMagic.size()> magic{};
  std::uint32_t encoding_version = 0U;
  IndexManifest decoded;
  std::uint16_t coefficient_codec = 0U;
  std::uint16_t norm_codec = 0U;
  std::uint64_t linkage_hash = 0U;
  std::uint64_t coefficient_hash = 0U;
  std::uint64_t norm_hash = 0U;
  std::uint64_t scaled_score_hash = 0U;
  std::uint32_t shard_count = 0U;

  if (!reader.ReadBytes(magic.data(), magic.size()) || magic != kMagic ||
      !reader.ReadU32(encoding_version) ||
      encoding_version != kEncodingVersion ||
      !reader.ReadU32(decoded.format_version) ||
      !reader.ReadU32(decoded.shape.dimension) ||
      !reader.ReadU32(decoded.shape.m) ||
      !reader.ReadU32(decoded.shape.nlist) ||
      !reader.ReadU32(decoded.shape.residual_codebook_size) ||
      !reader.ReadU64(decoded.counts.real) ||
      !reader.ReadU64(decoded.counts.virtual_roots) ||
      !reader.ReadU64(decoded.counts.real_roots) ||
      !reader.ReadU64(decoded.counts.linked_real) ||
      !reader.ReadU16(coefficient_codec) || !reader.ReadU16(norm_codec) ||
      !reader.ReadU64(linkage_hash) || !reader.ReadU64(coefficient_hash) ||
      !reader.ReadU64(norm_hash)) {
    return TruncatedManifest();
  }
  const bool references =
      (decoded.format_version == IndexManifest::kReferencedShardFormatVersion ||
       decoded.format_version >= IndexManifest::kNormCacheLayoutFormatVersion);
  const auto actual_format_version = decoded.format_version;
  if (references)
    decoded.format_version = IndexManifest::kIndependentNormFormatVersion;
  if ((decoded.format_version == IndexManifest::kScaledScoreFormatVersion ||
       decoded.format_version ==
           IndexManifest::kConfigurableCoefficientFormatVersion ||
       decoded.format_version ==
           IndexManifest::kConfigurableCodecFormatVersion ||
       decoded.format_version == IndexManifest::kFloatScaledFormatVersion ||
       decoded.format_version ==
           IndexManifest::kIndependentNormFormatVersion) &&
      !reader.ReadU64(scaled_score_hash)) {
    return TruncatedManifest();
  }
  if (decoded.format_version == IndexManifest::kScalarRecipeFormatVersion) {
    std::uint64_t coefficient_quantile_bits = 0U;
    IndexProducerRecipe recipe;
    if (!reader.ReadU64(coefficient_quantile_bits) ||
        !reader.ReadU32(recipe.norm8_lloyd_iterations)) {
      return TruncatedManifest();
    }
    double coefficient_quantile = 0.0;
    std::memcpy(&coefficient_quantile, &coefficient_quantile_bits,
                sizeof(coefficient_quantile_bits));
    recipe.coefficient_first_magnitude_quantiles = {coefficient_quantile};
    recipe.coefficient_rest_magnitude_quantiles = {coefficient_quantile};
    recipe.norm8_mode = ReconstructionNorm8Mode::kIdentityLloyd;
    decoded.producer_recipe = recipe;
  } else if (decoded.format_version ==
                 IndexManifest::kMultiCandidateRecipeFormatVersion ||
             decoded.format_version ==
                 IndexManifest::kNormModeRecipeFormatVersion ||
             decoded.format_version == IndexManifest::kCurrentFormatVersion ||
             decoded.format_version ==
                 IndexManifest::kScaledScoreFormatVersion ||
             decoded.format_version ==
                 IndexManifest::kConfigurableCoefficientFormatVersion ||
             decoded.format_version ==
                 IndexManifest::kConfigurableCodecFormatVersion ||
             decoded.format_version ==
                 IndexManifest::kFloatScaledFormatVersion ||
             decoded.format_version ==
                 IndexManifest::kIndependentNormFormatVersion ||
             decoded.format_version == IndexManifest::kFloatOnlyFormatVersion) {
    IndexProducerRecipe recipe;
    std::uint32_t has_recipe = 1U;
    if ((decoded.format_version == IndexManifest::kScaledScoreFormatVersion ||
         decoded.format_version == IndexManifest::kFloatScaledFormatVersion ||
         decoded.format_version ==
             IndexManifest::kIndependentNormFormatVersion ||
         decoded.format_version == IndexManifest::kFloatOnlyFormatVersion) &&
        (!reader.ReadU32(has_recipe) || has_recipe > 1U)) {
      return TruncatedManifest();
    }
    if (has_recipe != 0U) {
      if (decoded.format_version ==
              IndexManifest::kConfigurableCoefficientFormatVersion ||
          decoded.format_version ==
              IndexManifest::kConfigurableCodecFormatVersion ||
          decoded.format_version == IndexManifest::kFloatScaledFormatVersion ||
          decoded.format_version ==
              IndexManifest::kIndependentNormFormatVersion) {
        const auto read_widths = [&](std::vector<std::uint32_t>* widths,
                                     bool optional) {
          std::uint32_t count = 0U;
          if (!reader.ReadU32(count) || (!optional && count == 0U) ||
              count > kMaximumCoefficientLayerCount) {
            return false;
          }
          widths->resize(count);
          for (std::uint32_t& width : *widths) {
            if (!reader.ReadU32(width)) return false;
          }
          return true;
        };
        if (!read_widths(&recipe.coefficient_bits_per_layer, false) ||
            !read_widths(&recipe.coefficient_bits_per_layer_root, true) ||
            !read_widths(&recipe.coefficient_bits_per_layer_linkage, true)) {
          return TruncatedManifest();
        }
      }
      if (decoded.format_version ==
              IndexManifest::kConfigurableCodecFormatVersion ||
          decoded.format_version == IndexManifest::kFloatScaledFormatVersion ||
          decoded.format_version ==
              IndexManifest::kIndependentNormFormatVersion) {
        std::uint32_t weighted = 0U;
        std::uint32_t clipping = 0U;
        std::uint32_t fit_scale = 0U;
        if (!reader.ReadU32(weighted) || weighted > 1U ||
            !reader.ReadU32(clipping) || clipping > 1U ||
            !reader.ReadU32(fit_scale) || fit_scale > 1U ||
            !reader.ReadU32(recipe.coefficient_refinement_sweeps) ||
            !reader.ReadU32(recipe.coefficient_refinement_max_layer) ||
            !reader.ReadU32(recipe.coefficient_refinement_step_limit)) {
          return TruncatedManifest();
        }
        recipe.coefficient_weighted_quantile = weighted != 0U;
        recipe.coefficient_allow_clipping = clipping != 0U;
        recipe.coefficient_fit_scale = fit_scale != 0U;
      }
      const auto read_candidates = [&](std::vector<double>* candidates) {
        std::uint32_t count = 0U;
        if (!reader.ReadU32(count) || count == 0U ||
            count > kMaximumCodecCandidateCount) {
          return false;
        }
        candidates->resize(count);
        for (double& candidate : *candidates) {
          std::uint64_t bits = 0U;
          if (!reader.ReadU64(bits)) return false;
          std::memcpy(&candidate, &bits, sizeof(candidate));
        }
        return true;
      };
      std::uint32_t norm8_mode =
          static_cast<std::uint32_t>(ReconstructionNorm8Mode::kIdentityLloyd);
      if (!read_candidates(&recipe.coefficient_first_magnitude_quantiles) ||
          !read_candidates(&recipe.coefficient_rest_magnitude_quantiles) ||
          ((decoded.format_version ==
                IndexManifest::kNormModeRecipeFormatVersion ||
            decoded.format_version == IndexManifest::kCurrentFormatVersion ||
            decoded.format_version ==
                IndexManifest::kScaledScoreFormatVersion ||
            decoded.format_version ==
                IndexManifest::kConfigurableCoefficientFormatVersion ||
            decoded.format_version ==
                IndexManifest::kConfigurableCodecFormatVersion ||
            decoded.format_version ==
                IndexManifest::kFloatScaledFormatVersion ||
            decoded.format_version ==
                IndexManifest::kIndependentNormFormatVersion) &&
           !reader.ReadU32(norm8_mode)) ||
          !reader.ReadU32(recipe.norm8_lloyd_iterations)) {
        return TruncatedManifest();
      }
      recipe.norm8_mode = static_cast<ReconstructionNorm8Mode>(norm8_mode);
      decoded.producer_recipe = std::move(recipe);
    }
  } else if (decoded.format_version != IndexManifest::kLegacyFormatVersion) {
    return Status::DataLoss("unsupported STLQpp index format version");
  }
  if (!reader.ReadU32(shard_count)) {
    return TruncatedManifest();
  }
  if (shard_count > kMaximumShardCount) {
    return Status::ResourceExhausted("manifest declares too many shards");
  }

  decoded.coefficient_codec = static_cast<CoefficientCodec>(coefficient_codec);
  decoded.norm_codec = static_cast<NormCodec>(norm_codec);
  decoded.hashes = {
      .linkage_list_directory = ArtifactHash64::FromValue(linkage_hash),
      .coefficient_codec = ArtifactHash64::FromValue(coefficient_hash),
      .norm8_cache = ArtifactHash64::FromValue(norm_hash),
  };
  decoded.scaled_score8_hash = ArtifactHash64::FromValue(scaled_score_hash);
  decoded.shards.reserve(shard_count);
  for (std::uint32_t index = 0U; index < shard_count; ++index) {
    ShardDescriptor shard;
    std::uint16_t kind = 0U;
    std::uint16_t reserved = 0U;
    if (!reader.ReadU16(kind) || !reader.ReadU16(reserved) || reserved != 0U ||
        !reader.ReadU32(shard.ordinal) || !reader.ReadU64(shard.bytes) ||
        !reader.ReadU64(shard.elements) ||
        !reader.ReadString(shard.relative_path)) {
      return TruncatedManifest();
    }
    if (references && (!reader.ReadString(shard.reference_path) ||
                       !reader.ReadString(shard.reference_sha256) ||
                       !reader.ReadU64(shard.payload_offset)))
      return TruncatedManifest();
    shard.kind = static_cast<ShardKind>(kind);
    decoded.shards.push_back(std::move(shard));
  }
  if (!reader.empty()) {
    return Status::DataLoss("manifest contains trailing bytes");
  }
  if (references) decoded.format_version = actual_format_version;
  if (const Status status = decoded.Validate(); !status.ok()) {
    return status;
  }
  output = std::move(decoded);
  return Status::Ok();
}

}  // namespace stlqpp
