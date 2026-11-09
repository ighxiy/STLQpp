#include "stlqpp/storage/index_manifest.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <set>
#include <string>
#include <string_view>

#include "stlqpp/model/codec/coefficient_bit_width.h"

namespace stlqpp {
namespace {

constexpr std::size_t ShardIndex(ShardKind kind) {
  return static_cast<std::size_t>(kind) - 1U;
}

bool IsPortableRelativePath(std::string_view path) {
  if (path.empty() || path.front() == '/' || path.back() == '/') {
    return false;
  }
  std::size_t segment_begin = 0U;
  for (std::size_t index = 0U; index <= path.size(); ++index) {
    if (index != path.size() && path[index] != '/') {
      const unsigned char character = static_cast<unsigned char>(path[index]);
      const bool ascii_letter =
          (character >= static_cast<unsigned char>('a') &&
           character <= static_cast<unsigned char>('z')) ||
          (character >= static_cast<unsigned char>('A') &&
           character <= static_cast<unsigned char>('Z'));
      const bool ascii_digit = character >= static_cast<unsigned char>('0') &&
                               character <= static_cast<unsigned char>('9');
      if (!ascii_letter && !ascii_digit && character != '_' &&
          character != '-' && character != '.') {
        return false;
      }
      continue;
    }
    const std::string_view segment =
        path.substr(segment_begin, index - segment_begin);
    if (segment.empty() || segment == "." || segment == "..") {
      return false;
    }
    segment_begin = index + 1U;
  }
  return true;
}

}  // namespace

Status IndexManifest::Validate() const {
  if (format_version == kReferencedShardFormatVersion ||
      format_version == kNormCacheLayoutFormatVersion ||
      format_version == kCompletionOrderFormatVersion) {
    IndexManifest local = *this;
    if (format_version >= kNormCacheLayoutFormatVersion) {
      std::set<std::string> paths;
      for (const auto& shard : shards)
        if (!paths.insert(shard.relative_path).second)
          return Status::DataLoss("duplicate norm2 shard path");
      unsigned tables = 0;
      for (const auto& shard : shards)
        if (shard.kind == ShardKind::kNorm2Lut) {
          ++tables;
          if (shard.bytes != 1024U || shard.elements != 256U ||
              shard.ordinal != 0U ||
              !IsPortableRelativePath(shard.relative_path))
            return Status::DataLoss("invalid norm2 LUT shard");
        }
      if (tables != (has_quantized_forest() ? 1U : 0U))
        return Status::DataLoss("norm2 LUT capability mismatch");
    }
    local.format_version = kIndependentNormFormatVersion;
    // References may preserve older forests whose registered width was
    // implicit (six bits). Do not change their coefficient representation.
    if (local.producer_recipe &&
        local.producer_recipe->coefficient_bits_per_layer.empty())
      local.producer_recipe->coefficient_bits_per_layer.assign(
          shape.m, kDefaultCoefficientBits);
    if (local.has_quantized_forest() && !local.producer_recipe)
      local.format_version = local.has_scaled_score8()
                                 ? kScaledScoreFormatVersion
                             : local.has_float_forest() ? kCurrentFormatVersion
                                                        : kLegacyFormatVersion;
    for (auto& shard : local.shards) {
      if (shard.reference_path.empty()) {
        if (!shard.reference_sha256.empty() || shard.payload_offset != 0U)
          return Status::DataLoss(
              "local shard cannot carry reference metadata");
      } else {
        if (shard.reference_path.find('\0') != std::string::npos ||
            (!shard.reference_sha256.empty() && shard.reference_sha256.size() != 64U) ||
            !std::all_of(
                shard.reference_sha256.begin(), shard.reference_sha256.end(),
                [](char c) {
                  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                }) ||
            (shard.payload_offset != 0U && shard.kind != ShardKind::kCodebooks))
          return Status::DataLoss("invalid canonical shard reference");
      }
      shard.reference_path.clear();
      shard.reference_sha256.clear();
      shard.payload_offset = 0U;
    }
    if (format_version >= kNormCacheLayoutFormatVersion) {
      local.shards.erase(
          std::remove_if(local.shards.begin(), local.shards.end(),
                         [](const auto& shard) {
                           return shard.kind == ShardKind::kNorm2Lut;
                         }),
          local.shards.end());
    }
    return local.Validate();
  }
  for (const auto& shard : shards)
    if (!shard.reference_path.empty() || !shard.reference_sha256.empty() ||
        shard.payload_offset != 0U)
      return Status::DataLoss("shard references require canonical format v13");
  if (format_version != kLegacyFormatVersion &&
      format_version != kScalarRecipeFormatVersion &&
      format_version != kMultiCandidateRecipeFormatVersion &&
      format_version != kNormModeRecipeFormatVersion &&
      format_version != kCurrentFormatVersion &&
      format_version != kScaledScoreFormatVersion &&
      format_version != kConfigurableCoefficientFormatVersion &&
      format_version != kConfigurableCodecFormatVersion &&
      format_version != kFloatOnlyFormatVersion &&
      (format_version != kFloatScaledFormatVersion &&
       format_version != kIndependentNormFormatVersion)) {
    return Status::DataLoss("unsupported STLQpp index format version");
  }
  if (format_version == kLegacyFormatVersion && producer_recipe.has_value()) {
    return Status::DataLoss(
        "legacy index manifest cannot claim producer provenance");
  }
  const bool quantized = has_quantized_forest();
  if (format_version != kLegacyFormatVersion &&
      format_version != kFloatOnlyFormatVersion &&
      !((format_version == kFloatScaledFormatVersion ||
         format_version == kIndependentNormFormatVersion) &&
        !quantized) &&
      !(format_version == kScaledScoreFormatVersion &&
        !producer_recipe.has_value())) {
    const auto valid_candidates = [](const std::vector<double>& candidates) {
      return !candidates.empty() && candidates.size() <= 1024U &&
             std::all_of(
                 candidates.begin(), candidates.end(), [](double value) {
                   return std::isfinite(value) && value > 0.0 && value <= 1.0;
                 });
    };
    const auto valid_bits = [&](const std::vector<std::uint32_t>& bits,
                                bool optional) {
      return (optional && bits.empty()) ||
             (bits.size() == shape.m &&
              std::all_of(bits.begin(), bits.end(), [](std::uint32_t value) {
                return value >= kMinimumCoefficientBits &&
                       value <= kMaximumCoefficientBits;
              }));
    };
    if (!producer_recipe.has_value() ||
        !valid_candidates(
            producer_recipe->coefficient_first_magnitude_quantiles) ||
        !valid_candidates(
            producer_recipe->coefficient_rest_magnitude_quantiles) ||
        (format_version == kScalarRecipeFormatVersion &&
         (producer_recipe->coefficient_first_magnitude_quantiles.size() != 1U ||
          producer_recipe->coefficient_rest_magnitude_quantiles.size() != 1U ||
          producer_recipe->coefficient_first_magnitude_quantiles.front() !=
              producer_recipe->coefficient_rest_magnitude_quantiles.front())) ||
        ((format_version == kScalarRecipeFormatVersion ||
          format_version == kMultiCandidateRecipeFormatVersion) &&
         producer_recipe->norm8_mode !=
             ReconstructionNorm8Mode::kIdentityLloyd) ||
        (producer_recipe->norm8_mode !=
             ReconstructionNorm8Mode::kIdentityLloyd &&
         producer_recipe->norm8_mode !=
             ReconstructionNorm8Mode::kTailWeightedLloyd) ||
        producer_recipe->norm8_lloyd_iterations == 0U ||
        ((format_version == kConfigurableCoefficientFormatVersion ||
          format_version == kConfigurableCodecFormatVersion ||
          (format_version == kFloatScaledFormatVersion ||
           format_version == kIndependentNormFormatVersion)) &&
         (!valid_bits(producer_recipe->coefficient_bits_per_layer, false) ||
          !valid_bits(producer_recipe->coefficient_bits_per_layer_root, true) ||
          !valid_bits(producer_recipe->coefficient_bits_per_layer_linkage,
                      true))) ||
        (format_version != kConfigurableCoefficientFormatVersion &&
         format_version != kConfigurableCodecFormatVersion &&
         (format_version != kFloatScaledFormatVersion &&
          format_version != kIndependentNormFormatVersion) &&
         (!producer_recipe->coefficient_bits_per_layer.empty() ||
          !producer_recipe->coefficient_bits_per_layer_root.empty() ||
          !producer_recipe->coefficient_bits_per_layer_linkage.empty()))) {
      return Status::DataLoss(
          "current index manifest is missing a valid producer recipe");
    }
  }
  if (const Status shape_status = shape.Validate(); !shape_status.ok()) {
    return shape_status;
  }
  if (counts.real == 0U) {
    return Status::InvalidArgument(
        "index must contain at least one real vector");
  }
  if (counts.real_roots + counts.linked_real != counts.real) {
    return Status::DataLoss(
        "real root and linked counts do not sum to real count");
  }
  if (hashes.linkage_list_directory.empty() ||
      (quantized &&
       (hashes.coefficient_codec.empty() || hashes.norm8_cache.empty())) ||
      (!quantized &&
       (!hashes.coefficient_codec.empty() || !hashes.norm8_cache.empty()))) {
    return Status::DataLoss("manifest is missing a registered artifact hash");
  }
  if (quantized &&
      coefficient_codec != CoefficientCodec::kPerCidScalarHuffman) {
    return Status::Unimplemented(
        "coefficient codec is outside the initial product boundary");
  }
  if (quantized && norm_codec != NormCodec::kReconstructionNorm8 &&
      norm_codec != NormCodec::kReconstructionNorm8WithScaledScore8) {
    return Status::FailedPrecondition(
        "production search requires a supported one-byte score codec");
  }
  if (format_version != kConfigurableCoefficientFormatVersion &&
      format_version != kConfigurableCodecFormatVersion &&
      (format_version != kFloatScaledFormatVersion &&
       format_version != kIndependentNormFormatVersion) &&
      (norm_codec == NormCodec::kReconstructionNorm8WithScaledScore8) !=
          (format_version == kScaledScoreFormatVersion)) {
    return Status::DataLoss(
        "scaled score8 requires canonical format v7 and v7 requires both score "
        "codecs");
  }
  if ((norm_codec == NormCodec::kReconstructionNorm8WithScaledScore8) !=
      !scaled_score8_hash.empty()) {
    return Status::DataLoss(
        "scaled score8 manifest hash and codec capability disagree");
  }
  std::array<std::uint32_t, 10> next_ordinal{};
  std::set<std::string> paths;
  for (const ShardDescriptor& shard : shards) {
    const std::size_t index = ShardIndex(shard.kind);
    if (index >= next_ordinal.size()) {
      return Status::DataLoss("manifest contains an unknown shard kind");
    }
    if (next_ordinal[index] != 0U || shard.ordinal != 0U) {
      return Status::DataLoss(
          "canonical index requires at most one shard of each kind");
    }
    ++next_ordinal[index];
    if (!IsPortableRelativePath(shard.relative_path)) {
      return Status::DataLoss(
          "shard path must be canonical portable ASCII and relative");
    }
    if (!paths.insert(shard.relative_path).second) {
      return Status::DataLoss(
          "canonical shard paths must be unique across kinds");
    }
    if (shard.elements != 0U && shard.bytes == 0U) {
      return Status::DataLoss("non-empty shard must occupy bytes");
    }
  }
  for (std::size_t index = 0U; index < 4U; ++index) {
    if (next_ordinal[index] != 1U) {
      return Status::DataLoss(
          "canonical index requires its four topology shards");
    }
  }
  const bool has_coefficient_shard =
      next_ordinal[ShardIndex(ShardKind::kCoefficients)] == 1U;
  const bool has_norm8_shard =
      next_ordinal[ShardIndex(ShardKind::kNorm8)] == 1U;
  if (has_coefficient_shard != has_norm8_shard ||
      has_coefficient_shard != quantized) {
    return Status::DataLoss(
        "canonical coefficient and norm8 shards must be a capability pair");
  }
  const bool has_scaled_score8 =
      next_ordinal[ShardIndex(ShardKind::kScaledScore8)] == 1U;
  if (has_scaled_score8 !=
      (norm_codec == NormCodec::kReconstructionNorm8WithScaledScore8)) {
    return Status::DataLoss(
        "scaled score8 manifest must contain exactly one score-table shard");
  }
  const bool has_float_source_norm2 =
      next_ordinal[ShardIndex(ShardKind::kFloatSourceNorm2)] == 1U;
  if (format_version != kIndependentNormFormatVersion &&
      has_float_source_norm2 != (format_version == kFloatScaledFormatVersion)) {
    return Status::DataLoss(
        "float_scaled manifest must contain exactly one source-norm shard");
  }
  const bool has_float_coefficients =
      next_ordinal[ShardIndex(ShardKind::kFloatCoefficients)] == 1U;
  const bool has_float_norm2 =
      next_ordinal[ShardIndex(ShardKind::kFloatNorm2)] == 1U;
  if ((has_float_coefficients != has_float_norm2 &&
       format_version != kIndependentNormFormatVersion) ||
      (has_float_norm2 && !has_float_coefficients) ||
      ((has_float_coefficients || has_float_norm2) &&
       format_version != kCurrentFormatVersion &&
       format_version != kScaledScoreFormatVersion &&
       format_version != kConfigurableCoefficientFormatVersion &&
       format_version != kConfigurableCodecFormatVersion &&
       format_version != kFloatOnlyFormatVersion &&
       (format_version != kFloatScaledFormatVersion &&
        format_version != kIndependentNormFormatVersion))) {
    return Status::DataLoss(
        "native-float coefficient and norm shards must be a versioned pair");
  }
  if (has_float_source_norm2 && (!has_float_coefficients)) {
    return Status::DataLoss(
        "float_scaled requires a versioned native-float forest");
  }
  const bool float_only_format =
      format_version == kFloatOnlyFormatVersion ||
      ((format_version == kFloatScaledFormatVersion ||
        format_version == kIndependentNormFormatVersion) &&
       !quantized);
  if (float_only_format != !quantized ||
      (float_only_format &&
       (!has_float_coefficients || producer_recipe.has_value() ||
        !scaled_score8_hash.empty()))) {
    return Status::DataLoss(
        "canonical float-only format capability is inconsistent");
  }
  return Status::Ok();
}

bool IndexManifest::has_float_forest() const noexcept {
  bool coefficients = false;
  for (const ShardDescriptor& shard : shards) {
    coefficients = coefficients || shard.kind == ShardKind::kFloatCoefficients;
  }
  return coefficients;
}

bool IndexManifest::has_float_norm2() const noexcept {
  return std::any_of(shards.begin(), shards.end(), [](const auto& shard) {
    return shard.kind == ShardKind::kFloatNorm2;
  });
}

bool IndexManifest::has_quantized_forest() const noexcept {
  bool coefficients = false;
  bool norm8 = false;
  for (const ShardDescriptor& shard : shards) {
    coefficients = coefficients || shard.kind == ShardKind::kCoefficients;
    norm8 = norm8 || shard.kind == ShardKind::kNorm8;
  }
  return coefficients && norm8;
}

bool IndexManifest::has_float_source_norm2() const noexcept {
  return std::any_of(shards.begin(), shards.end(),
                     [](const ShardDescriptor& shard) {
                       return shard.kind == ShardKind::kFloatSourceNorm2;
                     });
}

std::uint32_t IndexManifest::coefficient_alphabet_size() const noexcept {
  if ((format_version != kConfigurableCoefficientFormatVersion &&
       format_version != kConfigurableCodecFormatVersion &&
       (format_version != kFloatScaledFormatVersion &&
        format_version != kIndependentNormFormatVersion &&
        format_version != kReferencedShardFormatVersion &&
        format_version < kNormCacheLayoutFormatVersion)) ||
      !producer_recipe.has_value() ||
      producer_recipe->coefficient_bits_per_layer.empty()) {
    return CoefficientAlphabetFromBits(kDefaultCoefficientBits);
  }
  const auto& root = producer_recipe->coefficient_bits_per_layer_root.empty()
                         ? producer_recipe->coefficient_bits_per_layer
                         : producer_recipe->coefficient_bits_per_layer_root;
  const auto& linkage =
      producer_recipe->coefficient_bits_per_layer_linkage.empty()
          ? producer_recipe->coefficient_bits_per_layer
          : producer_recipe->coefficient_bits_per_layer_linkage;
  return CoefficientAlphabetFromBits(MaximumCoefficientBits(
      producer_recipe->coefficient_bits_per_layer, root, linkage));
}

}  // namespace stlqpp
