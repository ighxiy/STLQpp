#pragma once

#include "stlqpp/core/artifact_hash.h"
#include "stlqpp/core/status.h"
#include "stlqpp/index/norm/reconstruction_norm8_recipe.h"
#include "stlqpp/model/model_shape.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace stlqpp {

enum class CoefficientCodec : std::uint16_t {
  kPerCidScalarHuffman = 1,
};

enum class NormCodec : std::uint16_t {
  kReconstructionNorm8 = 1,
  // Classic reconstruction norm8 plus an optional second score table sharing
  // the same one-byte labels. This is the production scaled representation.
  kReconstructionNorm8WithScaledScore8 = 3,
};

enum class ShardKind : std::uint16_t {
  kCodebooks = 1,
  kIvfDirectory,
  kCodes,
  kParents,
  kCoefficients,
  kNorm8,
  kFloatCoefficients,
  kFloatNorm2,
  kScaledScore8,
  // Optional original-vector norm2 companion used only by exact
  // native-float radial scaling.
  kFloatSourceNorm2,
  kNorm2Lut, // v14: raw 256 float32 centers; kNorm8 holds only labels.
};

struct ShardDescriptor final {
  ShardKind kind = ShardKind::kCodebooks;
  std::uint32_t ordinal = 0;
  std::string relative_path;
  std::uint64_t bytes = 0;
  std::uint64_t elements = 0;
  // A reference is metadata only: no copy, hard link or symbolic link is
  // published at relative_path. Paths are relative to the owning manifest.
  std::string reference_path = {};
  // Legacy checksum is decoded for compatibility, never rescanned on Open.
  // Newly published references leave it empty.
  std::string reference_sha256 = {};
  std::uint64_t payload_offset = 0U;
};

struct IndexCounts final {
  std::uint64_t real = 0;
  std::uint64_t virtual_roots = 0;
  std::uint64_t real_roots = 0;
  std::uint64_t linked_real = 0;
};

struct IndexArtifactHashes final {
  ArtifactHash64 linkage_list_directory;
  ArtifactHash64 coefficient_codec;
  // Identity of the classic reconstruction-norm8 representation. The v7
  // scaled score table has an independent hash so retaining it cannot change
  // the identity or availability of classic search.
  ArtifactHash64 norm8_cache;
};

// Producer parameters that materially determine the persisted coefficient
// and norm8 payloads. They are stored as provenance, not as search switches.
struct IndexProducerRecipe final {
  std::vector<std::uint32_t> coefficient_bits_per_layer = {};
  std::vector<std::uint32_t> coefficient_bits_per_layer_root = {};
  std::vector<std::uint32_t> coefficient_bits_per_layer_linkage = {};
  std::vector<double> coefficient_first_magnitude_quantiles;
  std::vector<double> coefficient_rest_magnitude_quantiles;
  bool coefficient_weighted_quantile = true;
  bool coefficient_allow_clipping = true;
  bool coefficient_fit_scale = true;
  std::uint32_t coefficient_refinement_sweeps = 3U;
  std::uint32_t coefficient_refinement_max_layer = 0U;
  std::uint32_t coefficient_refinement_step_limit = 1U;
  ReconstructionNorm8Mode norm8_mode =
      ReconstructionNorm8Mode::kIdentityLloyd;
  std::uint32_t norm8_lloyd_iterations = 0U;
};

struct IndexManifest final {
  static constexpr std::uint32_t kLegacyFormatVersion = 1U;
  static constexpr std::uint32_t kScalarRecipeFormatVersion = 2U;
  static constexpr std::uint32_t kMultiCandidateRecipeFormatVersion = 3U;
  static constexpr std::uint32_t kNormModeRecipeFormatVersion = 4U;
  static constexpr std::uint32_t kCurrentFormatVersion = 5U;
  static constexpr std::uint32_t kScaledScoreFormatVersion = 7U;
  static constexpr std::uint32_t kConfigurableCoefficientFormatVersion = 8U;
  static constexpr std::uint32_t kConfigurableCodecFormatVersion = 9U;
  // Canonical topology plus native-float coefficients/norms, with the
  // coefficient codec and norm8 representation deliberately absent.
  static constexpr std::uint32_t kFloatOnlyFormatVersion = 10U;
  static constexpr std::uint32_t kFloatScaledFormatVersion = 11U;
  // Native coefficients are durable; their evaluation norm is optional and
  // may be prepared independently for float or LUT scoring.
  static constexpr std::uint32_t kIndependentNormFormatVersion = 12U;
  static constexpr std::uint32_t kReferencedShardFormatVersion = 13U;

  static constexpr std::uint32_t kNormCacheLayoutFormatVersion = 14U;
  // Explicit native-float offsets and arbitrary physical CID record order.
  static constexpr std::uint32_t kCompletionOrderFormatVersion = 15U;

  std::uint32_t format_version = kCurrentFormatVersion;
  ModelShape shape;
  IndexCounts counts;
  IndexArtifactHashes hashes;
  CoefficientCodec coefficient_codec =
      CoefficientCodec::kPerCidScalarHuffman;
  NormCodec norm_codec = NormCodec::kReconstructionNorm8;
  // Empty unless the index has the v7 scaled score table. Classic norm8 keeps
  // its established identity in hashes.norm8_cache.
  ArtifactHash64 scaled_score8_hash;
  std::optional<IndexProducerRecipe> producer_recipe;
  std::vector<ShardDescriptor> shards;

  [[nodiscard]] bool has_float_forest() const noexcept;
  [[nodiscard]] bool has_float_norm2() const noexcept;
  [[nodiscard]] bool has_quantized_forest() const noexcept;
  [[nodiscard]] std::uint32_t coefficient_alphabet_size() const noexcept;
  [[nodiscard]] bool has_classic_norm8() const noexcept {
    return has_quantized_forest();
  }
  [[nodiscard]] bool has_scaled_score8() const noexcept {
    return norm_codec == NormCodec::kReconstructionNorm8WithScaledScore8;
  }
  [[nodiscard]] bool has_float_source_norm2() const noexcept;

  [[nodiscard]] Status Validate() const;
};

}  // namespace stlqpp
