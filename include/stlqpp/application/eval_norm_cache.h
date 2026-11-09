#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "stlqpp/application/source_norm_provider.h"
#include "stlqpp/core/artifact_hash.h"
#include "stlqpp/index/norm/reconstruction_norm8_recipe.h"
#include "stlqpp/search/row_score_data.h"

namespace stlqpp {

struct EvalNormCacheRequest final {
  std::filesystem::path index;
  std::filesystem::path cache_root;
  // Includes the resolved source slice identity when source norms are external.
  std::string external_source_identity = {};
  bool native_float = false;
  bool float_norm = false;
  bool scaled = false;
  bool use_cuda = false;
  bool cuda_compute_allowed = true;
  int cuda_device = 0;
  ReconstructionNorm8Mode mode = ReconstructionNorm8Mode::kIdentityLloyd;
  std::uint32_t iterations = 25U;
  const SourceNormProvider* source = nullptr;
  const ParallelForExecutor* parallel = nullptr;
};

struct PreparedEvalNormCache final {
  IndexRowScores scores;
  ArtifactHash64 hash;
  std::filesystem::path directory;
  bool cache_hit = false;
  std::vector<std::filesystem::path> query_payloads;
};

// Reconstructs from the selected coefficients, fits on base rows only, and
// publishes an immutable recipe/content-addressed norm cache. LUT caches keep
// one byte per real row; exact norms are persisted only for an explicit exact
// norm request. Cache hits validate recipe and extents without rescanning
// full payload identities.
[[nodiscard]] StatusOr<PreparedEvalNormCache> PrepareEvalNormCache(
    const EvalNormCacheRequest& request);

}  // namespace stlqpp
