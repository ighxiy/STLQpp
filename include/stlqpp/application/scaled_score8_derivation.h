#pragma once

#include "stlqpp/application/source_norm_provider.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/index/norm/reconstruction_norm8_recipe.h"
#include "stlqpp/model/scaled_score8_codebook.h"
#include "stlqpp/runtime/cpu/parallel_for.h"

#include <filesystem>

namespace stlqpp {

[[nodiscard]] StatusOr<ScaledScore8Codebook> FitScaledScore8Runtime(
    const std::filesystem::path& index, const SourceNormProvider* source,
    const ParallelForExecutor& parallel, bool use_cuda_norm = false,
    int cuda_device = 0);

// Optional scale publication consumes the actual new labels and the same
// norm population; nullptr original_norms opens the shared original-norm cache.
[[nodiscard]] Status PublishReconstructionNorm8Runtime(
    const std::filesystem::path& source,
    const std::filesystem::path& destination, ReconstructionNorm8Mode mode,
    std::uint32_t lloyd_iterations, bool use_cuda = true, int cuda_device = 0,
    bool publish_scaled = false, const SourceNormProvider* original_norms = nullptr);

// Fits a scaled score table for an existing compressed forest. When source is
// null, original-vector norms must be retained by the source index.
template <int M>
[[nodiscard]] StatusOr<ScaledScore8Codebook> FitScaledScore8(
    const std::filesystem::path& index,
    const SourceNormProvider* source,
    const ParallelForExecutor& parallel, bool use_cuda_norm = false,
    int cuda_device = 0);

extern template StatusOr<ScaledScore8Codebook> FitScaledScore8<5>(
    const std::filesystem::path&, const SourceNormProvider*,
    const ParallelForExecutor&, bool, int);
extern template StatusOr<ScaledScore8Codebook> FitScaledScore8<10>(
    const std::filesystem::path&, const SourceNormProvider*,
    const ParallelForExecutor&, bool, int);

// Create-only derived index: references immutable source shards and adds the
// fitted score table without rewriting the forest.
[[nodiscard]] Status PublishScaledScore8(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    ScaledScore8CodebookView table);

// Refit only the one-byte reconstruction-norm representation. Topology,
// coefficient bytes and native-float records remain unchanged.
template <int M>
[[nodiscard]] Status PublishReconstructionNorm8(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    ReconstructionNorm8Mode mode,
    std::uint32_t lloyd_iterations,
    bool use_cuda = true, int cuda_device = 0);

extern template Status PublishReconstructionNorm8<5>(
    const std::filesystem::path&, const std::filesystem::path&,
    ReconstructionNorm8Mode, std::uint32_t, bool, int);
extern template Status PublishReconstructionNorm8<10>(
    const std::filesystem::path&, const std::filesystem::path&,
    ReconstructionNorm8Mode, std::uint32_t, bool, int);

}  // namespace stlqpp
