#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/index/build/prepared_cluster_spool.h"
#include "stlqpp/index/codec/scalar_int6_encoder.h"
#include "stlqpp/index/norm/reconstruction_norm8.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/storage/canonical_index_writer.h"
#include "stlqpp/storage/canonical_model_codec.h"

namespace stlqpp {

template <int M>
struct FinalIndexPublishRequest final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::filesystem::path destination;
  CanonicalModelSource<M> model;
  IndexArtifactHashes hashes;
  std::vector<std::uint32_t> coefficient_bits_per_layer = {};
  std::vector<std::uint32_t> coefficient_bits_per_layer_root = {};
  std::vector<std::uint32_t> coefficient_bits_per_layer_linkage = {};
  std::vector<double> coefficient_first_magnitude_quantiles;
  std::vector<double> coefficient_rest_magnitude_quantiles;
  ScalarCoefficientCodecOptions coefficient_codec_options;
  ReconstructionNorm8Mode norm8_mode = ReconstructionNorm8Mode::kIdentityLloyd;
  std::uint32_t norm8_lloyd_iterations = 0U;
  bool use_cuda_norm8_fit = false;
  int cuda_norm_device = 0;
  bool retain_float_forest = true;
  bool retain_float_norm2 = true;
  bool publish_quantized_forest = true;
  bool publish_scaled_score8 = false;
  bool publish_float_scaled = false;
  std::filesystem::path model_reference = {};
};

struct FinalIndexPublicationStatistics final {
  CanonicalIndexWriteStatistics canonical;
  ReconstructionNorm8Mode norm8_mode = ReconstructionNorm8Mode::kIdentityLloyd;
  std::uint32_t norm8_lloyd_iterations = 0U;
  std::uint32_t norm8_center_count = 0U;
  // Absence means this producer did not measure the stage, not zero work.
  std::optional<ReconstructionNormPreparationTiming> reconstruction_norm_service;
  std::optional<double> norm8_fit_wall_seconds;
  // Sum of serial norm reads, label encoding and scaled-table fitting;
  // excludes canonical payload writes/commit and source-norm archival.
  std::optional<double> norm8_labels_scale_seconds;
  std::optional<double> coefficient_prepare_wall_seconds;
  std::optional<double> coefficient_service_seconds;
  // Delta writes are part of prepare/label wall, not an additional phase.
  std::optional<double> delta_write_seconds;
  std::optional<double> commit_wall_seconds;
  std::uint32_t coefficient_peak_workers = 0U;
  std::uint64_t coefficient_peak_window_bytes = 0U;
};

// Fits the one registered global norm8 model from the exact reconstruction
// norm stream. The public helper preserves the CPU reference path; product
// publication uses the explicitly requested GPU-norm backend for every
// population size, or fails if that backend cannot run.
[[nodiscard]] StatusOr<ReconstructionNorm8Codebook> FitPreparedSpoolNorm8(
    const PreparedClusterSpoolStorage& spool, ReconstructionNorm8Mode mode,
    std::uint32_t lloyd_iterations, const ParallelForExecutor& parallel);

template <int M>
[[nodiscard]] StatusOr<ReconstructionNorm8Codebook> FitPreparedSpoolNorm8(
    const PreparedClusterSpool<M>& spool, ReconstructionNorm8Mode mode,
    std::uint32_t lloyd_iterations, const ParallelForExecutor& parallel) {
  return FitPreparedSpoolNorm8(
      static_cast<const PreparedClusterSpoolStorage&>(spool), mode,
      lloyd_iterations, parallel);
}

template <int M>
[[nodiscard]] StatusOr<ReconstructionNorm8Codebook> FitPreparedSpoolNorm8(
    const PreparedClusterSpool<M>& spool, std::uint32_t lloyd_iterations,
    const ParallelForExecutor& parallel) {
  return FitPreparedSpoolNorm8(spool, ReconstructionNorm8Mode::kIdentityLloyd,
                               lloyd_iterations, parallel);
}

// Publishes norm metadata and the canonical manifest. Direct spools retain
// their payload files through one atomic directory rename; legacy spools copy
// their records. Temporary inputs are removed only after publication succeeds.
// Scaled publication preserves a CID-ordered source_norm2.pending stream for
// SourceNormProvider to bind to the source slice and consume into its stable-ID
// cache. This is a publication handoff, not a search payload or legacy shard.
template <int M>
[[nodiscard]] Status PublishPreparedClusterSpool(
    const std::filesystem::path& spool_directory,
    const FinalIndexPublishRequest<M>& request,
    const ParallelForExecutor& parallel,
    FinalIndexPublicationStatistics* statistics = nullptr);

extern template Status PublishPreparedClusterSpool<5>(
    const std::filesystem::path&, const FinalIndexPublishRequest<5>&,
    const ParallelForExecutor&, FinalIndexPublicationStatistics*);
extern template Status PublishPreparedClusterSpool<10>(
    const std::filesystem::path&, const FinalIndexPublishRequest<10>&,
    const ParallelForExecutor&, FinalIndexPublicationStatistics*);

}  // namespace stlqpp
