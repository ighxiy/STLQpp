#include "stlqpp/training/initialization/cuda_spherical_rvq_backend.h"

#include "stlqpp/training/initialization/rvq_assignment_store.h"
#include "stlqpp/training/initialization/cuda_normalized_source_cache.h"
#include "stlqpp/runtime/telemetry/logger.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <random>
#include <optional>
#include <utility>
#include <vector>

namespace stlqpp {

template <typename Scalar>
StatusOr<SphericalRvqBackendResult>
CudaSphericalRvqInitializationBackend<Scalar>::Initialize(
    InitializationBasicBlockSourceFactory<Scalar>& source,
    const SphericalRvqInitializationRequest& request) {
  if (const Status status = request.recipe.Validate(); !status.ok()) {
    return status;
  }
  if (const Status status = request.shape.Validate(); !status.ok()) {
    return status;
  }
  if (request.source_rows == 0U) {
    return Status::InvalidArgument(
        "CUDA spherical RVQ backend request is invalid");
  }
  // RVQ follows DEV's stream-kernel math policy without mutating the shared
  // context used by basic/linkage after initialization.
  auto rvq_context = CudaContext::Create(
      context_.device(), context_.allow_tf32(),
      context_.allow_tf32() ? CudaBlasAtomics::kAllowed
                           : CudaBlasAtomics::kDisallowed);
  if (!rvq_context.ok()) return rvq_context.status();
  CudaContext& context = rvq_context.value();
  auto assignment_result =
      RvqAssignmentStore::Create(request.shape, request.source_rows);
  if (!assignment_result.ok()) {
    return assignment_result.status();
  }
  auto assignments = std::move(assignment_result).value();

  auto owned_codebooks = std::vector<std::vector<float>>(request.shape.m);
  auto codebook_views = std::vector<ColumnMajorMatrixF32View>(request.shape.m);
  auto layer_statistics = std::vector<SphericalRvqLayerStatistics>(request.shape.m);
  std::optional<CudaNormalizedSourceCache<Scalar>> normalized_source;
  bool full_source_norm2_cache = false;
  std::mt19937 random(request.seed);
  for (std::uint32_t layer = 0U; layer < request.shape.m;
       ++layer) {
    const auto seeding_started = std::chrono::steady_clock::now();
    auto seed = CudaKmeansParallelSeeder<Scalar>::Seed(
        context, source, request, codebook_views, layer, assignments,
        random, parallel_, options_.seeding);
    if (!seed.ok()) {
      return seed.status();
    }
    const double seeding_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - seeding_started).count();
    const std::uint32_t entries = layer == 0U ? request.shape.nlist : 256U;
    double normalized_cache_wall_seconds = 0.0;
    if (!normalized_source.has_value()) {
      const auto cache_started = std::chrono::steady_clock::now();
      auto built = CudaNormalizedSourceCache<Scalar>::Build(
          context, source,
          {.source_rows = request.source_rows,
           .dimension = request.shape.dimension,
           .maximum_bytes =
               options_.lloyd.maximum_normalized_cache_bytes});
      if (!built.ok()) {
        return built.status();
      }
      normalized_source.emplace(std::move(built).value());
      full_source_norm2_cache =
          normalized_source->cached_rows() == request.source_rows;
      normalized_cache_wall_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - cache_started).count();
      constexpr std::uint64_t kMebibyte = 1024U * 1024U;
      const std::uint64_t configured_mib =
          options_.lloyd.maximum_normalized_cache_bytes / kMebibyte;
      const std::uint64_t normalized_bytes =
          normalized_source->cached_rows() * request.shape.dimension *
          sizeof(float);
      LogInfo(
          "[kmeans_cache] device_cache_mb=" +
          std::to_string(configured_mib) +
          " cache_mb=" + std::to_string(normalized_bytes / kMebibyte) +
          " cache_cols=" +
          std::to_string(normalized_source->cached_rows()) +
          " d=" + std::to_string(request.shape.dimension) +
          " block_cols=" + std::to_string(request.source_block_rows) +
          " n=" + std::to_string(request.source_rows));
    }
    auto trained =
        CudaStreamingSphericalLloyd<Scalar>::TrainWithNormalizedSource(
        context, parallel_, normalized_source.value(), request,
        codebook_views, layer, assignments,
        {seed.value().centers.data(), request.shape.dimension, entries},
        {seed.value().empty_cluster_fallback.data(), request.shape.dimension,
         entries},
        options_.lloyd, normalized_cache_wall_seconds);
    if (!trained.ok()) {
      return trained.status();
    }
    auto trained_value = std::move(trained).value();
    layer_statistics[layer] = trained_value.statistics;
    layer_statistics[layer].seeding_wall_seconds = seeding_wall_seconds;
    layer_statistics[layer].seed_sample_wall_seconds =
        seed.value().statistics.sample_wall_seconds;
    layer_statistics[layer].seed_candidate_wall_seconds =
        seed.value().statistics.candidate_wall_seconds;
    layer_statistics[layer].seed_reduction_wall_seconds =
        seed.value().statistics.reduction_wall_seconds;
    layer_statistics[layer].seed_fallback_wall_seconds =
        seed.value().statistics.fallback_wall_seconds;
    layer_statistics[layer].seed_rounds = seed.value().statistics.rounds;
    layer_statistics[layer].seed_candidates =
        seed.value().statistics.reduced_candidates;
    owned_codebooks[layer] = std::move(trained_value.centers);
    codebook_views[layer] = {
        owned_codebooks[layer].data(), request.shape.dimension, entries};
  }

  auto small = std::vector<std::vector<float>>(request.shape.m - 1U);
  for (std::uint32_t layer = 1U; layer < request.shape.m;
       ++layer) {
    small[layer - 1U] = std::move(owned_codebooks[layer]);
  }
  return SphericalRvqBackendResult{
      .codebooks = InitializedRootCodebooks(
          request.shape, std::move(owned_codebooks[0]), std::move(small)),
      .layers = layer_statistics,
      .cuda_diagnostics = SphericalRvqCudaDiagnostics{
          .overlapped_pass1 = false,
          .overlapped_pass2 = request.source_block_rows != 0U &&
              request.source_rows > request.source_block_rows,
          .full_source_norm2_cache = full_source_norm2_cache,
          .used_pinned_host_source = true,
      },
  };
}

template class CudaSphericalRvqInitializationBackend<std::uint8_t>;
template class CudaSphericalRvqInitializationBackend<float>;

}  // namespace stlqpp
