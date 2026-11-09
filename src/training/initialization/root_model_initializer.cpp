#include "stlqpp/training/initialization/root_model_initializer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

template <int M>
Status ValidateRequest(const SphericalRvqInitializationRequest& request) {
  if (const Status status = request.shape.Validate(); !status.ok()) {
    return status;
  }
  if ((M != 0 && request.shape.m != static_cast<std::uint32_t>(M)) ||
      request.source_rows == 0U || request.source_block_rows == 0U) {
    return Status::InvalidArgument(
        "root-model initializer received an invalid product source");
  }
  return request.recipe.Validate();
}

}  // namespace

Status SphericalRvqInitializationRecipe::Validate() const {
  if (seed_method != SphericalSeedMethod::kKmeansParallel &&
      seed_method != SphericalSeedMethod::kRandom &&
      seed_method != SphericalSeedMethod::kKmeansPlusPlus) {
    return Status::InvalidArgument("unknown spherical seeding method");
  }
  if (sample_vectors == 0U || maximum_iterations == 0U ||
      !std::isfinite(tolerance) || tolerance < 0.0F ||
      kmeans_parallel_rounds == 0U || kmeans_parallel_rounds > 16U ||
      !std::isfinite(kmeans_parallel_oversample) ||
      kmeans_parallel_oversample < 1.0F ||
      kmeans_parallel_oversample > 16.0F ||
      !std::isfinite(initial_weight) || initial_weight <= 0.0F ||
      initial_weight > 1.0F || !std::isfinite(minimum_weight) ||
      minimum_weight <= 0.0F || minimum_weight > initial_weight ||
      !std::isfinite(outlier_quantile) || outlier_quantile < 0.0F ||
      outlier_quantile > 1.0F || !std::isfinite(cost_threshold) ||
      cost_threshold < 0.0F || cost_threshold > 2.0F ||
      !std::isfinite(annealing_factor) || annealing_factor < 0.0F ||
      annealing_factor > 1.0F ||
      quantile_bins == 0U || large_layer_threshold == 0U ||
      large_layer_top_coarse == 0U || large_layer_top_coarse > 2U ||
      large_layer_minimum_coarse_count == 0U ||
      large_layer_maximum_coarse_count < large_layer_minimum_coarse_count) {
    return Status::InvalidArgument(
        "spherical RVQ initialization recipe is invalid");
  }
  return Status::Ok();
}

std::uint64_t SphericalRvqInitializationRecipe::EffectiveSampleVectors(
    std::uint64_t source_rows) const noexcept {
  return std::min(sample_vectors, source_rows);
}

std::uint32_t SphericalRvqInitializationRecipe::EffectiveCandidateCap(
    std::uint32_t center_count,
    std::uint32_t sample_count) const noexcept {
  const std::uint64_t automatic = std::min<std::uint64_t>(
      static_cast<std::uint64_t>(center_count) * 16U, 1'000'000U);
  const std::uint64_t requested =
      candidate_cap == 0U ? automatic : candidate_cap;
  const std::uint64_t bounded_below =
      std::max<std::uint64_t>(center_count, requested);
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(sample_count, bounded_below));
}

template <int M, typename Scalar>
StatusOr<RootModelInitializationResult<M>> InitializeRootTrainingModel(
    const SphericalRvqInitializationRequest& request,
    InitializationBasicBlockSourceFactory<Scalar>& source,
    SphericalRvqInitializationBackend<Scalar>& backend) {
  if (const Status status = ValidateRequest<M>(request); !status.ok()) {
    return status;
  }
  auto initialized = backend.Initialize(source, request);
  if (!initialized.ok()) {
    return initialized.status();
  }
  const ModelShape& shape = initialized.value().codebooks.shape();
  if (shape.dimension != request.shape.dimension ||
      shape.m != request.shape.m ||
      shape.nlist != request.shape.nlist ||
      shape.residual_codebook_size !=
          request.shape.residual_codebook_size) {
    return Status::DataLoss(
        "spherical RVQ backend changed the registered model shape");
  }
  const RootCodebookView<M> codebooks =
      initialized.value().codebooks.template view<M>();
  std::vector<float> identity(
      static_cast<std::size_t>(request.shape.dimension) *
          request.shape.dimension,
      0.0F);
  for (std::uint32_t axis = 0U; axis < request.shape.dimension; ++axis) {
    identity[static_cast<std::size_t>(axis) * request.shape.dimension + axis] =
        1.0F;
  }
  auto model = RootTrainingModel<M>::Create(
      {identity.data(), request.shape.dimension, request.shape.dimension},
      codebooks);
  if (!model.ok()) {
    return model.status();
  }
  auto layers = MakeModelLayers<SphericalRvqLayerStatistics, M>(request.shape.m);
  if (initialized.value().layers.size() != request.shape.m) {
    return Status::DataLoss("spherical RVQ backend changed the layer count");
  }
  std::copy(initialized.value().layers.begin(), initialized.value().layers.end(),
            layers.begin());
  return RootModelInitializationResult<M>{
      .model = std::move(model).value(),
      .layers = std::move(layers),
      .cuda_diagnostics = initialized.value().cuda_diagnostics,
  };
}

template StatusOr<RootModelInitializationResult<0>>
InitializeRootTrainingModel<0>(const SphericalRvqInitializationRequest&,
    InitializationU8BlockSourceFactory&,
    SphericalRvqInitializationBackend<std::uint8_t>&);
template StatusOr<RootModelInitializationResult<0>>
InitializeRootTrainingModel<0>(const SphericalRvqInitializationRequest&,
    InitializationF32BlockSourceFactory&,
    SphericalRvqInitializationBackend<float>&);

template StatusOr<RootModelInitializationResult<5>>
InitializeRootTrainingModel<5>(
    const SphericalRvqInitializationRequest&,
    InitializationU8BlockSourceFactory&,
    SphericalRvqInitializationBackend<std::uint8_t>&);
template StatusOr<RootModelInitializationResult<12>>
InitializeRootTrainingModel<12>(
    const SphericalRvqInitializationRequest&,
    InitializationU8BlockSourceFactory&,
    SphericalRvqInitializationBackend<std::uint8_t>&);
template StatusOr<RootModelInitializationResult<6>>
InitializeRootTrainingModel<6>(
    const SphericalRvqInitializationRequest&,
    InitializationU8BlockSourceFactory&,
    SphericalRvqInitializationBackend<std::uint8_t>&);
template StatusOr<RootModelInitializationResult<10>>
InitializeRootTrainingModel<10>(
    const SphericalRvqInitializationRequest&,
    InitializationU8BlockSourceFactory&,
    SphericalRvqInitializationBackend<std::uint8_t>&);
template StatusOr<RootModelInitializationResult<5>>
InitializeRootTrainingModel<5>(
    const SphericalRvqInitializationRequest&,
    InitializationF32BlockSourceFactory&,
    SphericalRvqInitializationBackend<float>&);
template StatusOr<RootModelInitializationResult<12>>
InitializeRootTrainingModel<12>(
    const SphericalRvqInitializationRequest&,
    InitializationF32BlockSourceFactory&,
    SphericalRvqInitializationBackend<float>&);
template StatusOr<RootModelInitializationResult<6>>
InitializeRootTrainingModel<6>(
    const SphericalRvqInitializationRequest&,
    InitializationF32BlockSourceFactory&,
    SphericalRvqInitializationBackend<float>&);
template StatusOr<RootModelInitializationResult<10>>
InitializeRootTrainingModel<10>(
    const SphericalRvqInitializationRequest&,
    InitializationF32BlockSourceFactory&,
    SphericalRvqInitializationBackend<float>&);

}  // namespace stlqpp
