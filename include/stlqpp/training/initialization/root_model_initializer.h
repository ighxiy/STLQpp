#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/io/basic_block_source.h"
#include "stlqpp/training/update/root_training_model.h"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace stlqpp {

enum class SphericalSeedMethod : std::uint8_t {
  kKmeansParallel = 0U,
  kRandom = 1U,
  kKmeansPlusPlus = 2U,
};

// Immutable registered spherical-RVQ recipe. sample_vectors is deliberately
// explicit because the source may contain 100K, 500K or 30M training rows;
// EffectiveSampleVectors applies the registered automatic shrink.
struct SphericalRvqInitializationRecipe final {
  SphericalSeedMethod seed_method = SphericalSeedMethod::kKmeansParallel;
  std::uint64_t sample_vectors = 0U;
  std::uint32_t maximum_iterations = 40U;
  float tolerance = 1.0e-6F;
  std::uint32_t kmeans_parallel_rounds = 8U;
  float kmeans_parallel_oversample = 16.0F;
  // DEV train.kmeansll_seed. Zero consumes the caller's training RNG;
  // nonzero is the explicit kmeans|| sampling seed.
  std::uint64_t kmeans_parallel_seed = 0U;
  // Zero is the Master/registered-DEV automatic policy:
  // min(16 * K, 1,000,000),
  // subsequently bounded below by K and above by the sample size. A nonzero
  // value is the explicit user cap before the same K/sample bounds.
  std::uint32_t candidate_cap = 0U;
  float initial_weight = 1.0F;
  float minimum_weight = 0.1F;
  float outlier_quantile = 0.85F;
  float cost_threshold = 0.115F;
  float annealing_factor = 0.9F;
  std::uint32_t warmup_iterations = 2U;
  std::uint32_t quantile_bins = 65'536U;
  std::uint32_t large_layer_threshold = 4'096U;
  // DEV's large-K hierarchical scorer searches one or two coarse groups.
  std::uint32_t large_layer_top_coarse = 1U;
  bool large_layer_hierarchy_enabled = true;
  std::uint32_t large_layer_fixed_coarse_count = 0U;
  std::uint32_t large_layer_minimum_coarse_count = 128U;
  std::uint32_t large_layer_maximum_coarse_count = 1'024U;
  bool large_layer_round_coarse_to_power_of_two = true;

  [[nodiscard]] Status Validate() const;
  [[nodiscard]] std::uint64_t EffectiveSampleVectors(
      std::uint64_t source_rows) const noexcept;
  [[nodiscard]] std::uint32_t EffectiveCandidateCap(
      std::uint32_t center_count,
      std::uint32_t sample_count) const noexcept;
};

struct SphericalRvqInitializationRequest final {

  ModelShape shape;
  std::uint64_t source_rows = 0U;
  // Effective dataset-reader block width. This is execution metadata used by
  // the stage owner for the DEV-compatible initialization transcript; it does
  // not alter the spherical-RVQ recipe or artifact identity.
  std::uint32_t source_block_rows = 0U;
  std::uint32_t seed = 0U;
  SphericalRvqInitializationRecipe recipe;
};

struct SphericalRvqLayerStatistics final {
  std::uint32_t iterations = 0U;
  double final_cost = 0.0;
  double final_mse_proxy = 0.0;
  double seeding_wall_seconds = 0.0;
  double seed_sample_wall_seconds = 0.0;
  double seed_candidate_wall_seconds = 0.0;
  double seed_reduction_wall_seconds = 0.0;
  double seed_fallback_wall_seconds = 0.0;
  std::uint32_t seed_rounds = 0U;
  std::uint32_t seed_candidates = 0U;
  double normalized_cache_wall_seconds = 0.0;
  double wall_seconds = 0.0;
};

// Backend-observed execution features rendered by the DEV-compatible RVQ
// diagnostic line. Absence means that the selected backend has no CUDA
// execution to report; values are never inferred from requested config.
struct SphericalRvqCudaDiagnostics final {
  bool overlapped_pass1 = false;
  bool overlapped_pass2 = false;
  bool full_source_norm2_cache = false;
  bool used_pinned_host_source = false;
};

class InitializedRootCodebooks final {
 public:
  InitializedRootCodebooks(ModelShape shape,
                           std::vector<float> root0,
                           std::vector<std::vector<float>> root_small)
      noexcept
      : shape_(shape),
        root0_(std::move(root0)),
        root_small_(std::move(root_small)) {}

  InitializedRootCodebooks(InitializedRootCodebooks&&) noexcept = default;
  InitializedRootCodebooks& operator=(InitializedRootCodebooks&&) noexcept =
      default;
  InitializedRootCodebooks(const InitializedRootCodebooks&) = delete;
  InitializedRootCodebooks& operator=(const InitializedRootCodebooks&) =
      delete;

  // Only the consumer's numerical view is typed; RVQ ownership is not.
  [[nodiscard]] const ModelShape& shape() const noexcept { return shape_; }

  template <int M = 0>
  [[nodiscard]] RootCodebookView<M> view() const noexcept(M != 0) {
    RootCodebookView<M> output;
    output.shape = shape_;
    if constexpr (M == 0) output.root_small.resize(shape_.m - 1U);
    output.root0 = {root0_.data(), shape_.dimension, shape_.nlist};
    const std::uint32_t layers = M == 0 ? shape_.m : M;
    for (std::uint32_t layer = 1U; layer < layers; ++layer) {
      output.root_small[layer - 1U] = {
          root_small_[layer - 1U].data(), shape_.dimension, 256U};
    }
    return output;
  }

 private:
  ModelShape shape_;
  std::vector<float> root0_;
  std::vector<std::vector<float>> root_small_;
};

struct SphericalRvqBackendResult final {

  InitializedRootCodebooks codebooks;
  std::vector<SphericalRvqLayerStatistics> layers;
  std::optional<SphericalRvqCudaDiagnostics> cuda_diagnostics;
};

template <typename Scalar = std::uint8_t>
class SphericalRvqInitializationBackend {
 public:
  virtual ~SphericalRvqInitializationBackend() = default;
  SphericalRvqInitializationBackend(
      const SphericalRvqInitializationBackend&) = delete;
  SphericalRvqInitializationBackend& operator=(
      const SphericalRvqInitializationBackend&) = delete;

  [[nodiscard]] virtual StatusOr<SphericalRvqBackendResult> Initialize(
      InitializationBasicBlockSourceFactory<Scalar>& source,
      const SphericalRvqInitializationRequest& request) = 0;

 protected:
  SphericalRvqInitializationBackend() = default;
};

template <int M>
struct RootModelInitializationResult final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  RootTrainingModel<M> model;
  ModelLayers<SphericalRvqLayerStatistics, M> layers{};
  std::optional<SphericalRvqCudaDiagnostics> cuda_diagnostics;
};

template <int M, typename Scalar>
[[nodiscard]] StatusOr<RootModelInitializationResult<M>>
InitializeRootTrainingModel(
    const SphericalRvqInitializationRequest& request,
    InitializationBasicBlockSourceFactory<Scalar>& source,
    SphericalRvqInitializationBackend<Scalar>& backend);

using RuntimeRootModelInitializationResult = RootModelInitializationResult<0>;

extern template StatusOr<RootModelInitializationResult<0>>
InitializeRootTrainingModel<0>(const SphericalRvqInitializationRequest&,
    InitializationU8BlockSourceFactory&,
    SphericalRvqInitializationBackend<std::uint8_t>&);
extern template StatusOr<RootModelInitializationResult<0>>
InitializeRootTrainingModel<0>(const SphericalRvqInitializationRequest&,
    InitializationF32BlockSourceFactory&,
    SphericalRvqInitializationBackend<float>&);

extern template StatusOr<RootModelInitializationResult<5>>
InitializeRootTrainingModel<5>(
    const SphericalRvqInitializationRequest&,
    InitializationU8BlockSourceFactory&,
    SphericalRvqInitializationBackend<std::uint8_t>&);
extern template StatusOr<RootModelInitializationResult<10>>
InitializeRootTrainingModel<10>(
    const SphericalRvqInitializationRequest&,
    InitializationU8BlockSourceFactory&,
    SphericalRvqInitializationBackend<std::uint8_t>&);
extern template StatusOr<RootModelInitializationResult<5>>
InitializeRootTrainingModel<5>(
    const SphericalRvqInitializationRequest&,
    InitializationF32BlockSourceFactory&,
    SphericalRvqInitializationBackend<float>&);
extern template StatusOr<RootModelInitializationResult<10>>
InitializeRootTrainingModel<10>(
    const SphericalRvqInitializationRequest&,
    InitializationF32BlockSourceFactory&,
    SphericalRvqInitializationBackend<float>&);

}  // namespace stlqpp
