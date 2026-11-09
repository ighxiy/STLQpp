#include "stlqpp/application/registered_artifact_recipes.h"

#include "stlqpp/index/codec/scalar_int6_encoder.h"

#include <utility>

namespace stlqpp {

CandidateSearchRecipe RegisteredCandidateSearchRecipe() noexcept {
  return {
      .good_parent = GoodParentCandidateRecipe::kCpuInnerExact,
      .bad_virtual = BadVirtualCandidateRecipe::kCpuExact,
      .bad_graph = BadClusterGraphRecipe::kExactOnly,
      .parent_quota =
          ParentCandidateQuotaRecipe::kEligibleLayerSlotsHalfK,
  };
}

CoefficientCodecRecipe RegisteredCoefficientCodecRecipe(
    double magnitude_quantile) noexcept {
  return RegisteredCoefficientCodecRecipe(
      {magnitude_quantile}, {magnitude_quantile});
}

CoefficientCodecRecipe RegisteredCoefficientCodecRecipe(
    std::vector<double> first_magnitude_quantiles,
    std::vector<double> rest_magnitude_quantiles) noexcept {
  return {
      .bits_per_layer = {},
      .bits_per_layer_root = {},
      .bits_per_layer_linkage = {},
      .weighted_quantile = true,
      .first_magnitude_quantiles = std::move(first_magnitude_quantiles),
      .rest_magnitude_quantiles = std::move(rest_magnitude_quantiles),
      .allow_clipping = true,
      .fit_reconstruction_scales = true,
      .refinement_sweeps = kScalarInt6RefinementSweeps,
      .refine_all_layers = true,
      .refinement_max_layer = 0U,
      .refinement_step_limit = kScalarInt6RefinementStepLimit,
  };
}

}  // namespace stlqpp
