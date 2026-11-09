#pragma once

#include "stlqpp/index/norm/reconstruction_norm8_recipe.h"
#include "stlqpp/storage/artifact_hash_factory.h"

#include <cstdint>

namespace stlqpp {

// Product semantics that affect persisted identities but are intentionally
// not runtime feature switches. The application composition layer is the one
// owner that translates the registered algorithms into hash recipes.
[[nodiscard]] CandidateSearchRecipe
RegisteredCandidateSearchRecipe() noexcept;

[[nodiscard]] CoefficientCodecRecipe
RegisteredCoefficientCodecRecipe(double magnitude_quantile) noexcept;

[[nodiscard]] CoefficientCodecRecipe RegisteredCoefficientCodecRecipe(
    std::vector<double> first_magnitude_quantiles,
    std::vector<double> rest_magnitude_quantiles) noexcept;

}  // namespace stlqpp
