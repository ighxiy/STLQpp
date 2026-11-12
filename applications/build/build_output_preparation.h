#pragma once

#include "build_command.h"
#include "effective_build_recipe.h"

#include "stlqpp/core/status.h"

namespace stlqpp::app {

// Prepare only the artifact directories that this invocation will rebuild.
// Checkpoint-backed base encoding keeps its existing store for resume.
[[nodiscard]] Status PrepareBuildOutputDestinations(
    const BuildCommand& command, const EffectiveBuildRecipe& recipe);

}  // namespace stlqpp::app
