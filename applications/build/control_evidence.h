#pragma once

#include "build_command.h"

#include "stlqpp/core/status.h"

namespace stlqpp::app {

struct EffectiveBuildRecipe;

// Validates a versioned guard record against the exact executable, resolved
// production manifest and input identities before a run session is created.
[[nodiscard]] Status ValidateControlEvidence(
    const BuildCommand& command,
    const EffectiveBuildRecipe& recipe);

}  // namespace stlqpp::app
