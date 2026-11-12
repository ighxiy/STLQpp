#pragma once

#include "build_command.h"
#include "effective_build_recipe.h"

#include <iosfwd>

namespace stlqpp::app {

// Machine-readable, stable view consumed by the experiment preflight.  The
// canonical config comes from the production parser; the effective section
// exposes resolver clamping/fallbacks; and the consumer section records phase
// resource ownership rather than trusting option names.
void WriteBuildControlManifest(
    const BuildCommand& command,
    const EffectiveBuildRecipe& recipe,
    std::ostream& output);

}  // namespace stlqpp::app
