#pragma once

#include "eval_command.h"

#include "stlqpp/core/status.h"

namespace stlqpp::app {

// Runs one fully parsed eval request, including manifest binding, affinity,
// logging and archive lifecycle. Both the standalone eval executable and
// advanced.eval_only in the product entry point use this exact path.
[[nodiscard]] Status RunExactDenseEval(
    EvalCommand command, char** argv, bool use_build_session = false);

}  // namespace stlqpp::app
