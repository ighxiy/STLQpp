#pragma once

#include "eval_command.h"

#include "stlqpp/core/status.h"

namespace stlqpp::app {

// Evaluate the durable base-basic product before linkage.  Coefficients are
// the float coefficients emitted by base encoding; only the reconstruction
// norm follows eval.disk_norm2_mode, matching DEV's base recall boundary.
[[nodiscard]] Status RunBaseBasicEval(const EvalCommand& command);

}  // namespace stlqpp::app
