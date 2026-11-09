#pragma once

#include "stlqpp/model/model_space_transform.h"

namespace stlqpp {

// Search-facing name retained at the public boundary; the representation is
// shared with offline training and index construction.
using QueryTransform = ModelSpaceTransform;

}  // namespace stlqpp
