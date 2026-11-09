#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"

#include <cstdint>
#include <vector>

namespace stlqpp::internal {

// Only for parent arrays already checked for causal BFS order by the
// canonical cluster codec. The public encoder remains the checked boundary.
[[nodiscard]] Status EncodeValidatedParentLouds(
    ArrayView<const std::uint32_t> parents,
    std::vector<std::uint64_t>& words);

}  // namespace stlqpp::internal
