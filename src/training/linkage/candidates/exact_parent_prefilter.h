#pragma once

#include <cstdint>

namespace stlqpp::linkage_candidates {

[[nodiscard]] std::uint32_t PrefilterEightParentCandidatesAvx2(
    float query_norm,
    const float* parent_norms,
    const float* dots,
    const std::uint32_t* local_ids,
    float worst_distance,
    std::uint32_t worst_local_id,
    float* distances) noexcept;

}  // namespace stlqpp::linkage_candidates
