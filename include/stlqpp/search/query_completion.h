#pragma once
#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include <cstdint>
namespace stlqpp {
// Invoked by the query owner after ADC top-k finalization. IDs are borrowed
// for this call. A failure stops new queries; active owners finish and join.
using DenseQueryCompletion = Status (*)(void*, std::uint32_t worker,
    std::uint32_t query, ArrayView<const std::uint64_t> candidates) noexcept;
}  // namespace stlqpp
