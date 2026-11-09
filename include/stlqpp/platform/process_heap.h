#pragma once

#include "stlqpp/core/status.h"

namespace stlqpp {

// Bounds glibc arena proliferation before offline-build worker threads start.
// Eight arenas preserve allocator concurrency for the fixed product topology
// while avoiding per-thread arena fragmentation. Other allocators use a no-op.
[[nodiscard]] Status ConfigureOfflineBuildProcessHeap();

}  // namespace stlqpp
