#include "stlqpp/platform/process_heap.h"

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace stlqpp {

Status ConfigureOfflineBuildProcessHeap() {
#if defined(__GLIBC__)
  constexpr int kMaximumArenaCount = 8;
  if (mallopt(M_ARENA_MAX, kMaximumArenaCount) == 0) {
    return Status::Internal(
        "failed to configure the offline-build malloc arena limit");
  }
#endif
  return Status::Ok();
}

}  // namespace stlqpp
