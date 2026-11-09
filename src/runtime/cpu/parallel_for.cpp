#include "stlqpp/runtime/cpu/parallel_for.h"

#include <cstdint>

namespace stlqpp {

void InlineParallelFor::Run(std::uint32_t item_count,
                            void* context,
                            ParallelForTask task) const noexcept {
  for (std::uint32_t item = 0U; item < item_count; ++item) {
    task(context, 0U, item);
  }
}

}  // namespace stlqpp
