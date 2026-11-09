#include "stlqpp/runtime/cpu/openmp_parallel_for.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include <omp.h>

namespace stlqpp {

std::uint32_t OpenMpDefaultWorkerCount() noexcept {
  return static_cast<std::uint32_t>(std::max(1, omp_get_max_threads()));
}

StatusOr<OpenMpParallelFor> OpenMpParallelFor::Create(
    std::uint32_t worker_count) {
  const std::uint32_t effective_worker_count = worker_count == 0U
      ? OpenMpDefaultWorkerCount() : worker_count;
  if (effective_worker_count > static_cast<std::uint32_t>(
                         std::numeric_limits<int>::max())) {
    return Status::InvalidArgument(
        "OpenMP worker count must fit a positive int");
  }
  omp_set_dynamic(0);
#if defined(_OPENMP) && (_OPENMP >= 200805)
  omp_set_max_active_levels(1);
#else
  omp_set_nested(0);
#endif
  return OpenMpParallelFor(effective_worker_count);
}

void OpenMpParallelFor::Run(std::uint32_t item_count,
                            void* context,
                            ParallelForTask task) const noexcept {
  const int threads = static_cast<int>(worker_count_);
  #pragma omp parallel for default(none) schedule(static) num_threads(threads) \
      shared(context, task, item_count)
  for (std::int64_t item = 0; item < static_cast<std::int64_t>(item_count);
       ++item) {
    task(context, static_cast<std::uint32_t>(omp_get_thread_num()),
         static_cast<std::uint32_t>(item));
  }
}

}  // namespace stlqpp
