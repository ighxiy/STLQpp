#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cpu/parallel_for.h"

#include <cstdint>

namespace stlqpp {

// Returns the positive OpenMP team size selected by the current process
// environment/runtime. Public config uses zero to request this value.
[[nodiscard]] std::uint32_t OpenMpDefaultWorkerCount() noexcept;

class OpenMpParallelFor final : public ParallelForExecutor {
 public:
  [[nodiscard]] static StatusOr<OpenMpParallelFor> Create(
      std::uint32_t worker_count);

  [[nodiscard]] std::uint32_t worker_count() const noexcept override {
    return worker_count_;
  }
  void Run(std::uint32_t item_count,
           void* context,
           ParallelForTask task) const noexcept override;

 private:
  explicit OpenMpParallelFor(std::uint32_t worker_count) noexcept
      : worker_count_(worker_count) {}

  std::uint32_t worker_count_ = 1U;
};

}  // namespace stlqpp
