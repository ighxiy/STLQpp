#pragma once

#include "stlqpp/core/status.h"

#include <cstdint>
#include <mutex>
#include <shared_mutex>

namespace stlqpp {

// Configures the process-wide vendor BLAS worker count. Product construction
// sets this to one before any OpenMP work, matching the registered runtime
// policy and preventing nested BLAS/OpenMP oversubscription.
[[nodiscard]] Status ConfigureBlasThreadCount(
    std::uint32_t thread_count);

// Stage-local override. MKL uses its calling-thread policy; OpenBLAS holds
// the process-wide budget for the entire stage, then restores the prior one.
class ScopedBlasThreadCount final {
 public:
  explicit ScopedBlasThreadCount(std::uint32_t thread_count) noexcept;
  ~ScopedBlasThreadCount();

  ScopedBlasThreadCount(const ScopedBlasThreadCount&) = delete;
  ScopedBlasThreadCount& operator=(const ScopedBlasThreadCount&) = delete;

 private:
  std::unique_lock<std::shared_mutex> global_lock_;
  int previous_local_ = 0;
  std::uint32_t previous_global_ = 0U;
  bool active_ = false;
};

// A positive CBLAS adapter budget applies for one call. Same-budget
// OpenBLAS calls may run concurrently; different budgets cannot overlap.
class ScopedBlasCallThreadCount final {
 public:
  explicit ScopedBlasCallThreadCount(std::uint32_t thread_count) noexcept;
  ~ScopedBlasCallThreadCount();

  ScopedBlasCallThreadCount(const ScopedBlasCallThreadCount&) = delete;
  ScopedBlasCallThreadCount& operator=(
      const ScopedBlasCallThreadCount&) = delete;

 private:
  std::shared_lock<std::shared_mutex> shared_lock_;
  std::unique_lock<std::shared_mutex> exclusive_lock_;
  int previous_local_ = 0;
  std::uint32_t previous_global_ = 0U;
  bool changed_ = false;
};

}  // namespace stlqpp
