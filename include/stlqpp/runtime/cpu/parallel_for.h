#pragma once

#include <cstdint>

namespace stlqpp {

using ParallelForTask = void (*)(void* context,
                                 std::uint32_t worker_index,
                                 std::uint32_t item_index) noexcept;

// Runtime-only scheduling boundary. Algorithms provide independent items and
// pre-sized worker-owned scratch; the runtime decides how workers execute.
class ParallelForExecutor {
 public:
  virtual ~ParallelForExecutor() = default;

  ParallelForExecutor(const ParallelForExecutor&) = delete;
  ParallelForExecutor& operator=(const ParallelForExecutor&) = delete;
  ParallelForExecutor(ParallelForExecutor&&) noexcept = default;
  ParallelForExecutor& operator=(ParallelForExecutor&&) noexcept = default;

  [[nodiscard]] virtual std::uint32_t worker_count() const noexcept = 0;
  virtual void Run(std::uint32_t item_count,
                   void* context,
                   ParallelForTask task) const noexcept = 0;

 protected:
  ParallelForExecutor() = default;
};

class InlineParallelFor final : public ParallelForExecutor {
 public:
  [[nodiscard]] std::uint32_t worker_count() const noexcept override {
    return 1U;
  }
  void Run(std::uint32_t item_count,
           void* context,
           ParallelForTask task) const noexcept override;
};

}  // namespace stlqpp
