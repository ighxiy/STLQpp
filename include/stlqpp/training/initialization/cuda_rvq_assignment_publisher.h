#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/training/initialization/rvq_assignment_store.h"

#include <cstdint>
#include <memory>

namespace stlqpp {

// DEV's two-slot retained-code publication protocol. The reusable scorer
// output is first staged D2D on the compute stream. D2H then runs on the same
// copy stream used by source uploads, and synchronizes only at slot reuse or
// final ordered drain.
class CudaRvqAssignmentPublisher final {
 public:
  [[nodiscard]] static StatusOr<CudaRvqAssignmentPublisher> Create(
      CudaContext& context,
      void* native_copy_stream,
      RvqAssignmentStore& destination,
      std::uint32_t layer);

  ~CudaRvqAssignmentPublisher();
  CudaRvqAssignmentPublisher(CudaRvqAssignmentPublisher&&) noexcept;
  CudaRvqAssignmentPublisher& operator=(
      CudaRvqAssignmentPublisher&&) noexcept;
  CudaRvqAssignmentPublisher(const CudaRvqAssignmentPublisher&) = delete;
  CudaRvqAssignmentPublisher& operator=(
      const CudaRvqAssignmentPublisher&) = delete;

  [[nodiscard]] Status Schedule(std::uint8_t slot,
                                std::uint64_t first_id,
                                const std::uint32_t* device_codes,
                                std::uint32_t count);
  [[nodiscard]] Status Drain();

 private:
  struct State;
  explicit CudaRvqAssignmentPublisher(std::unique_ptr<State> state) noexcept;
  [[nodiscard]] Status Flush(std::uint8_t slot);

  std::unique_ptr<State> state_;
};

}  // namespace stlqpp
