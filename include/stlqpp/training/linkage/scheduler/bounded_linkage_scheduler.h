#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <vector>

#include "stlqpp/core/status_or.h"
#include "stlqpp/training/linkage/scheduler/linkage_evaluation_service.h"
#include "stlqpp/training/linkage/scheduler/linkage_worker_backend.h"

namespace stlqpp {

inline constexpr std::uint32_t kMaximumSingleLinkageTaskPairs = 32768U;

struct LinkageSchedulerOptions final {
  std::size_t queue_capacity = 256U;
  // Generic backends must supply this explicitly. The production CUDA
  // factory derives it from real post-workspace device availability.
  std::uint64_t maximum_live_registration_bytes = 0U;
  std::size_t target_batch_tasks = 8U;
  std::uint32_t target_batch_nodes = 256U;
  std::uint32_t target_batch_pairs = 8192U;
  std::size_t maximum_batch_tasks = 32U;
  std::uint32_t maximum_batch_nodes = 1024U;
  std::uint32_t maximum_batch_pairs = kMaximumSingleLinkageTaskPairs;
  std::size_t maximum_update_requests = 256U;
  std::uint32_t maximum_update_columns = 4096U;
  std::chrono::microseconds gather_window{250};
  bool stable = false;
};

class BoundedLinkageScheduler final : public LinkageEvaluationService {
 public:
  [[nodiscard]] static StatusOr<std::unique_ptr<BoundedLinkageScheduler>>
  Create(std::vector<std::unique_ptr<LinkageWorkerBackend>> backends,
         LinkageSchedulerOptions options);

  ~BoundedLinkageScheduler() override;
  BoundedLinkageScheduler(const BoundedLinkageScheduler&) = delete;
  BoundedLinkageScheduler& operator=(const BoundedLinkageScheduler&) = delete;

  [[nodiscard]] StatusOr<LinkageClusterHandle> RegisterCluster(
      LinkageClusterRegistrationView registration) override;

  [[nodiscard]] StatusOr<LinkageUpdateTicket> SubmitUpdate(
      LinkageClusterHandle cluster,
      LinkageReconstructionUpdateView update) override;

  [[nodiscard]] StatusOr<std::future<StatusOr<LinkageEvaluationResult>>> Submit(
      LinkageEvaluationTask task) override;

  [[nodiscard]] Status UnregisterCluster(LinkageClusterHandle cluster) override;

  [[nodiscard]] LinkageSchedulerStats Stats() const override;
  [[nodiscard]] LinkageProducerTaskPolicy ProducerTaskPolicy()
      const noexcept override;
  [[nodiscard]] bool SupportsAsynchronousPreflush() const noexcept override {
    return true;
  }

 private:
  struct Impl;
  explicit BoundedLinkageScheduler(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace stlqpp
