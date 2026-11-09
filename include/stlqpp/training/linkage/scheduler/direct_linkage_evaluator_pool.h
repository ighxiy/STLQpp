#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "stlqpp/runtime/exclusive_index_pool.h"
#include "stlqpp/training/linkage/scheduler/linkage_evaluation_service.h"

namespace stlqpp {

struct DirectLinkageEvaluatorPoolOptions final {
  std::uint64_t maximum_live_registration_bytes = 0U;
  std::uint32_t maximum_task_nodes = 1024U;
  std::uint32_t maximum_task_pairs = 32768U;
  std::uint32_t maximum_update_columns = 4096U;
  bool asynchronous_evaluation = false;
};

// Scheduler-off service. A cluster exclusively owns one backend from
// registration through unregister, matching the old direct CUDA-pool
// lifetime. There are no scheduler threads, queues or cross-cluster merges.
class DirectLinkageEvaluatorPool final : public LinkageEvaluationService {
 public:
  [[nodiscard]] static StatusOr<std::unique_ptr<DirectLinkageEvaluatorPool>>
  Create(std::vector<std::unique_ptr<LinkageWorkerBackend>> backends,
         DirectLinkageEvaluatorPoolOptions options,
         std::shared_ptr<ExclusiveIndexPool> shared_lanes = {});

  ~DirectLinkageEvaluatorPool() override;
  DirectLinkageEvaluatorPool(const DirectLinkageEvaluatorPool&) = delete;
  DirectLinkageEvaluatorPool& operator=(const DirectLinkageEvaluatorPool&) =
      delete;

  [[nodiscard]] StatusOr<LinkageClusterHandle> RegisterCluster(
      LinkageClusterRegistrationView registration) override;
  [[nodiscard]] StatusOr<LinkageUpdateTicket> SubmitUpdate(
      LinkageClusterHandle cluster,
      LinkageReconstructionUpdateView update) override;
  [[nodiscard]] StatusOr<std::future<StatusOr<LinkageEvaluationResult>>> Submit(
      LinkageEvaluationTask task) override;
  [[nodiscard]] Status UnregisterCluster(LinkageClusterHandle cluster) override;
  [[nodiscard]] LinkageEvaluationServiceStats Stats() const override;
  [[nodiscard]] LinkageProducerTaskPolicy ProducerTaskPolicy()
      const noexcept override {
    return LinkageProducerTaskPolicy::kStable;
  }
  [[nodiscard]] bool SupportsAsynchronousPreflush() const noexcept override;

 private:
  struct Impl;
  explicit DirectLinkageEvaluatorPool(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace stlqpp
