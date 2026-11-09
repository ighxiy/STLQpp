#pragma once

#include <cstdint>
#include <future>

#include "stlqpp/core/status_or.h"
#include "stlqpp/training/linkage/scheduler/linkage_evaluation.h"
#include "stlqpp/training/linkage/scheduler/linkage_producer_task_policy.h"
#include "stlqpp/training/linkage/scheduler/linkage_worker_backend.h"

namespace stlqpp {

struct LinkageEvaluationServiceStats final {
  // A source task is the immutable numerical unit emitted by the topology
  // producer. A transport batch may carry several source tasks through shared
  // staging/copies/synchronization. Throughput mode may additionally
  // combine them into one numerical batch; stable scheduling never does.
  std::uint64_t source_tasks_submitted = 0U;
  std::uint64_t transport_batches = 0U;
  std::uint64_t cross_cluster_transport_batches = 0U;
  std::uint64_t numerical_batches = 0U;
  std::uint64_t cross_cluster_numerical_batches = 0U;
  std::uint64_t merged_numerical_source_tasks = 0U;
  std::uint64_t maximum_source_tasks_per_transport_batch = 0U;
  std::uint64_t maximum_source_tasks_per_numerical_batch = 0U;
  std::uint64_t nodes = 0U;
  std::uint64_t pairs = 0U;
  std::uint64_t registration_columns = 0U;
  std::uint64_t update_requests = 0U;
  std::uint64_t update_batches = 0U;
  std::uint64_t update_columns = 0U;
  std::uint64_t registration_admission_capacity_bytes = 0U;
  std::uint64_t registration_admission_peak_bytes = 0U;
  LinkageBackendResourceStats resources;
};

using LinkageSchedulerStats = LinkageEvaluationServiceStats;

struct LinkageUpdateTicket final {
  std::uint64_t generation = 0U;
  std::shared_future<Status> completion;
};

// Common topology-facing contract. A scheduled implementation may queue and
// merge tasks across clusters. A direct implementation owns a fixed evaluator
// pool and executes each request immediately without scheduler threads.
class LinkageEvaluationService {
 public:
  virtual ~LinkageEvaluationService() = default;

  [[nodiscard]] virtual StatusOr<LinkageClusterHandle> RegisterCluster(
      LinkageClusterRegistrationView registration) = 0;

  [[nodiscard]] virtual StatusOr<LinkageUpdateTicket> SubmitUpdate(
      LinkageClusterHandle cluster, LinkageReconstructionUpdateView update) = 0;

  [[nodiscard]] virtual StatusOr<std::future<StatusOr<LinkageEvaluationResult>>>
  Submit(LinkageEvaluationTask task) = 0;

  [[nodiscard]] virtual Status UnregisterCluster(
      LinkageClusterHandle cluster) = 0;

  [[nodiscard]] virtual LinkageEvaluationServiceStats Stats() const = 0;

  // The producer policy owns logical flush/preflush boundaries. Execution
  // capability is reported independently: direct pools need asynchronous
  // pinned issue/finish, while a scheduler can provide an asynchronous future
  // through its worker queue. Callers must require both stable policy and
  // asynchronous capability before issuing causal preflush work.
  [[nodiscard]] virtual LinkageProducerTaskPolicy ProducerTaskPolicy()
      const noexcept = 0;
  [[nodiscard]] virtual bool SupportsAsynchronousPreflush() const noexcept = 0;
};

}  // namespace stlqpp
