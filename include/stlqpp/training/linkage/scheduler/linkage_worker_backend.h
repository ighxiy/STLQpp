#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/training/linkage/scheduler/linkage_evaluation.h"

namespace stlqpp {

class DirectLinkageEvaluatorPool;

using LinkageBackendCluster = std::uint64_t;

struct LinkageBackendRegistration final {
  std::uint32_t dimension = 0U;
  std::uint32_t column_count = 0U;
  std::vector<float> reconstructions;
};

struct LinkageBackendUpdate final {
  LinkageBackendCluster cluster = 0U;
  std::uint64_t generation = 0U;
  std::uint32_t dimension = 0U;
  std::vector<std::uint32_t> columns;
  std::vector<float> reconstructions;
};

struct LinkageBackendEvaluation final {
  LinkageBackendCluster cluster = 0U;
  LinkageEvaluationTask task;
};

struct LinkageBackendResourceStats final {
  std::uint64_t live_registration_bytes = 0U;
  std::uint64_t idle_cache_buffers = 0U;
  std::uint64_t idle_cache_bytes = 0U;
  std::uint64_t idle_cache_peak_bytes = 0U;
  std::uint64_t idle_cache_evictions = 0U;
  std::uint64_t allocation_retries = 0U;
};

using LinkageBackendEvaluationCompletion =
    std::function<void(StatusOr<std::vector<LinkageEvaluationResult>>)>;

// One instance is owned by exactly one scheduler thread. Implementations may
// therefore keep a stream, workspace and bounded best-fit R_full cache without
// internal launch serialization.
class LinkageWorkerBackend {
 public:
  virtual ~LinkageWorkerBackend() = default;

  [[nodiscard]] virtual StatusOr<LinkageBackendCluster> Register(
      LinkageBackendRegistration registration) = 0;

  [[nodiscard]] virtual Status ApplyUpdates(
      ArrayView<const LinkageBackendUpdate> updates) = 0;

  [[nodiscard]] virtual StatusOr<std::vector<LinkageEvaluationResult>> Evaluate(
      ArrayView<const LinkageBackendEvaluation> tasks) = 0;

  // Stable scheduler entry point. Generic backends preserve the numerical
  // boundary by evaluating one producer task at a time. Backends may override
  // this to merge transport/staging while retaining each task's call shape.
  [[nodiscard]] virtual StatusOr<std::vector<LinkageEvaluationResult>>
  EvaluateStable(ArrayView<const LinkageBackendEvaluation> tasks) {
    std::vector<LinkageEvaluationResult> output;
    output.reserve(tasks.size());
    for (const LinkageBackendEvaluation& task : tasks) {
      auto result =
          Evaluate(ArrayView<const LinkageBackendEvaluation>(&task, 1U));
      if (!result.ok()) return result.status();
      if (result.value().size() != 1U) {
        return Status::Internal(
            "stable backend returned a different task count");
      }
      output.push_back(std::move(result.value().front()));
    }
    return output;
  }

  // Direct CUDA lanes may enqueue an evaluation and complete it from the
  // lane stream without introducing a scheduler thread. A successful begin
  // must invoke completion exactly once for success or asynchronous failure.
  // Generic and CPU backends retain a synchronous fallback.
  [[nodiscard]] virtual bool SupportsAsynchronousEvaluation() const noexcept {
    return false;
  }

  [[nodiscard]] virtual Status BeginEvaluate(
      ArrayView<const LinkageBackendEvaluation> tasks,
      LinkageBackendEvaluationCompletion completion) {
    completion(Evaluate(tasks));
    return Status::Ok();
  }

  [[nodiscard]] virtual Status Unregister(LinkageBackendCluster cluster) = 0;

  [[nodiscard]] virtual LinkageBackendResourceStats ResourceStats()
      const noexcept = 0;

 private:
  friend class DirectLinkageEvaluatorPool;
  // Submit has already validated this immutable task's CSR, order and
  // candidate extents. CPU backends may skip those scans but must still check
  // their own registration and generation state. Other backends use Evaluate.
  [[nodiscard]] virtual StatusOr<std::vector<LinkageEvaluationResult>>
  EvaluatePrevalidatedDirect(ArrayView<const LinkageBackendEvaluation> tasks) {
    return Evaluate(tasks);
  }
};

}  // namespace stlqpp
