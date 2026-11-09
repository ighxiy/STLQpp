#pragma once

#include <cstdint>

#include "stlqpp/core/status_or.h"
#include "stlqpp/training/linkage/scheduler/linkage_evaluation.h"

namespace stlqpp {

template <int M>
class ClusterTopologyState;

// Arithmetic endpoint for the <=8-pair forced tail in root-only
// initialization. The layer executor owns causal scheduling; this interface
// owns only the source-defined CPU candidate fits.
template <int M>
class RootOnlyInitializationTailEvaluator {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  virtual ~RootOnlyInitializationTailEvaluator() = default;
  RootOnlyInitializationTailEvaluator(
      const RootOnlyInitializationTailEvaluator&) = delete;
  RootOnlyInitializationTailEvaluator& operator=(
      const RootOnlyInitializationTailEvaluator&) = delete;

  [[nodiscard]] virtual StatusOr<LinkageEvaluationResult> Evaluate(
      const ClusterTopologyState<M>& state, const LinkageEvaluationTask& task,
      std::uint32_t cluster_seed, std::uint32_t real_local_id) = 0;

 protected:
  RootOnlyInitializationTailEvaluator() = default;
};

}  // namespace stlqpp
