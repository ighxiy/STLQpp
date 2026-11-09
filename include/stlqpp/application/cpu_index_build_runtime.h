#pragma once

#include <cstdint>
#include <memory>

#include "stlqpp/application/index_builder.h"
#include "stlqpp/application/index_source_traits.h"
#include "stlqpp/pipeline/cpu_training_topology_context_factory.h"
#include "stlqpp/pipeline/training_basic_pass.h"
#include "stlqpp/training/initialization/cpu_spherical_rvq_backend.h"
#include "stlqpp/training/linkage/initialization/cpu_root_only_initialization_context.h"

namespace stlqpp {

struct CpuIndexBuildRuntimeOptions final {
  bool create_root_initialization = true;
  bool create_training_initialization = true;
  bool create_fast_initialization = false;
  bool create_training = true;
  bool create_base_encode = true;
  bool create_final_linkage = true;
  std::uint32_t training_basic_rsd_rank = 2U;
  std::uint32_t base_basic_rsd_rank = 2U;
  IndexSourceSpec initialization_source;
  IndexSourceSpec training_source;
  IndexSourceSpec base_source;
  std::uint32_t cpu_worker_count = 1U;
  std::uint32_t base_list_worker_count = 1U;
  std::uint32_t root_moment_partition_count = 1U;
  std::uint32_t rotation_moment_partition_count = 1U;
  CpuSphericalRvqBackendOptions root_initialization;
  TrainingBasicPassOptions training_basic;
  CpuRootOnlyInitializationContextOptions initialization_topology;
  CpuTrainingTopologyContextOptions fast_initialization_topology;
  CpuTrainingTopologyContextOptions training_topology;
  CpuTrainingTopologyContextOptions final_topology;
};

template <int M>
class CpuIndexBuildRuntime final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<std::unique_ptr<CpuIndexBuildRuntime<M>>>
  Create(CpuIndexBuildRuntimeOptions options);

  ~CpuIndexBuildRuntime();
  CpuIndexBuildRuntime(const CpuIndexBuildRuntime&) = delete;
  CpuIndexBuildRuntime& operator=(const CpuIndexBuildRuntime&) = delete;

  [[nodiscard]] StatusOr<IndexBuildResult<M>> Build(
      const IndexBuildRequest<M>& request);

 private:
  struct State;
  explicit CpuIndexBuildRuntime(std::unique_ptr<State> state) noexcept
      : state_(std::move(state)) {}
  std::unique_ptr<State> state_;
};

extern template class CpuIndexBuildRuntime<5>;
extern template class CpuIndexBuildRuntime<10>;

}  // namespace stlqpp
