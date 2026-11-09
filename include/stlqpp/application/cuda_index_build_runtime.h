#pragma once

#include <cstdint>
#include <memory>

#include "stlqpp/application/index_builder.h"
#include "stlqpp/application/index_source_traits.h"
#include "stlqpp/application/rsd_basic_round_runtime.h"
#include "stlqpp/pipeline/cuda_training_topology_context_factory.h"
#include "stlqpp/pipeline/training_basic_pass.h"
#include "stlqpp/training/initialization/cuda_spherical_rvq_backend.h"
#include "stlqpp/training/linkage/initialization/cuda_root_only_initialization_context.h"

namespace stlqpp {

struct CudaIndexBuildRuntimeOptions final {
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
  int cuda_device = 0;
  bool cuda_allow_tf32 = true;
  CudaSphericalRvqBackendOptions root_initialization;
  TrainingBasicPassOptions training_basic;
  CudaRootOnlyInitializationContextOptions initialization_topology;
  CudaTrainingTopologyContextOptions fast_initialization_topology;
  CudaTrainingTopologyContextOptions training_topology;
  CudaTrainingTopologyContextOptions final_topology;
};

// Owns every long-lived concrete runtime used by BuildIndex<M>. The public
// product entry point cannot receive a partly wired collection of borrowed
// adapters, and CLI code does not know the construction order or lifetimes of
// CUDA, BLAS, LAPACK, source and topology resources.
template <int M>
class CudaIndexBuildRuntime final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<std::unique_ptr<CudaIndexBuildRuntime<M>>>
  Create(CudaIndexBuildRuntimeOptions options);

  ~CudaIndexBuildRuntime();
  CudaIndexBuildRuntime(const CudaIndexBuildRuntime&) = delete;
  CudaIndexBuildRuntime& operator=(const CudaIndexBuildRuntime&) = delete;

  [[nodiscard]] StatusOr<IndexBuildResult<M>> Build(
      const IndexBuildRequest<M>& request);

 private:
  struct State;
  explicit CudaIndexBuildRuntime(std::unique_ptr<State> state) noexcept
      : state_(std::move(state)) {}

  std::unique_ptr<State> state_;
};

extern template class CudaIndexBuildRuntime<5>;
extern template class CudaIndexBuildRuntime<10>;

}  // namespace stlqpp
