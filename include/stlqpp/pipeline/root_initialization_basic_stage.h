#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/index/emit/base_list_emitter.h"
#include "stlqpp/training/basic/basic_pipeline.h"
#include "stlqpp/training/basic/basic_round_runtime.h"
#include "stlqpp/training/io/basic_block_source.h"
#include "stlqpp/training/update/initialization_model_updater.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace stlqpp {

struct RootInitializationBasicStageOptions final {
  std::filesystem::path bucket_directory;
  std::filesystem::path list_directory;
  std::uint32_t cids_per_bucket = 0U;
  std::size_t bucket_flush_bytes = 256U * 1024U * 1024U;
  std::uint64_t source_rows = 0U;
  std::uint64_t source_block_rows = 0U;
  std::uint32_t cpu_threads = 0U;
  std::uint32_t seed = 0U;
  BasicSchedule schedule = BasicSchedule::Training();
  BasicPipelineOptions pipeline;
};

struct RootInitializationBasicStageResult final {
  BasicPipelineStatistics pipeline;
  std::uint64_t rows = 0U;
};

template <int M, typename Scalar>
class ScalarRootInitializationBasicStage final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<
      RootInitializationBasicStageResult> Run(
      const RootTrainingModel<M>& model,
      InitializationBasicBlockSourceFactory<Scalar>& source_factory,
      BasicRoundRuntimeFactory<M, Scalar>& runtime_factory,
      InitializationModelUpdater<M>& updater,
      const ParallelForExecutor& parallel,
      const RootInitializationBasicStageOptions& options);
};

// The runtime root stage publishes the same list format and accumulates the
// same Schur moments. It stops at the root-update barrier, before C_one exists.
template <typename Scalar>
class RuntimeRootInitializationBasicStage final {
 public:
  [[nodiscard]] static StatusOr<RootInitializationBasicStageResult> RunCpuRsd(
      const RuntimeRootTrainingModel& model,
      InitializationBasicBlockSourceFactory<Scalar>& source_factory,
      RuntimeRootCodebookMoments& moments, const F32GemmBackend& gemm,
      const ParallelForExecutor& encoding_parallel,
      const ParallelForExecutor& list_parallel, std::uint32_t rank,
      const RootInitializationBasicStageOptions& options);
};

extern template class RuntimeRootInitializationBasicStage<std::uint8_t>;
extern template class RuntimeRootInitializationBasicStage<float>;

template <int M>
using U8RootInitializationBasicStage =
    ScalarRootInitializationBasicStage<M, std::uint8_t>;
template <int M>
using F32RootInitializationBasicStage =
    ScalarRootInitializationBasicStage<M, float>;

extern template class ScalarRootInitializationBasicStage<5, std::uint8_t>;
extern template class ScalarRootInitializationBasicStage<10, std::uint8_t>;
extern template class ScalarRootInitializationBasicStage<5, float>;
extern template class ScalarRootInitializationBasicStage<10, float>;

}  // namespace stlqpp
