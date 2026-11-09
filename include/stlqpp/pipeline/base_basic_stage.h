#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/index/emit/base_list_emitter.h"
#include "stlqpp/training/basic/basic_pipeline.h"
#include "stlqpp/training/basic/basic_round_runtime.h"
#include "stlqpp/training/io/basic_block_source.h"
#include "stlqpp/training/update/training_model.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace stlqpp {

struct BaseBasicStageOptions final {
  std::filesystem::path bucket_directory;
  std::filesystem::path list_directory;
  std::uint32_t cids_per_bucket = 0U;
  std::size_t bucket_flush_bytes = 256U * 1024U * 1024U;
  std::uint32_t seed = 0U;
  // Resume partial encoding from buckets, or a completed stage from its list
  // identity. Checkpointing does not retain buckets after list publication.
  bool checkpoint_enabled = false;
  ArtifactHash64 checkpoint_identity{};
  ArtifactHash64 list_identity{};
  bool retain_bucket_directory = false;
  std::uint64_t source_rows = 0U;
  std::uint64_t source_block_rows = 0U;
  std::uint32_t cpu_threads = 0U;
  BasicSchedule schedule = BasicSchedule::BaseBuild();
  BasicPipelineOptions pipeline;
};

struct BaseBasicStageResult final {
  BasicPipelineStatistics pipeline;
  std::uint64_t rows = 0U;
  std::filesystem::path list_directory;
};

// Complete final-base basic encoding boundary. It executes the registered
// configured basic schedule, writes bounded CID buckets, then emits one
// list-major exact-vector store. No topology or coefficient codec is applied
// at this stage.
template <int M, typename Scalar>
class ScalarBaseBasicStage final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<BaseBasicStageResult> Run(
      const TrainingModel<M>& model,
      BaseBasicBlockSourceFactory<Scalar>& source_factory,
      BasicRoundRuntimeFactory<M, Scalar>& runtime_factory,
      const ParallelForExecutor& parallel,
      const BaseBasicStageOptions& options);
};

// Runtime-M CPU RSD stage, sharing checkpoint, bucket, ordered pipeline and
// list publication with the fixed-M stage. CUDA uses its corresponding RSD backend.
template <typename Scalar>
class RuntimeBaseBasicStage final {
 public:
  [[nodiscard]] static StatusOr<BaseBasicStageResult> RunCpuRsd(
      RuntimeRootCodebookView roots, ColumnMajorMatrixF32View rotation,
      BaseBasicBlockSourceFactory<Scalar>& source_factory,
      const F32GemmBackend& gemm, const ParallelForExecutor& encoding_parallel,
      const ParallelForExecutor& list_parallel, std::uint32_t rank,
      const BaseBasicStageOptions& options);
};
extern template class RuntimeBaseBasicStage<std::uint8_t>;
extern template class RuntimeBaseBasicStage<float>;

template <int M>
using U8BaseBasicStage = ScalarBaseBasicStage<M, std::uint8_t>;
template <int M>
using F32BaseBasicStage = ScalarBaseBasicStage<M, float>;

extern template class ScalarBaseBasicStage<5, std::uint8_t>;
extern template class ScalarBaseBasicStage<10, std::uint8_t>;
extern template class ScalarBaseBasicStage<5, float>;
extern template class ScalarBaseBasicStage<10, float>;

}  // namespace stlqpp
