#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/basic/basic_pipeline.h"
#include "stlqpp/training/basic/basic_round_runtime.h"
#include "stlqpp/training/io/basic_block_source.h"
#include "stlqpp/training/trainer.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <utility>

namespace stlqpp {

struct TrainingBasicPassOptions final {
  std::filesystem::path artifact_root;
  std::uint32_t cids_per_bucket = 0U;
  std::size_t bucket_flush_bytes = 256U * 1024U * 1024U;
  std::uint64_t source_rows = 0U;
  std::uint64_t source_block_rows = 0U;
  std::uint32_t cpu_threads = 0U;
  BasicSchedule schedule = BasicSchedule::Training();
  BasicPipelineOptions pipeline;
};

// Production training-basic composition. One pass accumulates the exact root
// moments and atomically publishes the same round's codes, coefficients and
// original typed rows in list-major order. Full/replay topology therefore cannot
// accidentally consume initialization-era basic encodings.
template <int M, typename Scalar>
class ScalarTrainingBasicPass final : public TrainingBasicPass<M> {
 public:
  ScalarTrainingBasicPass(
      TrainingBasicBlockSourceFactory<Scalar>& source,
      BasicRoundRuntimeFactory<M, Scalar>& runtime,
      const ParallelForExecutor& parallel,
      TrainingBasicPassOptions options) noexcept
      : source_(&source),
        runtime_(&runtime),
        parallel_(&parallel),
        options_(std::move(options)) {}

  [[nodiscard]] StatusOr<TrainingBasicRoundArtifact> Accumulate(
      const TrainingModel<M>& model,
      std::uint32_t round,
      std::uint32_t seed,
      RoundModelUpdater<M>& updater) override;

  [[nodiscard]] StatusOr<TrainingBasicRoundArtifact> AccumulateFixedRoots(
      const TrainingModel<M>& model,
      const TrainingCheckpoint<M>& previous_checkpoint,
      std::uint32_t round,
      std::uint32_t seed,
      RoundModelUpdater<M>& updater) override;

  [[nodiscard]] const std::optional<BasicPipelineStatistics>&
  last_statistics() const noexcept {
    return last_statistics_;
  }

 private:
  [[nodiscard]] StatusOr<TrainingBasicRoundArtifact> AccumulateImpl(
      const TrainingModel<M>& model,
      std::uint32_t round,
      std::uint32_t seed,
      ArrayView<const std::uint32_t> fixed_roots,
      RoundModelUpdater<M>& updater);

  TrainingBasicBlockSourceFactory<Scalar>* source_ = nullptr;
  BasicRoundRuntimeFactory<M, Scalar>* runtime_ = nullptr;
  const ParallelForExecutor* parallel_ = nullptr;
  TrainingBasicPassOptions options_;
  std::optional<BasicPipelineStatistics> last_statistics_;
};

template <int M>
using U8TrainingBasicPass = ScalarTrainingBasicPass<M, std::uint8_t>;
template <int M>
using F32TrainingBasicPass = ScalarTrainingBasicPass<M, float>;

extern template class ScalarTrainingBasicPass<5, std::uint8_t>;
extern template class ScalarTrainingBasicPass<10, std::uint8_t>;
extern template class ScalarTrainingBasicPass<5, float>;
extern template class ScalarTrainingBasicPass<10, float>;

}  // namespace stlqpp
