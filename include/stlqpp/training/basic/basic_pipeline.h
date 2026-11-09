#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "stlqpp/core/status_or.h"
#include "stlqpp/training/basic/basic_encoding.h"
#include "stlqpp/training/basic/basic_executor.h"
#include "stlqpp/training/basic/basic_schedule.h"
#include "stlqpp/training/io/basic_block_source.h"

namespace stlqpp {

template <typename Scalar>
struct EncodedBasicBlock final {
  EncodedBasicBlock(std::uint64_t input_first_id, std::uint32_t input_dimension,
                    std::vector<Scalar>&& input_values,
                    BasicEncodingBatch&& input_encoding) noexcept
      : first_id(input_first_id),
        dimension(input_dimension),
        values(std::move(input_values)),
        encoding(std::move(input_encoding)) {}

  std::uint64_t first_id = 0U;
  std::uint32_t dimension = 0U;
  std::vector<Scalar> values;
  BasicEncodingBatch encoding;
};

template <typename Scalar>
class BasicBlockSink {
 public:
  virtual ~BasicBlockSink() = default;
  BasicBlockSink(const BasicBlockSink&) = delete;
  BasicBlockSink& operator=(const BasicBlockSink&) = delete;

  [[nodiscard]] virtual Status Consume(EncodedBasicBlock<Scalar> block) = 0;

 protected:
  BasicBlockSink() = default;
};

template <typename Scalar>
struct EncodedTrainingBasicBlock final {
  EncodedTrainingBasicBlock(std::uint64_t input_first_id,
                            std::uint32_t input_dimension,
                            std::vector<Scalar>&& input_source_values,
                            std::vector<float>&& input_exact_model_space,
                            BasicEncodingBatch&& input_encoding) noexcept
      : first_id(input_first_id),
        dimension(input_dimension),
        source_values(std::move(input_source_values)),
        exact_model_space(std::move(input_exact_model_space)),
        encoding(std::move(input_encoding)) {}

  std::uint64_t first_id = 0U;
  std::uint32_t dimension = 0U;
  std::vector<Scalar> source_values;
  std::vector<float> exact_model_space;
  BasicEncodingBatch encoding;

  [[nodiscard]] ColumnMajorMatrixF32View model_space_view() const noexcept {
    return {exact_model_space.data(), dimension, encoding.sample_count()};
  }
};

template <typename Scalar>
class TrainingBasicBlockSink {
 public:
  virtual ~TrainingBasicBlockSink() = default;
  TrainingBasicBlockSink(const TrainingBasicBlockSink&) = delete;
  TrainingBasicBlockSink& operator=(const TrainingBasicBlockSink&) = delete;

  [[nodiscard]] virtual Status Consume(
      EncodedTrainingBasicBlock<Scalar> block) = 0;

 protected:
  TrainingBasicBlockSink() = default;
};

struct BasicPipelineOptions final {
  std::uint32_t maximum_inflight_blocks = 0U;
  // Zero preserves the legacy typed-call behavior and uses the overall
  // inflight bound. Production always supplies the separately resolved DEV
  // prefetch and ordered-write capacities.
  std::uint32_t maximum_prefetched_blocks = 0U;
  std::uint32_t maximum_pending_writes = 0U;
  // Identifies the single executor for telemetry; it does not route work.
  bool cpu_backend = false;
  std::uint64_t progress_total_blocks = 0U;
  bool log_progress = false;
  bool log_metrics = false;
};

struct BasicLaneStatistics final {
  std::uint64_t blocks = 0U;
  std::uint64_t vectors = 0U;
  double encode_seconds = 0.0;
};

struct BasicPipelineStatistics final {
  BasicLaneStatistics cpu;
  BasicLaneStatistics gpu;
  std::uint64_t metric_vectors = 0U;
  double initial_squared_error_sum = 0.0;
  double final_squared_error_sum = 0.0;
  double wall_seconds = 0.0;
  double sink_seconds = 0.0;
};

template <typename Scalar>
class BasicPipeline final {
 public:
  [[nodiscard]] static StatusOr<BasicPipelineStatistics> Run(
      BasicBlockSource<Scalar>& source, BasicBlockSink<Scalar>& sink,
      BasicBlockExecutor<Scalar>& executor,
      BasicSchedule schedule, std::uint32_t seed,
      BasicPipelineOptions options);
};

// Training counterpart to BasicPipeline. It preserves the computed
// model-space tile alongside the source rows for moments and artifact sinks.
// This is a separate semantic entry point, not a runtime mode flag.
template <typename Scalar>
class TrainingBasicPipeline final {
 public:
  [[nodiscard]] static StatusOr<BasicPipelineStatistics> Run(
      BasicBlockSource<Scalar>& source, TrainingBasicBlockSink<Scalar>& sink,
      BasicBlockExecutor<Scalar>& executor,
      BasicSchedule schedule, std::uint32_t seed,
      BasicPipelineOptions options);
};

// Replay counterpart: each source row is encoded with the CID frozen by the
// preceding full checkpoint.  It shares the same bounded single-executor pipeline
// and ordered sink, but its semantic entry point is distinct from ordinary
// root search.
template <typename Scalar>
class FixedRootTrainingBasicPipeline final {
 public:
  [[nodiscard]] static StatusOr<BasicPipelineStatistics> Run(
      BasicBlockSource<Scalar>& source, TrainingBasicBlockSink<Scalar>& sink,
      BasicBlockExecutor<Scalar>& executor,
      ArrayView<const std::uint32_t> fixed_roots, BasicSchedule schedule,
      std::uint32_t seed, BasicPipelineOptions options);
};

using EncodedU8BasicBlock = EncodedBasicBlock<std::uint8_t>;
using EncodedF32BasicBlock = EncodedBasicBlock<float>;
using U8BasicBlockSink = BasicBlockSink<std::uint8_t>;
using F32BasicBlockSink = BasicBlockSink<float>;
using EncodedTrainingU8BasicBlock = EncodedTrainingBasicBlock<std::uint8_t>;
using EncodedTrainingF32BasicBlock = EncodedTrainingBasicBlock<float>;
using U8TrainingBasicBlockSink = TrainingBasicBlockSink<std::uint8_t>;
using F32TrainingBasicBlockSink = TrainingBasicBlockSink<float>;
using U8BasicPipeline = BasicPipeline<std::uint8_t>;
using F32BasicPipeline = BasicPipeline<float>;
using U8TrainingBasicPipeline = TrainingBasicPipeline<std::uint8_t>;
using F32TrainingBasicPipeline = TrainingBasicPipeline<float>;

extern template class BasicPipeline<std::uint8_t>;
extern template class TrainingBasicPipeline<std::uint8_t>;
extern template class BasicPipeline<float>;
extern template class TrainingBasicPipeline<float>;
extern template class FixedRootTrainingBasicPipeline<std::uint8_t>;
extern template class FixedRootTrainingBasicPipeline<float>;

}  // namespace stlqpp
