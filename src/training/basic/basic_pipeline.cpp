#include "stlqpp/training/basic/basic_pipeline.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "stlqpp/runtime/concurrency/bounded_queue.h"
#include "stlqpp/runtime/cpu/openmp_affinity.h"
#include "stlqpp/runtime/telemetry/logger.h"

namespace stlqpp {
namespace {

template <typename Scalar>
struct WorkItem final {
  std::uint64_t sequence = 0U;
  BasicInputBlock<Scalar> block;
};

template <typename Scalar, typename Encoded>
struct CompletedItem final {
  CompletedItem(std::uint64_t input_sequence,
                BasicInputBlock<Scalar>&& input_block,
                Encoded&& input_encoded) noexcept
      : sequence(input_sequence),
        block(std::move(input_block)),
        encoded(std::move(input_encoded)) {}

  std::uint64_t sequence = 0U;
  BasicInputBlock<Scalar> block;
  Encoded encoded;
};

template <typename Scalar>
struct BaseBuildMode final {
  using Encoded = BasicEncodingBatch;
  using Sink = BasicBlockSink<Scalar>;

  static StatusOr<Encoded> Execute(BasicBlockExecutor<Scalar>& executor,
                                   ColumnMajorMatrixView<Scalar> samples,
                                   BasicSchedule schedule, std::uint32_t seed,
                                   std::uint64_t first_id,
                                   ArrayView<const std::uint32_t>) {
    return executor.Encode(samples, schedule, seed, first_id);
  }

  static Status Consume(Sink& sink, BasicInputBlock<Scalar> block,
                        Encoded encoded) {
    return sink.Consume(
        EncodedBasicBlock<Scalar>(block.first_id, block.dimension,
                                  std::move(block.values), std::move(encoded)));
  }

  static const BasicEncodingBatch& Encoding(const Encoded& encoded) noexcept {
    return encoded;
  }
};

template <typename Scalar>
struct TrainingMode final {
  using Encoded = RetainedModelSpaceBasicEncoding;
  using Sink = TrainingBasicBlockSink<Scalar>;

  static StatusOr<Encoded> Execute(BasicBlockExecutor<Scalar>& executor,
                                   ColumnMajorMatrixView<Scalar> samples,
                                   BasicSchedule schedule, std::uint32_t seed,
                                   std::uint64_t first_id,
                                   ArrayView<const std::uint32_t>) {
    return executor.EncodeRetainingModelSpace(samples, schedule, seed,
                                              first_id);
  }

  static Status Consume(Sink& sink, BasicInputBlock<Scalar> block,
                        Encoded encoded) {
    assert(encoded.dimension == block.dimension);
    assert(encoded.encoding.sample_count() == block.count);
    assert(encoded.exact_model_space.size() ==
           static_cast<std::size_t>(block.dimension) * block.count);
    return sink.Consume(EncodedTrainingBasicBlock<Scalar>(
        block.first_id, block.dimension, std::move(block.values),
        std::move(encoded.exact_model_space), std::move(encoded.encoding)));
  }

  static const BasicEncodingBatch& Encoding(const Encoded& encoded) noexcept {
    return encoded.encoding;
  }
};

template <typename Scalar>
struct FixedRootTrainingMode final {
  using Encoded = RetainedModelSpaceBasicEncoding;
  using Sink = TrainingBasicBlockSink<Scalar>;

  static StatusOr<Encoded> Execute(BasicBlockExecutor<Scalar>& executor,
                                   ColumnMajorMatrixView<Scalar> samples,
                                   BasicSchedule schedule, std::uint32_t seed,
                                   std::uint64_t first_id,
                                   ArrayView<const std::uint32_t> fixed_roots) {
    if (first_id > fixed_roots.size() ||
        samples.columns > fixed_roots.size() - first_id) {
      return Status::FailedPrecondition(
          "fixed-root basic rows exceed the checkpoint assignment");
    }
    return executor.EncodeFixedRootsRetainingModelSpace(
        samples,
        {fixed_roots.data() + static_cast<std::size_t>(first_id),
         samples.columns},
        schedule, seed, first_id);
  }

  static Status Consume(Sink& sink, BasicInputBlock<Scalar> block,
                        Encoded encoded) {
    assert(encoded.dimension == block.dimension);
    assert(encoded.encoding.sample_count() == block.count);
    assert(encoded.exact_model_space.size() ==
           static_cast<std::size_t>(block.dimension) * block.count);
    return sink.Consume(EncodedTrainingBasicBlock<Scalar>(
        block.first_id, block.dimension, std::move(block.values),
        std::move(encoded.exact_model_space), std::move(encoded.encoding)));
  }

  static const BasicEncodingBatch& Encoding(const Encoded& encoded) noexcept {
    return encoded.encoding;
  }
};

template <typename Scalar>
Status ValidateBlock(const BasicInputBlock<Scalar>& block,
                     std::optional<std::uint64_t> expected_first_id) {
  if (block.dimension == 0U || block.count == 0U ||
      block.count > std::numeric_limits<std::size_t>::max() / block.dimension ||
      block.values.size() !=
          static_cast<std::size_t>(block.dimension) * block.count) {
    return Status::InvalidArgument(
        "basic source returned an invalid U8 block");
  }
  if (expected_first_id.has_value() && block.first_id != *expected_first_id) {
    return Status::InvalidArgument(
        "basic source blocks are not globally contiguous");
  }
  if (block.first_id >
      std::numeric_limits<std::uint64_t>::max() - block.count) {
    return Status::ResourceExhausted(
        "basic source ID extent overflows u64");
  }
  return Status::Ok();
}

template <typename Scalar, typename Mode>
StatusOr<BasicPipelineStatistics> RunPipeline(
    BasicBlockSource<Scalar>& source, typename Mode::Sink& sink,
    BasicBlockExecutor<Scalar>& executor,
    BasicSchedule schedule, std::uint32_t seed,
    ArrayView<const std::uint32_t> fixed_roots,
    BasicPipelineOptions options) {
  if (options.maximum_inflight_blocks == 0U) {
    return Status::InvalidArgument(
        "basic pipeline requires a positive inflight bound");
  }
  const std::size_t capacity = options.maximum_inflight_blocks;
  const std::size_t prefetch_capacity =
      std::min<std::size_t>(capacity, options.maximum_prefetched_blocks == 0U
                                          ? capacity
                                          : options.maximum_prefetched_blocks);
  const std::size_t write_capacity =
      std::min<std::size_t>(capacity, options.maximum_pending_writes == 0U
                                          ? capacity
                                          : options.maximum_pending_writes);
  BoundedQueue<WorkItem<Scalar>> input_queue(prefetch_capacity);
  BoundedQueue<CompletedItem<Scalar, typename Mode::Encoded>> result_queue(
      write_capacity);
  std::atomic<bool> failed{false};
  std::mutex failure_mutex;
  Status failure = Status::Ok();

  auto fail = [&](Status status) {
    bool expected = false;
    if (failed.compare_exchange_strong(expected, true)) {
      std::lock_guard<std::mutex> lock(failure_mutex);
      failure = std::move(status);
    }
    input_queue.Close();
    result_queue.Close();
  };

  BasicLaneStatistics executor_statistics;
  const auto wall_begin = std::chrono::steady_clock::now();
  auto worker = [&](BasicBlockExecutor<Scalar>* worker_executor,
                    BoundedQueue<WorkItem<Scalar>>* queue,
                    BasicLaneStatistics* statistics) {
    const Status affinity = RestoreNativeThreadAffinity();
    if (!affinity.ok()) {
      fail(affinity);
    }
    while (affinity.ok() && !failed.load(std::memory_order_acquire)) {
      auto work = queue->Pop();
      if (!work.has_value()) {
        break;
      }
      const std::uint32_t vector_count = work->block.count;
      const auto encode_begin = std::chrono::steady_clock::now();
      auto encoded = Mode::Execute(*worker_executor, work->block.view(), schedule,
                                   seed, work->block.first_id, fixed_roots);
      statistics->encode_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        encode_begin)
              .count();
      if (!encoded.ok()) {
        fail(encoded.status());
        break;
      }
      ++statistics->blocks;
      statistics->vectors += vector_count;
      CompletedItem<Scalar, typename Mode::Encoded> completed(
          work->sequence, std::move(work->block), std::move(encoded).value());
      if (!result_queue.Push(std::move(completed))) {
        break;
      }
    }
    result_queue.Close();
  };

  std::thread encoder_worker(worker, &executor, &input_queue,
                             &executor_statistics);

  std::uint64_t next_sequence = 0U;
  std::uint64_t next_to_consume = 0U;
  std::uint64_t blocks_consumed = 0U;
  std::uint64_t metric_vectors = 0U;
  double initial_squared_error_sum = 0.0;
  double sink_seconds = 0.0;
  double final_squared_error_sum = 0.0;
  std::uint32_t inflight = 0U;
  bool source_exhausted = false;
  std::optional<std::uint64_t> expected_first_id;
  std::optional<WorkItem<Scalar>> pending_admission;

  auto fill_window = [&]() {
    while (!failed.load(std::memory_order_acquire) && !source_exhausted &&
           inflight < options.maximum_inflight_blocks) {
      if (!pending_admission.has_value()) {
        auto next = source.Next();
        if (!next.ok()) {
          fail(next.status());
          return;
        }
        if (!next.value().has_value()) {
          source_exhausted = true;
          input_queue.Close();
          return;
        }
        BasicInputBlock<Scalar> block = std::move(*next.value());
        if (const Status status = ValidateBlock(block, expected_first_id);
            !status.ok()) {
          fail(status);
          return;
        }
        expected_first_id = block.first_id + block.count;
        pending_admission.emplace(WorkItem<Scalar>{
            .sequence = next_sequence,
            .block = std::move(block),
        });
      }
      if (!input_queue.TryPush(&*pending_admission)) {
        // A full input lane is not a failure: return to the ordered result
        // consumer so a worker blocked on the independently bounded write
        // queue can make progress. The source block remains owned here and
        // is retried before reading another block.
        return;
      }
      pending_admission.reset();
      ++next_sequence;
      ++inflight;
    }
  };

  fill_window();
  while (!failed.load(std::memory_order_acquire) && inflight != 0U) {
    auto completed = result_queue.Pop();
    if (!completed.has_value()) {
      if (!failed.load(std::memory_order_acquire)) {
        fail(Status::Internal(
            "basic workers stopped before completing submitted work"));
      }
      break;
    }
    // One FIFO executor preserves source order without a reorder buffer.
    assert(completed->sequence == next_to_consume);
    auto item = std::move(*completed);
    if (options.log_metrics) {
      const BasicEncodingBatch& encoding = Mode::Encoding(item.encoded);
      for (const float value : encoding.initial_squared_errors()) {
        initial_squared_error_sum += value;
      }
      for (const float value : encoding.squared_errors()) {
        final_squared_error_sum += value;
      }
      metric_vectors += encoding.sample_count();
    }
    const auto sink_begin = std::chrono::steady_clock::now();
    const Status status =
        Mode::Consume(sink, std::move(item.block), std::move(item.encoded));
    sink_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sink_begin).count();
    if (!status.ok()) {
      fail(status);
      break;
    }
    ++next_to_consume;
    ++blocks_consumed;
    --inflight;
    if (options.log_progress && options.progress_total_blocks != 0U) {
      const std::string message =
          "Encoded block " + std::to_string(blocks_consumed) + " / " +
          std::to_string(options.progress_total_blocks);
      if (blocks_consumed == options.progress_total_blocks) {
        LogInfoProgressDone(message);
      } else {
        LogInfoProgress(message);
      }
    }
    fill_window();
    if (failed.load(std::memory_order_acquire)) {
      break;
    }
  }

  input_queue.Close();
  encoder_worker.join();
  if (failed.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(failure_mutex);
    return failure;
  }
  BasicPipelineStatistics statistics;
  if (options.cpu_backend) statistics.cpu = executor_statistics;
  else statistics.gpu = executor_statistics;
  statistics.sink_seconds = sink_seconds;
  statistics.metric_vectors = metric_vectors;
  statistics.initial_squared_error_sum = initial_squared_error_sum;
  statistics.final_squared_error_sum = final_squared_error_sum;
  statistics.wall_seconds = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - wall_begin)
                                .count();
  return statistics;
}

}  // namespace

template <typename Scalar>
StatusOr<BasicPipelineStatistics> BasicPipeline<Scalar>::Run(
    BasicBlockSource<Scalar>& source, BasicBlockSink<Scalar>& sink,
    BasicBlockExecutor<Scalar>& executor,
    BasicSchedule schedule, std::uint32_t seed,
    BasicPipelineOptions options) {
  return RunPipeline<Scalar, BaseBuildMode<Scalar>>(
      source, sink, executor, schedule, seed, {}, options);
}

template <typename Scalar>
StatusOr<BasicPipelineStatistics>
TrainingBasicPipeline<Scalar>::Run(
    BasicBlockSource<Scalar>& source, TrainingBasicBlockSink<Scalar>& sink,
    BasicBlockExecutor<Scalar>& executor,
    BasicSchedule schedule, std::uint32_t seed,
    BasicPipelineOptions options) {
  return RunPipeline<Scalar, TrainingMode<Scalar>>(
      source, sink, executor, schedule, seed, {}, options);
}

template <typename Scalar>
StatusOr<BasicPipelineStatistics>
FixedRootTrainingBasicPipeline<Scalar>::Run(
    BasicBlockSource<Scalar>& source, TrainingBasicBlockSink<Scalar>& sink,
    BasicBlockExecutor<Scalar>& executor,
    ArrayView<const std::uint32_t> fixed_roots, BasicSchedule schedule,
    std::uint32_t seed, BasicPipelineOptions options) {
  if (fixed_roots.empty()) {
    return Status::InvalidArgument(
        "fixed-root basic pipeline requires checkpoint assignments");
  }
  return RunPipeline<Scalar, FixedRootTrainingMode<Scalar>>(
      source, sink, executor, schedule, seed, fixed_roots, options);
}

template class BasicPipeline<std::uint8_t>;
template class BasicPipeline<float>;
template class TrainingBasicPipeline<std::uint8_t>;
template class TrainingBasicPipeline<float>;
template class FixedRootTrainingBasicPipeline<std::uint8_t>;
template class FixedRootTrainingBasicPipeline<float>;

}  // namespace stlqpp
