#include "stlqpp/pipeline/root_initialization_basic_stage.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include "stlqpp/index/emit/basic_bucket_store.h"
#include "stlqpp/training/basic/basic_stage_log.h"
#include "stlqpp/training/basic/detail/basic_encoding_access.h"
#include "stlqpp/training/basic/runtime_basic_executor.h"

namespace stlqpp {
namespace {

class InitializationBucketLifetime final {
 public:
  explicit InitializationBucketLifetime(
      std::filesystem::path directory) noexcept
      : directory_(std::move(directory)) {}

  ~InitializationBucketLifetime() {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

 private:
  std::filesystem::path directory_;
};

template <int M, typename Scalar, typename AccumulateBasic>
class InitializationBasicSink final : public TrainingBasicBlockSink<Scalar> {
 public:
  InitializationBasicSink(AccumulateBasic accumulate,
                          BasicBucketStoreWriter<M>& writer) noexcept
      : accumulate_(std::move(accumulate)), writer_(&writer) {}

  [[nodiscard]] Status Consume(
      EncodedTrainingBasicBlock<Scalar> block) override {
    if (block.first_id > std::numeric_limits<std::uint32_t>::max()) {
      return Status::ResourceExhausted(
          "initialization training row ID exceeds the canonical u32 domain");
    }
    const auto model_space = block.model_space_view();
    auto encoding =
        detail::BasicEncodingAccess::Adopt<M>(std::move(block.encoding));
    if (const Status status = accumulate_(model_space, encoding);
        !status.ok()) {
      return status;
    }
    // Initialization uses R=I. Keep native source values, including compact
    // U8 payloads; the list reader converts them when topology needs F32.
    return writer_->Append(EncodedBaseBlock<M, Scalar>{
        .first_id = static_cast<std::uint32_t>(block.first_id),
        .root_cids = encoding.root_codes(),
        .small_codes_row_major = encoding.small_codes_row_major(),
        .coefficients_row_major = encoding.coefficients_row_major(),
        .raw_vectors = block.source_values.data(),
        .raw_leading_dimension = block.dimension,
    });
  }

 private:
  AccumulateBasic accumulate_;
  BasicBucketStoreWriter<M>* writer_ = nullptr;
};

template <int M, typename Scalar, typename CreateRuntime,
          typename AccumulateBasic>
StatusOr<RootInitializationBasicStageResult> RunStage(
    const RootTrainingModel<M>& model,
    InitializationBasicBlockSourceFactory<Scalar>& source_factory,
    CreateRuntime create_runtime, AccumulateBasic accumulate,
    const ParallelForExecutor& parallel,
    const RootInitializationBasicStageOptions& options) {
  const ModelShape& shape = model.codebooks().shape();
  if (options.bucket_directory.empty() || options.list_directory.empty() ||
      options.bucket_directory == options.list_directory ||
      options.cids_per_bucket == 0U || options.source_rows == 0U ||
      options.source_block_rows == 0U) {
    return Status::InvalidArgument(
        "initialization basic-stage artifact options are incomplete");
  }
  auto created_writer = BasicBucketStoreWriter<M>::CreateTransient(
      options.bucket_directory,
      {
          .dimension = shape.dimension,
          .nlist = shape.nlist,
          .cids_per_bucket = options.cids_per_bucket,
          .vector_scalar = std::is_same_v<Scalar, std::uint8_t>
                               ? BaseVectorScalar::kU8 : BaseVectorScalar::kF32,
      },
      options.bucket_flush_bytes, shape.m);
  if (!created_writer.ok()) {
    return created_writer.status();
  }
  InitializationBucketLifetime bucket_lifetime(options.bucket_directory);
  std::optional<BasicRuntimeExecutionInfo> execution;
  auto statistics = [&]() -> StatusOr<BasicPipelineStatistics> {
    BasicBucketStoreWriter<M> writer = std::move(created_writer).value();
    auto source = source_factory.Open();
    if (!source.ok()) {
      return source.status();
    }
    auto runtime = create_runtime();
    if (!runtime.ok()) {
      return runtime.status();
    }
    execution = runtime.value()->execution_info();
    auto pipeline = options.pipeline;
    pipeline.cpu_backend = execution->primary_backend == BasicPrimaryBackend::kCpu;
    LogBasicExecution(*execution, options.schedule);
    InitializationBasicSink<M, Scalar, AccumulateBasic> sink(
        std::move(accumulate), writer);
    auto encoded = TrainingBasicPipeline<Scalar>::Run(
        *source.value(), sink, runtime.value()->primary(), options.schedule, options.seed,
        pipeline);
    if (!encoded.ok()) {
      return encoded.status();
    }
    if (const Status status = writer.Commit(); !status.ok()) {
      return status;
    }
    return std::move(encoded).value();
  }();
  if (!statistics.ok()) {
    return statistics.status();
  }
  LogBasicStageCompletion(
      {
          .shape = shape,
          .source_is_f32 = std::is_same_v<Scalar, float>,
          .source_rows = options.source_rows,
          .source_block_rows = options.source_block_rows,
          .cpu_threads = options.cpu_threads,
          .fixed_roots = false,
          .execution = *execution,
          .schedule = options.schedule,
          .seed = options.seed,
          .pipeline = options.pipeline,
      },
      statistics.value());
  auto buckets = BasicBucketStore<M>::Open(options.bucket_directory);
  if (!buckets.ok()) {
    return buckets.status();
  }
  const auto list_store_begin = std::chrono::steady_clock::now();
  if (const Status status = ParallelBaseListEmitter<M>::EmitTransient(
          buckets.value(), options.list_directory, parallel);
      !status.ok()) {
    return status;
  }
  LogBaseListStoreCompletion(
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    list_store_begin)
          .count());
  return RootInitializationBasicStageResult{
      .pipeline = statistics.value(),
      .rows = buckets.value().total_vectors(),
  };
}

}  // namespace

template <int M, typename Scalar>
StatusOr<RootInitializationBasicStageResult>
ScalarRootInitializationBasicStage<M, Scalar>::Run(
    const RootTrainingModel<M>& model,
    InitializationBasicBlockSourceFactory<Scalar>& source_factory,
    BasicRoundRuntimeFactory<M, Scalar>& runtime_factory,
    InitializationModelUpdater<M>& updater, const ParallelForExecutor& parallel,
    const RootInitializationBasicStageOptions& options) {
  return RunStage<M>(
      model, source_factory,
      [&] {
        return runtime_factory.Create(model.codebooks(), model.rotation());
      },
      [&](ColumnMajorMatrixF32View exact, const BasicEncoding<M>& encoding) {
        return updater.AccumulateBasic(exact, encoding);
      },
      parallel, options);
}

template <typename Scalar>
StatusOr<RootInitializationBasicStageResult>
RuntimeRootInitializationBasicStage<Scalar>::RunCpuRsd(
    const RuntimeRootTrainingModel& model,
    InitializationBasicBlockSourceFactory<Scalar>& source_factory,
    RuntimeRootCodebookMoments& moments, const F32GemmBackend& gemm,
    const ParallelForExecutor& encoding_parallel,
    const ParallelForExecutor& list_parallel, std::uint32_t rank,
    const RootInitializationBasicStageOptions& options) {
  const auto& shape = model.codebooks().shape();
  if (moments.m() != shape.m || moments.dimension() != shape.dimension ||
      moments.nlist() != shape.nlist)
    return Status::InvalidArgument(
        "initialization root moments differ from model shape");
  auto runtime = CreateRuntimeCpuRsdRound<Scalar>(
      model.codebooks().runtime_view(), model.rotation(), gemm,
      encoding_parallel, rank);
  if (!runtime.ok()) return runtime.status();
  return RunStage<0>(
      model, source_factory,
      [&]() -> StatusOr<std::unique_ptr<BasicRoundRuntime<Scalar>>> {
        return std::move(runtime).value();
      },
      [&](ColumnMajorMatrixF32View exact, const BasicEncoding<0>& encoding) {
        return moments.Accumulate(exact, encoding);
      },
      list_parallel, options);
}

template class RuntimeRootInitializationBasicStage<std::uint8_t>;
template class RuntimeRootInitializationBasicStage<float>;
template class ScalarRootInitializationBasicStage<0, std::uint8_t>;
template class ScalarRootInitializationBasicStage<0, float>;
template class ScalarRootInitializationBasicStage<5, std::uint8_t>;
template class ScalarRootInitializationBasicStage<12, std::uint8_t>;
template class ScalarRootInitializationBasicStage<6, std::uint8_t>;
template class ScalarRootInitializationBasicStage<10, std::uint8_t>;
template class ScalarRootInitializationBasicStage<5, float>;
template class ScalarRootInitializationBasicStage<12, float>;
template class ScalarRootInitializationBasicStage<6, float>;
template class ScalarRootInitializationBasicStage<10, float>;

}  // namespace stlqpp
