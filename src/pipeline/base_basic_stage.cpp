#include "stlqpp/pipeline/base_basic_stage.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include "stlqpp/core/filesystem_path.h"
#include "stlqpp/index/emit/base_list_emitter.h"
#include "stlqpp/index/emit/basic_bucket_store.h"
#include "stlqpp/runtime/telemetry/logger.h"
#include "stlqpp/storage/artifact_hash_factory.h"
#include "stlqpp/training/basic/basic_stage_log.h"
#include "stlqpp/training/basic/runtime_basic_executor.h"

namespace stlqpp {
namespace {

template <int M>
ArtifactHash64 BaseModelIdentity(const TrainingModel<M>& model) {
  TrainedModelHashView view;
  view.rotation = model.rotation();
  view.root_codebooks.reserve(model.codebooks().shape().m);
  view.root_codebooks.push_back(model.codebooks().root0());
  for (std::uint32_t layer = 1U; layer < model.codebooks().shape().m; ++layer) {
    view.root_codebooks.push_back(model.codebooks().root_small(layer));
  }
  return ArtifactHashFactory::BaseModel(view);
}

class BaseBucketLifetime final {
 public:
  explicit BaseBucketLifetime(std::filesystem::path directory) noexcept
      : directory_(std::move(directory)) {}

  ~BaseBucketLifetime() {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

 private:
  std::filesystem::path directory_;
};

// Before list publication, checkpoint buckets own recovery. After publication,
// the identity-bound list store owns it, including after a later linkage error.
Status CleanupCompletedCheckpointBuckets(const BaseBasicStageOptions& options) {
  if (!options.checkpoint_enabled || options.retain_bucket_directory)
    return Status::Ok();
  std::error_code error;
  const auto removed = std::filesystem::remove_all(options.bucket_directory, error);
  if (error)
    return Status::Internal("cannot release completed base checkpoint buckets: " +
                            options.bucket_directory.string() + ": " + error.message());
  if (removed != 0U)
    LogInfo("Cleanup: deleted completed base checkpoint buckets " +
            options.bucket_directory.string());
  return Status::Ok();
}

template <int M, typename Scalar>
class BaseBasicSink final : public BasicBlockSink<Scalar> {
 public:
  explicit BaseBasicSink(BasicBucketStoreWriter<M>& writer) noexcept
      : writer_(&writer) {}

  [[nodiscard]] Status Consume(EncodedBasicBlock<Scalar> block) override {
    if (block.first_id > std::numeric_limits<std::uint32_t>::max()) {
      return Status::ResourceExhausted(
          "base row ID exceeds the canonical u32 domain");
    }
    return writer_->Append(EncodedBaseBlock<M, Scalar>{
        .first_id = static_cast<std::uint32_t>(block.first_id),
        .root_cids = block.encoding.root_codes(),
        .small_codes_row_major = block.encoding.small_codes_row_major(),
        .coefficients_row_major = block.encoding.coefficients_row_major(),
        .raw_vectors = block.values.data(),
        .raw_leading_dimension = block.dimension,
    });
  }

 private:
  BasicBucketStoreWriter<M>* writer_ = nullptr;
};

template <int M, typename Scalar, typename CreateRuntime>
StatusOr<BaseBasicStageResult> RunStage(
    const ModelShape& shape, ArtifactHash64 source_model_identity,
    BaseBasicBlockSourceFactory<Scalar>& source_factory,
    CreateRuntime create_runtime, const ParallelForExecutor& parallel,
    const BaseBasicStageOptions& options) {
  const auto encoding_stage_begin = std::chrono::steady_clock::now();
  if (options.bucket_directory.empty() || options.list_directory.empty() ||
      options.bucket_directory == options.list_directory ||
      options.cids_per_bucket == 0U || options.list_identity.empty()) {
    return Status::InvalidArgument(
        "base basic-stage artifact options are incomplete");
  }
  const BaseBuildShape bucket_shape = {
      .dimension = shape.dimension,
      .nlist = shape.nlist,
      .cids_per_bucket = options.cids_per_bucket,
      .vector_scalar = std::is_same_v<Scalar, std::uint8_t>
                           ? BaseVectorScalar::kU8
                           : BaseVectorScalar::kF32,
  };
  if (options.checkpoint_enabled &&
      (options.checkpoint_identity.empty() || options.source_rows == 0U)) {
    return Status::InvalidArgument(
        "base basic-stage checkpoint identity or source extent is missing");
  }
  if (const auto status = ValidateDisjointFilesystemPaths(
          options.bucket_directory, "base buckets", options.list_directory,
          "base lists"); !status.ok())
    return status;
  // Test the published stage before opening (or creating) bucket checkpoints.
  // The list identity incorporates the basic identity; no bucket payload is
  // needed once the complete list has been atomically published.
  if (options.checkpoint_enabled) {
    std::error_code error;
    const bool lists_exist = std::filesystem::exists(options.list_directory, error);
    if (error)
      return Status::Internal("failed to inspect completed base-list checkpoint");
    if (lists_exist) {
      auto lists = BaseListStore<M>::Open(options.list_directory);
      if (!lists.ok()) return lists.status();
      if (lists.value().m() != shape.m ||
          lists.value().source_model_identity() != source_model_identity ||
          lists.value().total_vectors() != options.source_rows ||
          lists.value().identity() != options.list_identity ||
          lists.value().shape().dimension != bucket_shape.dimension ||
          lists.value().shape().nlist != bucket_shape.nlist ||
          lists.value().shape().vector_scalar != bucket_shape.vector_scalar) {
        return Status::DataLoss(
            "completed base-list checkpoint changed identity, shape, or row count");
      }
      if (auto status = CleanupCompletedCheckpointBuckets(options); !status.ok())
        return status;
      LogInfo("Reusing completed base-list checkpoint: " +
              options.list_directory.string() + " (no basic re-encoding)");
      return BaseBasicStageResult{
          .pipeline = {}, .rows = options.source_rows,
          .list_directory = options.list_directory};
    }
  }
  std::error_code path_error;
  const bool resume =
      options.checkpoint_enabled &&
      std::filesystem::exists(options.bucket_directory, path_error);
  if (path_error) {
    return Status::Internal(
        "failed to inspect base basic-stage checkpoint directory");
  }
  auto created_writer =
      options.checkpoint_enabled
          ? (resume ? BasicBucketStoreWriter<M>::OpenCheckpointed(
                          options.bucket_directory, bucket_shape,
                          options.checkpoint_identity,
                          options.bucket_flush_bytes, shape.m)
                    : BasicBucketStoreWriter<M>::CreateCheckpointed(
                          options.bucket_directory, bucket_shape,
                          options.checkpoint_identity,
                          options.bucket_flush_bytes, shape.m))
          : BasicBucketStoreWriter<M>::CreateTransient(
                options.bucket_directory, bucket_shape,
                options.bucket_flush_bytes, shape.m);
  if (!created_writer.ok()) {
    return created_writer.status();
  }
  auto bucket_lifetime =
      options.checkpoint_enabled ? std::unique_ptr<BaseBucketLifetime>{}
      : options.retain_bucket_directory
          ? std::unique_ptr<BaseBucketLifetime>{}
          : std::make_unique<BaseBucketLifetime>(options.bucket_directory);
  const std::uint64_t first_row = created_writer.value().next_id();
  if (options.checkpoint_enabled && first_row > options.source_rows) {
    return Status::DataLoss(
        "base basic-stage checkpoint exceeds the registered source");
  }
  std::optional<BasicRuntimeExecutionInfo> execution;
  auto statistics = [&]() -> StatusOr<BasicPipelineStatistics> {
    BasicBucketStoreWriter<M> writer = std::move(created_writer).value();
    BasicPipelineStatistics encoded_statistics;
    if (!options.checkpoint_enabled || first_row < options.source_rows) {
      auto source = options.checkpoint_enabled && first_row != 0U
                        ? source_factory.OpenRange(
                              first_row, options.source_rows - first_row)
                        : source_factory.Open();
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
      BaseBasicSink<M, Scalar> sink(writer);
      if (first_row != 0U && (options.source_block_rows == 0U ||
                              first_row % options.source_block_rows != 0U)) {
        return Status::DataLoss(
            "base basic checkpoint does not end at a configured block "
            "boundary");
      }
      auto encoded = BasicPipeline<Scalar>::Run(
          *source.value(), sink, runtime.value()->primary(), options.schedule, options.seed,
          pipeline);
      if (!encoded.ok()) {
        return encoded.status();
      }
      encoded_statistics = std::move(encoded).value();
    }
    if (const Status status = writer.Commit(); !status.ok()) {
      return status;
    }
    return encoded_statistics;
  }();
  if (!statistics.ok()) {
    return statistics.status();
  }
  if (execution.has_value()) {
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
  }
  LogLargeScaleBaseEncodingCompletion(
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    encoding_stage_begin)
          .count());
  auto buckets = BasicBucketStore<M>::Open(options.bucket_directory);
  if (!buckets.ok()) {
    return buckets.status();
  }
  const auto list_store_begin = std::chrono::steady_clock::now();
  LogInfo("Building list-order base store...");
  if (const Status status = ParallelBaseListEmitter<M>::EmitTransient(
          buckets.value(), options.list_directory, parallel,
          options.list_identity, source_model_identity);
      !status.ok()) {
    return status;
  }
  LogBaseListStoreCompletion(
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    list_store_begin)
          .count());
  if (auto status = CleanupCompletedCheckpointBuckets(options); !status.ok())
    return status;
  return BaseBasicStageResult{
      .pipeline = statistics.value(),
      .rows = buckets.value().total_vectors(),
      .list_directory = options.list_directory,
  };
}

}  // namespace

template <int M, typename Scalar>
StatusOr<BaseBasicStageResult> ScalarBaseBasicStage<M, Scalar>::Run(
    const TrainingModel<M>& model,
    BaseBasicBlockSourceFactory<Scalar>& source_factory,
    BasicRoundRuntimeFactory<M, Scalar>& runtime_factory,
    const ParallelForExecutor& parallel, const BaseBasicStageOptions& options) {
  return RunStage<M>(
      model.codebooks().shape(), BaseModelIdentity(model), source_factory,
      [&] {
        return runtime_factory.Create(model.codebooks().root_codebooks(),
                                      model.rotation());
      },
      parallel, options);
}

template <typename Scalar>
StatusOr<BaseBasicStageResult> RuntimeBaseBasicStage<Scalar>::RunCpuRsd(
    RuntimeRootCodebookView roots, ColumnMajorMatrixF32View rotation,
    BaseBasicBlockSourceFactory<Scalar>& source_factory,
    const F32GemmBackend& gemm, const ParallelForExecutor& encoding_parallel,
    const ParallelForExecutor& list_parallel, std::uint32_t rank,
    const BaseBasicStageOptions& options) {
  if (const auto s = roots.Validate(); !s.ok()) return s;
  TrainedModelHashView hash;
  hash.rotation = rotation;
  hash.root_codebooks.push_back(roots.root0);
  for (std::uint32_t layer = 1; layer < roots.shape.m; ++layer)
    hash.root_codebooks.push_back(
        {roots.root_small_flat.column((layer - 1U) * 256U),
         roots.shape.dimension, 256U});
  return RunStage<0>(
      roots.shape, ArtifactHashFactory::BaseModel(hash), source_factory,
      [&]() -> StatusOr<std::unique_ptr<BasicRoundRuntime<Scalar>>> {
        return CreateRuntimeCpuRsdRound<Scalar>(roots, rotation, gemm,
                                               encoding_parallel, rank);
      },
      list_parallel, options);
}

template class RuntimeBaseBasicStage<std::uint8_t>;
template class RuntimeBaseBasicStage<float>;
template class ScalarBaseBasicStage<0, std::uint8_t>;
template class ScalarBaseBasicStage<0, float>;
template class ScalarBaseBasicStage<5, std::uint8_t>;
template class ScalarBaseBasicStage<12, std::uint8_t>;
template class ScalarBaseBasicStage<6, std::uint8_t>;
template class ScalarBaseBasicStage<10, std::uint8_t>;
template class ScalarBaseBasicStage<5, float>;
template class ScalarBaseBasicStage<12, float>;
template class ScalarBaseBasicStage<6, float>;
template class ScalarBaseBasicStage<10, float>;

}  // namespace stlqpp
