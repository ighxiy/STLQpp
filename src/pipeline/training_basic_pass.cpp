#include "stlqpp/pipeline/training_basic_pass.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <type_traits>
#include <utility>

#include "stlqpp/index/emit/base_list_emitter.h"
#include "stlqpp/index/emit/basic_bucket_store.h"
#include "stlqpp/training/basic/basic_stage_log.h"
#include "stlqpp/training/basic/detail/basic_encoding_access.h"

namespace stlqpp {
namespace {

using Clock = std::chrono::steady_clock;

double ElapsedSeconds(Clock::time_point begin) noexcept {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

std::filesystem::path RoundDirectory(const std::filesystem::path& root,
                                     std::uint32_t round) {
  std::ostringstream name;
  name << "round_" << std::setw(8) << std::setfill('0') << round;
  return root / name.str();
}

class RoundArtifactLifetime final : public TrainingBasicArtifactLifetime {
 public:
  explicit RoundArtifactLifetime(std::filesystem::path directory) noexcept
      : directory_(std::move(directory)) {}

  ~RoundArtifactLifetime() override {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

 private:
  std::filesystem::path directory_;
};

}  // namespace

namespace detail {

template <int M, typename Scalar>
class RootMomentAndArtifactSink final : public TrainingBasicBlockSink<Scalar> {
 public:
  RootMomentAndArtifactSink(RoundModelUpdater<M>& updater,
                            BasicBucketStoreWriter<M>& writer) noexcept
      : updater_(&updater), writer_(&writer) {}

  [[nodiscard]] Status Consume(
      EncodedTrainingBasicBlock<Scalar> block) override {
    if (block.first_id > std::numeric_limits<std::uint32_t>::max()) {
      return Status::ResourceExhausted(
          "training row ID exceeds the canonical u32 domain");
    }
    const auto model_space = block.model_space_view();
    auto encoding =
        detail::BasicEncodingAccess::Adopt<M>(std::move(block.encoding));
    if (const Status status = updater_->AccumulateBasic(model_space, encoding);
        !status.ok()) {
      return status;
    }
    return writer_->AppendAfterBasicMoments(EncodedBaseBlock<M, Scalar>{
        .first_id = static_cast<std::uint32_t>(block.first_id),
        .root_cids = encoding.root_codes(),
        .small_codes_row_major = encoding.small_codes_row_major(),
        .coefficients_row_major = encoding.coefficients_row_major(),
        .raw_vectors = block.source_values.data(),
        .raw_leading_dimension = block.dimension,
    });
  }

 private:
  RoundModelUpdater<M>* updater_ = nullptr;
  BasicBucketStoreWriter<M>* writer_ = nullptr;
};

}  // namespace detail

template <int M, typename Scalar>
StatusOr<TrainingBasicRoundArtifact>
ScalarTrainingBasicPass<M, Scalar>::Accumulate(const TrainingModel<M>& model,
                                               std::uint32_t round,
                                               std::uint32_t seed,
                                               RoundModelUpdater<M>& updater) {
  return AccumulateImpl(model, round, seed, {}, updater);
}

template <int M>
StatusOr<std::vector<std::uint32_t>> BuildFixedRootAssignments(
    const TrainingCheckpoint<M>& checkpoint) {
  const std::uint32_t nlist = checkpoint.model().codebooks().shape().nlist;
  std::uint64_t row_count = 0U;
  for (std::uint32_t cid = 0U; cid < nlist; ++cid) {
    auto cluster = checkpoint.LoadCluster(cid);
    if (!cluster.ok()) {
      return cluster.status();
    }
    row_count += cluster.value().real_count();
  }
  if (row_count == 0U ||
      row_count > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "fixed-root checkpoint row domain exceeds u32");
  }
  std::vector<std::uint32_t> roots(static_cast<std::size_t>(row_count), nlist);
  for (std::uint32_t cid = 0U; cid < nlist; ++cid) {
    auto cluster = checkpoint.LoadCluster(cid);
    if (!cluster.ok()) {
      return cluster.status();
    }
    for (const std::uint64_t stable_id : cluster.value().real_sample_ids) {
      if (stable_id >= roots.size() || roots[stable_id] != nlist) {
        return Status::DataLoss(
            "fixed-root checkpoint stable IDs are not a permutation");
      }
      roots[stable_id] = cid;
    }
  }
  for (const std::uint32_t cid : roots) {
    if (cid == nlist) {
      return Status::DataLoss(
          "fixed-root checkpoint stable-ID domain has a gap");
    }
  }
  return roots;
}

template <int M, typename Scalar>
StatusOr<TrainingBasicRoundArtifact>
ScalarTrainingBasicPass<M, Scalar>::AccumulateFixedRoots(
    const TrainingModel<M>& model,
    const TrainingCheckpoint<M>& previous_checkpoint, std::uint32_t round,
    std::uint32_t seed, RoundModelUpdater<M>& updater) {
  auto fixed_roots = BuildFixedRootAssignments(previous_checkpoint);
  if (!fixed_roots.ok()) {
    return fixed_roots.status();
  }
  return AccumulateImpl(model, round, seed, fixed_roots.value(), updater);
}

template <int M, typename Scalar>
StatusOr<TrainingBasicRoundArtifact>
ScalarTrainingBasicPass<M, Scalar>::AccumulateImpl(
    const TrainingModel<M>& model, std::uint32_t round, std::uint32_t seed,
    ArrayView<const std::uint32_t> fixed_roots, RoundModelUpdater<M>& updater) {
  if (options_.artifact_root.empty() || options_.cids_per_bucket == 0U ||
      options_.source_rows == 0U || options_.source_block_rows == 0U) {
    return Status::InvalidArgument(
        "training basic-pass artifact policy is incomplete");
  }
  const ModelShape& shape = model.codebooks().shape();
  const ModelShape& moment_shape = updater.basic_model_shape();
  if (shape.dimension != moment_shape.dimension ||
      shape.nlist != moment_shape.nlist) {
    return Status::InvalidArgument(
        "training basic-pass model disagrees with root moments");
  }
  const std::filesystem::path round_directory =
      RoundDirectory(options_.artifact_root, round);
  const std::filesystem::path bucket_directory = round_directory / "buckets";
  const std::filesystem::path list_directory = round_directory / "lists";
  std::error_code cleanup_error;
  std::filesystem::remove_all(round_directory, cleanup_error);
  if (cleanup_error) {
    return Status::Internal(
        "failed to replace the round-local training basic workset");
  }
  auto lifetime = std::make_shared<RoundArtifactLifetime>(round_directory);
  const Clock::time_point preparation_begin = Clock::now();
  auto created_writer = BasicBucketStoreWriter<M>::CreateTransient(
      bucket_directory,
      {
          .dimension = shape.dimension,
          .nlist = shape.nlist,
          .cids_per_bucket = options_.cids_per_bucket,
          .vector_scalar = std::is_same_v<Scalar, std::uint8_t>
                               ? BaseVectorScalar::kU8
                               : BaseVectorScalar::kF32,
      },
      options_.bucket_flush_bytes, shape.m);
  if (!created_writer.ok()) {
    return created_writer.status();
  }
  double basic_open_seconds = 0.0;
  double precompute_root_seconds = 0.0;
  std::optional<BasicRuntimeExecutionInfo> execution;
  auto statistics = [&]() -> StatusOr<BasicPipelineStatistics> {
    BasicBucketStoreWriter<M> writer = std::move(created_writer).value();
    auto source = source_->Open(round);
    if (!source.ok()) {
      return source.status();
    }
    basic_open_seconds = ElapsedSeconds(preparation_begin);
    const Clock::time_point precompute_begin = Clock::now();
    auto runtime =
        runtime_->Create(model.codebooks().root_codebooks(), model.rotation());
    if (!runtime.ok()) {
      return runtime.status();
    }
    execution = runtime.value()->execution_info();
    auto pipeline = options_.pipeline;
    pipeline.cpu_backend = execution->primary_backend == BasicPrimaryBackend::kCpu;
    LogBasicExecution(*execution, options_.schedule);
    precompute_root_seconds = ElapsedSeconds(precompute_begin);
    LogTrainingRoundPreparation(precompute_root_seconds, basic_open_seconds,
                                ElapsedSeconds(preparation_begin));
    detail::RootMomentAndArtifactSink<M, Scalar> sink(updater, writer);
    auto encoded =
        fixed_roots.empty()
            ? TrainingBasicPipeline<Scalar>::Run(
                  *source.value(), sink, runtime.value()->primary(), options_.schedule, seed,
                  pipeline)
            : FixedRootTrainingBasicPipeline<Scalar>::Run(
                  *source.value(), sink, runtime.value()->primary(), fixed_roots, options_.schedule,
                  seed, pipeline);
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
          .source_rows = options_.source_rows,
          .source_block_rows = options_.source_block_rows,
          .cpu_threads = options_.cpu_threads,
          .fixed_roots = !fixed_roots.empty(),
          .execution = *execution,
          .schedule = options_.schedule,
          .seed = seed,
          .pipeline = options_.pipeline,
      },
      statistics.value());
  auto buckets = BasicBucketStore<M>::Open(bucket_directory);
  if (!buckets.ok()) {
    return buckets.status();
  }
  const Clock::time_point list_store_begin = Clock::now();
  if (const Status status = ParallelBaseListEmitter<M>::EmitTransient(
          buckets.value(), list_directory, *parallel_);
      !status.ok()) {
    return status;
  }
  LogBaseListStoreCompletion(ElapsedSeconds(list_store_begin));
  last_statistics_.emplace(statistics.value());
  return TrainingBasicRoundArtifact{
      .list_directory = list_directory,
      .rows = buckets.value().total_vectors(),
      .lifetime = std::move(lifetime),
      .metric_vectors = statistics.value().metric_vectors,
      .initial_squared_error_sum = statistics.value().initial_squared_error_sum,
      .final_squared_error_sum = statistics.value().final_squared_error_sum,
      .encode_wall_seconds = statistics.value().wall_seconds,
  };
}

template class ScalarTrainingBasicPass<0, std::uint8_t>;
template class ScalarTrainingBasicPass<0, float>;
template class ScalarTrainingBasicPass<5, std::uint8_t>;
template class ScalarTrainingBasicPass<12, std::uint8_t>;
template class ScalarTrainingBasicPass<6, std::uint8_t>;
template class ScalarTrainingBasicPass<10, std::uint8_t>;
template class ScalarTrainingBasicPass<5, float>;
template class ScalarTrainingBasicPass<12, float>;
template class ScalarTrainingBasicPass<6, float>;
template class ScalarTrainingBasicPass<10, float>;

}  // namespace stlqpp
