#include "stlqpp/pipeline/cpu_training_topology_context_factory.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "stlqpp/index/emit/base_list_reader.h"
#include "stlqpp/pipeline/base_list_full_topology_source.h"
#include "stlqpp/pipeline/stable_id_replay_source.h"
#include "stlqpp/runtime/exclusive_index_pool.h"
#include "stlqpp/training/basic/cpu_virtual_root_support_encoder.h"
#include "stlqpp/training/basic/large_root_algebra.h"
#include "stlqpp/training/basic/virtual_root_support_encoder_pool.h"
#include "stlqpp/training/checkpoint/checkpoint_replay_loader.h"
#include "stlqpp/training/linkage/candidates/cpu_exact_knn.h"
#include "stlqpp/training/linkage/candidates/cpu_exact_parent_candidates.h"
#include "stlqpp/training/linkage/replay/cpu_fixed_edge_replay_encoder.h"
#include "stlqpp/training/linkage/replay/fixed_topology_replay_executor.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/rsd/ranked_single_deviation.h"
#include "stlqpp/training/linkage/scheduler/cpu_linkage_worker_backend.h"
#include "stlqpp/training/linkage/scheduler/direct_linkage_evaluator_pool.h"
#include "stlqpp/training/linkage/topology/causal_cpu_linkage_evaluator.h"
#include "stlqpp/training/linkage/topology/full_topology_cluster_builder.h"

namespace stlqpp {
namespace {

template <int M>
Status ValidateArtifact(const BaseListReader<M>& lists,
                        const TrainingModel<M>& model,
                        const TrainingBasicRoundArtifact& basic) {
  const BaseBuildShape& shape = lists.store().shape();
  const ModelShape& model_shape = model.codebooks().shape();
  if (lists.store().directory() != basic.list_directory ||
      lists.store().total_vectors() != basic.rows ||
      shape.dimension != model_shape.dimension ||
      shape.nlist != model_shape.nlist || lists.store().m() != model_shape.m ||
      (M != 0 && model_shape.m != static_cast<std::uint32_t>(M))) {
    return Status::FailedPrecondition(
        "round basic artifact does not match its CPU topology model");
  }
  return Status::Ok();
}

template <int M>
LinkageResourceStatistics ObserveCpuResources(
    std::size_t lanes, std::uint64_t workspace_reserve_bytes,
    const LinkageEvaluationServiceStats& stats) {
  LinkageResourceStatistics output;
  output.direct_backend_count = lanes;
  output.live_registration_capacity_bytes =
      stats.registration_admission_capacity_bytes;
  output.live_registration_peak_bytes = stats.registration_admission_peak_bytes;
  output.deferred_workspace_reserve_bytes = workspace_reserve_bytes;
  output.evaluation_source_tasks = stats.source_tasks_submitted;
  output.evaluation_transport_batches = stats.transport_batches;
  output.evaluation_cross_cluster_transport_batches =
      stats.cross_cluster_transport_batches;
  output.evaluation_numerical_batches = stats.numerical_batches;
  output.evaluation_cross_cluster_numerical_batches =
      stats.cross_cluster_numerical_batches;
  output.merged_numerical_source_tasks = stats.merged_numerical_source_tasks;
  output.maximum_source_tasks_per_transport_batch =
      stats.maximum_source_tasks_per_transport_batch;
  output.maximum_source_tasks_per_numerical_batch =
      stats.maximum_source_tasks_per_numerical_batch;
  output.evaluation_nodes = stats.nodes;
  output.evaluation_pairs = stats.pairs;
  output.update_requests = stats.update_requests;
  output.update_batches = stats.update_batches;
  output.update_columns = stats.update_columns;
  output.ordinary_direct_lane_count = lanes;
  return output;
}

template <int M>
class CpuFullTopologyRoundContext final : public FullTopologyRoundContext<M> {
 public:
  static StatusOr<std::unique_ptr<FullTopologyRoundContext<M>>> Create(
      BaseListReader<M> lists, const TrainingModel<M>& model,
      const F32GemmBackend& gemm,
      const ParallelForExecutor& parallel,
      ArrayView<const std::uint8_t> bad_cluster_mask, std::uint32_t rsd_rank,
      std::uint32_t icm_rounds, bool collect_source_norm2,
      const CpuTrainingTopologyContextOptions& options) {
    if (options.full_topology_worker_count == 0U ||
        options.direct_lane_count == 0U ||
        options.maximum_registration_and_workspace_bytes == 0U ||
        options.maximum_task_nodes == 0U || options.maximum_task_pairs == 0U ||
        options.maximum_update_columns == 0U) {
      return Status::InvalidArgument(
          "CPU full topology requires bounded nonzero worker resources");
    }
    auto root_algebra =
        BasicCodebookAlgebra<M>::CreateForVirtualRoots(model.codebooks(), gemm, &parallel);
    if (!root_algebra.ok()) return root_algebra.status();
    auto edge_algebra =
        LinkageCodebookAlgebra<M>::Create(model.codebooks(), gemm);
    if (!edge_algebra.ok()) return edge_algebra.status();
    auto context = std::unique_ptr<CpuFullTopologyRoundContext<M>>(
        new CpuFullTopologyRoundContext<M>(
            std::move(lists), std::move(root_algebra).value(),
            std::move(edge_algebra).value(), gemm));
    if (const Status status = context->Initialize(
            model, parallel, bad_cluster_mask, rsd_rank, icm_rounds,
            collect_source_norm2, options);
        !status.ok()) {
      return status;
    }
    return std::unique_ptr<FullTopologyRoundContext<M>>(std::move(context));
  }

  [[nodiscard]] FullTopologyClusterSource<M>& source() noexcept override {
    return *source_;
  }

  [[nodiscard]] ArrayView<FullTopologyClusterExecutor<M>*> executors() noexcept
      override {
    return executor_views_;
  }

  [[nodiscard]] std::optional<LinkageResourceStatistics> resource_statistics()
      const noexcept override {
    return ObserveCpuResources<M>(direct_lane_count_, workspace_reserve_bytes_,
                                  evaluator_->Stats());
  }

 private:
  CpuFullTopologyRoundContext(BaseListReader<M> lists,
                              BasicCodebookAlgebra<M> root_algebra,
                              LinkageCodebookAlgebra<M> edge_algebra,
                              const F32GemmBackend& gemm) noexcept
      : lists_(std::move(lists)),
        root_algebra_(std::move(root_algebra)),
        edge_algebra_(std::move(edge_algebra)),
        good_candidates_(gemm),
        exact_knn_(gemm),
        gemm_(&gemm) {}

  Status Initialize(const TrainingModel<M>& model,
                    const ParallelForExecutor& parallel,
                    ArrayView<const std::uint8_t> bad_cluster_mask,
                    std::uint32_t rsd_rank, std::uint32_t icm_rounds,
                    bool collect_source_norm2,
                    const CpuTrainingTopologyContextOptions& options) {
    auto source = BaseListFullTopologySource<M>::Create(
        lists_, model, *gemm_, bad_cluster_mask, collect_source_norm2);
    if (!source.ok()) return source.status();
    source_ = std::make_unique<BaseListFullTopologySource<M>>(
        std::move(source).value());

    auto cpu_tail = CausalCpuLinkageEvaluator<M>::Create(
        model.codebooks(), edge_algebra_, rsd_rank,
        options.least_squares_cost_mode, icm_rounds);
    if (!cpu_tail.ok()) return cpu_tail.status();
    cpu_tail_ = std::make_unique<CausalCpuLinkageEvaluator<M>>(
        std::move(cpu_tail).value());

    auto lanes = ExclusiveIndexPool::Create(options.direct_lane_count);
    if (!lanes.ok()) return lanes.status();
    const std::uint64_t workspace_per_lane =
        CpuEdgeLinkageMatrixWorkspaceBytes<M>(
            model.codebooks().shape().dimension, options.maximum_task_pairs,
            model.codebooks().shape().m);
    if (options.direct_lane_count >
        std::numeric_limits<std::uint64_t>::max() / workspace_per_lane) {
      return Status::ResourceExhausted(
          "CPU linkage matrix workspace reservation overflows");
    }
    workspace_reserve_bytes_ = workspace_per_lane * options.direct_lane_count;
    if (workspace_reserve_bytes_ >=
        options.maximum_registration_and_workspace_bytes) {
      return Status::ResourceExhausted(
          "CPU linkage matrix workspace exhausts the phase budget");
    }
    const std::uint64_t registration_capacity =
        options.maximum_registration_and_workspace_bytes -
        workspace_reserve_bytes_;
    std::vector<std::unique_ptr<LinkageWorkerBackend>> backends;
    std::vector<std::unique_ptr<VirtualRootSupportEncoder<M>>> roots;
    backends.reserve(options.direct_lane_count);
    roots.reserve(options.direct_lane_count);
    for (std::size_t lane = 0U; lane < options.direct_lane_count; ++lane) {
      backends.push_back(std::make_unique<CpuRsdLinkageWorkerBackend<M>>(
          edge_algebra_, *gemm_, parallel, rsd_rank,
          options.least_squares_cost_mode, icm_rounds));
      roots.push_back(std::make_unique<CpuVirtualRootSupportEncoder<M>>(
          root_algebra_, *gemm_, options.virtual_root_schedule));
    }
    auto evaluator = DirectLinkageEvaluatorPool::Create(
        std::move(backends),
        {
            .maximum_live_registration_bytes = registration_capacity,
            .maximum_task_nodes = options.maximum_task_nodes,
            .maximum_task_pairs = options.maximum_task_pairs,
            .maximum_update_columns = options.maximum_update_columns,
            .asynchronous_evaluation = false,
        },
        lanes.value());
    if (!evaluator.ok()) return evaluator.status();
    evaluator_ = std::move(evaluator).value();
    auto root_pool = VirtualRootSupportEncoderPool<M>::Create(
        std::move(roots), std::move(lanes).value());
    if (!root_pool.ok()) return root_pool.status();
    root_pool_ = std::move(root_pool).value();
    direct_lane_count_ = options.direct_lane_count;

    builders_.reserve(options.full_topology_worker_count);
    executor_views_.reserve(options.full_topology_worker_count);
    for (std::size_t worker = 0U; worker < options.full_topology_worker_count;
         ++worker) {
      builders_.push_back(std::make_unique<FullTopologyClusterBuilder<M>>(
          model.codebooks(), good_candidates_, exact_knn_, *root_pool_,
          *evaluator_, cpu_tail_.get()));
      executor_views_.push_back(builders_.back().get());
    }
    return Status::Ok();
  }

  BaseListReader<M> lists_;
  BasicCodebookAlgebra<M> root_algebra_;
  LinkageCodebookAlgebra<M> edge_algebra_;
  CpuExactParentCandidateProvider good_candidates_;
  CpuExactKnnProvider exact_knn_;
  const F32GemmBackend* gemm_ = nullptr;
  std::unique_ptr<BaseListFullTopologySource<M>> source_;
  std::unique_ptr<LinkageEvaluationService> evaluator_;
  std::unique_ptr<VirtualRootSupportEncoderPool<M>> root_pool_;
  std::unique_ptr<CausalCpuLinkageEvaluator<M>> cpu_tail_;
  std::size_t direct_lane_count_ = 0U;
  std::uint64_t workspace_reserve_bytes_ = 0U;
  std::vector<std::unique_ptr<FullTopologyClusterBuilder<M>>> builders_;
  std::vector<FullTopologyClusterExecutor<M>*> executor_views_;
};

template <int M>
class CpuReplayTopologyRoundContext final
    : public ReplayTopologyRoundContext<M> {
 public:
  static StatusOr<std::unique_ptr<ReplayTopologyRoundContext<M>>> Create(
      BaseListReader<M> lists, const TrainingCheckpoint<M>& checkpoint,
      const TrainingModel<M>& model, const F32GemmBackend& gemm,
      const ParallelForExecutor& parallel, std::uint32_t rsd_rank,
      std::uint32_t icm_rounds, std::uint32_t seed,
      LeastSquaresCostMode cost_mode) {
    if (!IsSupportedRsdRank(rsd_rank)) {
      return Status::InvalidArgument("CPU replay RSD rank is unsupported");
    }
    auto algebra = LinkageCodebookAlgebra<M>::Create(model.codebooks(), gemm);
    if (!algebra.ok()) return algebra.status();
    auto context = std::unique_ptr<CpuReplayTopologyRoundContext<M>>(
        new CpuReplayTopologyRoundContext<M>(std::move(lists),
                                             std::move(algebra).value()));
    if (const Status status =
            context->Initialize(checkpoint, model, gemm, parallel, rsd_rank,
                                icm_rounds, seed, cost_mode);
        !status.ok()) {
      return status;
    }
    return std::unique_ptr<ReplayTopologyRoundContext<M>>(std::move(context));
  }

  [[nodiscard]] FixedTopologyReplayWindowLoader& loader() noexcept override {
    return *loader_;
  }

  [[nodiscard]] const FixedTopologyReplayExecutor<M>& executor()
      const noexcept override {
    return *executor_;
  }

 private:
  CpuReplayTopologyRoundContext(BaseListReader<M> lists,
                                LinkageCodebookAlgebra<M> algebra) noexcept
      : lists_(std::move(lists)), algebra_(std::move(algebra)) {}

  Status Initialize(const TrainingCheckpoint<M>& checkpoint,
                    const TrainingModel<M>& model, const F32GemmBackend& gemm,
                    const ParallelForExecutor& parallel, std::uint32_t rsd_rank,
                    std::uint32_t icm_rounds, std::uint32_t seed,
                    LeastSquaresCostMode cost_mode) {
    real_source_ = std::make_unique<StableIdReplaySource<M>>(lists_);
    auto loader = CheckpointReplayWindowLoader<M>::Create(
        checkpoint, model, *real_source_, gemm, seed);
    if (!loader.ok()) return loader.status();
    loader_ = std::make_unique<CheckpointReplayWindowLoader<M>>(
        std::move(loader).value());
    edge_encoder_ = std::make_unique<CpuFixedEdgeReplayEncoder<M>>(
        algebra_, gemm, parallel, rsd_rank, cost_mode, icm_rounds);
    executor_ = std::make_unique<FixedTopologyReplayExecutor<M>>(
        model.codebooks(), *edge_encoder_);
    return Status::Ok();
  }

  BaseListReader<M> lists_;
  LinkageCodebookAlgebra<M> algebra_;
  std::unique_ptr<StableIdReplaySource<M>> real_source_;
  std::unique_ptr<CheckpointReplayWindowLoader<M>> loader_;
  std::unique_ptr<CpuFixedEdgeReplayEncoder<M>> edge_encoder_;
  std::unique_ptr<FixedTopologyReplayExecutor<M>> executor_;
};

template <int M>
StatusOr<BaseListReader<M>> OpenRoundArtifact(
    const TrainingBasicRoundArtifact& basic, const TrainingModel<M>& model) {
  auto lists = BaseListReader<M>::Open(basic.list_directory);
  if (!lists.ok()) return lists.status();
  if (const Status status = ValidateArtifact(lists.value(), model, basic);
      !status.ok()) {
    return status;
  }
  return std::move(lists).value();
}

}  // namespace

template <int M>
StatusOr<std::unique_ptr<FullTopologyRoundContext<M>>>
CpuTrainingTopologyContextFactory<M>::CreateFull(
    const TrainingModel<M>& topology_model,
    ArrayView<const std::uint8_t> bad_cluster_mask,
    const TrainingBasicRoundArtifact& basic, std::uint32_t,
    const FullTopologyPolicy& policy) {
  auto lists = OpenRoundArtifact<M>(basic, topology_model);
  if (!lists.ok()) return lists.status();
  return CpuFullTopologyRoundContext<M>::Create(
      std::move(lists).value(), topology_model, *gemm_,
      direct_lane_parallel_, bad_cluster_mask, policy.rsd_rank,
      policy.icm_rounds, policy.collect_source_norm2, options_);
}

template <int M>
StatusOr<std::unique_ptr<ReplayTopologyRoundContext<M>>>
CpuTrainingTopologyContextFactory<M>::CreateReplay(
    const TrainingCheckpoint<M>& previous_checkpoint,
    const TrainingModel<M>& topology_model,
    const TrainingBasicRoundArtifact& basic, std::uint32_t,
    std::uint32_t rsd_rank, std::uint32_t icm_rounds, std::uint32_t seed) {
  auto lists = OpenRoundArtifact<M>(basic, topology_model);
  if (!lists.ok()) return lists.status();
  return CpuReplayTopologyRoundContext<M>::Create(
      std::move(lists).value(), previous_checkpoint, topology_model, *replay_gemm_,
      *parallel_, rsd_rank, icm_rounds, seed, options_.least_squares_cost_mode);
}

template class CpuTrainingTopologyContextFactory<0>;
template class CpuTrainingTopologyContextFactory<5>;
template class CpuTrainingTopologyContextFactory<12>;
template class CpuTrainingTopologyContextFactory<6>;
template class CpuTrainingTopologyContextFactory<10>;

}  // namespace stlqpp
