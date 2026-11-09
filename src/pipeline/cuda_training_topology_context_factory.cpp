#include "stlqpp/pipeline/cuda_training_topology_context_factory.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "stlqpp/index/emit/base_list_reader.h"
#include "stlqpp/pipeline/base_list_full_topology_source.h"
#include "stlqpp/pipeline/stable_id_replay_source.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/training/basic/cuda_virtual_root_support_encoder.h"
#include "stlqpp/training/basic/virtual_root_support_encoder_pool.h"
#include "stlqpp/training/checkpoint/checkpoint_replay_loader.h"
#include "stlqpp/training/linkage/candidates/cpu_exact_knn.h"
#include "stlqpp/training/linkage/candidates/cpu_exact_parent_candidates.h"
#include "stlqpp/training/linkage/replay/cuda_fixed_edge_replay_encoder.h"
#include "stlqpp/training/linkage/replay/fixed_topology_replay_executor.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/topology/causal_cpu_linkage_evaluator.h"
#include "stlqpp/training/linkage/topology/full_topology_cluster_builder.h"

namespace stlqpp {
namespace {

template <int M>
Status ValidateArtifact(const BaseListReader<M>& lists,
                        const TrainingModel<M>& model,
                        const TrainingBasicRoundArtifact& basic) {
  const BaseBuildShape& list_shape = lists.store().shape();
  const ModelShape& model_shape = model.codebooks().shape();
  if (lists.store().directory() != basic.list_directory ||
      lists.store().total_vectors() != basic.rows ||
      list_shape.dimension != model_shape.dimension ||
      list_shape.nlist != model_shape.nlist ||
      lists.store().m() != model_shape.m ||
      (M != 0 && model_shape.m != static_cast<std::uint32_t>(M))) {
    return Status::FailedPrecondition(
        "round basic artifact does not match its topology model");
  }
  return Status::Ok();
}

template <int M>
class CudaFullTopologyRoundContext final : public FullTopologyRoundContext<M> {
 public:
  static StatusOr<std::unique_ptr<FullTopologyRoundContext<M>>> Create(
      BaseListReader<M> lists, const TrainingModel<M>& model,
      const F32GemmBackend& gemm,
      ArrayView<const std::uint8_t> bad_cluster_mask,
      std::uint32_t maximum_virtual_rows, std::uint32_t rsd_rank,
      std::uint32_t icm_rounds, bool collect_source_norm2,
      const CudaTrainingTopologyContextOptions& options) {
    auto edge_algebra =
        LinkageCodebookAlgebra<M>::Create(model.codebooks(), gemm);
    if (!edge_algebra.ok()) {
      return edge_algebra.status();
    }
    auto context = std::unique_ptr<CudaFullTopologyRoundContext<M>>(
        new CudaFullTopologyRoundContext<M>(
            std::move(lists), std::move(edge_algebra).value(), gemm));
    if (const Status status = context->Initialize(
            model, bad_cluster_mask, options.full_topology_worker_count,
            options.ordinary_direct_lane_count, maximum_virtual_rows, rsd_rank,
            icm_rounds, collect_source_norm2, options.virtual_root_schedule,
            options.linkage_scheduler);
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
    LinkageResourceStatistics output =
        ObserveCudaLinkageResources(resources_, evaluator_->Stats());
    output.ordinary_direct_lane_count = ordinary_direct_lane_count_;
    output.virtual_root_model_resident_bytes =
        virtual_root_model_resident_bytes_;
    output.virtual_root_workspace_resident_bytes =
        virtual_root_workspace_resident_bytes_;
    return output;
  }

 private:
  CudaFullTopologyRoundContext(BaseListReader<M> lists,
                               LinkageCodebookAlgebra<M> edge_algebra,
                               const F32GemmBackend& gemm) noexcept
      : lists_(std::move(lists)),
        edge_algebra_(std::move(edge_algebra)),
        good_candidates_(gemm),
        exact_knn_(gemm),
        gemm_(&gemm) {}

  Status Initialize(const TrainingModel<M>& model,
                    ArrayView<const std::uint8_t> bad_cluster_mask,
                    std::size_t outer_worker_count,
                    std::size_t ordinary_direct_lane_count,
                    std::uint32_t maximum_virtual_rows, std::uint32_t rsd_rank,
                    std::uint32_t icm_rounds, bool collect_source_norm2,
                    BasicSchedule virtual_root_schedule,
                    CudaLinkageSchedulerFactoryOptions linkage_options) {
    if (outer_worker_count == 0U || ordinary_direct_lane_count == 0U ||
        maximum_virtual_rows == 0U) {
      return Status::InvalidArgument(
          "full topology requires outer workers and ordinary direct lanes");
    }
    if (!linkage_options.scheduler_enabled &&
        ordinary_direct_lane_count != linkage_options.direct_worker_count) {
      return Status::InvalidArgument(
          "scheduler-off requires one shared root/evaluation direct lane "
          "count");
    }
    auto source = BaseListFullTopologySource<M>::Create(
        lists_, model, *gemm_, bad_cluster_mask, collect_source_norm2);
    if (!source.ok()) {
      return source.status();
    }
    source_ = std::make_unique<BaseListFullTopologySource<M>>(
        std::move(source).value());

    auto cpu_evaluator = CausalCpuLinkageEvaluator<M>::Create(
        model.codebooks(), edge_algebra_, rsd_rank,
        linkage_options.least_squares_cost_mode, icm_rounds);
    if (!cpu_evaluator.ok()) {
      return cpu_evaluator.status();
    }
    cpu_evaluator_ = std::make_unique<CausalCpuLinkageEvaluator<M>>(
        std::move(cpu_evaluator).value());

    std::vector<std::unique_ptr<VirtualRootSupportEncoder<M>>> root_encoders;
    root_encoders.reserve(ordinary_direct_lane_count);
    root_cuda_contexts_.reserve(ordinary_direct_lane_count);
    for (std::size_t worker = 0U; worker < ordinary_direct_lane_count;
         ++worker) {
      auto cuda = CudaContext::Create(linkage_options.device,
                                      linkage_options.allow_tf32);
      if (!cuda.ok()) {
        return cuda.status();
      }
      root_cuda_contexts_.push_back(
          std::make_shared<CudaContext>(std::move(cuda).value()));
      if (root_device_model_ == nullptr) {
        auto uploaded_root_model = CudaFixedRootDeviceModel<M>::Create(
            model.codebooks().root_codebooks(), *gemm_,
            *root_cuda_contexts_.back());
        if (!uploaded_root_model.ok()) {
          return uploaded_root_model.status();
        }
        root_device_model_ = std::move(uploaded_root_model).value();
        virtual_root_model_resident_bytes_ =
            root_device_model_->logical_bytes();
      }
      auto encoder = CudaVirtualRootSupportEncoder<M>::Create(
          root_device_model_, *root_cuda_contexts_.back(),
          virtual_root_schedule);
      if (!encoder.ok()) {
        return encoder.status();
      }
      if (const Status status =
              encoder.value().ReserveWorkspace(maximum_virtual_rows);
          !status.ok()) {
        return status;
      }
      virtual_root_workspace_resident_bytes_ +=
          encoder.value().workspace_resident_bytes();
      root_encoders.push_back(
          std::make_unique<CudaVirtualRootSupportEncoder<M>>(
              std::move(encoder).value()));
    }
    ordinary_direct_lane_count_ = ordinary_direct_lane_count;
    linkage_options.rsd_rank = rsd_rank;
    linkage_options.icm_rounds = icm_rounds;

    StatusOr<CudaLinkageSchedulerFactoryResult> evaluation =
        linkage_options.scheduler_enabled
            ? CreateCudaLinkageScheduler<M>(edge_algebra_, linkage_options)
            : CreateCudaDirectLinkageEvaluator<M>(
                  edge_algebra_, root_cuda_contexts_, linkage_options);
    if (!evaluation.ok()) {
      return evaluation.status();
    }
    evaluator_ = std::move(evaluation.value().evaluator);
    resources_ = evaluation.value().resources;
    auto root_pool =
        linkage_options.scheduler_enabled
            ? VirtualRootSupportEncoderPool<M>::Create(std::move(root_encoders))
            : VirtualRootSupportEncoderPool<M>::Create(
                  std::move(root_encoders),
                  std::move(evaluation.value().shared_direct_lanes));
    if (!root_pool.ok()) {
      return root_pool.status();
    }
    root_encoder_pool_ = std::move(root_pool).value();

    builders_.reserve(outer_worker_count);
    executor_views_.reserve(outer_worker_count);
    for (std::size_t worker = 0U; worker < outer_worker_count; ++worker) {
      builders_.push_back(std::make_unique<FullTopologyClusterBuilder<M>>(
          model.codebooks(), good_candidates_, exact_knn_, *root_encoder_pool_,
          *evaluator_, cpu_evaluator_.get()));
      executor_views_.push_back(builders_.back().get());
    }
    return Status::Ok();
  }

  BaseListReader<M> lists_;
  LinkageCodebookAlgebra<M> edge_algebra_;
  CpuExactParentCandidateProvider good_candidates_;
  CpuExactKnnProvider exact_knn_;
  std::unique_ptr<LinkageEvaluationService> evaluator_;
  CudaLinkageSchedulerResourcePlan resources_;
  const F32GemmBackend* gemm_ = nullptr;
  std::unique_ptr<BaseListFullTopologySource<M>> source_;
  std::size_t ordinary_direct_lane_count_ = 0U;
  std::uint64_t virtual_root_model_resident_bytes_ = 0U;
  std::uint64_t virtual_root_workspace_resident_bytes_ = 0U;
  std::shared_ptr<const CudaFixedRootDeviceModel<M>> root_device_model_;
  std::vector<std::shared_ptr<CudaContext>> root_cuda_contexts_;
  std::unique_ptr<VirtualRootSupportEncoderPool<M>> root_encoder_pool_;
  std::unique_ptr<CausalCpuLinkageEvaluator<M>> cpu_evaluator_;
  std::vector<std::unique_ptr<FullTopologyClusterBuilder<M>>> builders_;
  std::vector<FullTopologyClusterExecutor<M>*> executor_views_;
};

template <int M>
class CudaReplayTopologyRoundContext final
    : public ReplayTopologyRoundContext<M> {
 public:
  static StatusOr<std::unique_ptr<ReplayTopologyRoundContext<M>>> Create(
      BaseListReader<M> lists, const TrainingCheckpoint<M>& checkpoint,
      const TrainingModel<M>& model, const F32GemmBackend& gemm,
      std::uint32_t seed, std::uint32_t rsd_rank, std::uint32_t icm_rounds,
      int device, bool allow_tf32, LeastSquaresCostMode cost_mode) {
    auto edge_algebra =
        LinkageCodebookAlgebra<M>::Create(model.codebooks(), gemm);
    if (!edge_algebra.ok()) {
      return edge_algebra.status();
    }
    auto context = std::unique_ptr<CudaReplayTopologyRoundContext<M>>(
        new CudaReplayTopologyRoundContext<M>(std::move(lists)));
    if (const Status status =
            context->Initialize(checkpoint, model, edge_algebra.value(), gemm,
                                seed, rsd_rank, icm_rounds, device, allow_tf32,
                                cost_mode);
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
  explicit CudaReplayTopologyRoundContext(BaseListReader<M> lists) noexcept
      : lists_(std::move(lists)) {}

  Status Initialize(const TrainingCheckpoint<M>& checkpoint,
                    const TrainingModel<M>& model,
                    const LinkageCodebookAlgebra<M>& edge_algebra,
                    const F32GemmBackend& gemm,
                    std::uint32_t seed, std::uint32_t rsd_rank,
                    std::uint32_t icm_rounds, int device, bool allow_tf32,
                    LeastSquaresCostMode cost_mode) {
    real_source_ = std::make_unique<StableIdReplaySource<M>>(lists_);
    auto loader = CheckpointReplayWindowLoader<M>::Create(
        checkpoint, model, *real_source_, gemm, seed);
    if (!loader.ok()) {
      return loader.status();
    }
    loader_ = std::make_unique<CheckpointReplayWindowLoader<M>>(
        std::move(loader).value());

    auto cuda = CudaContext::Create(device, allow_tf32);
    if (!cuda.ok()) {
      return cuda.status();
    }
    cuda_ = std::make_unique<CudaContext>(std::move(cuda).value());
    auto edges = CudaFixedEdgeReplayEncoder<M>::Create(
        edge_algebra, *cuda_, rsd_rank, cost_mode, icm_rounds);
    if (!edges.ok()) {
      return edges.status();
    }
    edge_encoder_ = std::make_unique<CudaFixedEdgeReplayEncoder<M>>(
        std::move(edges).value());
    executor_ = std::make_unique<FixedTopologyReplayExecutor<M>>(
        model.codebooks(), *edge_encoder_);
    return Status::Ok();
  }

  BaseListReader<M> lists_;
  std::unique_ptr<StableIdReplaySource<M>> real_source_;
  std::unique_ptr<CheckpointReplayWindowLoader<M>> loader_;
  std::unique_ptr<CudaContext> cuda_;
  std::unique_ptr<CudaFixedEdgeReplayEncoder<M>> edge_encoder_;
  std::unique_ptr<FixedTopologyReplayExecutor<M>> executor_;
};

template <int M>
StatusOr<BaseListReader<M>> OpenRoundArtifact(
    const TrainingBasicRoundArtifact& basic, const TrainingModel<M>& model) {
  auto lists = BaseListReader<M>::Open(basic.list_directory);
  if (!lists.ok()) {
    return lists.status();
  }
  if (const Status status = ValidateArtifact(lists.value(), model, basic);
      !status.ok()) {
    return status;
  }
  return std::move(lists).value();
}

}  // namespace

template <int M>
StatusOr<std::unique_ptr<FullTopologyRoundContext<M>>>
CudaTrainingTopologyContextFactory<M>::CreateFull(
    const TrainingModel<M>& topology_model,
    ArrayView<const std::uint8_t> bad_cluster_mask,
    const TrainingBasicRoundArtifact& basic, std::uint32_t,
    const FullTopologyPolicy& policy) {
  auto lists = OpenRoundArtifact<M>(basic, topology_model);
  if (!lists.ok()) {
    return lists.status();
  }
  return CudaFullTopologyRoundContext<M>::Create(
      std::move(lists).value(), topology_model, *gemm_,
      bad_cluster_mask, policy.maximum_virtual_roots, policy.rsd_rank,
      policy.icm_rounds, policy.collect_source_norm2, options_);
}

template <int M>
StatusOr<std::unique_ptr<ReplayTopologyRoundContext<M>>>
CudaTrainingTopologyContextFactory<M>::CreateReplay(
    const TrainingCheckpoint<M>& previous_checkpoint,
    const TrainingModel<M>& topology_model,
    const TrainingBasicRoundArtifact& basic, std::uint32_t,
    std::uint32_t rsd_rank, std::uint32_t icm_rounds, std::uint32_t seed) {
  auto lists = OpenRoundArtifact<M>(basic, topology_model);
  if (!lists.ok()) {
    return lists.status();
  }
  return CudaReplayTopologyRoundContext<M>::Create(
      std::move(lists).value(), previous_checkpoint, topology_model, *gemm_,
      seed, rsd_rank, icm_rounds, options_.linkage_scheduler.device,
      options_.linkage_scheduler.allow_tf32, options_.least_squares_cost_mode);
}

template class CudaTrainingTopologyContextFactory<0>;
template class CudaTrainingTopologyContextFactory<5>;
template class CudaTrainingTopologyContextFactory<12>;
template class CudaTrainingTopologyContextFactory<6>;
template class CudaTrainingTopologyContextFactory<10>;

}  // namespace stlqpp
