#include "stlqpp/training/checkpoint/checkpoint_replay_loader.h"

#include "stlqpp/training/linkage/rsd/icm_layer_orders.h"

#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

constexpr std::uint64_t kTargetReplayWindowRealRows = 32768U;

Status ValidateRealCluster(const TrainingReplayRealCluster& cluster,
                           std::uint32_t expected_cid,
                           std::uint32_t dimension,
                           std::uint32_t expected_real_count, std::uint32_t m) {
  const std::uint64_t value_count =
      static_cast<std::uint64_t>(dimension) * expected_real_count;
  if (cluster.cid != expected_cid || cluster.dimension != dimension ||
      cluster.real_count != expected_real_count ||
      value_count > std::numeric_limits<std::size_t>::max() ||
      cluster.unrotated_exact_real.size() != value_count ||
      cluster.basic_small_codes.size() !=
          static_cast<std::size_t>(expected_real_count) * (m - 1U) ||
      cluster.basic_coefficients.size() !=
          static_cast<std::size_t>(expected_real_count) * m) {
    return Status::FailedPrecondition(
        "training source CID disagrees with the checkpoint forest");
  }
  for (const float value : cluster.unrotated_exact_real) {
    if (!std::isfinite(value)) {
      return Status::DataLoss(
          "training source CID contains a non-finite coordinate");
    }
  }
  for (const float value : cluster.basic_coefficients) {
    if (!std::isfinite(value)) {
      return Status::DataLoss(
          "training basic artifact contains a non-finite coefficient");
    }
  }
  return ValidateIcmLayerOrders<0>(
      expected_real_count, cluster.edge_icm_layer_orders, m);
}

bool SameShape(const ModelShape& left, const ModelShape& right) noexcept {
  return left.dimension == right.dimension && left.m == right.m &&
      left.nlist == right.nlist &&
      left.residual_codebook_size == right.residual_codebook_size;
}

}  // namespace

CheckpointReplayWindowLoaderBase::CheckpointReplayWindowLoaderBase(
    ModelShape shape, const void* checkpoint, LoadForest load_forest,
    TrainingReplayRealClusterSource& real_source,
    const F32GemmBackend& gemm,
    ModelSpaceTransform transform,
    std::uint32_t round_seed) noexcept
    : shape_(shape),
      checkpoint_(checkpoint),
      load_forest_(load_forest),
      real_source_(&real_source),
      gemm_(&gemm),
      transform_(std::move(transform)),
      round_seed_(round_seed) {}

template <int M>
CheckpointReplayWindowLoader<M>::CheckpointReplayWindowLoader(
    const TrainingCheckpoint<M>& checkpoint,
    TrainingReplayRealClusterSource& real_source,
    const F32GemmBackend& gemm,
    ModelSpaceTransform transform,
    std::uint32_t round_seed) noexcept
    : CheckpointReplayWindowLoaderBase(
          checkpoint.model().codebooks().shape(), &checkpoint,
          &CheckpointReplayWindowLoader<M>::LoadForestCluster,
          real_source, gemm, std::move(transform), round_seed) {}

template <int M>
StatusOr<TrainingForestCluster>
CheckpointReplayWindowLoader<M>::LoadForestCluster(
    const void* checkpoint, std::uint32_t cid) {
  return static_cast<const TrainingCheckpoint<M>*>(checkpoint)->LoadCluster(cid);
}

template <int M>
StatusOr<CheckpointReplayWindowLoader<M>>
CheckpointReplayWindowLoader<M>::Create(
    const TrainingCheckpoint<M>& checkpoint,
    const TrainingModel<M>& topology_model,
    TrainingReplayRealClusterSource& real_source,
    const F32GemmBackend& gemm,
    std::uint32_t round_seed) {
  const ModelShape& checkpoint_shape = checkpoint.model().codebooks().shape();
  const ModelShape& topology_shape = topology_model.codebooks().shape();
  if (!SameShape(checkpoint_shape, topology_shape)) {
    return Status::FailedPrecondition(
        "replay topology model shape differs from the checkpoint");
  }
  auto transform = ModelSpaceTransform::Create(topology_model.rotation());
  if (!transform.ok()) {
    return Status::FailedPrecondition(transform.status().message());
  }
  return CheckpointReplayWindowLoader<M>(
      checkpoint, real_source, gemm, std::move(transform).value(), round_seed);
}

Status CheckpointReplayWindowLoaderBase::Rotate(
    ColumnMajorMatrixF32View source,
    std::vector<float>* destination) const {
  const std::uint32_t dimension = shape_.dimension;
  if (source.rows != dimension ||
      (source.columns != 0U && source.values == nullptr)) {
    return Status::InvalidArgument(
        "checkpoint replay source matrix has an invalid shape");
  }
  destination->resize(
      static_cast<std::size_t>(dimension) * source.columns);
  if (source.columns == 0U) {
    return Status::Ok();
  }
  return gemm_->MultiplyLeft({
      .left = transform_.matrix(),
      .right = source.values,
      .query_count = source.columns,
      .right_leading_dimension = dimension,
      .output = destination->data(),
      .output_leading_dimension = dimension,
  });
}

StatusOr<bool> CheckpointReplayWindowLoaderBase::LoadNext(
    FixedTopologyReplayWindow* window) {
  using Clock = std::chrono::steady_clock;
  const Clock::time_point wall_begin = Clock::now();
  if (window == nullptr || window->cluster_count() != 0U) {
    return Status::InvalidArgument(
        "checkpoint replay requires one empty destination window");
  }
  const ModelShape& shape = shape_;
  if (next_cid_ == shape.nlist) {
    return false;
  }

  struct PendingCluster final {
    TrainingForestCluster forest;
    std::vector<std::uint8_t> basic_small_codes;
    std::vector<float> basic_coefficients;
    std::vector<std::uint8_t> edge_icm_layer_orders;
    std::uint32_t real_begin = 0U;
  };
  std::vector<PendingCluster> pending;
  pending.reserve(kMaximumReplayWindowClusters);
  std::vector<float>& unrotated_real = window->unrotated_exact_real_;
  std::vector<float>& rotated_real = window->exact_real_;
  if (shape.dimension > std::numeric_limits<std::size_t>::max() /
                            kTargetReplayWindowRealRows) {
    return Status::ResourceExhausted(
        "checkpoint replay real-window allocation exceeds size_t");
  }
  const Clock::time_point reserve_begin = Clock::now();
  const std::size_t target_real_values =
      static_cast<std::size_t>(kTargetReplayWindowRealRows) *
      shape.dimension;
  unrotated_real.reserve(target_real_values);
  rotated_real.reserve(target_real_values);
  statistics_.assembly_seconds += std::chrono::duration<double>(
      Clock::now() - reserve_begin).count();
  std::uint64_t window_real_rows = 0U;
  std::uint64_t window_virtual_rows = 0U;
  while (next_cid_ < shape.nlist &&
         pending.size() <
             kMaximumReplayWindowClusters) {
    Clock::time_point phase_begin = Clock::now();
    auto forest = load_forest_(checkpoint_, next_cid_);
    statistics_.forest_decode_seconds += std::chrono::duration<double>(
        Clock::now() - phase_begin).count();
    if (!forest.ok()) {
      return forest.status();
    }
    phase_begin = Clock::now();
    auto real = real_source_->Load(
        next_cid_, forest.value().real_sample_ids, round_seed_);
    statistics_.real_source_seconds += std::chrono::duration<double>(
        Clock::now() - phase_begin).count();
    if (!real.ok()) {
      return real.status();
    }
    if (const Status status = ValidateRealCluster(
            real.value(), next_cid_, shape.dimension,
            forest.value().real_count(), shape.m);
        !status.ok()) {
      return status;
    }

    phase_begin = Clock::now();
    const std::uint32_t real_count = forest.value().real_count();
    const std::uint32_t virtual_count = forest.value().virtual_count();
    if (window_real_rows > std::numeric_limits<std::uint32_t>::max() ||
        window_virtual_rows > std::numeric_limits<std::uint32_t>::max()) {
      return Status::ResourceExhausted(
          "checkpoint replay window row domain exceeds uint32");
    }
    const ColumnMajorMatrixF32View real_matrix = real.value().matrix();
    if (real_count != 0U) {
      unrotated_real.insert(
          unrotated_real.end(), real_matrix.values,
          real_matrix.values + static_cast<std::size_t>(shape.dimension) *
                                   real_count);
    }
    // Only these payloads are needed after window assembly. Let this CID's
    // source matrix die now; its coordinates already belong to unrotated_real.
    pending.push_back({
        .forest = std::move(forest).value(),
        .basic_small_codes = std::move(real.value().basic_small_codes),
        .basic_coefficients = std::move(real.value().basic_coefficients),
        .edge_icm_layer_orders = std::move(real.value().edge_icm_layer_orders),
        .real_begin = static_cast<std::uint32_t>(window_real_rows),
    });
    window_real_rows += real_count;
    window_virtual_rows += virtual_count;
    statistics_.assembly_seconds += std::chrono::duration<double>(
        Clock::now() - phase_begin).count();
    ++next_cid_;
    if (window_real_rows >= kTargetReplayWindowRealRows) {
      break;
    }
  }

  if (window_real_rows > std::numeric_limits<std::uint32_t>::max() ||
      window_virtual_rows > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "checkpoint replay window row domain exceeds uint32");
  }
  const std::uint32_t real_rows =
      static_cast<std::uint32_t>(window_real_rows);
  const Clock::time_point rotation_begin = Clock::now();
  if (const Status status = Rotate(
          {unrotated_real.data(), shape.dimension, real_rows},
          &rotated_real);
      !status.ok()) {
    return status;
  }
  statistics_.rotation_seconds += std::chrono::duration<double>(
      Clock::now() - rotation_begin).count();
  const Clock::time_point append_begin = Clock::now();
  for (const PendingCluster& cluster : pending) {
    const std::uint32_t real_count = cluster.forest.real_count();
    const float* exact_real = real_count == 0U
        ? nullptr
        : rotated_real.data() +
            static_cast<std::size_t>(cluster.real_begin) * shape.dimension;
    const float* source_real = real_count == 0U
        ? nullptr
        : unrotated_real.data() +
            static_cast<std::size_t>(cluster.real_begin) * shape.dimension;
    const FixedTopologyReplayRequest request = {
        .m = shape.m,
        .cid = cluster.forest.cid,
        .exact_real = {exact_real, shape.dimension, real_count},
        .unrotated_exact_real = {
            source_real, shape.dimension, real_count},
        .real_sample_ids = cluster.forest.real_sample_ids,
        .virtual_sample_ids = cluster.forest.virtual_sample_ids,
        .basic_small_codes = cluster.basic_small_codes,
        .basic_coefficients = cluster.basic_coefficients,
        .frozen_virtual_small_codes =
            cluster.forest.virtual_small_codes,
        .frozen_virtual_coefficients =
            cluster.forest.virtual_coefficients,
        .frozen_roles = cluster.forest.real_roles,
        .frozen_parents = cluster.forest.real_parents,
        .frozen_depths = cluster.forest.real_depths,
        .edge_icm_layer_orders = cluster.edge_icm_layer_orders,
        .seed = round_seed_,
    };
    // The four matrices already reside in this reusable window.  Append only
    // the cluster slices and non-matrix payloads; copying the same rotated and
    // source matrices here would double the loader's memory traffic.
    if (const Status status = window->AppendImpl(request, false);
        !status.ok()) {
      return status;
    }
  }
  window->validated_checkpoint_payload_ = true;
  statistics_.append_seconds += std::chrono::duration<double>(
      Clock::now() - append_begin).count();
  statistics_.wall_seconds += std::chrono::duration<double>(
      Clock::now() - wall_begin).count();
  return true;
}

template class CheckpointReplayWindowLoader<0>;
template class CheckpointReplayWindowLoader<5>;
template class CheckpointReplayWindowLoader<12>;
template class CheckpointReplayWindowLoader<6>;
template class CheckpointReplayWindowLoader<10>;

}  // namespace stlqpp
