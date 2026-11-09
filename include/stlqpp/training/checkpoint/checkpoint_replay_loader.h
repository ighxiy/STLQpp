#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/model/model_space_transform.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/training/checkpoint/training_checkpoint.h"
#include "stlqpp/training/linkage/replay/fixed_topology_replay_pipeline.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

struct CheckpointReplayLoaderStatistics final {
  double wall_seconds = 0.0;
  double forest_decode_seconds = 0.0;
  double real_source_seconds = 0.0;
  double assembly_seconds = 0.0;
  double rotation_seconds = 0.0;
  double append_seconds = 0.0;
};

// One source-owned real CID in original coordinates. The training source is
// identified by the checkpoint but remains external to it, avoiding a second
// copy of every real vector. Layer orders are semantic round inputs and are
// generated from stable IDs by the source/orchestrator.
struct TrainingReplayRealCluster final {

  std::uint32_t cid = 0U;
  std::uint32_t dimension = 0U;
  std::uint32_t real_count = 0U;
  std::vector<float> unrotated_exact_real;
  std::vector<std::uint8_t> basic_small_codes;
  std::vector<float> basic_coefficients;
  std::vector<std::uint8_t> edge_icm_layer_orders;

  [[nodiscard]] ColumnMajorMatrixF32View matrix() const noexcept {
    return {
        unrotated_exact_real.data(), dimension,
        real_count};
  }
};

class TrainingReplayRealClusterSource {
 public:
  virtual ~TrainingReplayRealClusterSource() = default;
  TrainingReplayRealClusterSource(const TrainingReplayRealClusterSource&) =
      delete;
  TrainingReplayRealClusterSource& operator=(
      const TrainingReplayRealClusterSource&) = delete;

  [[nodiscard]] virtual StatusOr<TrainingReplayRealCluster> Load(
      std::uint32_t cid,
      ArrayView<const std::uint64_t> stable_ids,
      std::uint32_t round_seed) = 0;

 protected:
  TrainingReplayRealClusterSource() = default;
};

// Width-independent replay window assembly. The typed wrapper supplies only
// checkpoint identity and the forest reader; no numerical work is dispatched
// through this type-erased boundary.
class CheckpointReplayWindowLoaderBase : public FixedTopologyReplayWindowLoader {
 public:
  [[nodiscard]] StatusOr<bool> LoadNext(
      FixedTopologyReplayWindow* window) override;

  [[nodiscard]] const CheckpointReplayLoaderStatistics& statistics()
      const noexcept {
    return statistics_;
  }

 protected:
  using LoadForest = StatusOr<TrainingForestCluster> (*)(
      const void*, std::uint32_t);

  CheckpointReplayWindowLoaderBase(
      ModelShape shape, const void* checkpoint, LoadForest load_forest,
      TrainingReplayRealClusterSource& real_source,
      const F32GemmBackend& gemm, ModelSpaceTransform transform,
      std::uint32_t round_seed) noexcept;

  [[nodiscard]] Status Rotate(ColumnMajorMatrixF32View source,
                              std::vector<float>* destination) const;

 private:
  ModelShape shape_;
  const void* checkpoint_ = nullptr;
  LoadForest load_forest_ = nullptr;
  TrainingReplayRealClusterSource* real_source_ = nullptr;
  const F32GemmBackend* gemm_ = nullptr;
  ModelSpaceTransform transform_;
  std::uint32_t round_seed_ = 0U;
  std::uint32_t next_cid_ = 0U;
  CheckpointReplayLoaderStatistics statistics_;
};

// Reconstructs fixed-topology replay from the previous in-memory round or a
// published restart checkpoint. Real rows are loaded by stable ID; virtual
// roots need only frozen codes and coefficients, not original vectors.
template <int M>
class CheckpointReplayWindowLoader final
    : public CheckpointReplayWindowLoaderBase {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<CheckpointReplayWindowLoader<M>> Create(
      const TrainingCheckpoint<M>& checkpoint,
      const TrainingModel<M>& topology_model,
      TrainingReplayRealClusterSource& real_source,
      const F32GemmBackend& gemm,
      std::uint32_t round_seed);

  CheckpointReplayWindowLoader(CheckpointReplayWindowLoader&&) noexcept =
      default;
  CheckpointReplayWindowLoader& operator=(
      CheckpointReplayWindowLoader&&) noexcept = default;
  CheckpointReplayWindowLoader(const CheckpointReplayWindowLoader&) = delete;
  CheckpointReplayWindowLoader& operator=(
      const CheckpointReplayWindowLoader&) = delete;

 private:
  CheckpointReplayWindowLoader(
      const TrainingCheckpoint<M>& checkpoint,
      TrainingReplayRealClusterSource& real_source,
      const F32GemmBackend& gemm,
      ModelSpaceTransform transform,
      std::uint32_t round_seed) noexcept;

  static StatusOr<TrainingForestCluster> LoadForestCluster(
      const void* checkpoint, std::uint32_t cid);
};

extern template class CheckpointReplayWindowLoader<5>;
extern template class CheckpointReplayWindowLoader<10>;

}  // namespace stlqpp
