#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/training/checkpoint/training_checkpoint.h"
#include "stlqpp/training/update/training_model.h"

#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

namespace stlqpp {

template <int M>
class InitializedTrainingCheckpoint final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static Status Write(
      const std::filesystem::path& destination,
      const TrainingCheckpointIdentity& identity,
      const TrainingModel<M>& model,
      ArrayView<const std::uint8_t> bad_cluster_mask);

  [[nodiscard]] static StatusOr<InitializedTrainingCheckpoint<M>> Open(
      const std::filesystem::path& directory,
      const TrainingCheckpointIdentity& expected_identity);

  InitializedTrainingCheckpoint(InitializedTrainingCheckpoint&&) noexcept =
      default;
  InitializedTrainingCheckpoint& operator=(
      InitializedTrainingCheckpoint&&) noexcept = default;
  InitializedTrainingCheckpoint(const InitializedTrainingCheckpoint&) =
      delete;
  InitializedTrainingCheckpoint& operator=(
      const InitializedTrainingCheckpoint&) = delete;

  [[nodiscard]] const TrainingModel<M>& model() const noexcept {
    return model_;
  }
  [[nodiscard]] ArrayView<const std::uint8_t> bad_cluster_mask()
      const noexcept {
    return bad_cluster_mask_;
  }

 private:
  InitializedTrainingCheckpoint(
      TrainingModel<M> model,
      std::vector<std::uint8_t> bad_cluster_mask) noexcept
      : model_(std::move(model)),
        bad_cluster_mask_(std::move(bad_cluster_mask)) {}

  TrainingModel<M> model_;
  std::vector<std::uint8_t> bad_cluster_mask_;
};

extern template class InitializedTrainingCheckpoint<5>;
extern template class InitializedTrainingCheckpoint<10>;

}  // namespace stlqpp
