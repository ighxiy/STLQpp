#pragma once

#include "stlqpp/core/artifact_hash.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/training/checkpoint/training_forest_codec.h"
#include "stlqpp/training/update/training_model.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace stlqpp {

struct TrainingCheckpointIdentity final {
  ArtifactHash64 training_source;
  ArtifactHash64 training_recipe;
};

template <int M>
class TrainingCheckpoint;

template <int M>
struct TrainingCheckpointWriteRequest final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  std::uint32_t completed_round = 0U;
  TrainingCheckpointIdentity identity;
  ModelShape shape;
  ArrayView<const std::uint8_t> bad_cluster_mask;
  std::uint64_t expected_real_rows = 0U;
  // Replay only: the previous validated checkpoint owns an immutable forest.
  // Empty for full topology, which must encode its newly chosen forest.
  const TrainingCheckpoint<M>* reuse_forest_from = nullptr;
};

template <int M>
class TrainingCheckpointWriter final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<TrainingCheckpointWriter<M>> Create(
      const TrainingCheckpointWriteRequest<M>& request);

  ~TrainingCheckpointWriter();
  TrainingCheckpointWriter(TrainingCheckpointWriter&&) noexcept;
  TrainingCheckpointWriter& operator=(TrainingCheckpointWriter&&) noexcept;
  TrainingCheckpointWriter(const TrainingCheckpointWriter&) = delete;
  TrainingCheckpointWriter& operator=(const TrainingCheckpointWriter&) =
      delete;

  [[nodiscard]] Status Append(
      const TrainingForestClusterSource& cluster,
      CanonicalBfsForestOrder* order = nullptr);
  [[nodiscard]] StatusOr<TrainingCheckpoint<M>> FinishMemory(
      TrainingModel<M> model);

 private:
  struct State;
  explicit TrainingCheckpointWriter(std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

struct TrainingCheckpointOpenRequest final {
  std::filesystem::path directory;
  TrainingCheckpointIdentity expected_identity;
};

template <int M>
class TrainingCheckpoint final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<TrainingCheckpoint<M>> Open(
      const TrainingCheckpointOpenRequest& request);

  ~TrainingCheckpoint();
  TrainingCheckpoint(TrainingCheckpoint&&) noexcept;
  TrainingCheckpoint& operator=(TrainingCheckpoint&&) noexcept;
  TrainingCheckpoint(const TrainingCheckpoint&) = delete;
  TrainingCheckpoint& operator=(const TrainingCheckpoint&) = delete;

  [[nodiscard]] std::uint32_t completed_round() const noexcept;
  [[nodiscard]] const std::filesystem::path& directory() const noexcept;
  [[nodiscard]] const TrainingCheckpointIdentity& identity() const noexcept;
  [[nodiscard]] ArrayView<const TrainingForestEntry> forest_entries()
      const noexcept;
  [[nodiscard]] std::uint64_t forest_bytes() const noexcept;
  [[nodiscard]] std::uint64_t directory_bytes() const noexcept;
  [[nodiscard]] bool has_forest() const noexcept;
  [[nodiscard]] Status Publish(const std::filesystem::path& destination)
      const;
  // Formal trained-model artifact: no replay forest or CID directory.
  [[nodiscard]] Status PublishModelOnly(
      const std::filesystem::path& destination) const;
  [[nodiscard]] const TrainingModel<M>& model() const noexcept;
  [[nodiscard]] ArrayView<const std::uint8_t> bad_cluster_mask()
      const noexcept;
  [[nodiscard]] StatusOr<TrainingForestCluster> LoadCluster(
      std::uint32_t cid) const;

 private:
  friend class TrainingCheckpointWriter<M>;
  struct State;
  explicit TrainingCheckpoint(std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

extern template class TrainingCheckpointWriter<5>;
extern template class TrainingCheckpointWriter<10>;
extern template class TrainingCheckpoint<5>;
extern template class TrainingCheckpoint<10>;

}  // namespace stlqpp
