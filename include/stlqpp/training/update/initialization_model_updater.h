#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/runtime/cpu/spd_solver_f32.h"
#include "stlqpp/training/basic/basic_encoding.h"
#include "stlqpp/training/linkage/initialization/root_to_edge_code_map.h"
#include "stlqpp/training/update/edge_codebook_moments.h"
#include "stlqpp/training/update/root_codebook_moments.h"
#include "stlqpp/training/update/root_codebook_solver.h"
#include "stlqpp/training/update/root_training_model.h"
#include "stlqpp/training/update/training_model.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace stlqpp {

template <int M>
class RootOnlyInitializationMomentReplay;

struct InitializationModelUpdateStatistics final {
  std::uint64_t basic_rows = 0U;
  std::uint64_t linked_rows = 0U;
  float root_diagonal_bump = 0.0F;
  float edge_diagonal_bump = 0.0F;
};

template <int M>
struct InitializationModelUpdate final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  TrainingModel<M> model;
  InitializationModelUpdateStatistics statistics;
};

// Two-barrier initializer. It updates C_root from the first basic encoding,
// then learns the first C_one from the C_root-only forest. R is immutable
// throughout this stage and no edge codebook exists before Finish().
template <int M>
class InitializationModelUpdater final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<InitializationModelUpdater<M>> Create(
      const RootTrainingModel<M>& initial,
      const SpdSolverF32& solver,
      const ParallelForExecutor& root_moment_parallel,
      std::uint32_t edge_moment_shard_count);
  [[nodiscard]] static StatusOr<InitializationModelUpdater<M>> Create(
      const RootTrainingModel<M>& initial,
      const SpdSolverF32& solver,
      const ParallelForExecutor& root_accumulation_parallel,
      const ParallelForExecutor& root_conversion_parallel,
      std::uint32_t edge_moment_shard_count);

  InitializationModelUpdater(InitializationModelUpdater&&) noexcept = default;
  InitializationModelUpdater& operator=(
      InitializationModelUpdater&&) noexcept = default;
  InitializationModelUpdater(const InitializationModelUpdater&) = delete;
  InitializationModelUpdater& operator=(
      const InitializationModelUpdater&) = delete;

  [[nodiscard]] Status AccumulateBasic(
      ColumnMajorMatrixF32View exact_model_space,
      const BasicEncoding<M>& encoding);
  [[nodiscard]] Status EnablePostRootMetric();
  [[nodiscard]] std::optional<double> post_root_squared_error_sum()
      const noexcept;

  [[nodiscard]] StatusOr<RootTrainingModel<M>> FinishRootUpdate();

  [[nodiscard]] Status AccumulateRootOnlyCluster(
      const LinkedTrainingClusterView<M>& cluster,
      const RootToEdgeCodeMap& root_to_edge,
      double sample_weight);

  [[nodiscard]] Status AccumulateFrozenEdgeCluster(
      const ProductCodebooks<M>& codebooks,
      const FrozenLinkedTrainingClusterView<M>& cluster);

  [[nodiscard]] StatusOr<InitializationModelUpdate<M>> Finish();

 private:
  friend class RootOnlyInitializationMomentReplay<M>;

  [[nodiscard]] Status AccumulateRootOnlyClusterAfterReplayValidation(
      const LinkedTrainingClusterView<M>& cluster,
      const RootToEdgeCodeMap& root_to_edge,
      double sample_weight);

  template <bool Prevalidated>
  [[nodiscard]] Status AccumulateRootOnlyClusterImpl(
      const LinkedTrainingClusterView<M>& cluster,
      const RootToEdgeCodeMap& root_to_edge,
      double sample_weight);

  enum class Phase : std::uint8_t {
    kBasic,
    kTopology,
    kComplete,
    kFailed,
  };

  InitializationModelUpdater(
      const RootTrainingModel<M>& initial,
      const SpdSolverF32& solver,
      const ParallelForExecutor& edge_moment_parallel,
      RootCodebookMoments<M> root_moments,
      EdgeCodebookMoments<M> edge_moments,
      std::vector<EdgeCodebookMoments<M>> edge_moment_shards) noexcept
      : initial_(&initial),
        solver_(&solver),
        edge_moment_parallel_(&edge_moment_parallel),
        root_moments_(std::move(root_moments)),
        edge_moments_(std::move(edge_moments)),
        edge_moment_shards_(std::move(edge_moment_shards)) {}

  [[nodiscard]] Status Fail(Status status) noexcept;

  const RootTrainingModel<M>* initial_ = nullptr;
  const SpdSolverF32* solver_ = nullptr;
  const ParallelForExecutor* edge_moment_parallel_ = nullptr;
  // Root moments are dead immediately after the first publication barrier.
  // Keeping them through root-only topology retains a large, unused
  // partitioned Gram/RHS workspace.
  std::optional<RootCodebookMoments<M>> root_moments_;
  EdgeCodebookMoments<M> edge_moments_;
  std::vector<EdgeCodebookMoments<M>> edge_moment_shards_;
  std::optional<RootCodebookSolveResult<M>> root_result_;
  std::uint64_t basic_rows_ = 0U;
  Phase phase_ = Phase::kBasic;
};

using RuntimeInitializationModelUpdater = InitializationModelUpdater<0>;
extern template class InitializationModelUpdater<0>;
extern template class InitializationModelUpdater<5>;
extern template class InitializationModelUpdater<10>;

}  // namespace stlqpp
