#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/basic/basic_encoding.h"

namespace stlqpp {

// Streaming sufficient statistics for the exact root/small Schur solve.
// Large-root cross moments intentionally use FP32, matching the registered
// bounded-memory numerical path; primary Gram/RHS moments accumulate in FP64.
template <int M>
class RootCodebookMoments final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  static constexpr std::uint32_t kSmallAtoms = (M == 0 ? 0U : M - 1U) * 256U;
  static constexpr std::uint32_t kMaskWordsPerLayer = 4U;

  [[nodiscard]] static StatusOr<RootCodebookMoments<M>> Create(
      std::uint32_t dimension, std::uint32_t nlist,
      std::uint32_t partition_count, std::uint32_t m = FixedModelLayers<M>());
  [[nodiscard]] static StatusOr<RootCodebookMoments<M>> Create(
      std::uint32_t dimension, std::uint32_t nlist,
      const ParallelForExecutor& partition_parallel,
      std::uint32_t m = FixedModelLayers<M>());
  [[nodiscard]] static StatusOr<RootCodebookMoments<M>> Create(
      std::uint32_t dimension, std::uint32_t nlist,
      const ParallelForExecutor& partition_parallel,
      const ParallelForExecutor& conversion_parallel,
      std::uint32_t m = FixedModelLayers<M>());

  [[nodiscard]] Status Accumulate(ColumnMajorMatrixF32View exact,
                                  const BasicEncoding<M>& encoding);

  // Collect the model-space source norm needed by the post-root metric.
  // Must be selected before the first basic block is accumulated.
  [[nodiscard]] Status EnablePostRootMetric();
  [[nodiscard]] bool collects_post_root_metric() const noexcept {
    return collect_post_root_metric_;
  }
  [[nodiscard]] double source_squared_norm_sum() const noexcept;

  // Performs the ordered partition reduction and mandatory FP32 boundary,
  // then releases phase-A FP64 storage before Schur correction storage is
  // allocated. Accumulation is closed after this transition.
  [[nodiscard]] Status PrepareForSolve();

  [[nodiscard]] std::uint32_t m() const noexcept {
    if constexpr (M != 0) {
      return M;
    } else {
      return m_;
    }
  }
  [[nodiscard]] std::uint32_t small_atoms() const noexcept {
    return (m() - 1U) * 256U;
  }
  [[nodiscard]] std::uint32_t dimension() const noexcept { return dimension_; }
  [[nodiscard]] std::uint32_t nlist() const noexcept { return nlist_; }
  [[nodiscard]] std::uint32_t partition_count() const noexcept {
    return partition_count_;
  }
  [[nodiscard]] std::uint64_t sample_count() const noexcept {
    return sample_count_;
  }
  [[nodiscard]] ArrayView<const double> root_diagonal() const noexcept {
    return root_diagonal_;
  }
  [[nodiscard]] ArrayView<const double> root_atom_major_rhs() const noexcept {
    return root_atom_major_rhs_;
  }
  [[nodiscard]] ArrayView<const float> root_small_cross() const noexcept {
    return root_small_cross_;
  }
  [[nodiscard]] ArrayView<const std::uint64_t> cross_masks() const noexcept {
    return cross_masks_;
  }
  [[nodiscard]] ArrayView<const double> packed_small_gram(
      std::uint32_t partition) const noexcept {
    if (partition >= partition_count_ || packed_small_gram_.empty()) {
      return {};
    }
    const std::size_t count = PackedSmallGramCount();
    return {
        packed_small_gram_.data() + static_cast<std::size_t>(partition) * count,
        count};
  }
  [[nodiscard]] ArrayView<const double> small_atom_major_rhs(
      std::uint32_t partition) const noexcept {
    if (partition >= partition_count_ || small_atom_major_rhs_.empty()) {
      return {};
    }
    const std::size_t count = SmallRhsCount();
    return {small_atom_major_rhs_.data() +
                static_cast<std::size_t>(partition) * count,
            count};
  }
  [[nodiscard]] ArrayView<const float> reduced_small_gram() const noexcept {
    return reduced_small_gram_;
  }
  [[nodiscard]] ArrayView<const float> reduced_small_atom_major_rhs()
      const noexcept {
    return reduced_small_atom_major_rhs_;
  }

  [[nodiscard]] static constexpr std::size_t PackedUpperIndex(
      std::uint32_t row, std::uint32_t column) noexcept {
    return static_cast<std::size_t>(column) * (column + 1U) / 2U + row;
  }

  void RunPartitions(void* context, ParallelForTask task) const noexcept;
  [[nodiscard]] std::uint32_t output_worker_count() const noexcept;
  void RunOutputTasks(std::uint32_t task_count, void* context,
                      ParallelForTask task) const noexcept;

 private:
  struct AccumulationWork;

  template <std::uint32_t Capacity>
  [[nodiscard]] Status AccumulateWithCapacity(ColumnMajorMatrixF32View exact,
                                              const BasicEncoding<M>& encoding);

  RootCodebookMoments(std::uint32_t dimension, std::uint32_t nlist,
                      std::uint32_t partition_count,
                      const ParallelForExecutor* partition_parallel,
                      const ParallelForExecutor* conversion_parallel,
                      std::uint32_t m);

  template <std::uint32_t Capacity>
  static void AccumulatePartition(void* opaque, std::uint32_t worker,
                                  std::uint32_t partition) noexcept;
  template <std::uint32_t Capacity>
  void AccumulateSample(std::uint32_t sample, ColumnMajorMatrixF32View exact,
                        const BasicEncoding<M>& encoding,
                        std::uint32_t partition,
                        double* source_squared_norm) noexcept;

  [[nodiscard]] std::size_t PackedSmallGramCount() const noexcept {
    return static_cast<std::size_t>(small_atoms()) * (small_atoms() + 1U) / 2U;
  }
  [[nodiscard]] std::size_t SmallRhsCount() const noexcept {
    return static_cast<std::size_t>(small_atoms()) * dimension_;
  }

  std::uint32_t m_ = 0U;
  std::uint32_t dimension_ = 0U;
  std::uint32_t nlist_ = 0U;
  std::uint32_t partition_count_ = 0U;
  std::uint32_t cids_per_partition_ = 0U;
  std::uint64_t sample_count_ = 0U;
  bool collect_post_root_metric_ = false;
  std::vector<double> source_squared_norm_partitions_;
  std::vector<double> root_diagonal_;
  std::vector<double> root_atom_major_rhs_;
  std::vector<float> root_small_cross_;
  std::vector<std::uint64_t> cross_masks_;
  std::vector<double> packed_small_gram_;
  std::vector<double> small_atom_major_rhs_;
  std::vector<float> reduced_small_gram_;
  std::vector<float> reduced_small_atom_major_rhs_;
  const ParallelForExecutor* partition_parallel_ = nullptr;
  const ParallelForExecutor* conversion_parallel_ = nullptr;
  bool prepared_for_solve_ = false;
};

using RuntimeRootCodebookMoments = RootCodebookMoments<0>;
extern template class RootCodebookMoments<0>;
extern template class RootCodebookMoments<5>;
extern template class RootCodebookMoments<10>;

}  // namespace stlqpp
