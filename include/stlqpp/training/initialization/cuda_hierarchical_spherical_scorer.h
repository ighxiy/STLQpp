#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/runtime/cuda/device_matrix.h"
#include "stlqpp/training/initialization/spherical_hierarchical_layout.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace stlqpp {

struct CudaHierarchicalSphericalScoreView final {
  const std::uint32_t* codes = nullptr;
  const float* best_dots = nullptr;
  std::uint32_t count = 0U;
};

// CUDA owner of DEV's registered contiguous coarse-to-fine large-K scorer.
// It does not own residual construction or annealing; those remain shared with
// the flat scorer. Fine IDs are always the original flat center IDs.
class CudaHierarchicalSphericalScorer final {
 public:
  [[nodiscard]] static StatusOr<CudaHierarchicalSphericalScorer> Create(
      CudaContext& context,
      const ParallelForExecutor& parallel,
      std::uint32_t dimension,
      std::uint32_t center_count,
      std::uint32_t top_coarse,
      std::size_t maximum_score_bytes,
      SphericalHierarchicalSplit split = {});

  ~CudaHierarchicalSphericalScorer();
  CudaHierarchicalSphericalScorer(
      CudaHierarchicalSphericalScorer&&) noexcept;
  CudaHierarchicalSphericalScorer& operator=(
      CudaHierarchicalSphericalScorer&&) noexcept;
  CudaHierarchicalSphericalScorer(
      const CudaHierarchicalSphericalScorer&) = delete;
  CudaHierarchicalSphericalScorer& operator=(
      const CudaHierarchicalSphericalScorer&) = delete;

  [[nodiscard]] Status SetFineCenters(ColumnMajorMatrixF32View fine);
  [[nodiscard]] StatusOr<CudaHierarchicalSphericalScoreView> Score(
      CudaMatrixF32View residuals);

  [[nodiscard]] const SphericalHierarchicalLayout& layout() const noexcept;

 private:
  struct State;
  explicit CudaHierarchicalSphericalScorer(
      std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

}  // namespace stlqpp
