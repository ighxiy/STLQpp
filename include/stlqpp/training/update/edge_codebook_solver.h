#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/spd_solver_f32.h"
#include "stlqpp/training/update/edge_codebook_moments.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

template <int M>
class EdgeCodebookSolveResult final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] ColumnMajorMatrixF32View layer(
      std::uint32_t layer) const noexcept {
    return {
        values_.data() + static_cast<std::size_t>(layer) * 256U * dimension_,
        dimension_, 256U};
  }
  [[nodiscard]] std::uint32_t m() const noexcept { return m_; }
  [[nodiscard]] float diagonal_bump() const noexcept { return bump_; }

 private:
  template <int N>
  friend class EdgeCodebookSolver;

  std::uint32_t m_ = M;
  std::uint32_t dimension_ = 0U;
  float bump_ = 0.0F;
  std::vector<float> values_;
};

template <int M>
class EdgeCodebookSolver final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  explicit EdgeCodebookSolver(const SpdSolverF32& solver) noexcept
      : solver_(&solver) {}

  [[nodiscard]] StatusOr<EdgeCodebookSolveResult<M>> Solve(
      const EdgeCodebookMoments<M>& moments) const;

 private:
  const SpdSolverF32* solver_ = nullptr;
};

using RuntimeEdgeCodebookSolver = EdgeCodebookSolver<0>;
extern template class EdgeCodebookSolver<0>;
extern template class EdgeCodebookSolver<5>;
extern template class EdgeCodebookSolver<10>;

}  // namespace stlqpp
