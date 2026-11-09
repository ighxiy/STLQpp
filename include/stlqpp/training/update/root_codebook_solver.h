#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/spd_solver_f32.h"
#include "stlqpp/training/update/root_codebook_moments.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace stlqpp {

template <int M>
class RoundModelUpdater;
template <int M>
class InitializationModelUpdater;

template <int M>
class RootCodebookSolveResult final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] ColumnMajorMatrixF32View root0() const noexcept {
    return {root0_.data(), dimension_, nlist_};
  }
  [[nodiscard]] ColumnMajorMatrixF32View small_layer(
      std::uint32_t layer) const noexcept {
    return {
        small_.data() +
            static_cast<std::size_t>(layer - 1U) * 256U * dimension_,
        dimension_, 256U};
  }
  [[nodiscard]] std::uint32_t m() const noexcept { return m_; }
  [[nodiscard]] float diagonal_bump() const noexcept { return bump_; }
  [[nodiscard]] std::optional<double> post_root_squared_error_sum()
      const noexcept {
    return post_root_squared_error_sum_;
  }

 private:
  template <int N>
  friend class RootCodebookSolver;

  std::uint32_t m_ = M;
  std::uint32_t dimension_ = 0U;
  std::uint32_t nlist_ = 0U;
  float bump_ = 0.0F;
  std::optional<double> post_root_squared_error_sum_;
  std::vector<float> root0_;
  std::vector<float> small_;
};

template <int M>
class RootCodebookSolver final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  explicit RootCodebookSolver(const SpdSolverF32& solver) noexcept
      : solver_(&solver) {}

  [[nodiscard]] StatusOr<RootCodebookSolveResult<M>> Solve(
      RootCodebookMoments<M>& moments,
      ColumnMajorMatrixF32View previous_root0) const;

 private:
  friend class RoundModelUpdater<M>;
  friend class InitializationModelUpdater<M>;

  // The updater supplies the immutable root0 of a model whose constructor
  // already checked shape and finiteness. Standalone Solve remains checked.
  [[nodiscard]] StatusOr<RootCodebookSolveResult<M>>
  SolveValidatedPreviousRoot(
      RootCodebookMoments<M>& moments,
      ColumnMajorMatrixF32View previous_root0) const;

  template <bool PreviousRootValidated>
  [[nodiscard]] StatusOr<RootCodebookSolveResult<M>> SolveImpl(
      RootCodebookMoments<M>& moments,
      ColumnMajorMatrixF32View previous_root0) const;

  const SpdSolverF32* solver_ = nullptr;
};

using RuntimeRootCodebookSolver = RootCodebookSolver<0>;
extern template class RootCodebookSolver<0>;
extern template class RootCodebookSolver<5>;
extern template class RootCodebookSolver<10>;

}  // namespace stlqpp
