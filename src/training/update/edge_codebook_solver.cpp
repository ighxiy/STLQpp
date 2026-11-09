#include "stlqpp/training/update/edge_codebook_solver.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {

template <int M>
StatusOr<EdgeCodebookSolveResult<M>> EdgeCodebookSolver<M>::Solve(
    const EdgeCodebookMoments<M>& moments) const {
  if (moments.sample_count() == 0U) {
    return Status::FailedPrecondition(
        "C_one solve requires at least one linked training node");
  }
  const std::uint32_t kOrder = moments.atom_count();
  const std::uint32_t dimension = moments.dimension();
  const auto packed = moments.packed_upper_gram();
  const auto atom_major_rhs = moments.atom_major_rhs();
  std::vector<float> base_matrix(
      static_cast<std::size_t>(kOrder) * kOrder, 0.0F);
  std::vector<float> base_rhs(
      static_cast<std::size_t>(kOrder) * dimension, 0.0F);
  for (std::uint32_t column = 0U; column < kOrder; ++column) {
    for (std::uint32_t row = 0U; row <= column; ++row) {
      base_matrix[static_cast<std::size_t>(row) +
                  static_cast<std::size_t>(column) * kOrder] =
          static_cast<float>(packed[
              EdgeCodebookMoments<M>::PackedUpperIndex(row, column)]);
    }
    const double* source = atom_major_rhs.data() +
        static_cast<std::size_t>(column) * dimension;
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      base_rhs[static_cast<std::size_t>(column) +
               static_cast<std::size_t>(axis) * kOrder] =
          static_cast<float>(source[axis]);
    }
  }

  float bump = 0.0F;
  for (std::uint32_t attempt = 0U; attempt < 8U; ++attempt) {
    std::vector<float> matrix = base_matrix;
    std::vector<float> solution = base_rhs;
    if (bump != 0.0F) {
      for (std::uint32_t atom = 0U; atom < kOrder; ++atom) {
        matrix[static_cast<std::size_t>(atom) * (kOrder + 1U)] += bump;
      }
    }
    auto solved = solver_->FactorAndSolve({
        .matrix = matrix.data(),
        .order = kOrder,
        .matrix_leading_dimension = kOrder,
        .right_hand_sides = solution.data(),
        .right_hand_side_count = dimension,
        .rhs_leading_dimension = kOrder,
        .allow_phase_parallelism = true,
    });
    if (!solved.ok()) {
      return solved.status();
    }
    if (solved.value() == SpdSolveOutcome::kSolved) {
      EdgeCodebookSolveResult<M> result;
      result.m_ = moments.m();
      result.dimension_ = dimension;
      result.bump_ = bump;
      result.values_.resize(
          static_cast<std::size_t>(moments.m()) * 256U * dimension);
      for (std::uint32_t atom = 0U; atom < kOrder; ++atom) {
        float* destination = result.values_.data() +
            static_cast<std::size_t>(atom) * dimension;
        for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
          destination[axis] = solution[
              static_cast<std::size_t>(atom) +
              static_cast<std::size_t>(axis) * kOrder];
        }
      }
      return result;
    }
    bump = bump == 0.0F ? 1.0e-6F : bump * 10.0F;
  }
  return Status::FailedPrecondition(
      "C_one normal equations remained non-positive-definite");
}

template class EdgeCodebookSolver<0>;
template class EdgeCodebookSolver<5>;
template class EdgeCodebookSolver<12>;
template class EdgeCodebookSolver<6>;
template class EdgeCodebookSolver<10>;

}  // namespace stlqpp
