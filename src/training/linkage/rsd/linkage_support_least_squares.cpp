#include "stlqpp/training/linkage/rsd/linkage_support_least_squares.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace stlqpp {
namespace {

constexpr float kInitialDiagonalBump = 1.0e-6F;

template <int M, std::uint32_t Capacity>
bool CholeskyUpper(
    std::array<float, Capacity * Capacity>* matrix,
    std::uint32_t m) noexcept {
  for (std::uint32_t column = 0U; column < m; ++column) {
    float diagonal = (*matrix)[static_cast<std::size_t>(column) * m + column];
    for (std::uint32_t prior = 0U; prior < column; ++prior) {
      const float value =
          (*matrix)[static_cast<std::size_t>(prior) * m + column];
      diagonal -= value * value;
    }
    // Match basic, CUDA and LAPACK: a positive pivot is solvable. The constant
    // controls the existing retry bump, not an absolute SPD cutoff.
    if (!(diagonal > 0.0F)) {
      return false;
    }
    diagonal = std::sqrt(diagonal);
    (*matrix)[static_cast<std::size_t>(column) * m + column] = diagonal;
    for (std::uint32_t next = column + 1U; next < m; ++next) {
      float value = (*matrix)[static_cast<std::size_t>(column) * m + next];
      for (std::uint32_t prior = 0U; prior < column; ++prior) {
        value -= (*matrix)[static_cast<std::size_t>(prior) * m + column] *
                 (*matrix)[static_cast<std::size_t>(prior) * m + next];
      }
      (*matrix)[static_cast<std::size_t>(column) * m + next] = value / diagonal;
    }
  }
  return true;
}

template <int M, std::uint32_t Capacity>
void SolveUpper(const std::array<float, Capacity *
                                            Capacity>& factor,
                const std::array<float, Capacity>& rhs,
                float* solution, std::uint32_t m) noexcept {
  std::array<float, Capacity> forward{};
  for (std::uint32_t row = 0U; row < m; ++row) {
    float value = rhs[row];
    for (std::uint32_t prior = 0U; prior < row; ++prior) {
      value -=
          factor[static_cast<std::size_t>(prior) * m + row] * forward[prior];
    }
    forward[row] = value / factor[static_cast<std::size_t>(row) * m + row];
  }
  for (std::uint32_t reverse = m; reverse > 0U; --reverse) {
    const std::uint32_t row = reverse - 1U;
    float value = forward[row];
    for (std::uint32_t next = row + 1U; next < m; ++next) {
      value -=
          factor[static_cast<std::size_t>(row) * m + next] * solution[next];
    }
    solution[row] = value / factor[static_cast<std::size_t>(row) * m + row];
  }
}

template <int M, std::uint32_t Capacity>
void FillSystem(
    const LinkageCodebookAlgebra<M>& algebra, const LinkageSupport<M, Capacity>& support,
    const float* atom_dots, float diagonal_bump,
    std::array<float, Capacity * Capacity>* gram,
    std::array<float, Capacity>* rhs) noexcept {
  const std::uint32_t m = M == 0 ? algebra.shape().m : M;
  gram->fill(0.0F);
  for (std::uint32_t column = 0U; column < m; ++column) {
    const std::uint32_t column_atom =
        algebra.layer_offset(column) + support.codes[column];
    (*rhs)[column] = atom_dots[column_atom];
    for (std::uint32_t row = 0U; row <= column; ++row) {
      const std::uint32_t row_atom =
          algebra.layer_offset(row) + support.codes[row];
      (*gram)[static_cast<std::size_t>(row) * m + column] =
          algebra.dot(row_atom, column_atom);
    }
    (*gram)[static_cast<std::size_t>(column) * m + column] += diagonal_bump;
  }
}

}  // namespace

template <int M, std::uint32_t Capacity>
bool LinkageSupportLeastSquares<M, Capacity>::Solve(
    const LinkageCodebookAlgebra<M>& algebra, const LinkageSupport<M, Capacity>& support,
    const float* atom_dots, float* coefficients,
    float* applied_diagonal_bump) noexcept {
  const std::uint32_t m = M == 0 ? algebra.shape().m : M;
  std::array<float, Capacity * Capacity> gram{};
  std::array<float, Capacity> rhs{};
  FillSystem<M, Capacity>(algebra, support, atom_dots, 0.0F, &gram, &rhs);
  bool solved = CholeskyUpper<M, Capacity>(&gram, m);
  float solved_bump = 0.0F;
  float bump = kInitialDiagonalBump;
  for (std::uint32_t attempt = 0U; attempt < 3U && !solved; ++attempt) {
    FillSystem<M, Capacity>(algebra, support, atom_dots, bump, &gram, &rhs);
    solved = CholeskyUpper<M, Capacity>(&gram, m);
    solved_bump = bump;
    bump *= 10.0F;
  }
  if (!solved) {
    return false;
  }
  SolveUpper<M, Capacity>(gram, rhs, coefficients, m);
  if (applied_diagonal_bump != nullptr) {
    *applied_diagonal_bump = solved_bump;
  }
  return true;
}

template <int M, std::uint32_t Capacity>
float LinkageSupportLeastSquares<M, Capacity>::SquaredCost(
    const LinkageCodebookAlgebra<M>& algebra, const LinkageSupport<M, Capacity>& support,
    const float* atom_dots, const float* coefficients,
    float residual_squared_norm) noexcept {
  const std::uint32_t m = M == 0 ? algebra.shape().m : M;
  float data_dot = 0.0F;
  float reconstructed_norm = 0.0F;
  for (std::uint32_t column = 0U; column < m; ++column) {
    const std::uint32_t column_atom =
        algebra.layer_offset(column) + support.codes[column];
    data_dot += coefficients[column] * atom_dots[column_atom];
    for (std::uint32_t row = 0U; row < m; ++row) {
      const std::uint32_t row_atom =
          algebra.layer_offset(row) + support.codes[row];
      const float gram = row <= column ? algebra.dot(row_atom, column_atom)
                                       : algebra.dot(column_atom, row_atom);
      reconstructed_norm += coefficients[column] * coefficients[row] * gram;
    }
  }
  return residual_squared_norm - 2.0F * data_dot + reconstructed_norm;
}

template <int M, std::uint32_t Capacity>
float LinkageSupportLeastSquares<M, Capacity>::NormalEquationSquaredCost(
    const LinkageCodebookAlgebra<M>& algebra, const LinkageSupport<M, Capacity>& support,
    const float* atom_dots, const float* coefficients,
    float residual_squared_norm, float applied_diagonal_bump) noexcept {
  const std::uint32_t m = M == 0 ? algebra.shape().m : M;
  float data_dot = 0.0F;
  for (std::uint32_t layer = 0U; layer < m; ++layer) {
    const std::uint32_t atom =
        algebra.layer_offset(layer) + support.codes[layer];
    data_dot += coefficients[layer] * atom_dots[atom];
  }
  float squared_error = residual_squared_norm - data_dot;
  if (applied_diagonal_bump == 0.0F) {
    return squared_error;
  }
  float coefficient_norm = 0.0F;
  for (std::uint32_t layer = 0U; layer < m; ++layer) {
    coefficient_norm += coefficients[layer] * coefficients[layer];
  }
  return squared_error - applied_diagonal_bump * coefficient_norm;
}

template class LinkageSupportLeastSquares<0>;
template class LinkageSupportLeastSquares<0, kWideRuntimeScratchLayers>;
template class LinkageSupportLeastSquares<5>;
template class LinkageSupportLeastSquares<12>;
template class LinkageSupportLeastSquares<6>;
template class LinkageSupportLeastSquares<10>;

}  // namespace stlqpp
