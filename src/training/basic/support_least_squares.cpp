#include "stlqpp/training/basic/support_least_squares.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace stlqpp {
namespace {

constexpr float kNumericalEpsilon = 1.0e-6F;

template <int FixedM>
using SolveAlgebra = std::conditional_t<FixedM == 0,
    RuntimeBasicCodebookAlgebra, BasicCodebookAlgebra<FixedM>>;

bool CholeskyUpper(std::uint32_t M, float* matrix) noexcept {
  for (std::uint32_t column = 0U;
       column < static_cast<std::uint32_t>(M); ++column) {
    float diagonal = matrix[static_cast<std::size_t>(column) * M +
                               column];
    for (std::uint32_t prior = 0U; prior < column; ++prior) {
      const float value =
          matrix[static_cast<std::size_t>(prior) * M + column];
      diagonal -= value * value;
    }
    if (!(diagonal > 0.0F)) {
      return false;
    }
    diagonal = std::sqrt(diagonal);
    matrix[static_cast<std::size_t>(column) * M + column] = diagonal;
    for (std::uint32_t next = column + 1U;
         next < static_cast<std::uint32_t>(M); ++next) {
      float value = matrix[static_cast<std::size_t>(column) * M + next];
      for (std::uint32_t prior = 0U; prior < column; ++prior) {
        value -= matrix[static_cast<std::size_t>(prior) * M + column] *
                 matrix[static_cast<std::size_t>(prior) * M + next];
      }
      matrix[static_cast<std::size_t>(column) * M + next] =
          value / diagonal;
    }
  }
  return true;
}

template <std::size_t Capacity>
void SolveUpper(std::uint32_t M, const float* factor,
                const float* rhs, float* solution) noexcept {
  std::array<float, (Capacity > 20U ? Capacity : 20U)> forward{};
  for (std::uint32_t row = 0U;
       row < static_cast<std::uint32_t>(M); ++row) {
    float value = rhs[row];
    for (std::uint32_t prior = 0U; prior < row; ++prior) {
      value -= factor[static_cast<std::size_t>(prior) * M + row] *
               forward[prior];
    }
    forward[row] =
        value / factor[static_cast<std::size_t>(row) * M + row];
  }
  for (std::uint32_t reverse = static_cast<std::uint32_t>(M);
       reverse > 0U; --reverse) {
    const std::uint32_t row = reverse - 1U;
    float value = forward[row];
    for (std::uint32_t next = row + 1U;
         next < static_cast<std::uint32_t>(M); ++next) {
      value -= factor[static_cast<std::size_t>(row) * M + next] *
               solution[next];
    }
    solution[row] =
        value / factor[static_cast<std::size_t>(row) * M + row];
  }
}

template <typename Algebra>
void FillSystem(std::uint32_t M, const Algebra& algebra,
                std::uint32_t root,
                const std::uint8_t* small_codes,
                float root_dot,
                const float* small_dots,
                float diagonal_bump,
                float* gram, float* rhs) noexcept {
  std::fill_n(gram, M * M, 0.0F);
  rhs[0] = root_dot;
  gram[0] = algebra.root_norm2(root) + diagonal_bump;
  for (std::uint32_t column = 1U;
       column < static_cast<std::uint32_t>(M); ++column) {
    const std::uint32_t flat_column =
        algebra.small_offset(column) + small_codes[column - 1U];
    rhs[column] = small_dots[flat_column];
    gram[column] = algebra.root_small_dot(root, flat_column);
    for (std::uint32_t row = 1U; row <= column; ++row) {
      const std::uint32_t flat_row =
          algebra.small_offset(row) + small_codes[row - 1U];
      gram[static_cast<std::size_t>(row) * M + column] =
          algebra.small_dot(flat_row, flat_column);
    }
    gram[static_cast<std::size_t>(column) * M + column] +=
        diagonal_bump;
  }
}

template <int FixedM, std::size_t Capacity = kModelScratchLayers<FixedM>>
bool SolveImpl(
    const SolveAlgebra<FixedM>& algebra,
    std::uint32_t root,
    const std::uint8_t* small_codes,
    float root_dot,
    const float* small_dots,
    float* coefficients,
    float* applied_diagonal_bump) noexcept {
  constexpr std::size_t kCapacity = Capacity;
  const std::uint32_t M = FixedM == 0 ? algebra.shape().m : FixedM;
  std::array<float, kCapacity * kCapacity> gram{};
  std::array<float, kCapacity> rhs{};
  FillSystem(M, algebra, root, small_codes, root_dot, small_dots, 0.0F,
                gram.data(), rhs.data());
  bool solved = CholeskyUpper(M, gram.data());
  float solved_bump = 0.0F;
  float bump = kNumericalEpsilon;
  for (std::uint32_t attempt = 0U; attempt < 3U && !solved; ++attempt) {
    FillSystem(M, algebra, root, small_codes, root_dot, small_dots, bump,
                  gram.data(), rhs.data());
    solved = CholeskyUpper(M, gram.data());
    solved_bump = bump;
    bump *= 10.0F;
  }
  if (!solved) {
    return false;
  }
  SolveUpper<Capacity>(M, gram.data(), rhs.data(), coefficients);
  if (applied_diagonal_bump != nullptr) {
    *applied_diagonal_bump = solved_bump;
  }
  return true;
}

template <int FixedM>
float SquaredCostImpl(
    const SolveAlgebra<FixedM>& algebra,
    std::uint32_t root,
    const std::uint8_t* small_codes,
    float root_dot,
    const float* small_dots,
    const float* coefficients,
    float sample_squared_norm) noexcept {
  const std::uint32_t M = FixedM == 0 ? algebra.shape().m : FixedM;
  float data_dot = coefficients[0] * root_dot;
  for (std::uint32_t layer = 1U;
       layer < static_cast<std::uint32_t>(M); ++layer) {
    const std::uint32_t flat =
        algebra.small_offset(layer) + small_codes[layer - 1U];
    data_dot += coefficients[layer] * small_dots[flat];
  }

  float reconstructed_norm =
      coefficients[0] * coefficients[0] * algebra.root_norm2(root);
  for (std::uint32_t column = 1U;
       column < static_cast<std::uint32_t>(M); ++column) {
    const float column_coefficient = coefficients[column];
    const std::uint32_t flat_column =
        algebra.small_offset(column) + small_codes[column - 1U];
    reconstructed_norm += 2.0F * coefficients[0] * column_coefficient *
                          algebra.root_small_dot(root, flat_column);
    for (std::uint32_t row = 1U;
         row < static_cast<std::uint32_t>(M); ++row) {
      const std::uint32_t flat_row =
          algebra.small_offset(row) + small_codes[row - 1U];
      reconstructed_norm +=
          column_coefficient * coefficients[row] *
          algebra.small_dot(flat_column, flat_row);
    }
  }
  return sample_squared_norm - 2.0F * data_dot + reconstructed_norm;
}

template <int FixedM>
float NormalEquationSquaredCostImpl(
    const SolveAlgebra<FixedM>& algebra,
    const std::uint8_t* small_codes,
    float root_dot,
    const float* small_dots,
    const float* coefficients,
    float sample_squared_norm,
    float applied_diagonal_bump) noexcept {
  const std::uint32_t M = FixedM == 0 ? algebra.shape().m : FixedM;
  float data_dot = coefficients[0] * root_dot;
  for (std::uint32_t layer = 1U;
       layer < static_cast<std::uint32_t>(M); ++layer) {
    const std::uint32_t flat =
        algebra.small_offset(layer) + small_codes[layer - 1U];
    data_dot += coefficients[layer] * small_dots[flat];
  }
  float squared_error = sample_squared_norm - data_dot;
  if (applied_diagonal_bump == 0.0F) {
    return squared_error;
  }
  float coefficient_norm = 0.0F;
  for (std::uint32_t layer = 0U;
       layer < static_cast<std::uint32_t>(M); ++layer) {
    coefficient_norm += coefficients[layer] * coefficients[layer];
  }
  return squared_error - applied_diagonal_bump * coefficient_norm;
}

}  // namespace

template <int M>
bool BasicSupportLeastSquares<M>::Solve(
    const BasicCodebookAlgebra<M>& algebra, std::uint32_t root,
    const std::uint8_t* small_codes, float root_dot, const float* small_dots,
    float* coefficients, float* applied_diagonal_bump) noexcept {
  return SolveImpl<M>(algebra, root, small_codes, root_dot, small_dots,
                        coefficients, applied_diagonal_bump);
}

template <int M>
float BasicSupportLeastSquares<M>::SquaredCost(
    const BasicCodebookAlgebra<M>& algebra, std::uint32_t root,
    const std::uint8_t* small_codes, float root_dot, const float* small_dots,
    const float* coefficients, float sample_squared_norm) noexcept {
  return SquaredCostImpl<M>(algebra, root, small_codes, root_dot, small_dots,
                              coefficients, sample_squared_norm);
}

template <int M>
float BasicSupportLeastSquares<M>::NormalEquationSquaredCost(
    const BasicCodebookAlgebra<M>& algebra,
    const std::uint8_t* small_codes, float root_dot, const float* small_dots,
    const float* coefficients, float sample_squared_norm,
    float applied_diagonal_bump) noexcept {
  return NormalEquationSquaredCostImpl<M>(algebra, small_codes, root_dot,
      small_dots, coefficients, sample_squared_norm, applied_diagonal_bump);
}

bool RuntimeBasicSupportLeastSquares::Solve(
    const RuntimeBasicCodebookAlgebra& algebra, std::uint32_t root,
    const std::uint8_t* small_codes, float root_dot, const float* small_dots,
    float* coefficients, float* applied_diagonal_bump) noexcept {
  return algebra.shape().m > kLegacyRuntimeScratchLayers
      ? SolveWithCapacity<kWideRuntimeScratchLayers>(algebra, root, small_codes, root_dot, small_dots,
                        coefficients, applied_diagonal_bump)
      : SolveWithCapacity<kLegacyRuntimeScratchLayers>(algebra, root, small_codes, root_dot, small_dots,
                        coefficients, applied_diagonal_bump);
}

template <std::uint32_t Capacity>
bool RuntimeBasicSupportLeastSquares::SolveWithCapacity(
    const RuntimeBasicCodebookAlgebra& algebra, std::uint32_t root,
    const std::uint8_t* codes, float root_dot, const float* small_dots,
    float* coefficients, float* bump) noexcept {
  return SolveImpl<0, Capacity>(algebra, root, codes, root_dot, small_dots, coefficients, bump);
}
template bool RuntimeBasicSupportLeastSquares::SolveWithCapacity<20>(
    const RuntimeBasicCodebookAlgebra&,std::uint32_t,const std::uint8_t*,float,const float*,float*,float*) noexcept;
template bool RuntimeBasicSupportLeastSquares::SolveWithCapacity<24>(
    const RuntimeBasicCodebookAlgebra&,std::uint32_t,const std::uint8_t*,float,const float*,float*,float*) noexcept;

float RuntimeBasicSupportLeastSquares::SquaredCost(
    const RuntimeBasicCodebookAlgebra& algebra, std::uint32_t root,
    const std::uint8_t* small_codes, float root_dot, const float* small_dots,
    const float* coefficients, float sample_squared_norm) noexcept {
  return SquaredCostImpl<0>(algebra, root, small_codes, root_dot, small_dots,
                              coefficients, sample_squared_norm);
}

float RuntimeBasicSupportLeastSquares::NormalEquationSquaredCost(
    const RuntimeBasicCodebookAlgebra& algebra,
    const std::uint8_t* small_codes, float root_dot, const float* small_dots,
    const float* coefficients, float sample_squared_norm,
    float applied_diagonal_bump) noexcept {
  return NormalEquationSquaredCostImpl<0>(algebra, small_codes, root_dot,
      small_dots, coefficients, sample_squared_norm, applied_diagonal_bump);
}

template class BasicSupportLeastSquares<5>;
template class BasicSupportLeastSquares<12>;
template class BasicSupportLeastSquares<6>;
template class BasicSupportLeastSquares<10>;

}  // namespace stlqpp
