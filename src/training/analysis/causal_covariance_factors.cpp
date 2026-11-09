#include "stlqpp/training/analysis/causal_covariance_factors.h"

#include <algorithm>
#include <cassert>
#include <cstddef>

namespace stlqpp::detail {
namespace {
void AddScaled(const float* source, float* destination, float coefficient,
               std::uint32_t dimension) noexcept {
#pragma omp simd
  for (std::uint32_t axis = 0U; axis < dimension; ++axis)
    destination[axis] += coefficient * source[axis];
}
}

template <int M>
void BuildCausalCovarianceFactors(
    const RoundUpdateTranscriptCluster<M>& forest,
    ArrayView<const std::uint32_t> causal_order, std::uint32_t dimension,
    ArrayView<float> real_subtrees, ArrayView<float> virtual_subtrees,
    ArrayView<float> factors) noexcept {
  const auto m = forest.m;
  assert(causal_order.size() == forest.real_count());
  assert(real_subtrees.size() == static_cast<std::size_t>(dimension) * forest.real_count());
  assert(virtual_subtrees.size() == static_cast<std::size_t>(dimension) * forest.virtual_count());
  std::fill(virtual_subtrees.begin(), virtual_subtrees.end(), 0.0F);
  std::fill(factors.begin(), factors.end(), 0.0F);
  for (std::size_t position = causal_order.size(); position != 0U; --position) {
    const auto real = causal_order[position - 1U];
    const auto& parent = forest.parents[real];
    if (!parent.has_value()) continue;
    float* destination = parent->kind == LinkageParentKind::kVirtual
        ? virtual_subtrees.data() + static_cast<std::size_t>(parent->local_id) * dimension
        : real_subtrees.data() + static_cast<std::size_t>(parent->local_id) * dimension;
    AddScaled(real_subtrees.data() + static_cast<std::size_t>(real) * dimension,
              destination, 1.0F, dimension);
  }
  const auto add_root = [&](const float* sum, const float* coefficients,
                            const std::uint8_t* codes) {
    AddScaled(sum, factors.data(), coefficients[0], dimension);
    for (std::uint32_t layer = 1U; layer < m; ++layer) {
      const auto atom = 1U + (layer - 1U) * 256U + codes[layer - 1U];
      AddScaled(sum, factors.data() + static_cast<std::size_t>(atom) * dimension,
                coefficients[layer], dimension);
    }
  };
  for (std::uint32_t virtual_id = 0U; virtual_id < forest.virtual_count(); ++virtual_id)
    add_root(virtual_subtrees.data() + static_cast<std::size_t>(virtual_id) * dimension,
             forest.virtual_coefficients.data() + static_cast<std::size_t>(virtual_id) * m,
             forest.virtual_small_codes.data() + static_cast<std::size_t>(virtual_id) * (m - 1U));
  for (std::uint32_t real = 0U; real < forest.real_count(); ++real) {
    const float* sum = real_subtrees.data() + static_cast<std::size_t>(real) * dimension;
    const float* coefficients = forest.coefficients.data() + static_cast<std::size_t>(real) * m;
    const auto& support = forest.supports[real];
    if (forest.roles[real] == LinkageNodeRole::kRoot) {
      add_root(sum, coefficients, support.trailing.data());
    } else {
      for (std::uint32_t layer = 0U; layer < m; ++layer) {
        const auto code = layer == 0U ? support.first : support.trailing[layer - 1U];
        const auto atom = 1U + (m - 1U) * 256U + layer * 256U + code;
        assert(static_cast<std::size_t>(atom + 1U) * dimension <= factors.size());
        AddScaled(sum, factors.data() + static_cast<std::size_t>(atom) * dimension,
                  coefficients[layer], dimension);
      }
    }
  }
}

template void BuildCausalCovarianceFactors<0>(const RoundUpdateTranscriptCluster<0>&, ArrayView<const std::uint32_t>, std::uint32_t, ArrayView<float>, ArrayView<float>, ArrayView<float>) noexcept;
template void BuildCausalCovarianceFactors<5>(const RoundUpdateTranscriptCluster<5>&, ArrayView<const std::uint32_t>, std::uint32_t, ArrayView<float>, ArrayView<float>, ArrayView<float>) noexcept;
template void BuildCausalCovarianceFactors<6>(const RoundUpdateTranscriptCluster<6>&, ArrayView<const std::uint32_t>, std::uint32_t, ArrayView<float>, ArrayView<float>, ArrayView<float>) noexcept;
template void BuildCausalCovarianceFactors<10>(const RoundUpdateTranscriptCluster<10>&, ArrayView<const std::uint32_t>, std::uint32_t, ArrayView<float>, ArrayView<float>, ArrayView<float>) noexcept;
template void BuildCausalCovarianceFactors<12>(const RoundUpdateTranscriptCluster<12>&, ArrayView<const std::uint32_t>, std::uint32_t, ArrayView<float>, ArrayView<float>, ArrayView<float>) noexcept;
}
