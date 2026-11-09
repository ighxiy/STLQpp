#include "stlqpp/model/model_shape.h"
#include "stlqpp/index/codec/scalar_int6_encoder.h"

#include "../../model/codec/canonical_huffman_internal.h"

#include <cassert>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

struct WeightedMagnitude final {
  float magnitude = 0.0F;
  float weight = 0.0F;
};

float Type7Quantile(const std::vector<WeightedMagnitude>& values,
                    double quantile) noexcept {
  if (values.empty()) return 0.0F;
  if (quantile <= 0.0) return values.front().magnitude;
  if (quantile >= 1.0) return values.back().magnitude;
  const double h = static_cast<double>(values.size() - 1U) * quantile + 1.0;
  const std::size_t one_based = static_cast<std::size_t>(std::floor(h));
  const double fraction = h - static_cast<double>(one_based);
  if (one_based >= values.size()) return values.back().magnitude;
  const std::size_t lower = one_based == 0U ? 0U : one_based - 1U;
  const std::size_t upper = std::min(values.size() - 1U, one_based);
  return static_cast<float>(
      (1.0 - fraction) * values[lower].magnitude +
      fraction * values[upper].magnitude);
}

bool ValidMagnitudeQuantiles(const std::vector<double>& candidates) {
  return !candidates.empty() &&
      std::all_of(candidates.begin(), candidates.end(), [](double value) {
        return std::isfinite(value) && value > 0.0 && value <= 1.0;
      });
}

template <int M>
class MaterializedGramView final {
 public:
  explicit MaterializedGramView(ArrayView<const float> values, std::uint32_t width) noexcept
      : values_(values), m_(width) {}
  [[nodiscard]] std::uint32_t m() const noexcept { return M == 0 ? m_ : M; }

  [[nodiscard]] float At(std::uint32_t row,
                         std::uint32_t layer,
                         std::uint32_t other) const noexcept {
    return values_[static_cast<std::size_t>(row) * m() * m() + layer * m() +
                   other];
  }

  [[nodiscard]] float Weight(std::uint32_t row,
                             std::uint32_t layer) const noexcept {
    return At(row, layer, layer);
  }

 private:
  ArrayView<const float> values_;
  std::uint32_t m_;
};

template <int M, std::uint32_t Capacity, typename GramView>
double ReconstructionSse(
    std::uint32_t row_begin,
    std::uint32_t row_end,
    std::uint32_t node_count,
    ArrayView<const float> coefficients,
    const GramView& grams,
    const std::vector<std::int8_t>& quantized,
    const std::array<float, Capacity>& scales) noexcept {
  const auto m = grams.m();
  double output = 0.0;
  for (std::uint32_t row = row_begin; row < row_end; ++row) {
    std::array<double, Capacity> error{};
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      const std::size_t offset =
          static_cast<std::size_t>(layer) * node_count + row;
      error[layer] = static_cast<double>(coefficients[offset]) -
                     static_cast<double>(scales[layer]) *
                         static_cast<double>(quantized[offset]);
    }
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      for (std::uint32_t other = 0U;
           other < m; ++other) {
        output += error[layer] * error[other] *
                  static_cast<double>(grams.At(row, layer, other));
      }
    }
  }
  return std::max(0.0, output);
}

template <int M, std::uint32_t Capacity>
StatusOr<bool> SolveSymPosdefCholeskyRetry(
    const std::array<double, Capacity * Capacity>& matrix,
    const std::array<double, Capacity>& rhs,
    const SpdSolverF64& solver,
    std::array<double, Capacity>& solution,
    std::uint32_t m) {
  constexpr std::array<double, 4> kDiagonalBumps = {
      0.0, 1.0e-6, 1.0e-5, 1.0e-4};
  for (const double bump : kDiagonalBumps) {
    std::array<double, Capacity * Capacity> column_major{};
    for (std::uint32_t row = 0U;
         row < m; ++row) {
      for (std::uint32_t column = 0U;
           column < m; ++column) {
        column_major[static_cast<std::size_t>(row) +
                     static_cast<std::size_t>(column) * m] =
            matrix[static_cast<std::size_t>(row) * m + column];
      }
      column_major[static_cast<std::size_t>(row) * m + row] += bump;
    }
    std::array<double, Capacity> candidate = rhs;
    auto solved = solver.FactorAndSolve({
        .matrix = column_major.data(),
        .order = m,
        .matrix_leading_dimension = m,
        .right_hand_sides = candidate.data(),
        .right_hand_side_count = 1U,
        .rhs_leading_dimension = m,
    });
    if (!solved.ok()) {
      return solved.status();
    }
    if (solved.value() == SpdSolveOutcome::kSolved) {
      solution = candidate;
      return true;
    }
  }
  return false;
}

template <int M, std::uint32_t Capacity, typename GramView>
StatusOr<std::array<float, Capacity>> FitReconstructionScales(
    std::uint32_t row_begin,
    std::uint32_t row_end,
    std::uint32_t node_count,
    ArrayView<const float> coefficients,
    const GramView& grams,
    const std::vector<std::int8_t>& quantized,
    const std::array<float, Capacity>& fallback,
    const SpdSolverF64& solver) {
  const auto m = grams.m();
  if (row_begin == row_end) {
    return fallback;
  }
  std::array<double, Capacity * Capacity> matrix{};
  std::array<double, Capacity> rhs{};
  for (std::uint32_t row = row_begin; row < row_end; ++row) {
    std::array<double, Capacity> target_dots{};
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      for (std::uint32_t other = 0U;
           other < m; ++other) {
        target_dots[layer] +=
            static_cast<double>(coefficients[
                static_cast<std::size_t>(other) * node_count + row]) *
            static_cast<double>(grams.At(row, layer, other));
      }
    }
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      const double q_layer = static_cast<double>(quantized[
          static_cast<std::size_t>(layer) * node_count + row]);
      rhs[layer] += q_layer * target_dots[layer];
      for (std::uint32_t other = 0U;
           other < m; ++other) {
        const double q_other = static_cast<double>(quantized[
            static_cast<std::size_t>(other) * node_count + row]);
        matrix[layer * m + other] +=
            q_layer * q_other *
            static_cast<double>(grams.At(row, layer, other));
      }
    }
  }

  std::array<double, Capacity> solution{};
  auto solved = SolveSymPosdefCholeskyRetry<M, Capacity>(
      matrix, rhs, solver, solution, m);
  if (!solved.ok()) {
    return solved.status();
  }
  if (!solved.value()) {
    return fallback;
  }
  std::array<float, Capacity> output{};
  for (std::uint32_t layer = 0U;
       layer < m; ++layer) {
    output[layer] = static_cast<float>(solution[layer]);
  }
  return output;
}

template <int M, std::uint32_t Capacity, typename GramView>
std::array<float, Capacity> InitializeIntegerCoefficients(
    std::uint32_t row_begin,
    std::uint32_t row_end,
    std::uint32_t node_count,
    ArrayView<const float> coefficients,
    const GramView& grams,
    const std::array<std::int32_t, Capacity>& limits,
    const std::vector<double>& first_magnitude_quantiles,
    const std::vector<double>& rest_magnitude_quantiles,
    bool weighted_quantile,
    bool allow_clipping,
    bool fit_scale,
    std::vector<std::int8_t>& quantized) {
  const auto m = grams.m();
  std::array<float, Capacity> scales{};
  scales.fill(1.0F);
  if (row_begin == row_end) {
    return scales;
  }
  std::vector<WeightedMagnitude> values;
  values.reserve(row_end - row_begin);
  for (std::uint32_t layer = 0U;
       layer < m; ++layer) {
    const std::int32_t limit = limits[layer];
    values.clear();
    float maximum_rounded_magnitude = 0.0F;
    for (std::uint32_t row = row_begin; row < row_end; ++row) {
      const float coefficient = coefficients[
          static_cast<std::size_t>(layer) * node_count + row];
      values.push_back({
          .magnitude = std::abs(coefficient),
          .weight = weighted_quantile
              ? std::max(0.0F, grams.Weight(row, layer)) : 1.0F,
      });
      maximum_rounded_magnitude = std::max(
          maximum_rounded_magnitude, std::abs(std::round(coefficient)));
    }
    std::sort(values.begin(), values.end(),
              [](const WeightedMagnitude& left,
                 const WeightedMagnitude& right) {
                return left.magnitude < right.magnitude;
              });
    std::vector<float> cumulative_weight;
    double running_weight = 0.0;
    if (weighted_quantile) {
      cumulative_weight.resize(values.size());
      for (std::size_t index = 0U; index < values.size(); ++index) {
        running_weight += values[index].weight;
        cumulative_weight[index] = static_cast<float>(running_weight);
      }
    }
    const double total_weight = weighted_quantile
        ? cumulative_weight.back() : 0.0;
    const auto& candidates = layer == 0U
        ? first_magnitude_quantiles : rest_magnitude_quantiles;
    double best_cost = std::numeric_limits<double>::infinity();
    float best_scale = 1.0F;
    std::vector<std::int8_t> best_quantized(row_end - row_begin, 0);
    std::vector<std::int8_t> candidate_quantized(row_end - row_begin, 0);
    bool selected_candidate = false;
    for (const double magnitude_quantile : candidates) {
      float maximum = values.back().magnitude;
      if (magnitude_quantile < 1.0 && weighted_quantile &&
          total_weight > 0.0) {
        const double target = magnitude_quantile * total_weight;
        const auto selected = std::lower_bound(
            cumulative_weight.begin(), cumulative_weight.end(),
            static_cast<float>(target));
        const std::size_t index = static_cast<std::size_t>(
            selected - cumulative_weight.begin());
        maximum = values[std::min(index, values.size() - 1U)].magnitude;
      } else if (magnitude_quantile < 1.0 && !weighted_quantile) {
        maximum = Type7Quantile(values, magnitude_quantile);
      }
      if (!allow_clipping && maximum < maximum_rounded_magnitude) continue;
      if (!(maximum > 0.0F)) continue;
      const float delta = maximum > 0.0F
          ? maximum / static_cast<float>(limit)
          : 1.0F;
      double coefficient_squared = 0.0;
      double coefficient_q = 0.0;
      double q_squared = 0.0;
      for (std::uint32_t row = row_begin; row < row_end; ++row) {
        const double coefficient = coefficients[
            static_cast<std::size_t>(layer) * node_count + row];
        const std::int32_t q = std::clamp(
            static_cast<std::int32_t>(std::lrint(
                coefficient / static_cast<double>(delta))),
            -limit, limit);
        candidate_quantized[row - row_begin] =
            static_cast<std::int8_t>(q);
        const double weight = weighted_quantile
            ? std::max(0.0,
                       static_cast<double>(grams.Weight(row, layer)))
            : 1.0;
        coefficient_squared += weight * coefficient * coefficient;
        coefficient_q += weight * coefficient * q;
        q_squared += weight * static_cast<double>(q) * q;
      }
      // Score the scale that will actually be published. With fitting off,
      // every quantile candidate keeps its fixed quantization step.
      const double cost = q_squared > 0.0
          ? (fit_scale
                 ? coefficient_squared - coefficient_q * coefficient_q / q_squared
                 : coefficient_squared - 2.0 * static_cast<double>(delta) * coefficient_q +
                       static_cast<double>(delta) * delta * q_squared)
          : coefficient_squared;
      if (cost < best_cost) {
        best_cost = cost;
        best_scale = q_squared > 0.0
            ? (fit_scale
                   ? static_cast<float>(coefficient_q / q_squared)
                   : delta)
            : 1.0F;
        best_quantized = candidate_quantized;
        selected_candidate = true;
      }
    }
    if (!selected_candidate) {
      for (std::uint32_t row = row_begin; row < row_end; ++row) {
        const double coefficient = coefficients[
            static_cast<std::size_t>(layer) * node_count + row];
        best_quantized[row - row_begin] = static_cast<std::int8_t>(
            std::clamp(static_cast<std::int32_t>(std::lrint(coefficient)),
                       -limit, limit));
      }
    }
    scales[layer] = best_scale;
    for (std::uint32_t row = row_begin; row < row_end; ++row) {
      quantized[static_cast<std::size_t>(layer) * node_count + row] =
          best_quantized[row - row_begin];
    }
  }
  return scales;
}

template <int M, std::uint32_t Capacity, typename GramView>
bool RefineIntegerCoefficients(
    std::uint32_t row_begin,
    std::uint32_t row_end,
    std::uint32_t node_count,
    ArrayView<const float> coefficients,
    const GramView& grams,
    const std::array<float, Capacity>& scales,
    const std::array<std::int32_t, Capacity>& limits,
    std::uint32_t sweeps,
    std::uint32_t maximum_layer,
    std::uint32_t step_limit,
    std::vector<std::int8_t>& quantized) noexcept {
  if (sweeps == 0U || row_begin == row_end) return false;
  bool changed = false;
  const auto m = grams.m();
  const std::uint32_t layer_count = maximum_layer == 0U
      ? m
      : std::min(maximum_layer, m);
  for (std::uint32_t row = row_begin; row < row_end; ++row) {
    std::array<double, Capacity> target_dots{};
    std::array<std::int32_t, Capacity> q{};
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      q[layer] = quantized[
          static_cast<std::size_t>(layer) * node_count + row];
      for (std::uint32_t other = 0U;
           other < m; ++other) {
        target_dots[layer] +=
            static_cast<double>(coefficients[
                static_cast<std::size_t>(other) * node_count + row]) *
            static_cast<double>(grams.At(row, layer, other));
      }
    }
    for (std::uint32_t sweep = 0U; sweep < sweeps; ++sweep) {
      for (std::uint32_t layer = 0U; layer < layer_count; ++layer) {
        const std::int32_t limit = limits[layer];
        const double scale = scales[layer];
        const double diagonal = grams.At(row, layer, layer);
        if (scale == 0.0 || diagonal == 0.0) {
          q[layer] = 0;
          continue;
        }
        double other_sum = 0.0;
        for (std::uint32_t other = 0U;
             other < m; ++other) {
          if (other != layer) {
            other_sum += static_cast<double>(scales[other]) * q[other] *
                         static_cast<double>(grams.At(row, layer, other));
          }
        }
        const double beta = scale * (target_dots[layer] - other_sum);
        const double alpha = scale * scale * diagonal;
        if (alpha == 0.0) {
          continue;
        }
        const std::int32_t current = q[layer];
        std::int32_t best = current;
        double best_cost = alpha * current * current -
                           2.0 * beta * current;
        if (step_limit == 0U) {
          best = std::clamp(
              static_cast<std::int32_t>(std::llround(beta / alpha)),
              -limit, limit);
        } else {
          const std::int64_t wide_current = current;
          const std::int64_t wide_step = step_limit;
          const std::int32_t low = static_cast<std::int32_t>(std::max<
              std::int64_t>(-limit, wide_current - wide_step));
          const std::int32_t high = static_cast<std::int32_t>(std::min<
              std::int64_t>(limit, wide_current + wide_step));
          for (std::int32_t candidate = low; candidate <= high; ++candidate) {
            const double cost = alpha * candidate * candidate -
                                2.0 * beta * candidate;
            if (cost < best_cost) {
              best_cost = cost;
              best = candidate;
            }
          }
        }
        q[layer] = best;
      }
    }
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      const auto offset = static_cast<std::size_t>(layer) * node_count + row;
      changed |= q[layer] != quantized[offset];
      quantized[offset] = static_cast<std::int8_t>(q[layer]);
    }
  }
  return changed;
}

template <int M>
Status ValidateInput(const ScalarInt6EncodingInput<M>& input,
                     std::uint32_t& node_count,
                     std::uint32_t& root_count) {
  const auto m = input.m;
  if (m < 2U || m > stlqpp::kMaximumModelLayers || (M != 0 && m != M))
    return Status::InvalidArgument("invalid coefficient layer count");
  const std::uint64_t roots =
      static_cast<std::uint64_t>(input.virtual_root_count) +
      input.real_root_count;
  const std::uint64_t nodes = roots + input.linked_real_count;
  if (roots > std::numeric_limits<std::uint32_t>::max() ||
      nodes > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "scalar int6 cluster exceeds the canonical u32 row range");
  }
  root_count = static_cast<std::uint32_t>(roots);
  node_count = static_cast<std::uint32_t>(nodes);
  const std::uint64_t coefficient_count = nodes * m;
  const std::uint64_t gram_count = coefficient_count * m;
  if (coefficient_count != input.coefficients_layer_major.size() ||
      gram_count != input.selected_atom_grams_row_major.size()) {
    return Status::InvalidArgument(
        "scalar int6 coefficients or selected-atom Gram shape is invalid");
  }
  for (const float coefficient : input.coefficients_layer_major) {
    if (!std::isfinite(coefficient)) {
      return Status::InvalidArgument(
          "scalar int6 input contains a non-finite coefficient");
    }
  }
  for (const float value : input.selected_atom_grams_row_major) {
    if (!std::isfinite(value)) {
      return Status::InvalidArgument(
          "scalar int6 input contains a non-finite Gram value");
    }
  }
  return Status::Ok();
}

template <int M>
Status ValidateInput(const SelectedScalarInt6EncodingInput<M>& input,
                     std::uint32_t& node_count,
                     std::uint32_t& root_count,
                     bool validated_coefficients = false) {
  const auto m = input.selected_atom_grams == nullptr ? 0U : input.selected_atom_grams->m();
  const std::uint64_t roots =
      static_cast<std::uint64_t>(input.virtual_root_count) +
      input.real_root_count;
  const std::uint64_t nodes = roots + input.linked_real_count;
  if (roots > std::numeric_limits<std::uint32_t>::max() ||
      nodes > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "selected scalar int6 cluster exceeds the canonical u32 row range");
  }
  root_count = static_cast<std::uint32_t>(roots);
  node_count = static_cast<std::uint32_t>(nodes);
  if (input.selected_atom_grams == nullptr ||
      input.selected_atom_grams->root_count() != root_count ||
      input.selected_atom_grams->node_count() != node_count ||
      input.coefficients_layer_major.size() !=
          static_cast<std::uint64_t>(node_count) * m) {
    return Status::InvalidArgument(
        "selected scalar int6 coefficient or Gram shape is invalid");
  }
  if (!validated_coefficients) {
    for (const float coefficient : input.coefficients_layer_major) {
      if (!std::isfinite(coefficient)) {
        return Status::InvalidArgument(
            "selected scalar int6 input contains a non-finite coefficient");
      }
    }
  }
#ifndef NDEBUG
  if (validated_coefficients) {
    for (const float coefficient : input.coefficients_layer_major) {
      assert(std::isfinite(coefficient));
    }
  }
#endif
  return Status::Ok();
}

template <int M, std::uint32_t Capacity>
StatusOr<EncodedCanonicalHuffman> EncodeRole(
    std::uint32_t row_begin,
    std::uint32_t row_end,
    std::uint32_t node_count,
    const std::vector<std::int8_t>& quantized,
    std::int32_t symbol_base,
    std::uint32_t alphabet_size, std::uint32_t m) {
  std::vector<std::uint8_t> symbols;
  symbols.reserve(static_cast<std::size_t>(row_end - row_begin) * m);
  for (std::uint32_t layer = 0U;
       layer < m; ++layer) {
    for (std::uint32_t row = row_begin; row < row_end; ++row) {
      const std::int32_t q = quantized[
          static_cast<std::size_t>(layer) * node_count + row];
      const std::int32_t symbol = q + symbol_base;
      assert(symbol >= 0 &&
             symbol < static_cast<std::int32_t>(alphabet_size));
      symbols.push_back(static_cast<std::uint8_t>(symbol));
    }
  }
  return internal::EncodeCanonicalHuffmanMsbFirstTrusted(symbols, alphabet_size);
}

template <int M, std::uint32_t Capacity>
struct ScalarInt6Work final {
  std::array<float, Capacity> root_scales{};
  std::array<float, Capacity> linkage_scales{};
  std::vector<std::int8_t> coefficients_layer_major;
  EncodedCanonicalHuffman root;
  EncodedCanonicalHuffman linkage;
  ScalarInt6EncodingStatistics statistics;
};

template <int M, std::uint32_t Capacity, typename RootGramView, typename LinkageGramView>
StatusOr<ScalarInt6Work<M, Capacity>> EncodeGroups(
    std::uint32_t root_count,
    std::uint32_t node_count,
    ArrayView<const float> coefficients,
    const RootGramView& root_grams,
    const LinkageGramView& linkage_grams,
    const std::vector<std::uint32_t>& base_bits,
    const std::vector<std::uint32_t>& root_bits,
    const std::vector<std::uint32_t>& linkage_bits,
    const std::vector<double>& first_magnitude_quantiles,
    const std::vector<double>& rest_magnitude_quantiles,
    const ScalarCoefficientCodecOptions& options,
    const SpdSolverF64& solver) {
  const auto m = root_grams.m();
  ScalarInt6Work<M, Capacity> output;
  if (base_bits.size() != m ||
      root_bits.size() != m ||
      linkage_bits.size() != m) {
    return Status::InvalidArgument(
        "coefficient bit-width vectors must contain one value per layer");
  }
  std::array<std::int32_t, Capacity> root_limits{};
  std::array<std::int32_t, Capacity> linkage_limits{};
  for (std::uint32_t layer = 0U; layer < m;
       ++layer) {
    if (base_bits[layer] < kMinimumCoefficientBits ||
        base_bits[layer] > kMaximumCoefficientBits ||
        root_bits[layer] < kMinimumCoefficientBits ||
        root_bits[layer] > kMaximumCoefficientBits ||
        linkage_bits[layer] < kMinimumCoefficientBits ||
        linkage_bits[layer] > kMaximumCoefficientBits) {
      return Status::InvalidArgument(
          "coefficient bit widths must be in [2,8]");
    }
    root_limits[layer] = CoefficientLimitFromBits(root_bits[layer]);
    linkage_limits[layer] = CoefficientLimitFromBits(linkage_bits[layer]);
  }
  const std::uint32_t maximum_bits =
      MaximumCoefficientBits(base_bits, root_bits, linkage_bits);
  const std::int32_t symbol_base =
      CoefficientLimitFromBits(maximum_bits);
  const std::uint32_t alphabet_size =
      CoefficientAlphabetFromBits(maximum_bits);
  output.coefficients_layer_major.assign(
      static_cast<std::size_t>(node_count) * m, 0);
  output.root_scales = InitializeIntegerCoefficients<M, Capacity>(
      0U, root_count, node_count, coefficients, root_grams,
      root_limits,
      first_magnitude_quantiles, rest_magnitude_quantiles,
      options.weighted_quantile, options.allow_clipping,
      options.fit_reconstruction_scales,
      output.coefficients_layer_major);
  output.linkage_scales = InitializeIntegerCoefficients<M, Capacity>(
      root_count, node_count, node_count, coefficients, linkage_grams,
      linkage_limits,
      first_magnitude_quantiles, rest_magnitude_quantiles,
      options.weighted_quantile, options.allow_clipping,
      options.fit_reconstruction_scales,
      output.coefficients_layer_major);

  if (options.fit_reconstruction_scales) {
    auto root_scale_result = FitReconstructionScales<M, Capacity>(
        0U, root_count, node_count, coefficients, root_grams,
        output.coefficients_layer_major, output.root_scales, solver);
    if (!root_scale_result.ok()) return root_scale_result.status();
    output.root_scales = std::move(root_scale_result).value();
    auto linkage_scale_result = FitReconstructionScales<M, Capacity>(
        root_count, node_count, node_count, coefficients, linkage_grams,
        output.coefficients_layer_major, output.linkage_scales, solver);
    if (!linkage_scale_result.ok()) return linkage_scale_result.status();
    output.linkage_scales = std::move(linkage_scale_result).value();
  }
  const double before_root_sse = options.collect_sse_statistics ? ReconstructionSse<M, Capacity>(
      0U, root_count, node_count, coefficients, root_grams,
      output.coefficients_layer_major, output.root_scales) : 0.0;
  const double before_linkage_sse = options.collect_sse_statistics ? ReconstructionSse<M, Capacity>(
      root_count, node_count, node_count, coefficients, linkage_grams,
      output.coefficients_layer_major, output.linkage_scales) : 0.0;
  output.statistics.before_integer_refinement_sse = before_root_sse + before_linkage_sse;

  const bool roots_changed = RefineIntegerCoefficients<M, Capacity>(
      0U, root_count, node_count, coefficients, root_grams,
      output.root_scales, root_limits, options.refinement_sweeps,
      options.refinement_max_layer, options.refinement_step_limit,
      output.coefficients_layer_major);
  const bool linkage_changed = RefineIntegerCoefficients<M, Capacity>(
      root_count, node_count, node_count, coefficients, linkage_grams,
      output.linkage_scales, linkage_limits, options.refinement_sweeps,
      options.refinement_max_layer, options.refinement_step_limit,
      output.coefficients_layer_major);
  if (options.fit_reconstruction_scales && roots_changed) {
    auto fitted = FitReconstructionScales<M, Capacity>(
        0U, root_count, node_count, coefficients, root_grams,
        output.coefficients_layer_major, output.root_scales, solver);
    if (!fitted.ok()) return fitted.status();
    output.root_scales = std::move(fitted).value();
  }
  if (options.fit_reconstruction_scales && linkage_changed) {
    auto fitted = FitReconstructionScales<M, Capacity>(
        root_count, node_count, node_count, coefficients, linkage_grams,
        output.coefficients_layer_major, output.linkage_scales, solver);
    if (!fitted.ok()) return fitted.status();
    output.linkage_scales = std::move(fitted).value();
  }
  if (options.collect_sse_statistics) {
  output.statistics.final_sse =
      (roots_changed ? ReconstructionSse<M, Capacity>(
          0U, root_count, node_count, coefficients, root_grams,
          output.coefficients_layer_major, output.root_scales) : before_root_sse) +
      (linkage_changed ? ReconstructionSse<M, Capacity>(
          root_count, node_count, node_count, coefficients, linkage_grams,
          output.coefficients_layer_major, output.linkage_scales) : before_linkage_sse);
  }

  auto root_result = EncodeRole<M, Capacity>(
      0U, root_count, node_count, output.coefficients_layer_major,
      symbol_base, alphabet_size, m);
  if (!root_result.ok()) {
    return root_result.status();
  }
  output.root = std::move(root_result).value();
  auto linkage_result = EncodeRole<M, Capacity>(
      root_count, node_count, node_count, output.coefficients_layer_major,
      symbol_base, alphabet_size, m);
  if (!linkage_result.ok()) {
    return linkage_result.status();
  }
  output.linkage = std::move(linkage_result).value();
  output.statistics.root_payload_bits =
      static_cast<std::uint64_t>(output.root.payload.size()) * 8U;
  output.statistics.linkage_payload_bits =
      static_cast<std::uint64_t>(output.linkage.payload.size()) * 8U;
  return output;
}

}  // namespace

template <int M>
StatusOr<EncodedScalarInt6Cluster<M>> ScalarInt6Encoder<M>::Encode(
    const ScalarInt6EncodingInput<M>& input) const {
  if (!ValidMagnitudeQuantiles(first_magnitude_quantiles_) ||
      !ValidMagnitudeQuantiles(rest_magnitude_quantiles_)) {
    return Status::InvalidArgument(
        "scalar int6 magnitude-quantile candidates are invalid");
  }
  std::uint32_t node_count = 0U;
  std::uint32_t root_count = 0U;
  if (const Status status = ValidateInput(input, node_count, root_count);
      !status.ok()) {
    return status;
  }

  if constexpr (M == 0) {
    if (base_bits_.empty() && root_bits_.empty() && linkage_bits_.empty()) {
      const std::vector<std::uint32_t> bits(input.m, kDefaultCoefficientBits);
      return ScalarInt6Encoder<0>(*solver_, bits, bits, bits,
          first_magnitude_quantiles_, rest_magnitude_quantiles_, options_).Encode(input);
    }
  }
  const MaterializedGramView<M> grams(
      input.selected_atom_grams_row_major, input.m);
  const auto run = [&]<std::uint32_t Capacity>() -> StatusOr<EncodedScalarInt6Cluster<M>> {
  auto work_result = EncodeGroups<M, Capacity>(
      root_count, node_count, input.coefficients_layer_major, grams, grams,
      base_bits_, root_bits_, linkage_bits_,
      first_magnitude_quantiles_, rest_magnitude_quantiles_,
      options_, *solver_);
  if (!work_result.ok()) {
    return work_result.status();
  }
  auto work = std::move(work_result).value();
  EncodedScalarInt6Cluster<M> output;
  output.virtual_root_count_ = input.virtual_root_count;
  output.real_root_count_ = input.real_root_count;
  output.linked_real_count_ = input.linked_real_count;
  const auto m = static_cast<std::uint32_t>(base_bits_.size());
  if constexpr (M == 0) {
    output.root_scales_.resize(m);
    output.linkage_scales_.resize(m);
  }
  std::copy_n(work.root_scales.data(), m, output.root_scales_.data());
  std::copy_n(work.linkage_scales.data(), m, output.linkage_scales_.data());
  output.coefficients_layer_major_ =
      std::move(work.coefficients_layer_major);
  output.root_ = std::move(work.root);
  output.linkage_ = std::move(work.linkage);
  output.statistics_ = work.statistics;
  return output;
  };
  if constexpr (M == 0) {
    if (base_bits_.size() > kLegacyRuntimeScratchLayers)
      return run.template operator()<kWideRuntimeScratchLayers>();
  }
  return run.template operator()<kModelScratchLayers<M>>();
}

template <int M>
StatusOr<EncodedScalarInt6Cluster<M>> ScalarInt6Encoder<M>::EncodeSelected(
    const SelectedScalarInt6EncodingInput<M>& input) const {
  return EncodeSelectedImpl(input, false);
}

template <int M>
StatusOr<EncodedScalarInt6Cluster<M>>
ScalarInt6Encoder<M>::EncodeSelectedValidated(
    const SelectedScalarInt6EncodingInput<M>& input) const {
  return EncodeSelectedImpl(input, true);
}

template <int M>
StatusOr<EncodedScalarInt6Cluster<M>> ScalarInt6Encoder<M>::EncodeSelectedImpl(
    const SelectedScalarInt6EncodingInput<M>& input,
    bool validated_coefficients) const {
  if (!ValidMagnitudeQuantiles(first_magnitude_quantiles_) ||
      !ValidMagnitudeQuantiles(rest_magnitude_quantiles_)) {
    return Status::InvalidArgument(
        "selected scalar int6 magnitude-quantile candidates are invalid");
  }
  std::uint32_t node_count = 0U;
  std::uint32_t root_count = 0U;
  if (const Status status = ValidateInput(
          input, node_count, root_count, validated_coefficients);
      !status.ok()) {
    return status;
  }
  if constexpr (M == 0) {
    if (base_bits_.empty() && root_bits_.empty() && linkage_bits_.empty()) {
      const std::vector<std::uint32_t> bits(input.selected_atom_grams->m(),
                                          kDefaultCoefficientBits);
      return ScalarInt6Encoder<0>(*solver_, bits, bits, bits,
          first_magnitude_quantiles_, rest_magnitude_quantiles_, options_)
          .EncodeSelectedImpl(input, validated_coefficients);
    }
  }
  const auto run = [&]<std::uint32_t Capacity>() -> StatusOr<EncodedScalarInt6Cluster<M>> {
  auto work_result = EncodeGroups<M, Capacity>(
      root_count, node_count, input.coefficients_layer_major,
      input.selected_atom_grams->roots(),
      input.selected_atom_grams->linkage(), base_bits_, root_bits_,
      linkage_bits_,
      first_magnitude_quantiles_,
      rest_magnitude_quantiles_, options_, *solver_);
  if (!work_result.ok()) {
    return work_result.status();
  }
  auto work = std::move(work_result).value();
  EncodedScalarInt6Cluster<M> output;
  output.virtual_root_count_ = input.virtual_root_count;
  output.real_root_count_ = input.real_root_count;
  output.linked_real_count_ = input.linked_real_count;
  const auto m = static_cast<std::uint32_t>(base_bits_.size());
  if constexpr (M == 0) {
    output.root_scales_.resize(m);
    output.linkage_scales_.resize(m);
  }
  std::copy_n(work.root_scales.data(), m, output.root_scales_.data());
  std::copy_n(work.linkage_scales.data(), m, output.linkage_scales_.data());
  output.coefficients_layer_major_ =
      std::move(work.coefficients_layer_major);
  output.root_ = std::move(work.root);
  output.linkage_ = std::move(work.linkage);
  output.statistics_ = work.statistics;
  return output;
  };
  if constexpr (M == 0) {
    if (base_bits_.size() > kLegacyRuntimeScratchLayers)
      return run.template operator()<kWideRuntimeScratchLayers>();
  }
  return run.template operator()<kModelScratchLayers<M>>();
}

template class EncodedScalarInt6Cluster<0>;
template class EncodedScalarInt6Cluster<5>;
template class EncodedScalarInt6Cluster<12>;
template class EncodedScalarInt6Cluster<6>;
template class EncodedScalarInt6Cluster<10>;
template class ScalarInt6Encoder<0>;
template class ScalarInt6Encoder<5>;
template class ScalarInt6Encoder<12>;
template class ScalarInt6Encoder<6>;
template class ScalarInt6Encoder<10>;

}  // namespace stlqpp
