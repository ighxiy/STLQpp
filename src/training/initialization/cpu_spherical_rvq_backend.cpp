#include "stlqpp/training/initialization/cpu_spherical_rvq_backend.h"

#include "stlqpp/training/initialization/spherical_rvq_log.h"

#include "stlqpp/training/initialization/rvq_assignment_store.h"
#include "stlqpp/training/initialization/spherical_kmeans_plus_plus.h"
#include "stlqpp/training/initialization/spherical_lloyd_code_emission.h"
#include "stlqpp/training/initialization/spherical_hierarchical_layout.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

constexpr float kNormalizationEpsilon = 1.0e-6F;
constexpr std::uint64_t kRoundIncrement = 0x9e3779b97f4a7c15ULL;
constexpr std::uint32_t kSampleTileColumns = 256U;

std::uint32_t Xorshift32(std::uint32_t value) noexcept {
  value ^= value << 13U;
  value ^= value >> 17U;
  value ^= value << 5U;
  return value;
}

struct ProjectedBlock final {
  std::uint64_t first_id = 0U;
  std::uint32_t dimension = 0U;
  std::uint32_t count = 0U;
  std::vector<float> residuals;
  std::vector<float> source_squared_norms;
  std::vector<float> projected_squared_norms;
};

template <typename Scalar>
StatusOr<ProjectedBlock> ProjectBlock(
    const BasicInputBlock<Scalar>& block,
    ArrayView<const ColumnMajorMatrixF32View> prior_codebooks,
    std::uint32_t layer_count,
    const RvqAssignmentStore& emitted_codes,
    const ParallelForExecutor& parallel) {
  if (block.count == 0U || block.dimension == 0U ||
      block.values.size() !=
          static_cast<std::size_t>(block.count) * block.dimension) {
    return Status::DataLoss(
        "CPU spherical RVQ source produced an invalid block");
  }
  ProjectedBlock result;
  result.first_id = block.first_id;
  result.dimension = block.dimension;
  result.count = block.count;
  result.residuals.resize(
      static_cast<std::size_t>(block.count) * block.dimension);
  result.source_squared_norms.resize(block.count);
  result.projected_squared_norms.resize(block.count);
  struct Projection final {
    const BasicInputBlock<Scalar>& input;
    ArrayView<const ColumnMajorMatrixF32View> codebooks;
    std::uint32_t prior_layers;
    const RvqAssignmentStore& assignments;
    ProjectedBlock& output;
  } projection{block, prior_codebooks, layer_count, emitted_codes, result};
  // STLQ normalizes/projects independent sample columns in parallel. Keep
  // each column's arithmetic order and use the existing reducer tile size.
  parallel.Run(1U + (block.count - 1U) / kSampleTileColumns, &projection,
      [](void* opaque, std::uint32_t, std::uint32_t tile) noexcept {
    auto& state = *static_cast<Projection*>(opaque);
    const auto& input = state.input;
    auto& output = state.output;
    const auto codebooks = state.codebooks;
    const auto prior_layers = state.prior_layers;
    const auto& assignments = state.assignments;
    const std::uint32_t begin = tile * kSampleTileColumns;
    const std::uint32_t end = std::min(
        input.count, begin + kSampleTileColumns);
    for (std::uint32_t column = begin; column < end; ++column) {
      const Scalar* source = input.values.data() +
          static_cast<std::size_t>(column) * input.dimension;
      float* residual = output.residuals.data() +
          static_cast<std::size_t>(column) * input.dimension;
      float source_norm2 = 0.0F;
      for (std::uint32_t axis = 0U; axis < input.dimension; ++axis) {
        const float value = static_cast<float>(source[axis]);
        residual[axis] = value;
        source_norm2 += value * value;
      }
      output.source_squared_norms[column] = source_norm2;
      if (source_norm2 > 1.0e-12F) {
        const float inverse = 1.0F / std::sqrt(source_norm2);
        for (std::uint32_t axis = 0U; axis < input.dimension; ++axis) {
          residual[axis] *= inverse;
        }
      }
      const std::uint64_t row = input.first_id + column;
      for (std::uint32_t layer = 0U; layer < prior_layers; ++layer) {
        const float* center = codebooks[layer].column(
            assignments.code(layer, row));
        float dot = 0.0F;
        for (std::uint32_t axis = 0U; axis < input.dimension; ++axis) {
          dot += residual[axis] * center[axis];
        }
        for (std::uint32_t axis = 0U; axis < input.dimension; ++axis) {
          residual[axis] -= dot * center[axis];
        }
      }
      float projected_norm2 = 0.0F;
      for (std::uint32_t axis = 0U; axis < input.dimension; ++axis) {
        projected_norm2 += residual[axis] * residual[axis];
      }
      output.projected_squared_norms[column] = projected_norm2;
      if (projected_norm2 > 1.0e-12F) {
        const float inverse = 1.0F / std::sqrt(projected_norm2);
        for (std::uint32_t axis = 0U; axis < input.dimension; ++axis) {
          residual[axis] *= inverse;
        }
      }
    }
  });
  return result;
}

template <typename Scalar>
StatusOr<std::vector<float>> ReadSample(
    InitializationBasicBlockSourceFactory<Scalar>& source_factory,
    const SphericalRvqInitializationRequest& request,
    ArrayView<const ColumnMajorMatrixF32View> codebooks,
    std::uint32_t layer,
    const RvqAssignmentStore& assignments,
    std::uint32_t sample_count,
    std::mt19937& random,
    std::uint64_t* sample_start,
    const ParallelForExecutor& parallel) {
  *sample_start = 0U;
  if (request.source_rows > sample_count) {
    std::uniform_int_distribution<std::int64_t> start(
        0, static_cast<std::int64_t>(request.source_rows - sample_count));
    *sample_start = static_cast<std::uint64_t>(start(random));
  }
  auto source = source_factory.OpenRange(*sample_start, sample_count);
  if (!source.ok() && source.status().code() == StatusCode::kUnimplemented) {
    *sample_start = 0U;
    source = source_factory.OpenRange(0U, sample_count);
  }
  if (!source.ok()) return source.status();
  std::vector<float> sample(
      static_cast<std::size_t>(sample_count) * request.shape.dimension);
  std::uint64_t expected = *sample_start;
  std::uint32_t filled = 0U;
  while (filled < sample_count) {
    auto next = source.value()->Next();
    if (!next.ok()) return next.status();
    if (!next.value().has_value()) {
      return Status::DataLoss(
          "CPU kmeans|| source ended before its sample window");
    }
    BasicInputBlock<Scalar>& block = *next.value();
    if (block.first_id != expected ||
        block.dimension != request.shape.dimension) {
      return Status::DataLoss(
          "CPU kmeans|| source order or dimension changed");
    }
    const std::uint32_t take =
        std::min(block.count, sample_count - filled);
    block.count = take;
    block.values.resize(
        static_cast<std::size_t>(take) * block.dimension);
    auto projected = ProjectBlock<Scalar>(
        block, codebooks, layer, assignments, parallel);
    if (!projected.ok()) return projected.status();
    std::copy(
        projected.value().residuals.begin(),
        projected.value().residuals.end(),
        sample.begin() +
            static_cast<std::size_t>(filled) * request.shape.dimension);
    filled += take;
    expected += take;
  }
  return sample;
}

void UpdateBestDots(
    ColumnMajorMatrixF32View points,
    ArrayView<const std::uint32_t> candidate_ids,
    std::vector<float>* best,
    const ParallelForExecutor& parallel) {
  struct DotUpdate final {
    ColumnMajorMatrixF32View sample;
    ArrayView<const std::uint32_t> candidates;
    float* best;
  } update{points, candidate_ids, best->data()};
  // Same sample-wise parallel boundary as STLQ's k-means++ dot update.
  // Sampling, phi accumulation and candidate order remain sequential.
  parallel.Run(1U + (points.columns - 1U) / kSampleTileColumns, &update,
      [](void* opaque, std::uint32_t, std::uint32_t tile) noexcept {
    auto& state = *static_cast<DotUpdate*>(opaque);
    const auto sample = state.sample;
    const auto candidates = state.candidates;
    const std::uint32_t begin = tile * kSampleTileColumns;
    const std::uint32_t end = std::min(
        sample.columns, begin + kSampleTileColumns);
    for (std::uint32_t column = begin; column < end; ++column) {
      const float* value = sample.column(column);
      float current = state.best[column];
      for (const std::uint32_t candidate : candidates) {
        const float* center = sample.column(candidate);
        float dot = 0.0F;
        for (std::uint32_t axis = 0U; axis < sample.rows; ++axis) {
          dot += value[axis] * center[axis];
        }
        current = std::max(current, dot);
      }
      state.best[column] = current;
    }
  });
}

struct CpuSeedResult final {
  std::vector<float> centers;
  std::vector<float> fallback;
  std::uint64_t sample_start = 0U;
  std::uint32_t rounds = 0U;
  std::uint32_t candidates = 0U;
  std::uint32_t picked_raw = 0U;
  std::uint32_t picked_new = 0U;
  std::uint32_t unique_candidates = 0U;
  std::uint32_t padded_candidates = 0U;
  std::uint32_t capped_candidates = 0U;
  double sample_seconds = 0.0;
  double candidate_seconds = 0.0;
  double reduction_seconds = 0.0;
  double fallback_seconds = 0.0;
};

template <typename Scalar>
StatusOr<CpuSeedResult> SeedLayer(
    InitializationBasicBlockSourceFactory<Scalar>& source_factory,
    const SphericalRvqInitializationRequest& request,
    ArrayView<const ColumnMajorMatrixF32View> codebooks,
    std::uint32_t layer,
    const RvqAssignmentStore& assignments,
    std::mt19937& random,
    const ParallelForExecutor& parallel) {
  const std::uint64_t sample64 =
      request.recipe.EffectiveSampleVectors(request.source_rows);
  if (sample64 == 0U ||
      sample64 > static_cast<std::uint64_t>(
          std::numeric_limits<int>::max())) {
    return Status::InvalidArgument(
        "CPU kmeans|| sample exceeds the registered index domain");
  }
  const std::uint32_t sample_count = static_cast<std::uint32_t>(sample64);
  const std::uint32_t centers =
      layer == 0U ? request.shape.nlist : 256U;
  const std::uint32_t candidate_cap =
      request.recipe.EffectiveCandidateCap(centers, sample_count);
  if (request.recipe.seed_method == SphericalSeedMethod::kKmeansParallel) {
    LogSphericalRvqSeedStart({
      .gpu = false,
      .sample_count = sample_count,
      .center_count = centers,
      .requested_rounds = request.recipe.kmeans_parallel_rounds,
      .oversample_factor = request.recipe.kmeans_parallel_oversample,
      .candidate_cap = candidate_cap,
    });
  }
  const std::uint64_t base_seed =
      request.recipe.seed_method != SphericalSeedMethod::kKmeansParallel
      ? 0U : request.recipe.kmeans_parallel_seed == 0U
      ? (static_cast<std::uint64_t>(random()) << 32U) ^ random()
      : request.recipe.kmeans_parallel_seed;

  CpuSeedResult output;
  const auto sample_started = std::chrono::steady_clock::now();
  auto sample = ReadSample<Scalar>(
      source_factory, request, codebooks, layer, assignments, sample_count,
      random, &output.sample_start, parallel);
  if (!sample.ok()) return sample.status();
  output.sample_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - sample_started).count();
  const ColumnMajorMatrixF32View sample_view{
      sample.value().data(), request.shape.dimension, sample_count};

  if (request.recipe.seed_method != SphericalSeedMethod::kKmeansParallel) {
    const auto seed_begin = std::chrono::steady_clock::now();
    if (request.recipe.seed_method == SphericalSeedMethod::kKmeansPlusPlus) {
      auto seeded = ReduceSphericalKmeansPlusPlus(
          sample_view, centers, random, parallel);
      if (!seeded.ok()) return seeded.status();
      output.centers = std::move(seeded).value();
      output.candidates = sample_count;
      output.unique_candidates = sample_count;
      output.reduction_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - seed_begin).count();
    } else {
      std::vector<std::uint32_t> indices(sample_count);
      std::iota(indices.begin(), indices.end(), 0U);
      std::shuffle(indices.begin(), indices.end(), random);
      output.centers.resize(
          static_cast<std::size_t>(request.shape.dimension) * centers);
      for (std::uint32_t center = 0U; center < centers; ++center) {
        std::copy_n(sample_view.column(indices[center % sample_count]),
                    request.shape.dimension,
                    output.centers.data() +
                        static_cast<std::size_t>(center) * request.shape.dimension);
      }
      output.candidates = std::min(centers, sample_count);
      output.unique_candidates = output.candidates;
      output.candidate_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - seed_begin).count();
    }
    std::uniform_int_distribution<int> fallback_index(
        0, static_cast<int>(sample_count - 1U));
    output.fallback.resize(
        static_cast<std::size_t>(request.shape.dimension) * centers);
    for (std::uint32_t center = 0U; center < centers; ++center) {
      std::copy_n(sample_view.column(static_cast<std::uint32_t>(
                      fallback_index(random))), request.shape.dimension,
                  output.fallback.data() +
                      static_cast<std::size_t>(center) * request.shape.dimension);
    }
    return output;
  }

  const auto candidate_started = std::chrono::steady_clock::now();
  std::uniform_int_distribution<int> sample_index(
      0, static_cast<int>(sample_count - 1U));
  std::vector<std::uint32_t> selected{
      static_cast<std::uint32_t>(sample_index(random))};
  std::vector<std::uint8_t> seen(sample_count, 0U);
  seen[selected.front()] = 1U;
  output.picked_raw = 1U;
  output.picked_new = 1U;
  std::vector<float> best(
      sample_count, -std::numeric_limits<float>::infinity());
  UpdateBestDots(sample_view, selected, &best, parallel);
  const float oversample =
      request.recipe.kmeans_parallel_oversample *
      static_cast<float>(centers);
  for (std::uint32_t round = 0U;
       round < request.recipe.kmeans_parallel_rounds &&
       selected.size() < candidate_cap;
       ++round) {
    std::vector<float> distances(sample_count);
    float phi = 0.0F;
    for (std::uint32_t row = 0U; row < sample_count; ++row) {
      const float dot = std::min(1.0F, std::max(-1.0F, best[row]));
      distances[row] = std::max(0.0F, 2.0F - 2.0F * dot);
      phi += distances[row];
    }
    if (!(phi > 0.0F)) break;
    std::vector<std::uint32_t> added;
    const std::uint64_t round_seed = base_seed + kRoundIncrement * round;
    for (std::uint32_t row = 0U;
         row < sample_count && selected.size() < candidate_cap; ++row) {
      const float probability = std::min(
          1.0F, std::max(0.0F, oversample * distances[row] / phi));
      std::uint32_t value = static_cast<std::uint32_t>(
          round_seed ^ (static_cast<std::uint64_t>(row) * kRoundIncrement));
      value = Xorshift32(value);
      const float uniform =
          (static_cast<float>(value) + 1.0F) / 4294967296.0F;
      if (probability > 0.0F && uniform < probability) {
        ++output.picked_raw;
        if (seen[row] != 0U) continue;
        seen[row] = 1U;
        selected.push_back(row);
        added.push_back(row);
        ++output.picked_new;
      }
    }
    output.rounds = round + 1U;
    if (added.empty()) break;
    UpdateBestDots(sample_view, added, &best, parallel);
  }
  output.unique_candidates = static_cast<std::uint32_t>(selected.size());
  while (selected.size() < centers) {
    selected.push_back(static_cast<std::uint32_t>(sample_index(random)));
  }
  output.padded_candidates = static_cast<std::uint32_t>(
      selected.size() - output.unique_candidates);
  const std::size_t before_cap = selected.size();
  if (selected.size() > candidate_cap) selected.resize(candidate_cap);
  output.capped_candidates = static_cast<std::uint32_t>(
      before_cap - selected.size());
  output.candidates = static_cast<std::uint32_t>(selected.size());
  LogSphericalRvqSeedFinish({
      .center_count = centers,
      .rounds_ran = output.rounds,
      .picked_raw = output.picked_raw,
      .picked_new = output.picked_new,
      .unique_candidates = output.unique_candidates,
      .padded_candidates = output.padded_candidates,
      .capped_candidates = output.capped_candidates,
      .reducer_input_candidates = output.candidates,
  });
  output.candidate_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - candidate_started).count();

  const auto reduction_started = std::chrono::steady_clock::now();
  std::vector<float> candidates(
      static_cast<std::size_t>(request.shape.dimension) * selected.size());
  for (std::size_t index = 0U; index < selected.size(); ++index) {
    std::copy_n(
        sample_view.column(selected[index]), request.shape.dimension,
        candidates.data() + index * request.shape.dimension);
  }
  auto reduced = ReduceSphericalKmeansPlusPlus(
      {candidates.data(), request.shape.dimension,
       static_cast<std::uint32_t>(selected.size())},
      centers, random, parallel);
  if (!reduced.ok()) return reduced.status();
  output.centers = std::move(reduced).value();
  output.reduction_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - reduction_started).count();

  const auto fallback_started = std::chrono::steady_clock::now();
  output.fallback.resize(
      static_cast<std::size_t>(request.shape.dimension) * centers);
  for (std::uint32_t center = 0U; center < centers; ++center) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(sample_index(random));
    std::copy_n(
        sample_view.column(row), request.shape.dimension,
        output.fallback.data() +
            static_cast<std::size_t>(center) * request.shape.dimension);
  }
  output.fallback_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - fallback_started).count();
  return output;
}

struct AssignmentContext final {
  ColumnMajorMatrixF32View residuals;
  ColumnMajorMatrixF32View centers;
  ColumnMajorMatrixF32View coarse_centers;
  SphericalHierarchicalLayout layout;
  std::uint32_t top_coarse = 1U;
  std::uint32_t* codes = nullptr;
  float* best_dots = nullptr;
};

void AssignOne(void* opaque, std::uint32_t, std::uint32_t row) noexcept {
  auto& context = *static_cast<AssignmentContext*>(opaque);
  const float* residual = context.residuals.column(row);
  std::uint32_t best_code = 0U;
  float best_dot = -std::numeric_limits<float>::infinity();
  const auto dot_center = [&](const float* value) {
    float dot = 0.0F;
    for (std::uint32_t axis = 0U; axis < context.centers.rows; ++axis) {
      dot += residual[axis] * value[axis];
    }
    return dot;
  };
  const auto score_group = [&](std::uint32_t group) {
    const std::uint32_t begin = context.layout.group_begin(group);
    const std::uint32_t end = begin + context.layout.group_size(group);
    for (std::uint32_t center = begin; center < end; ++center) {
      const float dot = dot_center(context.centers.column(center));
      if (dot > best_dot || (dot == best_dot && center < best_code)) {
        best_dot = dot;
        best_code = center;
      }
    }
  };
  if (context.coarse_centers.values == nullptr) {
    for (std::uint32_t center = 0U; center < context.centers.columns;
         ++center) {
      const float dot = dot_center(context.centers.column(center));
      if (dot > best_dot) {
        best_dot = dot;
        best_code = center;
      }
    }
  } else {
    std::uint32_t first = 0U;
    std::uint32_t second = 0U;
    float first_dot = -std::numeric_limits<float>::infinity();
    float second_dot = -std::numeric_limits<float>::infinity();
    for (std::uint32_t coarse = 0U;
         coarse < context.layout.non_empty_coarse_count(); ++coarse) {
      const float dot = dot_center(context.coarse_centers.column(coarse));
      if (dot > first_dot || (dot == first_dot && coarse < first)) {
        second = first;
        second_dot = first_dot;
        first = coarse;
        first_dot = dot;
      } else if (coarse != first &&
                 (dot > second_dot ||
                  (dot == second_dot && coarse < second))) {
        second = coarse;
        second_dot = dot;
      }
    }
    score_group(first);
    if (context.top_coarse == 2U && second != first) {
      score_group(second);
    }
  }
  context.codes[row] = best_code;
  context.best_dots[row] = best_dot;
}

struct FinalizeContext final {
  const float* sums = nullptr;
  const float* center_weights = nullptr;
  const float* fallback = nullptr;
  std::uint32_t dimension = 0U;
  float* centers = nullptr;
};

struct AccumulateContext final {
  ColumnMajorMatrixF32View residuals;
  ArrayView<const std::uint32_t> offsets;
  ArrayView<const std::uint32_t> order;
  const float* weights = nullptr;
  float* sums = nullptr;
  float* center_weights = nullptr;
};

void AccumulateCenter(void* opaque, std::uint32_t,
                      std::uint32_t center) noexcept {
  auto& state = *static_cast<AccumulateContext*>(opaque);
  float* sum = state.sums +
      static_cast<std::size_t>(center) * state.residuals.rows;
  float weight_sum = state.center_weights[center];
  // STLQ groups rows by assignment and parallelizes across centers. Append
  // rows directly to the running sums, preserving PP's order across blocks.
  for (std::uint32_t slot = state.offsets[center];
       slot < state.offsets[center + 1U]; ++slot) {
    const std::uint32_t row = state.order[slot];
    const float weight = state.weights[row];
    weight_sum += weight;
    const float* residual = state.residuals.column(row);
    for (std::uint32_t axis = 0U; axis < state.residuals.rows; ++axis) {
      sum[axis] += weight * residual[axis];
    }
  }
  state.center_weights[center] = weight_sum;
}

void FinalizeOne(void* opaque, std::uint32_t, std::uint32_t center) noexcept {
  auto& context = *static_cast<FinalizeContext*>(opaque);
  const std::size_t offset =
      static_cast<std::size_t>(center) * context.dimension;
  if (context.center_weights[center] <= kNormalizationEpsilon) {
    std::copy_n(
        context.fallback + offset, context.dimension,
        context.centers + offset);
    return;
  }
  const float inverse_weight = 1.0F / context.center_weights[center];
  float norm2 = 0.0F;
  for (std::uint32_t axis = 0U; axis < context.dimension; ++axis) {
    const float value = context.sums[offset + axis] * inverse_weight;
    context.centers[offset + axis] = value;
    norm2 += value * value;
  }
  const float inverse_norm =
      1.0F / std::sqrt(std::max(norm2, kNormalizationEpsilon));
  for (std::uint32_t axis = 0U; axis < context.dimension; ++axis) {
    context.centers[offset + axis] *= inverse_norm;
  }
}

template <typename Scalar, typename WriteCodes>
StatusOr<std::pair<std::vector<float>, SphericalRvqLayerStatistics>>
TrainLayer(
    InitializationBasicBlockSourceFactory<Scalar>& source_factory,
    const SphericalRvqInitializationRequest& request,
    ArrayView<const ColumnMajorMatrixF32View> codebooks,
    std::uint32_t layer,
    RvqAssignmentStore* assignments,
    CpuSeedResult seed,
    const ParallelForExecutor& parallel,
    bool collect_metrics,
    const WriteCodes& write_codes) {
  const std::uint32_t dimension = request.shape.dimension;
  const std::uint32_t center_count =
      layer == 0U ? request.shape.nlist : 256U;
  std::vector<float> centers = std::move(seed.centers);
  std::vector<float> weights(
      static_cast<std::size_t>(request.source_rows),
      request.recipe.initial_weight);
  double previous_cost = std::numeric_limits<double>::infinity();
  float previous_threshold = request.recipe.cost_threshold;
  double final_cost = previous_cost;
  double final_mse = collect_metrics ? 0.0 : -1.0;
  std::uint32_t iterations = 0U;
  const bool retain_codes = layer + 1U < request.shape.m;
  const bool use_hierarchy = request.recipe.large_layer_hierarchy_enabled &&
      center_count >= request.recipe.large_layer_threshold;
  const SphericalHierarchicalLayout layout = use_hierarchy
      ? MakeSphericalHierarchicalLayout(
            center_count,
            {.fixed_coarse_count =
                 request.recipe.large_layer_fixed_coarse_count,
             .minimum_coarse_count =
                 request.recipe.large_layer_minimum_coarse_count,
             .maximum_coarse_count =
                 request.recipe.large_layer_maximum_coarse_count,
             .round_to_power_of_two =
                 request.recipe.large_layer_round_coarse_to_power_of_two})
      : SphericalHierarchicalLayout{};
  SphericalLloydCodeEmission emission(retain_codes);
  std::vector<std::uint32_t> offsets(center_count + 1U);
  std::vector<std::uint32_t> cursor(center_count);
  std::vector<std::uint32_t> order;
  const auto started = std::chrono::steady_clock::now();

  for (std::uint32_t iteration = 1U;
       iteration <= request.recipe.maximum_iterations; ++iteration) {
    std::vector<float> coarse;
    if (use_hierarchy) {
      auto built = BuildSphericalCoarseCenters(
          {centers.data(), dimension, center_count}, layout, parallel);
      if (!built.ok()) return built.status();
      coarse = std::move(built).value();
    }
    std::vector<float> sums(
        static_cast<std::size_t>(center_count) * dimension, 0.0F);
    std::vector<float> center_weights(center_count, 0.0F);
    std::vector<std::uint32_t> histogram(
        request.recipe.quantile_bins, 0U);
    double total_weight = 0.0;
    double weighted_cost = 0.0;
    double mse_sum = 0.0;
    std::uint64_t row_count = 0U;
    const bool emit = emission.BeginIteration(
        iteration, request.recipe.maximum_iterations);
    const bool adaptive = iteration > request.recipe.warmup_iterations;
    const float denominator = static_cast<float>(
        std::max<std::uint32_t>(
            1U, request.recipe.maximum_iterations -
                    request.recipe.warmup_iterations));
    const float annealing = adaptive
        ? request.recipe.annealing_factor *
              (1.0F - std::exp(
                  -static_cast<float>(
                      iteration - request.recipe.warmup_iterations) /
                  denominator))
        : 0.0F;
    const float threshold =
        std::max(previous_threshold, request.recipe.cost_threshold);
    auto source = source_factory.Open();
    if (!source.ok()) return source.status();
    std::uint64_t expected = 0U;
    while (true) {
      auto next = source.value()->Next();
      if (!next.ok()) return next.status();
      if (!next.value().has_value()) break;
      BasicInputBlock<Scalar>& input = *next.value();
      if (input.first_id != expected || input.dimension != dimension) {
        return Status::DataLoss(
            "CPU spherical Lloyd source order or dimension changed");
      }
      auto projected = ProjectBlock<Scalar>(
          input, codebooks, layer, *assignments, parallel);
      if (!projected.ok()) return projected.status();
      std::vector<std::uint32_t> codes(input.count);
      std::vector<float> best_dots(input.count);
      AssignmentContext assignment{
          .residuals = {
              projected.value().residuals.data(), dimension, input.count},
          .centers = {centers.data(), dimension, center_count},
          .coarse_centers = {
              coarse.empty() ? nullptr : coarse.data(), dimension,
              layout.coarse_count},
          .layout = layout,
          .top_coarse = request.recipe.large_layer_top_coarse,
          .codes = codes.data(),
          .best_dots = best_dots.data(),
      };
      parallel.Run(input.count, &assignment, AssignOne);
      if (emit) {
        if (const Status status = write_codes(layer, input.first_id, codes);
            !status.ok()) {
          return status;
        }
      }
      std::fill(offsets.begin(), offsets.end(), 0U);
      for (std::uint32_t row = 0U; row < input.count; ++row) {
        const std::size_t global =
            static_cast<std::size_t>(input.first_id + row);
        const float cost = 1.0F - best_dots[row];
        const float bounded = std::min(2.0F, std::max(0.0F, cost));
        std::uint32_t bin = static_cast<std::uint32_t>(
            bounded *
            (0.5F * static_cast<float>(request.recipe.quantile_bins)));
        bin = std::min(request.recipe.quantile_bins - 1U, bin);
        ++histogram[bin];
        float weight = weights[global];
        if (adaptive) {
          weight = cost > threshold
              ? std::max(
                    request.recipe.minimum_weight,
                    weight * (1.0F - annealing))
              : std::min(1.0F, weight * (1.0F + 0.5F * annealing));
          weights[global] = weight;
        }
        total_weight += weight;
        weighted_cost += static_cast<double>(weight) * cost;
        const std::uint32_t center = codes[row];
        ++offsets[center + 1U];
        if (collect_metrics) {
          const double dot = best_dots[row];
          mse_sum += projected.value().source_squared_norms[row] *
              projected.value().projected_squared_norms[row] *
              (1.0 - dot * dot);
        }
      }
      std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
      std::copy_n(offsets.data(), center_count, cursor.data());
      order.resize(input.count);
      for (std::uint32_t row = 0U; row < input.count; ++row) {
        order[cursor[codes[row]]++] = row;
      }
      AccumulateContext accumulation{
          .residuals = {projected.value().residuals.data(), dimension,
                        input.count},
          .offsets = offsets,
          .order = order,
          .weights = weights.data() +
              static_cast<std::size_t>(input.first_id),
          .sums = sums.data(),
          .center_weights = center_weights.data(),
      };
      parallel.Run(center_count, &accumulation, AccumulateCenter);
      expected += input.count;
      row_count += input.count;
    }
    if (row_count != request.source_rows) {
      return Status::DataLoss(
          "CPU spherical Lloyd source row count changed");
    }
    if (emit) {
      if (const Status status = assignments->FinishLayer(layer);
          !status.ok()) {
        return status;
      }
    }
    std::vector<float> next_centers(
        static_cast<std::size_t>(center_count) * dimension);
    FinalizeContext finalize{
        .sums = sums.data(),
        .center_weights = center_weights.data(),
        .fallback = seed.fallback.data(),
        .dimension = dimension,
        .centers = next_centers.data(),
    };
    parallel.Run(center_count, &finalize, FinalizeOne);
    centers = std::move(next_centers);
    const std::uint64_t quantile_target = static_cast<std::uint64_t>(
        request.recipe.outlier_quantile *
        static_cast<float>(std::max<std::uint64_t>(
            1U, request.source_rows - 1U)));
    std::uint64_t cumulative = 0U;
    std::uint32_t selected_bin = request.recipe.quantile_bins - 1U;
    for (std::uint32_t bin = 0U; bin < request.recipe.quantile_bins; ++bin) {
      cumulative += histogram[bin];
      if (cumulative > quantile_target) {
        selected_bin = bin;
        break;
      }
    }
    if (adaptive) {
      previous_threshold =
          2.0F * (static_cast<float>(selected_bin) + 0.5F) /
          static_cast<float>(request.recipe.quantile_bins);
    }
    final_cost =
        total_weight > 0.0 ? weighted_cost / total_weight : 0.0;
    final_mse =
        collect_metrics && row_count != 0U
        ? mse_sum / static_cast<double>(row_count) : -1.0;
    iterations = iteration;
    const bool converged =
        std::abs(previous_cost - final_cost) < request.recipe.tolerance;
    if (emission.EndIteration(emit, converged)) break;
    previous_cost = final_cost;
  }
  if (retain_codes &&
      (!emission.complete() || !assignments->complete(layer))) {
    return Status::Internal(
        "CPU spherical Lloyd did not publish retained assignments");
  }
  SphericalRvqLayerStatistics statistics{
      .iterations = iterations,
      .final_cost = final_cost,
      .final_mse_proxy = final_mse,
      .seeding_wall_seconds =
          seed.sample_seconds + seed.candidate_seconds +
          seed.reduction_seconds + seed.fallback_seconds,
      .seed_sample_wall_seconds = seed.sample_seconds,
      .seed_candidate_wall_seconds = seed.candidate_seconds,
      .seed_reduction_wall_seconds = seed.reduction_seconds,
      .seed_fallback_wall_seconds = seed.fallback_seconds,
      .seed_rounds = seed.rounds,
      .seed_candidates = seed.candidates,
      .normalized_cache_wall_seconds = 0.0,
      .wall_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count(),
  };
  return std::make_pair(std::move(centers), statistics);
}

}  // namespace

template <typename Scalar>
StatusOr<SphericalRvqBackendResult>
CpuSphericalRvqInitializationBackend<Scalar>::Initialize(
    InitializationBasicBlockSourceFactory<Scalar>& source,
    const SphericalRvqInitializationRequest& request) {
  if (const Status status = request.recipe.Validate(); !status.ok()) {
    return status;
  }
  if (const Status status = request.shape.Validate(); !status.ok()) {
    return status;
  }
  if (request.source_rows == 0U) {
    return Status::InvalidArgument(
        "CPU spherical RVQ backend request is invalid");
  }
  auto assignment_result =
      RvqAssignmentStore::Create(request.shape, request.source_rows);
  if (!assignment_result.ok()) return assignment_result.status();
  RvqAssignmentStore assignments =
      std::move(assignment_result).value();
  const auto write_codes = [&assignments](
                               std::uint32_t layer,
                               std::uint64_t first_id,
                               const std::vector<std::uint32_t>& codes) {
    return assignments.WriteCpuAssignmentBlock(layer, first_id, codes);
  };
  auto owned = std::vector<std::vector<float>>(request.shape.m);
  auto views = std::vector<ColumnMajorMatrixF32View>(request.shape.m);
  auto statistics = std::vector<SphericalRvqLayerStatistics>(request.shape.m);
  std::mt19937 random(request.seed);
  for (std::uint32_t layer = 0U; layer < request.shape.m;
       ++layer) {
    auto seeded = SeedLayer<Scalar>(
        source, request, views, layer, assignments, random, *parallel_);
    if (!seeded.ok()) return seeded.status();
    auto trained = TrainLayer<Scalar>(
        source, request, views, layer, &assignments,
        std::move(seeded).value(), *parallel_, options_.collect_metrics,
        write_codes);
    if (!trained.ok()) return trained.status();
    owned[layer] = std::move(trained.value().first);
    statistics[layer] = trained.value().second;
    views[layer] = {
        owned[layer].data(), request.shape.dimension,
        layer == 0U ? request.shape.nlist : 256U};
  }
  auto small = std::vector<std::vector<float>>(request.shape.m - 1U);
  for (std::uint32_t layer = 1U; layer < request.shape.m;
       ++layer) {
    small[layer - 1U] = std::move(owned[layer]);
  }
  return SphericalRvqBackendResult{
      .codebooks = InitializedRootCodebooks(
          request.shape, std::move(owned[0]), std::move(small)),
      .layers = std::move(statistics),
      .cuda_diagnostics = std::nullopt,
  };
}

template class CpuSphericalRvqInitializationBackend<std::uint8_t>;
template class CpuSphericalRvqInitializationBackend<float>;

}  // namespace stlqpp
