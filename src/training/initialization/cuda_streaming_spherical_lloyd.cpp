#include "stlqpp/training/initialization/cuda_streaming_spherical_lloyd.h"

#include "stlqpp/training/initialization/cuda_normalized_source_cache.h"
#include "stlqpp/training/initialization/cuda_rvq_assignment_publisher.h"
#include "stlqpp/training/initialization/cuda_spherical_accumulator.h"
#include "stlqpp/training/initialization/spherical_lloyd_code_emission.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

struct PassResult final {
  CudaSphericalIterationResult update;
  std::uint64_t rows = 0U;
};

template <typename Scalar>
StatusOr<PassResult> RunPass(
    CudaContext& context,
    CudaNormalizedSourceCache<Scalar>& source,
    CudaSphericalAssignment<Scalar>& assignment,
    CudaSphericalAccumulator& accumulator,
    RvqAssignmentStore& retained,
    std::uint32_t layer,
    bool emit_codes,
    bool update_weights,
    float threshold,
    float annealing_factor,
    const SphericalRvqInitializationRequest& request,
    ColumnMajorMatrixF32View empty_cluster_fallback) {
  if (const Status status = accumulator.BeginIteration(); !status.ok()) {
    return status;
  }
  auto source_pass = source.OpenPass(layer != 0U);
  if (!source_pass.ok()) {
    return source_pass.status();
  }
  std::optional<CudaRvqAssignmentPublisher> publisher;
  if (emit_codes) {
    auto created = CudaRvqAssignmentPublisher::Create(
        context, source_pass.value().native_upload_stream(),
        retained, layer);
    if (!created.ok()) {
      return created.status();
    }
    publisher.emplace(std::move(created).value());
  }
  auto PrepareNext = [&]()
      -> StatusOr<std::optional<CudaNormalizedSourceBlock>> {
    auto next = source_pass.value().Next();
    if (!next.ok()) {
      return next.status();
    }
    if (!next.value().has_value()) {
      return std::optional<CudaNormalizedSourceBlock>{};
    }
    const CudaNormalizedSourceBlock& block = *next.value();
    if (const Status status = assignment.StagePriorCodes(
            block.first_id, block.normalized.columns, block.slot,
            source_pass.value().native_upload_stream()); !status.ok()) {
      return status;
    }
    if (const Status status = source_pass.value().Publish(block);
        !status.ok()) {
      return status;
    }
    return next.value();
  };

  auto current = PrepareNext();
  if (!current.ok()) {
    return current.status();
  }
  std::uint64_t rows = 0U;
  while (current.value().has_value()) {
    auto following = PrepareNext();
    if (!following.ok()) {
      return following.status();
    }
    const CudaNormalizedSourceBlock block = *current.value();
    assert(block.first_id == rows);
    if (const Status status = source_pass.value().WaitReady(block);
        !status.ok()) {
      return status;
    }
    auto residuals = assignment.ProjectPriorInPlace(
        block.normalized, block.slot);
    if (!residuals.ok()) {
      return residuals.status();
    }
    auto device = assignment.AssignPreparedDevice(residuals.value().residuals);
    if (!device.ok()) {
      return device.status();
    }
    if (publisher.has_value()) {
      if (const Status status = publisher->Schedule(
              block.slot, block.first_id, device.value().codes,
              device.value().count); !status.ok()) {
        return status;
      }
    }
    if (const Status status = accumulator.Accumulate(
            device.value(), block.first_id, update_weights, threshold,
            annealing_factor, request.recipe.minimum_weight,
            block.source_squared_norms,
            residuals.value().squared_norms);
        !status.ok()) {
      return status;
    }
    if (const Status status = source_pass.value().MarkConsumed(block);
        !status.ok()) {
      return status;
    }
    rows += block.normalized.columns;
    current = std::move(following);
  }
  if (const Status status = source_pass.value().Finish(); !status.ok()) {
    return status;
  }
  assert(rows == request.source_rows);
  if (publisher.has_value()) {
    if (const Status status = publisher->Drain(); !status.ok()) {
      return status;
    }
  }
  auto update = accumulator.FinishIteration(
      empty_cluster_fallback, request.recipe.outlier_quantile);
  if (!update.ok()) {
    return update.status();
  }
  if (emit_codes) {
    if (const Status status = retained.FinishLayer(layer); !status.ok()) {
      return status;
    }
  }
  return PassResult{
      .update = std::move(update).value(),
      .rows = rows,
  };
}

}  // namespace

template <typename Scalar>
StatusOr<CudaStreamingSphericalLloydResult>
CudaStreamingSphericalLloyd<Scalar>::Train(
    CudaContext& context,
    const ParallelForExecutor& parallel,
    InitializationBasicBlockSourceFactory<Scalar>& source,
    const SphericalRvqInitializationRequest& request,
    ArrayView<const ColumnMajorMatrixF32View> prior_codebooks,
    std::uint32_t layer,
    RvqAssignmentStore& assignments,
    ColumnMajorMatrixF32View initial_centers,
    ColumnMajorMatrixF32View empty_cluster_fallback,
    CudaStreamingSphericalLloydOptions options) {
  const auto cache_started = std::chrono::steady_clock::now();
  auto normalized_source = CudaNormalizedSourceCache<Scalar>::Build(
      context, source,
      {.source_rows = request.source_rows,
       .dimension = request.shape.dimension,
       .maximum_bytes = options.maximum_normalized_cache_bytes});
  if (!normalized_source.ok()) {
    return normalized_source.status();
  }
  const double normalized_cache_wall_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - cache_started).count();
  return TrainWithNormalizedSource(
      context, parallel, normalized_source.value(), request,
      prior_codebooks, layer, assignments, initial_centers,
      empty_cluster_fallback, options, normalized_cache_wall_seconds);
}

template <typename Scalar>
StatusOr<CudaStreamingSphericalLloydResult>
CudaStreamingSphericalLloyd<Scalar>::TrainWithNormalizedSource(
    CudaContext& context,
    const ParallelForExecutor& parallel,
    CudaNormalizedSourceCache<Scalar>& normalized_source,
    const SphericalRvqInitializationRequest& request,
    ArrayView<const ColumnMajorMatrixF32View> prior_codebooks,
    std::uint32_t layer,
    RvqAssignmentStore& assignments,
    ColumnMajorMatrixF32View initial_centers,
    ColumnMajorMatrixF32View empty_cluster_fallback,
    CudaStreamingSphericalLloydOptions options,
    double normalized_cache_wall_seconds) {
  if (const Status status = request.recipe.Validate(); !status.ok()) {
    return status;
  }
  const std::uint32_t expected_centers = layer == 0U ? request.shape.nlist : 256U;
  if (layer >= request.shape.m ||
      request.source_rows != assignments.row_count() ||
      normalized_source.row_count() != request.source_rows ||
      initial_centers.values == nullptr ||
      initial_centers.rows != request.shape.dimension ||
      initial_centers.columns != expected_centers ||
      empty_cluster_fallback.values == nullptr ||
      empty_cluster_fallback.rows != request.shape.dimension ||
      empty_cluster_fallback.columns != expected_centers ||
      options.maximum_score_bytes < sizeof(float)) {
    return Status::InvalidArgument(
        "CUDA streaming spherical Lloyd request is invalid");
  }
  auto assignment = CudaSphericalAssignment<Scalar>::Create(
      context, request.shape, prior_codebooks, layer,
      layer == 0U ? nullptr : &assignments,
      {.maximum_score_bytes = options.maximum_score_bytes,
       .use_registered_hierarchy =
           request.recipe.large_layer_hierarchy_enabled &&
           expected_centers >= request.recipe.large_layer_threshold,
       .hierarchical_top_coarse =
           request.recipe.large_layer_top_coarse,
       .hierarchical_split = {
           .fixed_coarse_count =
               request.recipe.large_layer_fixed_coarse_count,
           .minimum_coarse_count =
               request.recipe.large_layer_minimum_coarse_count,
           .maximum_coarse_count =
               request.recipe.large_layer_maximum_coarse_count,
           .round_to_power_of_two =
               request.recipe.large_layer_round_coarse_to_power_of_two,
       },
       .parallel = &parallel});
  if (!assignment.ok()) {
    return assignment.status();
  }
  auto accumulator = CudaSphericalAccumulator::Create(
      context, parallel,
      {.source_rows = request.source_rows,
       .dimension = request.shape.dimension,
       .center_count = expected_centers,
       .quantile_bins = request.recipe.quantile_bins,
       .initial_weight = request.recipe.initial_weight,
       .collect_metrics = options.collect_metrics});
  if (!accumulator.ok()) {
    return accumulator.status();
  }

  std::vector<float> centers(
      initial_centers.values,
      initial_centers.values +
          static_cast<std::size_t>(initial_centers.rows) *
              initial_centers.columns);
  double previous_cost = std::numeric_limits<double>::infinity();
  float previous_threshold = request.recipe.cost_threshold;
  std::uint32_t completed_iterations = 0U;
  double final_cost = previous_cost;
  double final_mse_proxy = 0.0;
  const bool retain_layer_codes =
      layer + 1U < request.shape.m;
  SphericalLloydCodeEmission code_emission(retain_layer_codes);
  const auto started = std::chrono::steady_clock::now();

  for (std::uint32_t iteration = 1U;
       iteration <= request.recipe.maximum_iterations; ++iteration) {
    if (const Status status = assignment.value().SetCenters(
            {centers.data(), request.shape.dimension, expected_centers});
        !status.ok()) {
      return status;
    }
    const bool emit_codes_this_iteration =
        code_emission.BeginIteration(
            iteration, request.recipe.maximum_iterations);
    const bool adaptive = iteration > request.recipe.warmup_iterations;
    const float denominator = static_cast<float>(std::max<std::uint32_t>(
        1U, request.recipe.maximum_iterations -
                request.recipe.warmup_iterations));
    const float annealing = adaptive
        ? request.recipe.annealing_factor *
              (1.0F - std::exp(
                  -static_cast<float>(iteration -
                                      request.recipe.warmup_iterations) /
                  denominator))
        : 0.0F;
    const float threshold =
        std::max(previous_threshold, request.recipe.cost_threshold);
    auto pass = RunPass<Scalar>(
        context, normalized_source, assignment.value(), accumulator.value(),
        assignments, layer,
        emit_codes_this_iteration, adaptive,
        threshold, annealing, request,
        empty_cluster_fallback);
    if (!pass.ok()) {
      return pass.status();
    }
    auto pass_result = std::move(pass).value();
    final_cost = pass_result.update.mean_cost;
    final_mse_proxy = pass_result.update.mse_proxy;
    if (adaptive) {
      previous_threshold = pass_result.update.quantile_threshold;
    }
    centers = std::move(pass_result.update.centers);
    completed_iterations = iteration;
    const bool converged =
        std::abs(previous_cost - final_cost) < request.recipe.tolerance;
    if (code_emission.EndIteration(
            emit_codes_this_iteration, converged)) {
      break;
    }
    previous_cost = final_cost;
  }

  // The final layer intentionally has no retained code store. A non-final
  // layer reaches here only through the at-limit pass, which already closed it.
  if (retain_layer_codes &&
      (!code_emission.complete() || !assignments.complete(layer))) {
    return Status::Internal(
        "spherical Lloyd did not publish retained layer assignments");
  }
  const double wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  return CudaStreamingSphericalLloydResult{
      .centers = std::move(centers),
      .statistics = {
          .iterations = completed_iterations,
          .final_cost = final_cost,
          .final_mse_proxy = final_mse_proxy,
          .normalized_cache_wall_seconds =
              normalized_cache_wall_seconds,
          .wall_seconds = wall_seconds,
      },
  };
}

template class CudaStreamingSphericalLloyd<std::uint8_t>;
template class CudaStreamingSphericalLloyd<float>;

}  // namespace stlqpp
