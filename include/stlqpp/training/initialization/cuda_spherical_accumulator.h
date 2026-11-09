#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/training/initialization/cuda_spherical_assignment.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace stlqpp {

struct CudaSphericalAccumulatorOptions final {
  std::uint64_t source_rows = 0U;
  std::uint32_t dimension = 0U;
  std::uint32_t center_count = 0U;
  std::uint32_t quantile_bins = 0U;
  float initial_weight = 1.0F;
  bool collect_metrics = true;
};

struct CudaSphericalIterationResult final {
  std::vector<float> centers;
  double mean_cost = 0.0;
  double mse_proxy = 0.0;
  float quantile_threshold = 0.0F;
  double total_weight = 0.0;
};

// Device-resident one-pass annealing and center-update owner. Global sample
// weights persist across iterations; center moments and the quantile histogram
// are reset at each BeginIteration boundary.
class CudaSphericalAccumulator final {
 public:
  [[nodiscard]] static StatusOr<CudaSphericalAccumulator> Create(
      CudaContext& context,
      const ParallelForExecutor& parallel,
      CudaSphericalAccumulatorOptions options);

  ~CudaSphericalAccumulator();
  CudaSphericalAccumulator(CudaSphericalAccumulator&&) noexcept;
  CudaSphericalAccumulator& operator=(CudaSphericalAccumulator&&) noexcept;
  CudaSphericalAccumulator(const CudaSphericalAccumulator&) = delete;
  CudaSphericalAccumulator& operator=(const CudaSphericalAccumulator&) =
      delete;

  [[nodiscard]] Status BeginIteration();
  [[nodiscard]] Status Accumulate(
      const CudaSphericalDeviceAssignment& assigned,
      std::uint64_t first_id,
      bool update_weights,
      float effective_threshold,
      float annealing_factor,
      float minimum_weight,
      const float* source_squared_norms,
      const float* projected_squared_norms);
  [[nodiscard]] StatusOr<CudaSphericalIterationResult> FinishIteration(
      ColumnMajorMatrixF32View empty_cluster_fallback,
      float quantile);

 private:
  struct State;
  explicit CudaSphericalAccumulator(std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

}  // namespace stlqpp
