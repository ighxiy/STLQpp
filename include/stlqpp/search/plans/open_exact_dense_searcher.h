#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/search/plans/dense_cpu_plan.h"
#include "stlqpp/search/plans/exact_dense_searcher.h"
#include "stlqpp/search/parent_access_mode.h"
#include "stlqpp/search/score_mode.h"
#include "stlqpp/search/row_score_data.h"

#include <filesystem>
#include <memory>

namespace stlqpp {

template <int M>
[[nodiscard]] StatusOr<std::unique_ptr<ExactDenseSearcher<M>>>
OpenExactDenseSearcher(
    const std::filesystem::path& directory,
    std::shared_ptr<const F32GemmBackend> gemm,
    std::shared_ptr<const ParallelForExecutor> parallel,
    const DenseCpuPlan& plan,
    ExactDenseSearchRuntime runtime,
    SearchCoefficientMode coefficient_mode = SearchCoefficientMode::kInt8,
    ParentAccessMode parent_access_mode = ParentAccessMode::kMaterialized,
    SearchScoreMode score_mode = SearchScoreMode::kClassic,
    const IndexRowScores* row_scores = nullptr);

extern template StatusOr<std::unique_ptr<ExactDenseSearcher<5>>>
OpenExactDenseSearcher<5>(
    const std::filesystem::path& directory,
    std::shared_ptr<const F32GemmBackend> gemm,
    std::shared_ptr<const ParallelForExecutor> parallel,
    const DenseCpuPlan& plan,
    ExactDenseSearchRuntime runtime, SearchCoefficientMode, ParentAccessMode,
    SearchScoreMode, const IndexRowScores*);
extern template StatusOr<std::unique_ptr<ExactDenseSearcher<10>>>
OpenExactDenseSearcher<10>(
    const std::filesystem::path& directory,
    std::shared_ptr<const F32GemmBackend> gemm,
    std::shared_ptr<const ParallelForExecutor> parallel,
    const DenseCpuPlan& plan,
    ExactDenseSearchRuntime runtime, SearchCoefficientMode, ParentAccessMode,
    SearchScoreMode, const IndexRowScores*);

}  // namespace stlqpp
