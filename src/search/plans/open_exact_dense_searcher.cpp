#include "stlqpp/search/plans/open_exact_dense_searcher.h"

#include "stlqpp/search/index/open_index.h"

#include <filesystem>
#include <memory>
#include <utility>

namespace stlqpp {

template <int M>
StatusOr<std::unique_ptr<ExactDenseSearcher<M>>> OpenExactDenseSearcher(
    const std::filesystem::path& directory,
    std::shared_ptr<const F32GemmBackend> gemm,
    std::shared_ptr<const ParallelForExecutor> parallel,
    const DenseCpuPlan& plan,
    ExactDenseSearchRuntime runtime,
    SearchCoefficientMode coefficient_mode,
    ParentAccessMode parent_access_mode,
    SearchScoreMode score_mode, const IndexRowScores* row_scores) {
  auto index = OpenIndex<M>(
      directory, coefficient_mode, parent_access_mode, score_mode,
      runtime.pretransform_query_codebooks ? gemm.get() : nullptr, row_scores);
  if (!index.ok()) {
    return index.status();
  }
  return ExactDenseSearcher<M>::Create(
      std::move(index).value(), std::move(gemm), std::move(parallel), plan,
      runtime);
}

template StatusOr<std::unique_ptr<ExactDenseSearcher<5>>>
OpenExactDenseSearcher<5>(
    const std::filesystem::path& directory,
    std::shared_ptr<const F32GemmBackend> gemm,
    std::shared_ptr<const ParallelForExecutor> parallel,
    const DenseCpuPlan& plan,
    ExactDenseSearchRuntime runtime, SearchCoefficientMode, ParentAccessMode,
    SearchScoreMode, const IndexRowScores*);
template StatusOr<std::unique_ptr<ExactDenseSearcher<12>>>
OpenExactDenseSearcher<12>(
    const std::filesystem::path& directory,
    std::shared_ptr<const F32GemmBackend> gemm,
    std::shared_ptr<const ParallelForExecutor> parallel,
    const DenseCpuPlan& plan,
    ExactDenseSearchRuntime runtime, SearchCoefficientMode, ParentAccessMode,
    SearchScoreMode, const IndexRowScores*);
template StatusOr<std::unique_ptr<ExactDenseSearcher<6>>>
OpenExactDenseSearcher<6>(
    const std::filesystem::path& directory,
    std::shared_ptr<const F32GemmBackend> gemm,
    std::shared_ptr<const ParallelForExecutor> parallel,
    const DenseCpuPlan& plan,
    ExactDenseSearchRuntime runtime, SearchCoefficientMode, ParentAccessMode,
    SearchScoreMode, const IndexRowScores*);
template StatusOr<std::unique_ptr<ExactDenseSearcher<10>>>
OpenExactDenseSearcher<10>(
    const std::filesystem::path& directory,
    std::shared_ptr<const F32GemmBackend> gemm,
    std::shared_ptr<const ParallelForExecutor> parallel,
    const DenseCpuPlan& plan,
    ExactDenseSearchRuntime runtime, SearchCoefficientMode, ParentAccessMode,
    SearchScoreMode, const IndexRowScores*);

}  // namespace stlqpp
