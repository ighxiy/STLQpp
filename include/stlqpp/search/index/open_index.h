#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/search/index/resident_index.h"
#include "stlqpp/search/parent_access_mode.h"
#include "stlqpp/search/score_mode.h"
#include "stlqpp/search/row_score_data.h"

#include <filesystem>
#include <memory>

namespace stlqpp {

// Opens canonical v1/v2, validates and decodes every cold artifact, then derives
// the sole canonical-BFS AoSoA16 resident representation used by production
// search. Canonical mappings are released before this function returns.
template <int M>
[[nodiscard]] StatusOr<std::shared_ptr<const ResidentIndex<M>>> OpenIndex(
    const std::filesystem::path& directory,
    SearchCoefficientMode coefficient_mode = SearchCoefficientMode::kInt8,
    ParentAccessMode parent_access_mode = ParentAccessMode::kMaterialized,
    SearchScoreMode score_mode = SearchScoreMode::kClassic,
    const F32GemmBackend* query_codebook_pretransform_gemm = nullptr,
    const IndexRowScores* row_scores = nullptr);

extern template StatusOr<std::shared_ptr<const ResidentIndex<5>>> OpenIndex<5>(
    const std::filesystem::path& directory, SearchCoefficientMode,
    ParentAccessMode, SearchScoreMode, const F32GemmBackend*, const IndexRowScores*);
extern template StatusOr<std::shared_ptr<const ResidentIndex<10>>>
OpenIndex<10>(const std::filesystem::path& directory, SearchCoefficientMode,
              ParentAccessMode, SearchScoreMode, const F32GemmBackend*, const IndexRowScores*);

}  // namespace stlqpp
