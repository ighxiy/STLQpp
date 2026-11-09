#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/parallel_for.h"

#include <cstdint>
#include <random>
#include <vector>

namespace stlqpp {

// Deterministic spherical k-means++ reduction used after kmeans|| candidate
// generation. Dot updates are independent and parallel; the total, random
// threshold and prefix walk remain serial so worker count cannot change the
// chosen center stream.
[[nodiscard]] StatusOr<std::vector<float>> ReduceSphericalKmeansPlusPlus(
    ColumnMajorMatrixF32View candidates,
    std::uint32_t center_count,
    std::mt19937& random,
    const ParallelForExecutor& parallel);

}  // namespace stlqpp
