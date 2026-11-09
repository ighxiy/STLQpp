#pragma once

#include <cstdint>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/scaled_score8_codebook.h"
#include "stlqpp/search/score_mode.h"
#include "stlqpp/search/row_score_data.h"
#include "stlqpp/storage/canonical_cluster_codec.h"
#include "stlqpp/storage/canonical_model_codec.h"

namespace stlqpp {

// Scalar fallback for code lengths without a fixed-M SIMD specialization.
// Query tables are built once per query and reused across all selected CIDs.
struct RuntimeCanonicalQueryTableView final {
  std::uint32_t m = 0U;
  std::uint32_t nlist = 0U;
  ArrayView<const float> root0;
  ArrayView<const float> root_small;
  ArrayView<const float> edge;
};

struct RuntimeCanonicalQueryTables final {
  std::uint32_t m = 0U;
  std::uint32_t nlist = 0U;
  std::vector<float> root0;
  std::vector<float> root_small;
  std::vector<float> edge;

  [[nodiscard]] RuntimeCanonicalQueryTableView view() const noexcept {
    return {m, nlist, root0, root_small, edge};
  }
};

// Native-float records are independent of the compressed representation.
// Multipliers are prepared once on open and never propagated to children.
struct RuntimeFloatCluster final {
  std::vector<float> coefficients_layer_major;
  std::vector<float> norm2;
  std::vector<float> source_norm2;
  std::vector<float> dot_multipliers;
};

[[nodiscard]] Status PrepareRuntimeDenseCluster(
    const RuntimeDecodedCanonicalCluster& cluster, const ModelShape& shape,
    std::uint32_t cid, Norm8CodebookView norms, RuntimeFloatCluster* native,
    SearchScoreMode mode);

// The searcher validates these views at its construction boundary. The scan
// only consumes immutable resident data and worker-owned reusable scratch.
void ScoreRuntimeCanonicalClusterTrusted(
    const RuntimeCanonicalQueryTableView& query,
    const RuntimeDecodedCanonicalCluster& cluster, Norm8CodebookView norm8,
    const RuntimeFloatCluster* native_float, SearchScoreMode mode,
    ScaledScore8CodebookView scaled,
    ArrayView<const std::uint32_t> materialized_parents,
    std::vector<float>& node_dots, std::vector<float>& real_scores,
    const ClusterRowScores* row_scores = nullptr,
    const float* cosine_inverse_norms = nullptr);

[[nodiscard]] StatusOr<RuntimeCanonicalQueryTables>
BuildRuntimeCanonicalQueryTables(const RuntimeCanonicalModel& model,
                                 ArrayView<const float> source_query);

// Model/query extents are established by the owning searcher. All four
// destinations are caller-owned reusable storage, sized D, nlist, (M-1)*256,
// and M*256 respectively.
void BuildRuntimeCanonicalQueryTablesTrusted(const RuntimeCanonicalModel& model,
                                             const float* source_query,
                                             float* transformed, float* roots,
                                             float* small, float* edges,
                                             bool skip_root0 = false);

// Emits one classic norm8 score per real row in canonical BFS order.
// Parent contributions are propagated in the validated LOUDS order; virtual
// rows participate in recursion but do not appear in the output.
[[nodiscard]] Status ScoreRuntimeCanonicalCluster(
    const RuntimeCanonicalQueryTables& query,
    const RuntimeDecodedCanonicalCluster& cluster, Norm8CodebookView norm8,
    std::vector<float>& real_scores,
    ArrayView<const std::uint32_t> materialized_parents = {});

}  // namespace stlqpp
