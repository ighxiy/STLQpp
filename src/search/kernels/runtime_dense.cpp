#include "stlqpp/model/model_shape.h"
#include "stlqpp/search/kernels/runtime_dense.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "stlqpp/model/topology/parent_louds.h"

namespace stlqpp {
namespace {

float Dot(ColumnMajorMatrixF32View book, std::uint32_t column,
          ArrayView<const float> query) noexcept {
  const float* atom =
      book.values + static_cast<std::size_t>(column) * book.rows;
  float result = 0.0F;
  for (std::uint32_t axis = 0U; axis < book.rows; ++axis) {
    result += atom[axis] * query[axis];
  }
  return result;
}

}  // namespace

Status PrepareRuntimeDenseCluster(const RuntimeDecodedCanonicalCluster& cluster,
                                  const ModelShape& shape, std::uint32_t cid,
                                  Norm8CodebookView norms,
                                  RuntimeFloatCluster* native,
                                  SearchScoreMode mode) {
  const auto& t = cluster.topology;
  const std::uint64_t nodes =
      t.virtual_count + static_cast<std::uint64_t>(t.real_ids.size());
  if (t.m != shape.m || t.cid != cid ||
      nodes > std::numeric_limits<std::uint32_t>::max() ||
      t.real_root_count > t.real_ids.size() ||
      t.real_codes_small.size() != t.real_ids.size() * (shape.m - 1U) ||
      t.virtual_codes_small.size() !=
          static_cast<std::size_t>(t.virtual_count) * (shape.m - 1U) ||
      t.edge_code0.size() != t.real_ids.size()) {
    return Status::DataLoss("runtime resident topology shape is invalid");
  }
  if (const auto status = ValidateParentLouds(
          t.parent_louds_words, static_cast<std::uint32_t>(nodes),
          t.virtual_count + t.real_root_count);
      !status.ok())
    return status;
  if (native) {
    if (native->coefficients_layer_major.size() != nodes * shape.m ||
        native->norm2.size() != t.real_ids.size() ||
        (IsScaledScoreMode(mode) &&
         native->source_norm2.size() != t.real_ids.size())) {
      return Status::DataLoss("runtime native-float record extent is invalid");
    }
    for (float value : native->coefficients_layer_major) {
      if (!std::isfinite(value))
        return Status::DataLoss("non-finite float coefficient");
    }
    for (float value : native->norm2) {
      if (!std::isfinite(value) || value < 0)
        return Status::DataLoss("invalid float norm");
    }
    if (IsScaledScoreMode(mode) && native->dot_multipliers.empty()) {
      native->dot_multipliers.resize(t.real_ids.size());
      for (std::size_t row = 0; row < t.real_ids.size(); ++row) {
        auto scale = ComputeRadialScoreScale(native->source_norm2[row],
                                             native->norm2[row]);
        if (!scale.ok()) return scale.status();
        native->dot_multipliers[row] = -2.0F * scale.value();
      }
    }
  } else {
    const auto& c = cluster.coefficients;
    if (c.m != shape.m || c.virtual_root_count != t.virtual_count ||
        c.real_root_count != t.real_root_count ||
        c.linked_real_count != t.real_ids.size() - t.real_root_count ||
        c.coefficients_layer_major.size() != nodes * shape.m ||
        c.root_scales.size() != shape.m || c.linkage_scales.size() != shape.m ||
        cluster.real_norm8.size() != t.real_ids.size()) {
      return Status::DataLoss("runtime quantized record extent is invalid");
    }
    for (std::uint32_t layer = 0; layer < shape.m; ++layer) {
      if (!std::isfinite(c.root_scales[layer]) ||
          !std::isfinite(c.linkage_scales[layer]))
        return Status::DataLoss("non-finite quantized scale");
    }
    for (auto label : cluster.real_norm8) {
      if (label >= norms.center_count)
        return Status::DataLoss("norm label exceeds codebook");
    }
  }
  return Status::Ok();
}

StatusOr<RuntimeCanonicalQueryTables> BuildRuntimeCanonicalQueryTables(
    const RuntimeCanonicalModel& model, ArrayView<const float> source_query) {
  const ModelShape& shape = model.shape();
  if (const auto status = model.Validate(); !status.ok()) return status;
  if (source_query.size() != shape.dimension) {
    return Status::InvalidArgument(
        "runtime query dimension disagrees with model");
  }
  RuntimeCanonicalQueryTables output;
  output.m = shape.m;
  output.nlist = shape.nlist;
  output.root0.resize(shape.nlist);
  output.root_small.resize(static_cast<std::size_t>(shape.m - 1U) * 256U);
  output.edge.resize(static_cast<std::size_t>(shape.m) * 256U);
  std::vector<float> transformed(shape.dimension);
  BuildRuntimeCanonicalQueryTablesTrusted(
      model, source_query.data(), transformed.data(), output.root0.data(),
      output.root_small.data(), output.edge.data());
  return output;
}

void BuildRuntimeCanonicalQueryTablesTrusted(const RuntimeCanonicalModel& model,
                                             const float* source_query,
                                             float* transformed, float* roots,
                                             float* small, float* edges,
                                             bool skip_root0) {
  const auto& shape = model.shape();
  std::fill_n(transformed, shape.dimension, 0.0F);
  const auto rotation = model.rotation();
  for (std::uint32_t source_axis = 0U; source_axis < shape.dimension;
       ++source_axis) {
    const float* column =
        rotation.values +
        static_cast<std::size_t>(source_axis) * shape.dimension;
    for (std::uint32_t target_axis = 0U; target_axis < shape.dimension;
         ++target_axis) {
      transformed[target_axis] +=
          column[target_axis] * source_query[source_axis];
    }
  }
  if (!skip_root0) {
    for (std::uint32_t cid = 0U; cid < shape.nlist; ++cid)
      roots[cid] = Dot(model.root0(), cid, {transformed, shape.dimension});
  }
  for (std::uint32_t layer = 1U; layer < shape.m; ++layer) {
    const auto book = model.root_small(layer);
    for (std::uint32_t code = 0U; code < 256U; ++code) {
      small[static_cast<std::size_t>(layer - 1U) * 256U + code] =
          Dot(book, code, {transformed, shape.dimension});
    }
  }
  for (std::uint32_t layer = 0U; layer < shape.m; ++layer) {
    const auto book = model.edge(layer);
    for (std::uint32_t code = 0U; code < 256U; ++code) {
      edges[static_cast<std::size_t>(layer) * 256U + code] =
          Dot(book, code, {transformed, shape.dimension});
    }
  }
}

Status ScoreRuntimeCanonicalCluster(
    const RuntimeCanonicalQueryTables& query,
    const RuntimeDecodedCanonicalCluster& cluster, Norm8CodebookView norm8,
    std::vector<float>& real_scores,
    ArrayView<const std::uint32_t> materialized_parents) {
  const auto& topology = cluster.topology;
  const auto& coefficients = cluster.coefficients;
  const std::uint32_t m = topology.m;
  const std::uint32_t real_count =
      static_cast<std::uint32_t>(topology.real_ids.size());
  const std::uint32_t node_count = topology.virtual_count + real_count;
  if (m != query.m || m != coefficients.m || m < 2U || m > stlqpp::kMaximumModelLayers ||
      topology.cid >= query.nlist || query.root0.size() != query.nlist ||
      query.root_small.size() != static_cast<std::size_t>(m - 1U) * 256U ||
      query.edge.size() != static_cast<std::size_t>(m) * 256U ||
      norm8.centers == nullptr || norm8.center_count == 0U ||
      cluster.real_norm8.size() != real_count ||
      (!materialized_parents.empty() &&
       materialized_parents.size() != node_count)) {
    return Status::InvalidArgument("runtime cluster and query tables disagree");
  }
  if (const auto status = PrepareRuntimeDenseCluster(
          cluster, {.dimension = 1U, .m = m, .nlist = query.nlist},
          topology.cid, norm8, nullptr, SearchScoreMode::kClassic);
      !status.ok())
    return status;
  if (!materialized_parents.empty()) {
    std::vector<std::uint32_t> decoded;
    if (const auto status =
            DecodeParentLouds(topology.parent_louds_words, node_count, decoded);
        !status.ok())
      return status;
    if (!std::equal(decoded.begin(), decoded.end(),
                    materialized_parents.begin())) {
      return Status::InvalidArgument(
          "materialized parents disagree with topology");
    }
  }
  std::vector<float> node_dots;
  ScoreRuntimeCanonicalClusterTrusted(
      query.view(), cluster, norm8, nullptr, SearchScoreMode::kClassic, {},
      materialized_parents, node_dots, real_scores);
  return Status::Ok();
}

void ScoreRuntimeCanonicalClusterTrusted(
    const RuntimeCanonicalQueryTableView& query,
    const RuntimeDecodedCanonicalCluster& cluster, Norm8CodebookView norm8,
    const RuntimeFloatCluster* native_float, SearchScoreMode mode,
    ScaledScore8CodebookView scaled,
    ArrayView<const std::uint32_t> materialized_parents,
    std::vector<float>& node_dots, std::vector<float>& real_scores,
    const ClusterRowScores* row_scores, const float* cosine_inverse_norms) {
  const auto& topology = cluster.topology;
  const auto& coefficients = cluster.coefficients;
  const std::uint32_t m = topology.m;
  const auto real_count = static_cast<std::uint32_t>(topology.real_ids.size());
  const auto root_count = topology.virtual_count + topology.real_root_count;
  const auto node_count = topology.virtual_count + real_count;
  node_dots.resize(node_count);
  const auto coefficient = [&](std::uint32_t node, std::uint32_t layer,
                               bool root) {
    const auto offset = static_cast<std::size_t>(layer) * node_count + node;
    if (native_float != nullptr) {
      return native_float->coefficients_layer_major[offset];
    }
    return static_cast<float>(coefficients.coefficients_layer_major[offset]) *
           (root ? coefficients.root_scales[layer]
                 : coefficients.linkage_scales[layer]);
  };
  for (std::uint32_t root = 0U; root < root_count; ++root) {
    const std::uint8_t* codes =
        root < topology.virtual_count
            ? topology.virtual_codes_small.data() +
                  static_cast<std::size_t>(root) * (m - 1U)
            : topology.real_codes_small.data() +
                  static_cast<std::size_t>(root - topology.virtual_count) *
                      (m - 1U);
    float dot = coefficient(root, 0U, true) * query.root0[topology.cid];
    for (std::uint32_t layer = 1U; layer < m; ++layer) {
      dot += coefficient(root, layer, true) *
             query.root_small[static_cast<std::size_t>(layer - 1U) * 256U +
                              codes[layer - 1U]];
    }
    node_dots[root] = dot;
  }
  ParentLoudsSequentialDecoder parents(topology.parent_louds_words);
  parents.InitializeAfterValidatedRootPrefix(root_count);
  for (std::uint32_t real = topology.real_root_count; real < real_count;
       ++real) {
    const std::uint32_t node = topology.virtual_count + real;
    const std::uint32_t parent = materialized_parents.empty()
                                     ? parents.NextParent1Based() - 1U
                                     : materialized_parents[node] - 1U;
    float edge_dot = 0.0F;
    for (std::uint32_t layer = 0U; layer < m; ++layer) {
      const std::uint8_t code =
          layer == 0U
              ? topology.edge_code0[real]
              : topology.real_codes_small[static_cast<std::size_t>(real) *
                                              (m - 1U) +
                                          layer - 1U];
      edge_dot += coefficient(node, layer, false) *
                  query.edge[static_cast<std::size_t>(layer) * 256U + code];
    }
    node_dots[node] = node_dots[parent] + edge_dot;
  }
  real_scores.resize(real_count);
  if (mode == SearchScoreMode::kMipsScaled) {
    const float* multipliers = row_scores != nullptr
        ? row_scores->dot_multipliers.data()
        : native_float != nullptr ? native_float->dot_multipliers.data() : nullptr;
    const float* dots = node_dots.data() + topology.virtual_count;
    if (multipliers != nullptr) {
      for (std::uint32_t real = 0U; real < real_count; ++real)
        real_scores[real] = multipliers[real] * dots[real];
    } else {
      for (std::uint32_t real = 0U; real < real_count; ++real)
        real_scores[real] = -scaled.scales[cluster.real_norm8[real]] * dots[real];
    }
    return;
  }
  for (std::uint32_t real = 0U; real < real_count; ++real) {
    const float dot = node_dots[topology.virtual_count + real];
    const float norm = row_scores != nullptr ? row_scores->norm2[real]
        : native_float != nullptr
                           ? native_float->norm2[real]
                           : norm8.centers[cluster.real_norm8[real]];
    switch (mode) {
      case SearchScoreMode::kClassic:
        real_scores[real] = norm - 2.0F * dot;
        break;
      case SearchScoreMode::kScaled:
        real_scores[real] =
            row_scores != nullptr
                ? std::fma(row_scores->dot_multipliers[real], dot, row_scores->biases[real])
                : native_float != nullptr
                ? std::fma(native_float->dot_multipliers[real], dot,
                           native_float->source_norm2[real])
                : std::fma(-2.0F * scaled.scales[cluster.real_norm8[real]], dot,
                           scaled.biases[cluster.real_norm8[real]]);
        break;
      case SearchScoreMode::kCosineRecon:
        real_scores[real] = row_scores == nullptr && native_float == nullptr &&
                cosine_inverse_norms != nullptr
            ? -dot * cosine_inverse_norms[cluster.real_norm8[real]]
            : -dot / std::sqrt(std::max(norm, 1.0e-20F));
        break;
      case SearchScoreMode::kMipsScaled:
        break;  // Gain-only loop above; never reads norm or bias.
      case SearchScoreMode::kMipsRecon:
        real_scores[real] = -dot;
        break;
    }
  }
}

}  // namespace stlqpp
