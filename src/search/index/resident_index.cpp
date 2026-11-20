#include "stlqpp/search/index/resident_index.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace stlqpp {

template <int M>
StatusOr<ResidentIndex<M>> ResidentIndex<M>::Create(
    QueryTransform query_transform,
    ProductCodebooks<M> codebooks,
    std::vector<float> norm8_centers,
    std::vector<ResidentCluster<M>> clusters,
    SearchCoefficientMode coefficient_mode,
    ParentAccessMode parent_access_mode,
    SearchScoreMode score_mode,
    ScaledScore8CodebookView scaled_score8,
    bool query_codebooks_pretransformed) {
  static_assert(M >= 2 && M <= 20);
  const ModelShape& shape = codebooks.shape();
  if (query_transform.dimension() != shape.dimension) {
    return Status::DataLoss(
        "query transform dimension does not match product codebooks");
  }
  if (clusters.size() != shape.nlist) {
    return Status::DataLoss(
        "resident CID table does not span native-IVF nlist");
  }
  if (coefficient_mode == SearchCoefficientMode::kInt8 &&
      (norm8_centers.empty() || norm8_centers.size() > 256U)) {
    return Status::DataLoss(
        "resident norm8 codebook must contain 1..256 centers");
  }
  for (float norm : norm8_centers) {
    if (!std::isfinite(norm) || norm < 0.0F) {
      return Status::DataLoss(
          "resident norm8 center is not finite and nonnegative");
    }
  }
  const bool table_scaled = coefficient_mode == SearchCoefficientMode::kInt8 &&
      IsScaledScoreMode(score_mode) &&
      !std::all_of(clusters.begin(), clusters.end(), [](const auto& cluster) {
        return cluster.empty() || cluster.has_float_scaled_scores();
      });
  if (table_scaled != scaled_score8.enabled() ||
      (scaled_score8.enabled() &&
       scaled_score8.center_count != norm8_centers.size())) {
    return Status::DataLoss(
        "resident scaled score8 table does not match the norm8 labels");
  }
  for (std::uint32_t label = 0U; label < scaled_score8.center_count;
       ++label) {
    if (!std::isfinite(scaled_score8.scales[label]) ||
        scaled_score8.scales[label] < 0.0F ||
        !std::isfinite(-2.0F * scaled_score8.scales[label]) ||
        !std::isfinite(scaled_score8.biases[label]) ||
        scaled_score8.biases[label] < 0.0F) {
      return Status::DataLoss(
          "resident scaled score8 value is not finite and nonnegative");
    }
  }

  ResidentIndex<M> output(std::move(query_transform), std::move(codebooks));
  output.norm8_centers_ = std::move(norm8_centers);
  if (score_mode == SearchScoreMode::kCosineRecon &&
      coefficient_mode == SearchCoefficientMode::kInt8 &&
      std::any_of(clusters.begin(), clusters.end(), [](const auto& cluster) {
        return !cluster.empty() && !cluster.has_float_forest();
      })) {
    output.norm8_inverse_norms_.resize(output.norm8_centers_.size());
    std::transform(output.norm8_centers_.begin(), output.norm8_centers_.end(),
        output.norm8_inverse_norms_.begin(),
        [](float norm) { return 1.0F / std::sqrt(std::max(norm, 1.0e-20F)); });
  }
  if (scaled_score8.enabled()) {
    output.scaled_score_biases_.assign(
        scaled_score8.biases,
        scaled_score8.biases + scaled_score8.center_count);
    output.scaled_score_dot_multipliers_.resize(
        scaled_score8.center_count);
    std::transform(
        scaled_score8.scales,
        scaled_score8.scales + scaled_score8.center_count,
        output.scaled_score_dot_multipliers_.begin(),
        [score_mode](float scale) noexcept {
          return (score_mode == SearchScoreMode::kMipsScaled ? -1.0F : -2.0F) * scale;
        });
  }
  output.clusters_ = std::move(clusters);
  output.coefficient_mode_ = coefficient_mode;
  output.parent_access_mode_ = parent_access_mode;
  output.score_mode_ = score_mode;
  output.query_codebooks_pretransformed_ =
      query_codebooks_pretransformed;
  bool saw_u16_parent = false;
  bool saw_u32_parent = false;
  for (std::uint32_t cid = 0U; cid < output.shape_.nlist; ++cid) {
    const ResidentCluster<M>& cluster = output.clusters_[cid];
    if (cluster.cid() != cid) {
      return Status::DataLoss(
          "resident cluster order does not match native-IVF CID order");
    }
    if (!cluster.empty() &&
        coefficient_mode == SearchCoefficientMode::kFloat &&
        !cluster.has_float_forest()) {
      return Status::DataLoss(
          "float resident index is missing a CID companion");
    }
    if (!cluster.empty() &&
        coefficient_mode == SearchCoefficientMode::kFloat &&
        (IsScaledScoreMode(score_mode)) !=
            cluster.has_float_scaled_scores()) {
      return Status::DataLoss(
          "float resident radial-score capability disagrees with index mode");
    }
    if (!cluster.empty() &&
        (cluster.parents_materialized() !=
         (parent_access_mode == ParentAccessMode::kMaterialized))) {
      return Status::DataLoss(
          "resident cluster parent representation disagrees with index mode");
    }
    if (!cluster.empty() && cluster.parents_materialized()) {
      saw_u16_parent = saw_u16_parent || cluster.parents_are_u16();
      saw_u32_parent = saw_u32_parent || !cluster.parents_are_u16();
    }
    if (output.norm8_centers_.size() != 256U) {
      for (std::uint8_t label : cluster.real_norm8()) {
        if (label >= output.norm8_centers_.size()) {
          return Status::DataLoss(
              "resident norm8 label exceeds the index-global codebook");
        }
      }
    }
    output.real_count_ += cluster.real_count();
    output.max_node_count_ =
        std::max(output.max_node_count_, cluster.node_count());
  }
  output.materialized_parent_width_ =
      saw_u16_parent && saw_u32_parent
          ? MaterializedParentWidth::kMixed
          : saw_u16_parent
              ? MaterializedParentWidth::kU16
              : saw_u32_parent
                  ? MaterializedParentWidth::kU32
                  : MaterializedParentWidth::kNone;
  if (score_mode == SearchScoreMode::kMipsScaled) {
    for (auto& cluster : output.clusters_) cluster.PrepareMipsScaledScores();
  }
  return output;
}

template class ResidentIndex<5>;
template class ResidentIndex<12>;
template class ResidentIndex<6>;
template class ResidentIndex<10>;

}  // namespace stlqpp
