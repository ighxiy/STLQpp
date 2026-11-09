#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/model/norm8_codebook.h"
#include "stlqpp/model/scaled_score8_codebook.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/model/query_transform.h"
#include "stlqpp/search/layout/resident_cluster.h"
#include "stlqpp/search/coefficient_mode.h"
#include "stlqpp/search/parent_access_mode.h"
#include "stlqpp/search/score_mode.h"
#include "stlqpp/search/routing/exact_ivf_router.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {

// Immutable owner of the complete CPU scan representation. Canonical storage
// is converted into this object once at OpenIndex. ParentLOUDS remains the
// canonical topology; the resident query path is selected explicitly between
// a one-time adaptive u16/u32 parent materialization and native LOUDS decode.
template <int M>
class ResidentIndex final {
 public:
  static_assert(M >= 2 && M <= 20);

  [[nodiscard]] static StatusOr<ResidentIndex<M>> Create(
      QueryTransform query_transform,
      ProductCodebooks<M> codebooks,
      std::vector<float> norm8_centers,
      std::vector<ResidentCluster<M>> clusters,
      SearchCoefficientMode coefficient_mode =
          SearchCoefficientMode::kInt8,
      ParentAccessMode parent_access_mode =
          ParentAccessMode::kNativeLouds,
      SearchScoreMode score_mode = SearchScoreMode::kClassic,
      ScaledScore8CodebookView scaled_score8 = {},
      bool query_codebooks_pretransformed = false);

  ResidentIndex(ResidentIndex&&) noexcept = default;
  ResidentIndex& operator=(ResidentIndex&&) noexcept = default;
  ResidentIndex(const ResidentIndex&) = delete;
  ResidentIndex& operator=(const ResidentIndex&) = delete;

  [[nodiscard]] const ModelShape& shape() const noexcept { return shape_; }
  [[nodiscard]] const ProductCodebooks<M>& codebooks() const noexcept {
    return codebooks_;
  }
  [[nodiscard]] const QueryTransform& query_transform() const noexcept {
    return query_transform_;
  }
  [[nodiscard]] std::uint32_t nlist() const noexcept { return shape_.nlist; }
  [[nodiscard]] std::uint64_t real_count() const noexcept {
    return real_count_;
  }
  [[nodiscard]] std::uint32_t max_node_count() const noexcept {
    return max_node_count_;
  }
  [[nodiscard]] SearchCoefficientMode coefficient_mode() const noexcept {
    return coefficient_mode_;
  }
  [[nodiscard]] ParentAccessMode parent_access_mode() const noexcept {
    return parent_access_mode_;
  }
  [[nodiscard]] SearchScoreMode score_mode() const noexcept {
    return score_mode_;
  }
  [[nodiscard]] bool query_codebooks_pretransformed() const noexcept {
    return query_codebooks_pretransformed_;
  }
  [[nodiscard]] MaterializedParentWidth materialized_parent_width() const
      noexcept {
    return materialized_parent_width_;
  }
  [[nodiscard]] const ResidentCluster<M>& cluster(
      std::uint32_t cid) const noexcept {
    return clusters_[cid];
  }
  [[nodiscard]] Norm8CodebookView norm8() const noexcept {
    return {norm8_centers_.data(),
            static_cast<std::uint32_t>(norm8_centers_.size())};
  }
  [[nodiscard]] const float* cosine_inverse_norms() const noexcept {
    return norm8_inverse_norms_.empty() ? nullptr : norm8_inverse_norms_.data();
  }
  [[nodiscard]] ScaledScoreTableView scaled_score() const noexcept {
    return {
        .dot_multipliers = scaled_score_dot_multipliers_.empty()
            ? nullptr
            : scaled_score_dot_multipliers_.data(),
        .biases = scaled_score_biases_.empty()
            ? nullptr : scaled_score_biases_.data(),
        .center_count = static_cast<std::uint32_t>(
            scaled_score_biases_.size()),
    };
  }
  [[nodiscard]] TrustedNativeIvfView native_ivf() const noexcept {
    const ArrayView<const float> inverse_norms =
        codebooks_.root_inverse_norms();
    return TrustedNativeIvfView(inverse_norms.data(), shape_.nlist);
  }

 private:
  ResidentIndex(QueryTransform query_transform,
                ProductCodebooks<M> codebooks) noexcept
      : shape_(codebooks.shape()),
        query_transform_(std::move(query_transform)),
        codebooks_(std::move(codebooks)) {}

  ModelShape shape_;
  QueryTransform query_transform_;
  ProductCodebooks<M> codebooks_;
  std::vector<float> norm8_centers_;
  std::vector<float> norm8_inverse_norms_;
  std::vector<float> scaled_score_biases_;
  std::vector<float> scaled_score_dot_multipliers_;
  std::vector<ResidentCluster<M>> clusters_;
  std::uint64_t real_count_ = 0U;
  std::uint32_t max_node_count_ = 0U;
  SearchCoefficientMode coefficient_mode_ = SearchCoefficientMode::kInt8;
  ParentAccessMode parent_access_mode_ = ParentAccessMode::kNativeLouds;
  SearchScoreMode score_mode_ = SearchScoreMode::kClassic;
  bool query_codebooks_pretransformed_ = false;
  MaterializedParentWidth materialized_parent_width_ =
      MaterializedParentWidth::kNone;
};

extern template class ResidentIndex<5>;
extern template class ResidentIndex<10>;

}  // namespace stlqpp
