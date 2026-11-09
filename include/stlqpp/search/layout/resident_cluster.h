#pragma once

#include "stlqpp/model/norm8_codebook.h"
#include "stlqpp/search/score_mode.h"
#include "stlqpp/search/row_score_data.h"
#include "stlqpp/search/parent_access_mode.h"
#include "stlqpp/storage/canonical_cluster.h"
#include "stlqpp/storage/canonical_cluster_codec.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

template <int M>
struct DenseAoSoA16Layout final {
  static_assert(M >= 2 && M <= 20);

  static constexpr std::uint32_t kTileRows = 16U;
  static constexpr std::uint32_t kRootBytesPerRow = 2U * M - 1U;
  static constexpr std::uint32_t kLinkedBytesPerRow = 2U * M;
  static constexpr std::uint32_t kRootTileBytes =
      kTileRows * kRootBytesPerRow;
  static constexpr std::uint32_t kLinkedTileBytes =
      kTileRows * kLinkedBytesPerRow;
};

template <int M>
class ResidentCluster;

template <int M>
struct NativeFloatBfsClusterSource final {
  static_assert(M >= 2 && M <= 20);

  std::uint32_t cid = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  ArrayView<const std::uint32_t> real_ids;
  ArrayView<const std::uint64_t> parent_louds_words;
  ArrayView<const std::uint8_t> real_codes_small;
  ArrayView<const std::uint8_t> edge_code0;
  ArrayView<const std::uint8_t> virtual_codes_small;
  ArrayView<const float> coefficients_layer_major;
  ArrayView<const float> real_norm2;
  // Optional for classic float search; required by float_scaled.
  ArrayView<const float> real_source_norm2;
};

template <int M>
class FloatResidentClusterView final {
 public:
  static_assert(M >= 2 && M <= 20);

  [[nodiscard]] std::uint32_t cid() const noexcept { return cid_; }
  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return real_count_;
  }
  [[nodiscard]] std::uint32_t virtual_count() const noexcept {
    return virtual_count_;
  }
  [[nodiscard]] std::uint32_t real_root_count() const noexcept {
    return real_root_count_;
  }
  [[nodiscard]] std::uint32_t root_count() const noexcept {
    return virtual_count_ + real_root_count_;
  }
  [[nodiscard]] std::uint32_t linked_count() const noexcept {
    return real_count_ - real_root_count_;
  }
  [[nodiscard]] std::uint32_t node_count() const noexcept {
    return real_count_ + virtual_count_;
  }
  [[nodiscard]] const float* root_coefficients(
      std::uint32_t layer) const noexcept {
    return root_coefficients_layer_major_ +
        static_cast<std::size_t>(layer) * root_count();
  }
  [[nodiscard]] const std::uint8_t* root_small_codes(
      std::uint32_t layer) const noexcept {
    return root_small_codes_layer_major_ +
        static_cast<std::size_t>(layer - 1U) * root_count();
  }
  [[nodiscard]] const float* linked_coefficients(
      std::uint32_t layer) const noexcept {
    return linked_coefficients_layer_major_ +
        static_cast<std::size_t>(layer) * linked_count();
  }
  [[nodiscard]] const std::uint8_t* linked_codes(
      std::uint32_t layer) const noexcept {
    return linked_codes_layer_major_ +
        static_cast<std::size_t>(layer) * linked_count();
  }
  [[nodiscard]] ArrayView<const std::uint64_t> parent_louds_words() const
      noexcept {
    return {parent_louds_words_, parent_louds_word_count_};
  }
  [[nodiscard]] const std::uint16_t* parent_1based_u16() const noexcept {
    return parent_1based_u16_;
  }
  [[nodiscard]] const std::uint32_t* parent_1based_u32() const noexcept {
    return parent_1based_u32_;
  }
  [[nodiscard]] const std::uint32_t* real_ids() const noexcept {
    return real_ids_;
  }
  [[nodiscard]] const float* real_norm2() const noexcept {
    return real_norm2_;
  }
  [[nodiscard]] const float* scaled_score_biases() const noexcept {
    return scaled_score_biases_;
  }
  [[nodiscard]] const float* scaled_score_dot_multipliers() const noexcept {
    return scaled_score_dot_multipliers_;
  }

 private:
  friend class ResidentCluster<M>;

  std::uint32_t cid_ = 0U;
  std::uint32_t real_count_ = 0U;
  std::uint32_t virtual_count_ = 0U;
  std::uint32_t real_root_count_ = 0U;
  const float* root_coefficients_layer_major_ = nullptr;
  const std::uint8_t* root_small_codes_layer_major_ = nullptr;
  const float* linked_coefficients_layer_major_ = nullptr;
  const std::uint8_t* linked_codes_layer_major_ = nullptr;
  const std::uint64_t* parent_louds_words_ = nullptr;
  std::size_t parent_louds_word_count_ = 0U;
  const std::uint16_t* parent_1based_u16_ = nullptr;
  const std::uint32_t* parent_1based_u32_ = nullptr;
  const std::uint32_t* real_ids_ = nullptr;
  const float* real_norm2_ = nullptr;
  const float* scaled_score_biases_ = nullptr;
  const float* scaled_score_dot_multipliers_ = nullptr;
};

template <int M>
class ResidentClusterView final {
 public:
  static_assert(M >= 2 && M <= 20);

  [[nodiscard]] std::uint32_t cid() const noexcept { return cid_; }
  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return real_count_;
  }
  [[nodiscard]] std::uint32_t virtual_count() const noexcept {
    return virtual_count_;
  }
  [[nodiscard]] std::uint32_t real_root_count() const noexcept {
    return real_root_count_;
  }
  [[nodiscard]] std::uint32_t root_count() const noexcept {
    return virtual_count_ + real_root_count_;
  }
  [[nodiscard]] std::uint32_t linked_count() const noexcept {
    return real_count_ - real_root_count_;
  }
  [[nodiscard]] std::uint32_t node_count() const noexcept {
    return real_count_ + virtual_count_;
  }

  [[nodiscard]] const std::uint8_t* root_payload() const noexcept {
    return root_payload_;
  }
  [[nodiscard]] const std::uint8_t* linked_payload() const noexcept {
    return linked_payload_;
  }
  [[nodiscard]] ArrayView<const std::uint64_t> parent_louds_words() const
      noexcept {
    return {parent_louds_words_, parent_louds_word_count_};
  }
  [[nodiscard]] const std::uint16_t* parent_1based_u16() const noexcept {
    return parent_1based_u16_;
  }
  [[nodiscard]] const std::uint32_t* parent_1based_u32() const noexcept {
    return parent_1based_u32_;
  }
  [[nodiscard]] const std::uint32_t* real_ids() const noexcept {
    return real_ids_;
  }
  [[nodiscard]] const std::uint8_t* real_norm8() const noexcept {
    return real_norm8_;
  }
  [[nodiscard]] const float* norm8_centers() const noexcept {
    return norm8_centers_;
  }
  [[nodiscard]] const float* scaled_score_dot_multipliers() const noexcept {
    return scaled_score_dot_multipliers_;
  }
  [[nodiscard]] std::uint32_t norm8_center_count() const noexcept {
    return norm8_center_count_;
  }
  [[nodiscard]] float root_scale(std::uint32_t layer) const noexcept {
    return root_scales_[layer];
  }
  [[nodiscard]] float linkage_scale(std::uint32_t layer) const noexcept {
    return linkage_scales_[layer];
  }

  [[nodiscard]] const float* row_norm2() const noexcept {
    return row_scores_ == nullptr ? nullptr : row_scores_->norm2.data();
  }
  [[nodiscard]] const float* row_biases() const noexcept {
    return row_scores_ == nullptr ? nullptr : row_scores_->biases.data();
  }
  [[nodiscard]] const float* row_dot_multipliers() const noexcept {
    return row_scores_ == nullptr || row_scores_->dot_multipliers.empty()
        ? nullptr : row_scores_->dot_multipliers.data();
  }

 private:
  friend class ResidentCluster<M>;

  std::uint32_t cid_ = 0U;
  std::uint32_t real_count_ = 0U;
  std::uint32_t virtual_count_ = 0U;
  std::uint32_t real_root_count_ = 0U;
  const std::uint8_t* root_payload_ = nullptr;
  const std::uint8_t* linked_payload_ = nullptr;
  const std::uint64_t* parent_louds_words_ = nullptr;
  std::size_t parent_louds_word_count_ = 0U;
  const std::uint16_t* parent_1based_u16_ = nullptr;
  const std::uint32_t* parent_1based_u32_ = nullptr;
  const std::uint32_t* real_ids_ = nullptr;
  const std::uint8_t* real_norm8_ = nullptr;
  const float* norm8_centers_ = nullptr;
  const float* scaled_score_dot_multipliers_ = nullptr;
  std::uint32_t norm8_center_count_ = 0U;
  const float* root_scales_ = nullptr;
  const float* linkage_scales_ = nullptr;
  const ClusterRowScores* row_scores_ = nullptr;
};

template <int M>
class ResidentCluster final {
 public:
  static_assert(M >= 2 && M <= 20);

  [[nodiscard]] std::uint32_t cid() const noexcept { return cid_; }
  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return real_count_;
  }
  [[nodiscard]] std::uint32_t virtual_count() const noexcept {
    return virtual_count_;
  }
  [[nodiscard]] std::uint32_t real_root_count() const noexcept {
    return real_root_count_;
  }
  [[nodiscard]] std::uint32_t root_count() const noexcept {
    return virtual_count_ + real_root_count_;
  }
  [[nodiscard]] std::uint32_t linked_count() const noexcept {
    return real_count_ - real_root_count_;
  }
  [[nodiscard]] std::uint32_t node_count() const noexcept {
    return real_count_ + virtual_count_;
  }
  [[nodiscard]] bool empty() const noexcept { return node_count() == 0U; }
  [[nodiscard]] const std::vector<std::uint8_t>& root_payload() const noexcept {
    return root_payload_;
  }
  [[nodiscard]] const std::vector<std::uint8_t>& linked_payload() const
      noexcept {
    return linked_payload_;
  }
  [[nodiscard]] const std::vector<std::uint64_t>& parent_louds_words() const
      noexcept {
    return parent_louds_words_;
  }
  [[nodiscard]] bool parents_materialized() const noexcept {
    return !parent_1based_u16_.empty() || !parent_1based_u32_.empty();
  }
  [[nodiscard]] bool parents_are_u16() const noexcept {
    return !parent_1based_u16_.empty();
  }
  [[nodiscard]] const std::vector<std::uint32_t>& real_ids() const noexcept {
    return real_ids_;
  }
  [[nodiscard]] const std::vector<std::uint8_t>& real_norm8() const noexcept {
    return real_norm8_;
  }
  [[nodiscard]] std::size_t packed_payload_bytes() const noexcept {
    return root_payload_.size() + linked_payload_.size();
  }
  [[nodiscard]] std::size_t logical_resident_bytes() const noexcept;
  [[nodiscard]] bool has_float_forest() const noexcept {
    return !float_scores_.norm2.empty();
  }
  [[nodiscard]] bool has_float_scaled_scores() const noexcept {
    return !float_scores_.biases.empty();
  }

  [[nodiscard]] ResidentClusterView<M> view(
      Norm8CodebookView norm8,
      ScaledScoreTableView scaled = {}) const noexcept;
  [[nodiscard]] FloatResidentClusterView<M> float_view() const noexcept;
  [[nodiscard]] Status ConfigureParentAccess(ParentAccessMode mode);
  [[nodiscard]] Status ConfigureRowScores(const ClusterRowScores& scores);

 private:
  template <int N>
  friend ResidentCluster<N> BuildBfsResidentCluster(
      const TrustedBfsClusterView<N>& source);
  template <int N>
  friend StatusOr<ResidentCluster<N>> BuildBfsResidentCluster(
      const TrustedClusterView<N>& source);
  template <int N>
  friend StatusOr<ResidentCluster<N>> BuildBfsResidentCluster(
      const TrustedBfsClusterView<N>& source,
      ArrayView<const float> coefficients_layer_major,
      ArrayView<const float> real_norm2);
  template <int N>
  friend StatusOr<ResidentCluster<N>> BuildFloatBfsResidentCluster(
      const NativeFloatBfsClusterSource<N>& source);
  template <int N>
  friend StatusOr<ResidentCluster<N>> BuildFloatBfsResidentClusterImpl(
      const NativeFloatBfsClusterSource<N>& source,
      bool validate_louds);

  [[nodiscard]] ResidentClusterView<M> MakeView(
      Norm8CodebookView norm8,
      ScaledScoreTableView scaled) const noexcept;

  std::uint32_t cid_ = 0U;
  std::uint32_t real_count_ = 0U;
  std::uint32_t virtual_count_ = 0U;
  std::uint32_t real_root_count_ = 0U;
  std::array<float, M> root_scales_{};
  std::array<float, M> linkage_scales_{};
  std::vector<std::uint32_t> real_ids_;
  std::vector<std::uint8_t> real_norm8_;
  std::vector<std::uint8_t> root_payload_;
  std::vector<std::uint8_t> linked_payload_;
  std::vector<std::uint64_t> parent_louds_words_;
  std::vector<std::uint16_t> parent_1based_u16_;
  std::vector<std::uint32_t> parent_1based_u32_;
  std::vector<float> float_root_coefficients_layer_major_;
  std::vector<std::uint8_t> float_root_small_codes_layer_major_;
  std::vector<float> float_linked_coefficients_layer_major_;
  std::vector<std::uint8_t> float_linked_codes_layer_major_;
  ClusterRowScores float_scores_;
};

template <int M>
[[nodiscard]] ResidentCluster<M> BuildBfsResidentCluster(
    const TrustedBfsClusterView<M>& source);

template <int M>
[[nodiscard]] StatusOr<ResidentCluster<M>> BuildBfsResidentCluster(
    const TrustedClusterView<M>& source);

template <int M>
[[nodiscard]] StatusOr<ResidentCluster<M>> BuildBfsResidentCluster(
    const TrustedBfsClusterView<M>& source,
    ArrayView<const float> coefficients_layer_major,
    ArrayView<const float> real_norm2);

template <int M>
[[nodiscard]] StatusOr<ResidentCluster<M>> BuildFloatBfsResidentCluster(
    const NativeFloatBfsClusterSource<M>& source);

// Restricted decoded-topology path: the LOUDS words and their validation
// provenance come from the same owning object.
template <int M>
[[nodiscard]] StatusOr<ResidentCluster<M>>
BuildFloatBfsResidentClusterFromDecodedTopology(
    const DecodedCanonicalTopology<M>& topology,
    ArrayView<const float> coefficients_layer_major,
    ArrayView<const float> real_norm2,
    ArrayView<const float> real_source_norm2);

extern template class ResidentCluster<5>;
extern template class ResidentCluster<10>;
extern template class ResidentClusterView<5>;
extern template class ResidentClusterView<10>;
extern template class FloatResidentClusterView<5>;
extern template class FloatResidentClusterView<10>;
extern template ResidentCluster<5> BuildBfsResidentCluster(
    const TrustedBfsClusterView<5>& source);
extern template ResidentCluster<10> BuildBfsResidentCluster(
    const TrustedBfsClusterView<10>& source);
extern template StatusOr<ResidentCluster<5>> BuildBfsResidentCluster(
    const TrustedClusterView<5>& source);
extern template StatusOr<ResidentCluster<10>> BuildBfsResidentCluster(
    const TrustedClusterView<10>& source);
extern template StatusOr<ResidentCluster<5>> BuildBfsResidentCluster(
    const TrustedBfsClusterView<5>& source,
    ArrayView<const float>, ArrayView<const float>);
extern template StatusOr<ResidentCluster<10>> BuildBfsResidentCluster(
    const TrustedBfsClusterView<10>& source,
    ArrayView<const float>, ArrayView<const float>);
extern template StatusOr<ResidentCluster<5>> BuildFloatBfsResidentCluster(
    const NativeFloatBfsClusterSource<5>& source);
extern template StatusOr<ResidentCluster<10>> BuildFloatBfsResidentCluster(
    const NativeFloatBfsClusterSource<10>& source);
extern template StatusOr<ResidentCluster<5>>
BuildFloatBfsResidentClusterFromDecodedTopology(
    const DecodedCanonicalTopology<5>&, ArrayView<const float>,
    ArrayView<const float>, ArrayView<const float>);
extern template StatusOr<ResidentCluster<10>>
BuildFloatBfsResidentClusterFromDecodedTopology(
    const DecodedCanonicalTopology<10>&, ArrayView<const float>,
    ArrayView<const float>, ArrayView<const float>);

}  // namespace stlqpp
