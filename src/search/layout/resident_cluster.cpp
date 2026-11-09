#include "stlqpp/search/layout/resident_cluster.h"

#include "stlqpp/model/scaled_score8_codebook.h"
#include "stlqpp/model/topology/parent_louds.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace stlqpp {
namespace {

[[nodiscard]] std::uint8_t CoefficientByte(std::int8_t value) noexcept {
  return static_cast<std::uint8_t>(value);
}

template <int M>
[[nodiscard]] const std::uint8_t* RootSmallCodes(
    const TrustedBfsClusterView<M>& source,
    std::uint32_t old_local) noexcept {
  if (old_local < source.virtual_count()) {
    return source.virtual_codes_small() +
           static_cast<std::size_t>(old_local) * (M - 1U);
  }
  const std::uint32_t old_real = old_local - source.virtual_count();
  return source.real_codes_small() +
         static_cast<std::size_t>(old_real) * (M - 1U);
}

template <int M>
void PackRootPayload(const TrustedBfsClusterView<M>& source,
                     std::vector<std::uint8_t>& output) {
  using Layout = DenseAoSoA16Layout<M>;
  const std::uint32_t root_count =
      source.virtual_count() + source.real_root_count();
  output.resize(static_cast<std::size_t>(root_count) *
                Layout::kRootBytesPerRow);
  const std::uint32_t tiled =
      root_count / Layout::kTileRows * Layout::kTileRows;

  for (std::uint32_t row = 0U; row < tiled; row += Layout::kTileRows) {
    std::uint8_t* tile = output.data() +
        static_cast<std::size_t>(row / Layout::kTileRows) *
            Layout::kRootTileBytes;
    for (std::uint32_t lane = 0U; lane < Layout::kTileRows; ++lane) {
      const std::uint32_t local = row + lane;
      tile[lane] = CoefficientByte(source.coefficients(0U)[local]);
    }
    for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(M);
         ++layer) {
      std::uint8_t* pair = tile + Layout::kTileRows +
          static_cast<std::size_t>(layer - 1U) * 2U * Layout::kTileRows;
      for (std::uint32_t lane = 0U; lane < Layout::kTileRows; ++lane) {
        const std::uint32_t local = row + lane;
        pair[lane] = CoefficientByte(source.coefficients(layer)[local]);
        pair[Layout::kTileRows + lane] =
            RootSmallCodes(source, local)[layer - 1U];
      }
    }
  }

  std::uint8_t* trailer = output.data() +
      static_cast<std::size_t>(tiled / Layout::kTileRows) *
          Layout::kRootTileBytes;
  for (std::uint32_t root = tiled; root < root_count; ++root) {
    std::uint8_t* record = trailer +
        static_cast<std::size_t>(root - tiled) * Layout::kRootBytesPerRow;
    record[0] = CoefficientByte(source.coefficients(0U)[root]);
    const std::uint8_t* codes = RootSmallCodes(source, root);
    for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(M);
         ++layer) {
      record[1U + 2U * (layer - 1U)] =
          CoefficientByte(source.coefficients(layer)[root]);
      record[2U + 2U * (layer - 1U)] = codes[layer - 1U];
    }
  }
}

template <int M>
void PackLinkedPayload(const TrustedBfsClusterView<M>& source,
                       std::uint32_t root_count,
                       std::vector<std::uint8_t>& output) {
  using Layout = DenseAoSoA16Layout<M>;
  const std::uint32_t linked_count = source.real_count() -
                                     source.real_root_count();
  output.resize(static_cast<std::size_t>(linked_count) *
                Layout::kLinkedBytesPerRow);
  const std::uint32_t tiled =
      linked_count / Layout::kTileRows * Layout::kTileRows;

  for (std::uint32_t row = 0U; row < tiled; row += Layout::kTileRows) {
    std::uint8_t* tile = output.data() +
        static_cast<std::size_t>(row / Layout::kTileRows) *
            Layout::kLinkedTileBytes;
    for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(M);
         ++layer) {
      std::uint8_t* pair = tile +
          static_cast<std::size_t>(layer) * 2U * Layout::kTileRows;
      for (std::uint32_t lane = 0U; lane < Layout::kTileRows; ++lane) {
        const std::uint32_t local = root_count + row + lane;
        const std::uint32_t real = local - source.virtual_count();
        pair[lane] = CoefficientByte(source.coefficients(layer)[local]);
        pair[Layout::kTileRows + lane] = layer == 0U
            ? source.edge_code0()[real]
            : source.real_codes_small()[
                  static_cast<std::size_t>(real) * (M - 1U) +
                  layer - 1U];
      }
    }
  }

  std::uint8_t* trailer = output.data() +
      static_cast<std::size_t>(tiled / Layout::kTileRows) *
          Layout::kLinkedTileBytes;
  for (std::uint32_t row = tiled; row < linked_count; ++row) {
    const std::uint32_t local = root_count + row;
    const std::uint32_t real = local - source.virtual_count();
    std::uint8_t* record = trailer +
        static_cast<std::size_t>(row - tiled) * Layout::kLinkedBytesPerRow;
    for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(M);
         ++layer) {
      record[2U * layer] =
          CoefficientByte(source.coefficients(layer)[local]);
      record[2U * layer + 1U] = layer == 0U
          ? source.edge_code0()[real]
          : source.real_codes_small()[
                static_cast<std::size_t>(real) * (M - 1U) +
                layer - 1U];
    }
  }
}

template <int M>
void FillBfsRows(const TrustedBfsClusterView<M>& source,
                 std::vector<std::uint32_t>& real_ids,
                 std::vector<std::uint8_t>& real_norm8) {
  if (source.real_count() != 0U) {
    std::copy_n(source.real_ids(), source.real_count(), real_ids.data());
    std::copy_n(source.real_norm8(), source.real_count(), real_norm8.data());
  }
}

}  // namespace

template <int M>
ResidentClusterView<M> ResidentCluster<M>::MakeView(
    Norm8CodebookView norm8,
    ScaledScoreTableView scaled) const noexcept {
  assert(norm8.centers != nullptr && norm8.center_count != 0U);
  assert(!scaled.enabled() || scaled.center_count == norm8.center_count);
  ResidentClusterView<M> view;
  view.cid_ = cid_;
  view.real_count_ = real_count_;
  view.virtual_count_ = virtual_count_;
  view.real_root_count_ = real_root_count_;
  view.root_payload_ = root_payload_.data();
  view.linked_payload_ = linked_payload_.data();
  view.parent_louds_words_ = parent_louds_words_.data();
  view.parent_louds_word_count_ = parent_louds_words_.size();
  view.parent_1based_u16_ = parent_1based_u16_.data();
  view.parent_1based_u32_ = parent_1based_u32_.data();
  view.real_ids_ = real_ids_.data();
  view.row_scores_ = float_scores_.norm2.empty() ? nullptr : &float_scores_;
  view.real_norm8_ = real_norm8_.data();
  view.norm8_centers_ = norm8.centers;
  // The presence of a multiplier selects scaled query tables for either
  // representation. The row pointer distinguishes row scores from norm8.
  view.scaled_score_dot_multipliers_ = float_scores_.dot_multipliers.empty()
      ? scaled.dot_multipliers : float_scores_.dot_multipliers.data();
  if (scaled.enabled()) view.norm8_centers_ = scaled.biases;
  view.norm8_center_count_ = norm8.center_count;
  view.root_scales_ = root_scales_.data();
  view.linkage_scales_ = linkage_scales_.data();
  return view;
}

template <int M>
ResidentClusterView<M> ResidentCluster<M>::view(
    Norm8CodebookView norm8,
    ScaledScoreTableView scaled) const noexcept {
  assert(!empty());
  return MakeView(norm8, scaled);
}

template <int M>
Status ResidentCluster<M>::ConfigureRowScores(const ClusterRowScores& scores) {
  if (const auto status = ValidateClusterRowScores(scores, real_count_); !status.ok())
    return status;
  float_scores_.norm2 = scores.norm2;
  float_scores_.biases = scores.biases;
  float_scores_.dot_multipliers = scores.dot_multipliers;
  return Status::Ok();
}

template <int M>
std::size_t ResidentCluster<M>::logical_resident_bytes() const noexcept {
  return root_payload_.size() + linked_payload_.size() + real_norm8_.size() +
         real_ids_.size() * sizeof(real_ids_[0]) +
         parent_louds_words_.size() * sizeof(parent_louds_words_[0]) +
         parent_1based_u16_.size() * sizeof(parent_1based_u16_[0]) +
         parent_1based_u32_.size() * sizeof(parent_1based_u32_[0]) +
         root_scales_.size() * sizeof(root_scales_[0]) +
         linkage_scales_.size() * sizeof(linkage_scales_[0]) +
         float_root_coefficients_layer_major_.size() * sizeof(float) +
         float_root_small_codes_layer_major_.size() +
         float_linked_coefficients_layer_major_.size() * sizeof(float) +
         float_linked_codes_layer_major_.size() +
         float_scores_.norm2.size() * sizeof(float) +
         float_scores_.biases.size() * sizeof(float) +
         float_scores_.dot_multipliers.size() * sizeof(float);
}

template <int M>
FloatResidentClusterView<M> ResidentCluster<M>::float_view() const noexcept {
  assert(!empty() && has_float_forest());
  FloatResidentClusterView<M> view;
  view.cid_ = cid_;
  view.real_count_ = real_count_;
  view.virtual_count_ = virtual_count_;
  view.real_root_count_ = real_root_count_;
  view.root_coefficients_layer_major_ =
      float_root_coefficients_layer_major_.data();
  view.root_small_codes_layer_major_ =
      float_root_small_codes_layer_major_.data();
  view.linked_coefficients_layer_major_ =
      float_linked_coefficients_layer_major_.data();
  view.linked_codes_layer_major_ = float_linked_codes_layer_major_.data();
  view.parent_louds_words_ = parent_louds_words_.data();
  view.parent_louds_word_count_ = parent_louds_words_.size();
  view.parent_1based_u16_ = parent_1based_u16_.data();
  view.parent_1based_u32_ = parent_1based_u32_.data();
  view.real_ids_ = real_ids_.data();
  view.real_norm2_ = float_scores_.norm2.data();
  view.scaled_score_biases_ = float_scores_.biases.data();
  view.scaled_score_dot_multipliers_ =
      float_scores_.dot_multipliers.data();
  return view;
}

template <int M>
Status ResidentCluster<M>::ConfigureParentAccess(ParentAccessMode mode) {
  // Parent access is selected once when opening an index in production, but
  // callers may explicitly reconfigure an already resident cluster.  Swapping
  // releases a previously materialized parent table instead of leaving its
  // capacity resident while reporting native LOUDS mode.
  std::vector<std::uint16_t>().swap(parent_1based_u16_);
  std::vector<std::uint32_t>().swap(parent_1based_u32_);
  if (mode == ParentAccessMode::kNativeLouds || empty()) {
    return Status::Ok();
  }
  if (parent_louds_words_.size() != ParentLoudsWordCount(node_count())) {
    return Status::FailedPrecondition(
        "resident parent stream is unavailable");
  }

  std::vector<std::uint32_t> decoded(node_count(), 0U);
  ParentLoudsSequentialDecoder decoder(parent_louds_words_);
  decoder.InitializeAfterValidatedRootPrefix(root_count());
  for (std::uint32_t node = root_count(); node < node_count(); ++node) {
    decoded[node] = decoder.NextParent1Based();
  }
  if (node_count() <= std::numeric_limits<std::uint16_t>::max()) {
    parent_1based_u16_.resize(decoded.size());
    std::transform(decoded.begin(), decoded.end(),
                   parent_1based_u16_.begin(),
                   [](std::uint32_t parent) noexcept {
                     return static_cast<std::uint16_t>(parent);
                   });
  } else {
    parent_1based_u32_ = std::move(decoded);
  }
  return Status::Ok();
}

template <int M>
ResidentCluster<M> BuildBfsResidentCluster(
    const TrustedBfsClusterView<M>& source) {
  ResidentCluster<M> output;
  output.cid_ = source.cid();
  output.real_count_ = source.real_count();
  output.virtual_count_ = source.virtual_count();
  output.real_root_count_ = source.real_root_count();
  if (source.node_count() == 0U) {
    return output;
  }

  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(M);
       ++layer) {
    output.root_scales_[layer] = source.root_scale(layer);
    output.linkage_scales_[layer] = source.linkage_scale(layer);
  }
  output.real_ids_.resize(source.real_count());
  output.real_norm8_.resize(source.real_count());
  output.parent_louds_words_.assign(source.parent_louds_words().begin(),
                                    source.parent_louds_words().end());

  const std::uint32_t root_count = output.root_count();
  PackRootPayload(source, output.root_payload_);
  PackLinkedPayload(source, root_count, output.linked_payload_);

  FillBfsRows(source, output.real_ids_, output.real_norm8_);
  return output;
}

template <int M>
StatusOr<ResidentCluster<M>> BuildBfsResidentCluster(
    const TrustedClusterView<M>& source) {
  if (source.node_count() == 0U) {
    ResidentCluster<M> output;
    output.cid_ = source.cid();
    return output;
  }
  std::vector<std::uint32_t> all_parents(source.node_count(), 0U);
  if (source.real_count() != 0U) {
    std::copy_n(source.parent_local_1based(), source.real_count(),
                all_parents.data() + source.virtual_count());
  }
  std::vector<std::uint64_t> words;
  if (const Status status = EncodeParentLouds(all_parents, words);
      !status.ok()) {
    return status;
  }
  std::array<float, M> root_scales{};
  std::array<float, M> linkage_scales{};
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(M);
       ++layer) {
    root_scales[layer] = source.root_scale(layer);
    linkage_scales[layer] = source.linkage_scale(layer);
  }
  auto trusted = TrustedBfsClusterView<M>::Create({
      .cid = source.cid(),
      .real_count = source.real_count(),
      .virtual_count = source.virtual_count(),
      .real_root_count = source.real_root_count(),
      .real_ids = source.real_ids(),
      .parent_louds_words = words,
      .real_codes_small = source.real_codes_small(),
      .edge_code0 = source.edge_code0(),
      .virtual_codes_small = source.virtual_codes_small(),
      .coefficients_layer_major = source.coefficients(0U),
      .root_scales = root_scales.data(),
      .linkage_scales = linkage_scales.data(),
      .real_norm8 = source.real_norm8(),
      .norm8_centers = source.norm8_centers(),
      .norm8_center_count = source.norm8_center_count(),
  });
  if (!trusted.ok()) {
    return trusted.status();
  }
  return BuildBfsResidentCluster(trusted.value());
}

template <int M>
StatusOr<ResidentCluster<M>> BuildBfsResidentCluster(
    const TrustedBfsClusterView<M>& source,
    ArrayView<const float> coefficients_layer_major,
    ArrayView<const float> real_norm2) {
  return BuildFloatBfsResidentCluster<M>({
      .cid = source.cid(),
      .virtual_count = source.virtual_count(),
      .real_root_count = source.real_root_count(),
      .real_ids = {source.real_ids(), source.real_count()},
      .parent_louds_words = source.parent_louds_words(),
      .real_codes_small = {
          source.real_codes_small(),
          static_cast<std::size_t>(source.real_count()) * (M - 1U)},
      .edge_code0 = {source.edge_code0(), source.real_count()},
      .virtual_codes_small = {
          source.virtual_codes_small(),
          static_cast<std::size_t>(source.virtual_count()) * (M - 1U)},
      .coefficients_layer_major = coefficients_layer_major,
      .real_norm2 = real_norm2,
      .real_source_norm2 = {},
  });
}

template <int M>
StatusOr<ResidentCluster<M>> BuildFloatBfsResidentClusterImpl(
    const NativeFloatBfsClusterSource<M>& source, bool validate_louds) {
  const std::uint64_t node_count_u64 =
      static_cast<std::uint64_t>(source.virtual_count) +
      source.real_ids.size();
  if (source.real_root_count > source.real_ids.size() ||
      node_count_u64 > std::numeric_limits<std::uint32_t>::max()) {
    return Status::DataLoss(
        "native-float resident topology counts are invalid");
  }
  const std::uint32_t real_count =
      static_cast<std::uint32_t>(source.real_ids.size());
  const std::uint32_t node_count =
      static_cast<std::uint32_t>(node_count_u64);
  const std::uint32_t root_count =
      source.virtual_count + source.real_root_count;
  const std::uint32_t linked_count = real_count - source.real_root_count;
  if (source.real_codes_small.size() !=
          static_cast<std::size_t>(real_count) * (M - 1U) ||
      source.edge_code0.size() != real_count ||
      source.virtual_codes_small.size() !=
          static_cast<std::size_t>(source.virtual_count) * (M - 1U) ||
      source.coefficients_layer_major.size() !=
          static_cast<std::size_t>(node_count) * M ||
      source.real_norm2.size() != real_count ||
      (!source.real_source_norm2.empty() &&
       source.real_source_norm2.size() != real_count)) {
    return Status::InvalidArgument(
        "native-float resident records disagree with canonical counts");
  }
  if (validate_louds) {
    if (const Status status = ValidateParentLouds(
            source.parent_louds_words, node_count, root_count);
        !status.ok()) {
      return status;
    }
  }
  for (const float value : source.coefficients_layer_major) {
    if (!std::isfinite(value)) {
      return Status::DataLoss(
          "native-float resident coefficient is not finite");
    }
  }
  for (const float value : source.real_norm2) {
    if (!std::isfinite(value) || value < 0.0F) {
      return Status::DataLoss(
          "native-float resident norm is invalid");
    }
  }
  ResidentCluster<M> output;
  output.cid_ = source.cid;
  output.real_count_ = real_count;
  output.virtual_count_ = source.virtual_count;
  output.real_root_count_ = source.real_root_count;
  if (node_count == 0U) return output;
  output.real_ids_.assign(source.real_ids.begin(), source.real_ids.end());
  output.parent_louds_words_.assign(
      source.parent_louds_words.begin(), source.parent_louds_words.end());
  output.float_root_coefficients_layer_major_.resize(
      static_cast<std::size_t>(root_count) * M);
  output.float_root_small_codes_layer_major_.resize(
      static_cast<std::size_t>(root_count) * (M - 1U));
  output.float_linked_coefficients_layer_major_.resize(
      static_cast<std::size_t>(linked_count) * M);
  output.float_linked_codes_layer_major_.resize(
      static_cast<std::size_t>(linked_count) * M);
  output.float_scores_.norm2.assign(
      source.real_norm2.begin(), source.real_norm2.end());
  if (!source.real_source_norm2.empty()) {
    output.float_scores_.biases.assign(
        source.real_source_norm2.begin(), source.real_source_norm2.end());
    output.float_scores_.dot_multipliers.resize(real_count);
    for (std::uint32_t row = 0U; row < real_count; ++row) {
      const float source_norm2 = source.real_source_norm2[row];
      const float reconstruction_norm2 = source.real_norm2[row];
      if (!std::isfinite(source_norm2) || source_norm2 < 0.0F) {
        return Status::DataLoss(
            "native-float source norm is invalid");
      }
      auto scale = ComputeRadialScoreScale(
          source_norm2, reconstruction_norm2);
      if (!scale.ok()) return scale.status();
      output.float_scores_.dot_multipliers[row] =
          -2.0F * scale.value();
    }
  }

  for (std::uint32_t layer = 0U;
       layer < static_cast<std::uint32_t>(M); ++layer) {
    const float* source_coefficients =
        source.coefficients_layer_major.data() +
        static_cast<std::size_t>(layer) * node_count;
    std::copy_n(
        source_coefficients, root_count,
        output.float_root_coefficients_layer_major_.data() +
            static_cast<std::size_t>(layer) * root_count);
    std::copy_n(
        source_coefficients + root_count, linked_count,
        output.float_linked_coefficients_layer_major_.data() +
            static_cast<std::size_t>(layer) * linked_count);
    for (std::uint32_t linked = 0U; linked < linked_count; ++linked) {
      const std::uint32_t real = source.real_root_count + linked;
      output.float_linked_codes_layer_major_[
          static_cast<std::size_t>(layer) * linked_count + linked] =
          layer == 0U
          ? source.edge_code0[real]
          : source.real_codes_small[
                static_cast<std::size_t>(real) * (M - 1U) + layer - 1U];
    }
  }
  for (std::uint32_t layer = 1U;
       layer < static_cast<std::uint32_t>(M); ++layer) {
    std::uint8_t* destination =
        output.float_root_small_codes_layer_major_.data() +
        static_cast<std::size_t>(layer - 1U) * root_count;
    for (std::uint32_t root = 0U; root < root_count; ++root) {
      destination[root] = root < source.virtual_count
          ? source.virtual_codes_small[
                static_cast<std::size_t>(root) * (M - 1U) + layer - 1U]
          : source.real_codes_small[
                static_cast<std::size_t>(root - source.virtual_count) *
                    (M - 1U) + layer - 1U];
    }
  }
  return output;
}

template <int M>
StatusOr<ResidentCluster<M>> BuildFloatBfsResidentCluster(
    const NativeFloatBfsClusterSource<M>& source) {
  return BuildFloatBfsResidentClusterImpl(source, true);
}

template <int M>
StatusOr<ResidentCluster<M>>
BuildFloatBfsResidentClusterFromDecodedTopology(
    const DecodedCanonicalTopology<M>& topology,
    ArrayView<const float> coefficients_layer_major,
    ArrayView<const float> real_norm2,
    ArrayView<const float> real_source_norm2) {
  const NativeFloatBfsClusterSource<M> source = {
      .cid = topology.cid(),
      .virtual_count = topology.virtual_count(),
      .real_root_count = topology.real_root_count(),
      .real_ids = topology.real_ids(),
      .parent_louds_words = topology.parent_louds_words(),
      .real_codes_small = topology.real_codes_small(),
      .edge_code0 = topology.edge_code0(),
      .virtual_codes_small = topology.virtual_codes_small(),
      .coefficients_layer_major = coefficients_layer_major,
      .real_norm2 = real_norm2,
      .real_source_norm2 = real_source_norm2,
  };
  return BuildFloatBfsResidentClusterImpl(
      source, !topology.has_validated_parent_louds());
}

template class ResidentCluster<5>;
template class ResidentCluster<12>;
template class ResidentCluster<6>;
template class ResidentCluster<10>;
template class ResidentClusterView<5>;
template class ResidentClusterView<12>;
template class ResidentClusterView<6>;
template class ResidentClusterView<10>;
template class FloatResidentClusterView<5>;
template class FloatResidentClusterView<12>;
template class FloatResidentClusterView<6>;
template class FloatResidentClusterView<10>;
template ResidentCluster<5> BuildBfsResidentCluster(
    const TrustedBfsClusterView<5>& source);
template ResidentCluster<12> BuildBfsResidentCluster(
    const TrustedBfsClusterView<12>& source);
template ResidentCluster<6> BuildBfsResidentCluster(
    const TrustedBfsClusterView<6>& source);
template ResidentCluster<10> BuildBfsResidentCluster(
    const TrustedBfsClusterView<10>& source);
template StatusOr<ResidentCluster<5>> BuildBfsResidentCluster(
    const TrustedClusterView<5>& source);
template StatusOr<ResidentCluster<12>> BuildBfsResidentCluster(
    const TrustedClusterView<12>& source);
template StatusOr<ResidentCluster<6>> BuildBfsResidentCluster(
    const TrustedClusterView<6>& source);
template StatusOr<ResidentCluster<10>> BuildBfsResidentCluster(
    const TrustedClusterView<10>& source);
template StatusOr<ResidentCluster<5>> BuildBfsResidentCluster(
    const TrustedBfsClusterView<5>& source,
    ArrayView<const float>, ArrayView<const float>);
template StatusOr<ResidentCluster<12>> BuildBfsResidentCluster(
    const TrustedBfsClusterView<12>& source,
    ArrayView<const float>, ArrayView<const float>);
template StatusOr<ResidentCluster<6>> BuildBfsResidentCluster(
    const TrustedBfsClusterView<6>& source,
    ArrayView<const float>, ArrayView<const float>);
template StatusOr<ResidentCluster<10>> BuildBfsResidentCluster(
    const TrustedBfsClusterView<10>& source,
    ArrayView<const float>, ArrayView<const float>);
template StatusOr<ResidentCluster<5>> BuildFloatBfsResidentCluster(
    const NativeFloatBfsClusterSource<5>& source);
template StatusOr<ResidentCluster<12>> BuildFloatBfsResidentCluster(
    const NativeFloatBfsClusterSource<12>& source);
template StatusOr<ResidentCluster<6>> BuildFloatBfsResidentCluster(
    const NativeFloatBfsClusterSource<6>& source);
template StatusOr<ResidentCluster<10>> BuildFloatBfsResidentCluster(
    const NativeFloatBfsClusterSource<10>& source);
template StatusOr<ResidentCluster<5>>
BuildFloatBfsResidentClusterFromDecodedTopology(
    const DecodedCanonicalTopology<5>&, ArrayView<const float>,
    ArrayView<const float>, ArrayView<const float>);
template StatusOr<ResidentCluster<12>>
BuildFloatBfsResidentClusterFromDecodedTopology(
    const DecodedCanonicalTopology<12>&, ArrayView<const float>,
    ArrayView<const float>, ArrayView<const float>);
template StatusOr<ResidentCluster<6>>
BuildFloatBfsResidentClusterFromDecodedTopology(
    const DecodedCanonicalTopology<6>&, ArrayView<const float>,
    ArrayView<const float>, ArrayView<const float>);
template StatusOr<ResidentCluster<10>>
BuildFloatBfsResidentClusterFromDecodedTopology(
    const DecodedCanonicalTopology<10>&, ArrayView<const float>,
    ArrayView<const float>, ArrayView<const float>);

}  // namespace stlqpp
