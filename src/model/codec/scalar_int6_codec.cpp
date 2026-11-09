#include "stlqpp/model/model_shape.h"
#include "stlqpp/model/codec/scalar_int6_codec.h"

#include "stlqpp/model/codec/canonical_huffman.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace stlqpp {
namespace {

Status ValidateEmptyGroup(const EncodedHuffmanView& group) {
  if (group.code_lengths.empty() || group.code_lengths.size() > 255U ||
      (group.code_lengths.size() & 1U) == 0U) {
    return Status::DataLoss(
        "empty coefficient group has an invalid alphabet size");
  }
  if (!group.payload.empty() ||
      std::any_of(group.code_lengths.begin(), group.code_lengths.end(),
                  [](std::uint8_t length) { return length != 0U; })) {
    return Status::DataLoss(
        "empty coefficient group must contain zero lengths and no payload");
  }
  return Status::Ok();
}

Status DecodeGroup(const EncodedHuffmanView& group,
                   std::uint32_t row_count,
                   std::uint32_t row_offset,
                   std::uint32_t node_count,
                   std::uint32_t layers,
                   std::vector<std::int8_t>& output) {
  if (row_count == 0U) {
    return ValidateEmptyGroup(group);
  }
  if (group.code_lengths.empty() || group.code_lengths.size() > 255U ||
      (group.code_lengths.size() & 1U) == 0U) {
    return Status::DataLoss(
        "coefficient Huffman alphabet must be odd and contain at most 255 symbols");
  }
  const std::int32_t symbol_base = static_cast<std::int32_t>(
      (group.code_lengths.size() - 1U) / 2U);
  const std::uint64_t symbol_count =
      static_cast<std::uint64_t>(row_count) * layers;
  if (symbol_count > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted(
        "coefficient group exceeds addressable memory");
  }

  auto decoder_result =
      CanonicalHuffmanDecoder::Create(group.code_lengths);
  if (!decoder_result.ok()) {
    return decoder_result.status();
  }
  const CanonicalHuffmanDecoder decoder =
      std::move(decoder_result).value();
  std::vector<std::uint8_t> symbols(static_cast<std::size_t>(symbol_count));
  if (const Status status = decoder.DecodeMsbFirst(group.payload, symbols);
      !status.ok()) {
    return status;
  }

  for (std::uint32_t layer = 0U; layer < layers; ++layer) {
    const std::size_t source_begin =
        static_cast<std::size_t>(layer) * row_count;
    const std::size_t destination_begin =
        static_cast<std::size_t>(layer) * node_count + row_offset;
    for (std::uint32_t row = 0U; row < row_count; ++row) {
      output[destination_begin + row] = static_cast<std::int8_t>(
          static_cast<std::int32_t>(symbols[source_begin + row]) -
          symbol_base);
    }
  }
  return Status::Ok();
}

}  // namespace

StatusOr<RuntimeDecodedScalarInt6Cluster> DecodeScalarInt6ClusterRuntime(
    const RuntimeScalarInt6ClusterView& source) {
  if (source.m < 2U || source.m > stlqpp::kMaximumModelLayers ||
      source.root_scales.size() != source.m ||
      source.linkage_scales.size() != source.m ||
      source.root_scales.data() == nullptr ||
      source.linkage_scales.data() == nullptr) {
    return Status::DataLoss(
        "runtime coefficient cluster scale count does not match M");
  }
  for (std::uint32_t layer = 0U; layer < source.m; ++layer) {
    if (!std::isfinite(source.root_scales[layer]) ||
        !std::isfinite(source.linkage_scales[layer])) {
      return Status::DataLoss(
          "runtime coefficient cluster contains an invalid scale");
    }
  }
  const std::uint64_t root_count =
      static_cast<std::uint64_t>(source.virtual_root_count) +
      source.real_root_count;
  const std::uint64_t node_count = root_count + source.linked_real_count;
  if (root_count > std::numeric_limits<std::uint32_t>::max() ||
      node_count > std::numeric_limits<std::uint32_t>::max() ||
      node_count > std::numeric_limits<std::size_t>::max() / source.m) {
    return Status::ResourceExhausted(
        "runtime coefficient cluster shape exceeds storage");
  }
  RuntimeDecodedScalarInt6Cluster output;
  output.m = source.m;
  output.virtual_root_count = source.virtual_root_count;
  output.real_root_count = source.real_root_count;
  output.linked_real_count = source.linked_real_count;
  output.root_scales.assign(source.root_scales.begin(),
                            source.root_scales.end());
  output.linkage_scales.assign(source.linkage_scales.begin(),
                               source.linkage_scales.end());
  output.coefficients_layer_major.resize(
      static_cast<std::size_t>(node_count) * source.m);
  const auto roots = static_cast<std::uint32_t>(root_count);
  const auto nodes = static_cast<std::uint32_t>(node_count);
  if (const Status status = DecodeGroup(
          source.root, roots, 0U, nodes, source.m,
          output.coefficients_layer_major);
      !status.ok()) {
    return status;
  }
  if (const Status status = DecodeGroup(
          source.linkage, source.linked_real_count, roots, nodes, source.m,
          output.coefficients_layer_major);
      !status.ok()) {
    return status;
  }
  return output;
}

template <int M>
StatusOr<DecodedScalarInt6Cluster<M>> DecodeScalarInt6Cluster(
    const ScalarInt6ClusterView<M>& source) {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  const auto m = M == 0 ? source.m : M;
  if (m < 2U || m > stlqpp::kMaximumModelLayers || source.m != m) return Status::DataLoss("coefficient M mismatch");
  if (source.root_scales.size() != m ||
      source.linkage_scales.size() != m) {
    return Status::DataLoss(
        "coefficient cluster scale count does not match M");
  }
  for (std::uint32_t layer = 0U; layer < m;
       ++layer) {
    if (!std::isfinite(source.root_scales[layer]) ||
        !std::isfinite(source.linkage_scales[layer])) {
      return Status::DataLoss(
          "coefficient cluster contains an invalid scale");
    }
  }

  const std::uint64_t root_count =
      static_cast<std::uint64_t>(source.virtual_root_count) +
      source.real_root_count;
  const std::uint64_t node_count = root_count + source.linked_real_count;
  if (root_count > std::numeric_limits<std::uint32_t>::max() ||
      node_count > std::numeric_limits<std::uint32_t>::max() ||
      node_count > std::numeric_limits<std::size_t>::max() / m) {
    return Status::ResourceExhausted(
        "coefficient cluster shape exceeds u32 storage");
  }

  DecodedScalarInt6Cluster<M> output;
  if constexpr (M == 0) {
    output.root_scales_.resize(m);
    output.linkage_scales_.resize(m);
  }
  output.virtual_root_count_ = source.virtual_root_count;
  output.real_root_count_ = source.real_root_count;
  output.linked_real_count_ = source.linked_real_count;
  std::copy(source.root_scales.begin(), source.root_scales.end(),
            output.root_scales_.begin());
  std::copy(source.linkage_scales.begin(), source.linkage_scales.end(),
            output.linkage_scales_.begin());
  output.coefficients_layer_major_.resize(
      static_cast<std::size_t>(node_count) * m);

  const auto roots = static_cast<std::uint32_t>(root_count);
  const auto nodes = static_cast<std::uint32_t>(node_count);
  if (const Status status =
          DecodeGroup(source.root, roots, 0U, nodes, m,
                      output.coefficients_layer_major_);
      !status.ok()) {
    return status;
  }
  if (const Status status =
          DecodeGroup(source.linkage, source.linked_real_count, roots, nodes,
                      m, output.coefficients_layer_major_);
      !status.ok()) {
    return status;
  }
  return output;
}

template class DecodedScalarInt6Cluster<0>;
template StatusOr<DecodedScalarInt6Cluster<0>> DecodeScalarInt6Cluster(
    const ScalarInt6ClusterView<0>& source);
template class DecodedScalarInt6Cluster<5>;
template class DecodedScalarInt6Cluster<12>;
template class DecodedScalarInt6Cluster<6>;
template class DecodedScalarInt6Cluster<10>;
template StatusOr<DecodedScalarInt6Cluster<5>> DecodeScalarInt6Cluster(
    const ScalarInt6ClusterView<5>& source);
template StatusOr<DecodedScalarInt6Cluster<12>> DecodeScalarInt6Cluster(
    const ScalarInt6ClusterView<12>& source);
template StatusOr<DecodedScalarInt6Cluster<6>> DecodeScalarInt6Cluster(
    const ScalarInt6ClusterView<6>& source);
template StatusOr<DecodedScalarInt6Cluster<10>> DecodeScalarInt6Cluster(
    const ScalarInt6ClusterView<10>& source);

}  // namespace stlqpp
