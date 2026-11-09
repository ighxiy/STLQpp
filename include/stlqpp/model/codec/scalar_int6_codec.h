#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/model/model_layers.h"
#include "stlqpp/core/status_or.h"

#include <array>
#include <cstdint>
#include <vector>

namespace stlqpp {

struct EncodedHuffmanView final {
  ArrayView<const std::uint8_t> code_lengths;
  ArrayView<const std::uint8_t> payload;
};

struct RuntimeScalarInt6ClusterView final {
  std::uint32_t m = 0U;
  std::uint32_t virtual_root_count = 0U;
  std::uint32_t real_root_count = 0U;
  std::uint32_t linked_real_count = 0U;
  ArrayView<const float> root_scales;
  ArrayView<const float> linkage_scales;
  EncodedHuffmanView root;
  EncodedHuffmanView linkage;
};

struct RuntimeDecodedScalarInt6Cluster final {
  std::uint32_t m = 0U;
  std::uint32_t virtual_root_count = 0U;
  std::uint32_t real_root_count = 0U;
  std::uint32_t linked_real_count = 0U;
  std::vector<float> root_scales;
  std::vector<float> linkage_scales;
  // Layer-major; each layer contains all virtual, real-root, then linked rows.
  std::vector<std::int8_t> coefficients_layer_major;
};

[[nodiscard]] StatusOr<RuntimeDecodedScalarInt6Cluster>
DecodeScalarInt6ClusterRuntime(const RuntimeScalarInt6ClusterView& source);

template <int M>
struct ScalarInt6ClusterView final {
  std::uint32_t m = M;
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t virtual_root_count = 0U;
  std::uint32_t real_root_count = 0U;
  std::uint32_t linked_real_count = 0U;
  ArrayView<const float> root_scales;
  ArrayView<const float> linkage_scales;
  EncodedHuffmanView root;
  EncodedHuffmanView linkage;
};

template <int M>
class DecodedScalarInt6Cluster final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] std::uint32_t virtual_root_count() const noexcept {
    return virtual_root_count_;
  }
  [[nodiscard]] std::uint32_t real_root_count() const noexcept {
    return real_root_count_;
  }
  [[nodiscard]] std::uint32_t linked_real_count() const noexcept {
    return linked_real_count_;
  }
  [[nodiscard]] std::uint32_t node_count() const noexcept {
    return virtual_root_count_ + real_root_count_ + linked_real_count_;
  }
  [[nodiscard]] const ModelLayers<float, M>& root_scales() const noexcept {
    return root_scales_;
  }
  [[nodiscard]] const ModelLayers<float, M>& linkage_scales() const noexcept {
    return linkage_scales_;
  }
  [[nodiscard]] const std::vector<std::int8_t>& coefficients() const noexcept {
    return coefficients_layer_major_;
  }

 private:
  template <int N>
  friend StatusOr<DecodedScalarInt6Cluster<N>> DecodeScalarInt6Cluster(
      const ScalarInt6ClusterView<N>& source);

  std::uint32_t virtual_root_count_ = 0U;
  std::uint32_t real_root_count_ = 0U;
  std::uint32_t linked_real_count_ = 0U;
  ModelLayers<float, M> root_scales_{};
  ModelLayers<float, M> linkage_scales_{};
  std::vector<std::int8_t> coefficients_layer_major_;
};

template <int M>
[[nodiscard]] StatusOr<DecodedScalarInt6Cluster<M>> DecodeScalarInt6Cluster(
    const ScalarInt6ClusterView<M>& source);

extern template class DecodedScalarInt6Cluster<5>;
extern template class DecodedScalarInt6Cluster<10>;
extern template StatusOr<DecodedScalarInt6Cluster<5>> DecodeScalarInt6Cluster(
    const ScalarInt6ClusterView<5>& source);
extern template StatusOr<DecodedScalarInt6Cluster<10>> DecodeScalarInt6Cluster(
    const ScalarInt6ClusterView<10>& source);

}  // namespace stlqpp
