#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"

#include <cstdint>

namespace stlqpp {

template <int M>
class DecodedCanonicalCluster;

struct MappedClusterView final {
  std::uint32_t cid = 0U;
  std::uint32_t real_count = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  const std::uint32_t* real_ids = nullptr;
  const std::uint32_t* parent_local_1based = nullptr;
  const std::uint8_t* real_codes_small = nullptr;
  const std::uint8_t* edge_code0 = nullptr;
  const std::uint8_t* virtual_codes_small = nullptr;
  const std::int8_t* coefficients_layer_major = nullptr;
  const float* root_scales = nullptr;
  const float* linkage_scales = nullptr;
  const std::uint8_t* real_norm8 = nullptr;
  const float* norm8_centers = nullptr;
  std::uint32_t norm8_center_count = 0U;
};

// Canonical BFS cluster view whose topology remains in its succinct on-disk
// representation. Unlike MappedClusterView, this boundary never materializes
// one parent integer per node.
struct MappedBfsClusterView final {
  std::uint32_t cid = 0U;
  std::uint32_t real_count = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  const std::uint32_t* real_ids = nullptr;
  ArrayView<const std::uint64_t> parent_louds_words;
  const std::uint8_t* real_codes_small = nullptr;
  const std::uint8_t* edge_code0 = nullptr;
  const std::uint8_t* virtual_codes_small = nullptr;
  const std::int8_t* coefficients_layer_major = nullptr;
  const float* root_scales = nullptr;
  const float* linkage_scales = nullptr;
  const std::uint8_t* real_norm8 = nullptr;
  const float* norm8_centers = nullptr;
  std::uint32_t norm8_center_count = 0U;
};

template <int M>
class TrustedClusterView final {
 public:
  [[nodiscard]] static StatusOr<TrustedClusterView<M>> Create(
      const MappedClusterView& source);

  [[nodiscard]] std::uint32_t cid() const noexcept { return view_.cid; }
  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return view_.real_count;
  }
  [[nodiscard]] std::uint32_t virtual_count() const noexcept {
    return view_.virtual_count;
  }
  [[nodiscard]] std::uint32_t real_root_count() const noexcept {
    return view_.real_root_count;
  }
  [[nodiscard]] std::uint32_t node_count() const noexcept {
    return view_.real_count + view_.virtual_count;
  }
  [[nodiscard]] const std::uint32_t* real_ids() const noexcept {
    return view_.real_ids;
  }
  [[nodiscard]] const std::uint32_t* parent_local_1based() const noexcept {
    return view_.parent_local_1based;
  }
  [[nodiscard]] const std::uint8_t* real_codes_small() const noexcept {
    return view_.real_codes_small;
  }
  [[nodiscard]] const std::uint8_t* edge_code0() const noexcept {
    return view_.edge_code0;
  }
  [[nodiscard]] const std::uint8_t* virtual_codes_small() const noexcept {
    return view_.virtual_codes_small;
  }
  [[nodiscard]] const std::int8_t* coefficients(std::uint32_t layer) const
      noexcept {
    return view_.coefficients_layer_major + layer * node_count();
  }
  [[nodiscard]] float root_scale(std::uint32_t layer) const noexcept {
    return view_.root_scales[layer];
  }
  [[nodiscard]] float linkage_scale(std::uint32_t layer) const noexcept {
    return view_.linkage_scales[layer];
  }
  [[nodiscard]] const std::uint8_t* real_norm8() const noexcept {
    return view_.real_norm8;
  }
  [[nodiscard]] const float* norm8_centers() const noexcept {
    return view_.norm8_centers;
  }
  [[nodiscard]] std::uint32_t norm8_center_count() const noexcept {
    return view_.norm8_center_count;
  }

 private:
  explicit TrustedClusterView(MappedClusterView view) noexcept : view_(view) {}

  MappedClusterView view_;
};

template <int M>
class TrustedBfsClusterView final {
 public:
  [[nodiscard]] static StatusOr<TrustedBfsClusterView<M>> Create(
      const MappedBfsClusterView& source);

  [[nodiscard]] std::uint32_t cid() const noexcept { return view_.cid; }
  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return view_.real_count;
  }
  [[nodiscard]] std::uint32_t virtual_count() const noexcept {
    return view_.virtual_count;
  }
  [[nodiscard]] std::uint32_t real_root_count() const noexcept {
    return view_.real_root_count;
  }
  [[nodiscard]] std::uint32_t root_count() const noexcept {
    return view_.virtual_count + view_.real_root_count;
  }
  [[nodiscard]] std::uint32_t node_count() const noexcept {
    return view_.real_count + view_.virtual_count;
  }
  [[nodiscard]] const std::uint32_t* real_ids() const noexcept {
    return view_.real_ids;
  }
  [[nodiscard]] ArrayView<const std::uint64_t> parent_louds_words() const
      noexcept {
    return view_.parent_louds_words;
  }
  [[nodiscard]] const std::uint8_t* real_codes_small() const noexcept {
    return view_.real_codes_small;
  }
  [[nodiscard]] const std::uint8_t* edge_code0() const noexcept {
    return view_.edge_code0;
  }
  [[nodiscard]] const std::uint8_t* virtual_codes_small() const noexcept {
    return view_.virtual_codes_small;
  }
  [[nodiscard]] const std::int8_t* coefficients(std::uint32_t layer) const
      noexcept {
    return view_.coefficients_layer_major + layer * node_count();
  }
  [[nodiscard]] float root_scale(std::uint32_t layer) const noexcept {
    return view_.root_scales[layer];
  }
  [[nodiscard]] float linkage_scale(std::uint32_t layer) const noexcept {
    return view_.linkage_scales[layer];
  }
  [[nodiscard]] const std::uint8_t* real_norm8() const noexcept {
    return view_.real_norm8;
  }
  [[nodiscard]] const float* norm8_centers() const noexcept {
    return view_.norm8_centers;
  }
  [[nodiscard]] std::uint32_t norm8_center_count() const noexcept {
    return view_.norm8_center_count;
  }

 private:
  template <int N>
  friend class DecodedCanonicalCluster;

  [[nodiscard]] static StatusOr<TrustedBfsClusterView<M>>
  CreateAfterValidatedLouds(const MappedBfsClusterView& source);
  [[nodiscard]] static StatusOr<TrustedBfsClusterView<M>> CreateImpl(
      const MappedBfsClusterView& source, bool validate_louds);

  explicit TrustedBfsClusterView(MappedBfsClusterView view) noexcept
      : view_(view) {}

  MappedBfsClusterView view_;
};

extern template class TrustedClusterView<5>;
extern template class TrustedClusterView<10>;
extern template class TrustedBfsClusterView<5>;
extern template class TrustedBfsClusterView<10>;

}  // namespace stlqpp
