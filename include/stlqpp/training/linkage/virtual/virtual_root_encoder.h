#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/training/linkage/virtual/umap_virtual_root_plan.h"
#include "stlqpp/training/basic/virtual_root_support_encoder.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {

class EncodedVirtualRoots;

// The virtual-root subset that remains live after topology construction.
// Exact centers, reconstructions and construction errors never cross the
// worker-to-publication boundary.
class EncodedVirtualRootsPublication final {
 public:
  [[nodiscard]] std::uint32_t dimension() const noexcept {
    return dimension_;
  }
  [[nodiscard]] std::uint32_t count() const noexcept {
    return static_cast<std::uint32_t>(sample_ids_.size());
  }
  [[nodiscard]] ArrayView<const std::uint64_t> sample_ids() const noexcept {
    return sample_ids_;
  }
  [[nodiscard]] ArrayView<const std::uint8_t> small_codes() const noexcept {
    return small_codes_;
  }
  [[nodiscard]] ArrayView<const float> coefficients() const noexcept {
    return coefficients_;
  }
  [[nodiscard]] std::uint64_t resident_bytes() const noexcept {
    return sizeof(*this) +
        static_cast<std::uint64_t>(sample_ids_.capacity()) * sizeof(std::uint64_t) +
        small_codes_.capacity() +
        static_cast<std::uint64_t>(coefficients_.capacity()) * sizeof(float);
  }
  [[nodiscard]] std::vector<std::uint8_t> TakeSmallCodes() && noexcept {
    return std::move(small_codes_);
  }
  [[nodiscard]] std::vector<float> TakeCoefficients() && noexcept {
    return std::move(coefficients_);
  }

 private:
  friend class EncodedVirtualRoots;

  std::uint32_t dimension_ = 0U;
  std::vector<std::uint64_t> sample_ids_;
  std::vector<std::uint8_t> small_codes_;
  std::vector<float> coefficients_;
};

class EncodedVirtualRoots final {
 public:
  [[nodiscard]] std::uint32_t dimension() const noexcept {
    return dimension_;
  }
  [[nodiscard]] std::uint32_t count() const noexcept {
    return static_cast<std::uint32_t>(sample_ids_.size());
  }
  [[nodiscard]] ColumnMajorMatrixF32View rotated_exact() const noexcept {
    return {rotated_exact_.data(), dimension_, count()};
  }
  [[nodiscard]] ColumnMajorMatrixF32View reconstructions() const noexcept {
    return {reconstructions_.data(), dimension_, count()};
  }
  [[nodiscard]] ArrayView<const std::uint64_t> sample_ids() const noexcept {
    return sample_ids_;
  }
  [[nodiscard]] ArrayView<const std::uint8_t> small_codes() const noexcept {
    return small_codes_;
  }
  [[nodiscard]] ArrayView<const float> coefficients() const noexcept {
    return coefficients_;
  }
  [[nodiscard]] ArrayView<const float> squared_errors() const noexcept {
    return squared_errors_;
  }
  [[nodiscard]] std::vector<float> TakeReconstructions() noexcept {
    return std::move(reconstructions_);
  }
  void ReleaseConstructionOnly() noexcept {
    std::vector<float>().swap(unrotated_exact_);
    std::vector<float>().swap(rotated_exact_);
    std::vector<float>().swap(squared_errors_);
  }
  [[nodiscard]] EncodedVirtualRootsPublication TakePublication()
      && noexcept {
    EncodedVirtualRootsPublication output;
    output.dimension_ = dimension_;
    output.sample_ids_ = std::move(sample_ids_);
    output.small_codes_ = std::move(small_codes_);
    output.coefficients_ = std::move(coefficients_);
    return output;
  }

 private:
  template <int N>
  friend StatusOr<EncodedVirtualRoots> EncodeVirtualRoots(
      std::uint32_t,
      ColumnMajorMatrixF32View,
      ColumnMajorMatrixF32View,
      const UmapVirtualRootPlan&,
      const ProductCodebooks<N>&,
      const VirtualRootSupportEncoder<N>&,
      std::uint32_t);

  std::uint32_t dimension_ = 0U;
  std::vector<float> unrotated_exact_;
  std::vector<float> rotated_exact_;
  std::vector<std::uint64_t> sample_ids_;
  std::vector<std::uint8_t> small_codes_;
  std::vector<float> coefficients_;
  std::vector<float> squared_errors_;
  std::vector<float> reconstructions_;
};

// Encodes the barycentric centers with the cluster's fixed native CID.
template <int M>
[[nodiscard]] StatusOr<EncodedVirtualRoots> EncodeVirtualRoots(
    std::uint32_t cid,
    ColumnMajorMatrixF32View unrotated_real,
    ColumnMajorMatrixF32View rotated_real,
    const UmapVirtualRootPlan& plan,
    const ProductCodebooks<M>& codebooks,
    const VirtualRootSupportEncoder<M>& encoder,
    std::uint32_t seed);

extern template StatusOr<EncodedVirtualRoots> EncodeVirtualRoots<5>(
    std::uint32_t,
    ColumnMajorMatrixF32View,
    ColumnMajorMatrixF32View,
    const UmapVirtualRootPlan&,
    const ProductCodebooks<5>&,
    const VirtualRootSupportEncoder<5>&,
    std::uint32_t);
extern template StatusOr<EncodedVirtualRoots> EncodeVirtualRoots<10>(
    std::uint32_t,
    ColumnMajorMatrixF32View,
    ColumnMajorMatrixF32View,
    const UmapVirtualRootPlan&,
    const ProductCodebooks<10>&,
    const VirtualRootSupportEncoder<10>&,
    std::uint32_t);

}  // namespace stlqpp
