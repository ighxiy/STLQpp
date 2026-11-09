#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"

namespace stlqpp {

// Model-owned algebra for the M linkage codebooks. Atom indices are flattened
// as layer * 256 + code. The full Gram matrix is paid once per model and is
// shared by RSD construction, exact support refits and ICM.
class RuntimeLinkageCodebookAlgebra {
 public:
  [[nodiscard]] static StatusOr<RuntimeLinkageCodebookAlgebra> Create(
      ModelShape shape, ColumnMajorMatrixF32View atoms,
      const F32GemmBackend& gemm);

  RuntimeLinkageCodebookAlgebra(RuntimeLinkageCodebookAlgebra&&) noexcept =
      default;
  RuntimeLinkageCodebookAlgebra& operator=(
      RuntimeLinkageCodebookAlgebra&&) noexcept = default;
  RuntimeLinkageCodebookAlgebra(const RuntimeLinkageCodebookAlgebra&) = delete;
  RuntimeLinkageCodebookAlgebra& operator=(
      const RuntimeLinkageCodebookAlgebra&) = delete;

  [[nodiscard]] const ModelShape& shape() const noexcept { return shape_; }
  [[nodiscard]] std::uint32_t atom_count() const noexcept {
    return shape_.m * 256U;
  }
  [[nodiscard]] static constexpr std::uint32_t layer_offset(
      std::uint32_t layer) noexcept {
    return layer * 256U;
  }
  [[nodiscard]] ColumnMajorMatrixF32View atoms() const noexcept {
    return {atoms_.data(), shape_.dimension, atom_count()};
  }
  [[nodiscard]] float squared_norm(std::uint32_t atom) const noexcept {
    return squared_norms_[atom];
  }
  [[nodiscard]] float inverse_norm(std::uint32_t atom) const noexcept {
    return inverse_norms_[atom];
  }
  [[nodiscard]] float dot(std::uint32_t row,
                          std::uint32_t column) const noexcept {
    return gram_[static_cast<std::size_t>(column) * atom_count() + row];
  }
  [[nodiscard]] ArrayView<const float> squared_norms() const noexcept {
    return squared_norms_;
  }
  [[nodiscard]] ArrayView<const float> inverse_norms() const noexcept {
    return inverse_norms_;
  }
  [[nodiscard]] ArrayView<const float> gram() const noexcept { return gram_; }

 protected:
  RuntimeLinkageCodebookAlgebra() = default;
  [[nodiscard]] Status Initialize(ModelShape shape,
                                  ColumnMajorMatrixF32View atoms,
                                  const F32GemmBackend& gemm);

 private:
  ModelShape shape_;
  std::vector<float> atoms_;
  std::vector<float> squared_norms_;
  std::vector<float> inverse_norms_;
  std::vector<float> gram_;
};

template <int M>
class LinkageCodebookAlgebra final : public RuntimeLinkageCodebookAlgebra {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  [[nodiscard]] static StatusOr<LinkageCodebookAlgebra<M>> Create(
      const ProductCodebooks<M>& codebooks, const F32GemmBackend& gemm);
  template <int N = M, std::enable_if_t<N == M && N == 0, int> = 0>
  [[nodiscard]] static StatusOr<LinkageCodebookAlgebra<M>> Create(
      ModelShape shape, ColumnMajorMatrixF32View atoms,
      const F32GemmBackend& gemm) {
    auto created = RuntimeLinkageCodebookAlgebra::Create(shape, atoms, gemm);
    if (!created.ok()) return created.status();
    return LinkageCodebookAlgebra(std::move(created).value());
  }
  [[nodiscard]] std::uint32_t atom_count() const noexcept {
    if constexpr (M == 0)
      return RuntimeLinkageCodebookAlgebra::atom_count();
    else
      return M * 256U;
  }
  [[nodiscard]] float dot(std::uint32_t row,
                          std::uint32_t column) const noexcept {
    return gram()[static_cast<std::size_t>(column) * atom_count() + row];
  }

 private:
  LinkageCodebookAlgebra() = default;
  explicit LinkageCodebookAlgebra(RuntimeLinkageCodebookAlgebra&& algebra)
      : RuntimeLinkageCodebookAlgebra(std::move(algebra)) {}
};

extern template class LinkageCodebookAlgebra<5>;
extern template class LinkageCodebookAlgebra<10>;

}  // namespace stlqpp
