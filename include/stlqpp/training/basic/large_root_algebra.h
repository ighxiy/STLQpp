#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/model/root_codebooks.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/parallel_for.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

template <int M>
class ProductCodebooks;

// Shared by host algebra and CUDA-only model construction. Keeping the norm
// reduction in one CPU translation unit avoids a second numerical path.
void ComputeBasicCodebookNorms(
    ColumnMajorMatrixF32View root0,
    ColumnMajorMatrixF32View small,
    std::vector<float>* root_norm2,
    std::vector<float>* root_inverse_norm,
    std::vector<float>* small_norm2,
    std::vector<float>* small_inverse_norm);

// CUDA virtual roots consume only G_small. Preserve the virtual-root GEMM
// shape for small root books without preparing the unused full root/small
// cross table for large IVF models.
[[nodiscard]] StatusOr<std::vector<float>> BuildVirtualRootSmallGram(
    RuntimeRootCodebookView codebooks, const F32GemmBackend& gemm);

// Owns model/Gram storage for one active runtime code length. Rebuild retains
// allocations; its ordinary and virtual-root GEMM contracts are identical to
// the fixed-M adapter below.
class RuntimeBasicCodebookAlgebra {
 public:
  [[nodiscard]] static StatusOr<RuntimeBasicCodebookAlgebra> Create(
      RuntimeRootCodebookView codebooks, const F32GemmBackend& gemm,
      const ParallelForExecutor* parallel = nullptr);
  [[nodiscard]] static StatusOr<RuntimeBasicCodebookAlgebra> CreateForVirtualRoots(
      RuntimeRootCodebookView codebooks, const F32GemmBackend& gemm,
      const ParallelForExecutor* parallel = nullptr);
  [[nodiscard]] Status Rebuild(
      RuntimeRootCodebookView codebooks, const F32GemmBackend& gemm,
      const ParallelForExecutor* parallel = nullptr);
  [[nodiscard]] Status RebuildForVirtualRoots(
      RuntimeRootCodebookView codebooks, const F32GemmBackend& gemm,
      const ParallelForExecutor* parallel = nullptr);
  RuntimeBasicCodebookAlgebra(RuntimeBasicCodebookAlgebra&&) noexcept = default;
  RuntimeBasicCodebookAlgebra& operator=(RuntimeBasicCodebookAlgebra&&) noexcept = default;
  RuntimeBasicCodebookAlgebra(const RuntimeBasicCodebookAlgebra&) = delete;
  RuntimeBasicCodebookAlgebra& operator=(const RuntimeBasicCodebookAlgebra&) = delete;

  [[nodiscard]] const ModelShape& shape() const noexcept { return shape_; }
  [[nodiscard]] ColumnMajorMatrixF32View root0() const noexcept {
    return {full_.data(), shape_.dimension, shape_.nlist};
  }
  [[nodiscard]] ColumnMajorMatrixF32View small_flat() const noexcept {
    return {full_.data() +
                static_cast<std::size_t>(shape_.dimension) * shape_.nlist,
            shape_.dimension,
            static_cast<std::uint32_t>((shape_.m - 1U) * 256U)};
  }
  [[nodiscard]] ColumnMajorMatrixF32View full_flat() const noexcept {
    return {full_.data(), shape_.dimension, shape_.nlist + small_count()};
  }
  [[nodiscard]] bool has_dense_full_gram() const noexcept {
    return !dense_full_gram_.empty();
  }
  [[nodiscard]] float dense_full_dot(
      std::uint32_t row, std::uint32_t column) const noexcept {
    const std::uint32_t full_count = shape_.nlist + small_count();
    return dense_full_gram_[static_cast<std::size_t>(column) * full_count +
                            row];
  }
  [[nodiscard]] std::uint32_t small_count() const noexcept {
    return static_cast<std::uint32_t>((shape_.m - 1U) * 256U);
  }
  [[nodiscard]] std::uint32_t small_offset(
      std::uint32_t layer) const noexcept {
    return (layer - 1U) * 256U;
  }
  [[nodiscard]] float root_norm2(std::uint32_t cid) const noexcept {
    return root_norm2_[cid];
  }
  [[nodiscard]] float root_inverse_norm(
      std::uint32_t cid) const noexcept {
    return root_inverse_norm_[cid];
  }
  [[nodiscard]] float small_norm2(std::uint32_t flat) const noexcept {
    return small_norm2_[flat];
  }
  [[nodiscard]] float small_inverse_norm(
      std::uint32_t flat) const noexcept {
    return small_inverse_norm_[flat];
  }
  [[nodiscard]] float root_small_dot(
      std::uint32_t cid, std::uint32_t flat) const noexcept {
    return root_small_gram_[static_cast<std::size_t>(cid) * small_count() +
                            flat];
  }
  [[nodiscard]] float small_dot(
      std::uint32_t row, std::uint32_t column) const noexcept {
    return small_gram_[static_cast<std::size_t>(column) * small_count() +
                       row];
  }
  [[nodiscard]] ArrayView<const float> root_norms2() const noexcept {
    return root_norm2_;
  }
  [[nodiscard]] ArrayView<const float> root_inverse_norms() const noexcept {
    return root_inverse_norm_;
  }
  [[nodiscard]] ArrayView<const float> small_norms2() const noexcept {
    return small_norm2_;
  }
  [[nodiscard]] ArrayView<const float> small_inverse_norms() const noexcept {
    return small_inverse_norm_;
  }
  [[nodiscard]] ArrayView<const float> small_gram() const noexcept {
    return small_gram_;
  }
  [[nodiscard]] ArrayView<const float> root_small_gram() const noexcept {
    return root_small_gram_;
  }

 protected:
  RuntimeBasicCodebookAlgebra() = default;
  [[nodiscard]] Status RebuildImpl(RuntimeRootCodebookView codebooks,
                                  const F32GemmBackend& gemm,
                                  bool build_dense_virtual_gram,
                                  const ParallelForExecutor* parallel);

 private:
  ModelShape shape_;
  // DEV forms virtual-root projections with one C_all^T X GEMM. Keeping the
  // root and small books contiguous preserves that numerical contract without
  // duplicating the model.
  std::vector<float> full_;
  std::vector<float> root_norm2_;
  std::vector<float> root_inverse_norm_;
  std::vector<float> small_norm2_;
  std::vector<float> small_inverse_norm_;
  std::vector<float> small_gram_;
  std::vector<float> root_small_gram_;
  std::vector<float> dense_full_gram_;
};

template <int M>
class BasicCodebookAlgebra final : public RuntimeBasicCodebookAlgebra {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<BasicCodebookAlgebra<M>> Create(
      const RootCodebooks<M>& codebooks,
      const F32GemmBackend& gemm,
      const ParallelForExecutor* parallel = nullptr);

  // The frozen DEV virtual-root initializer forms one dense C_all^T C_all
  // product when the root book is small enough. Ordinary/base basic encoding
  // instead uses the large-root split G_small/G0S construction. Keep those
  // numerical contracts explicit rather than selecting one from model shape.
  [[nodiscard]] static StatusOr<BasicCodebookAlgebra<M>>
  CreateForVirtualRoots(
      const RootCodebooks<M>& codebooks,
      const F32GemmBackend& gemm,
      const ParallelForExecutor* parallel = nullptr);

  [[nodiscard]] static StatusOr<BasicCodebookAlgebra<M>> Create(
      const ProductCodebooks<M>& codebooks,
      const F32GemmBackend& gemm,
      const ParallelForExecutor* parallel = nullptr);

  [[nodiscard]] static StatusOr<BasicCodebookAlgebra<M>>
  CreateForVirtualRoots(
      const ProductCodebooks<M>& codebooks,
      const F32GemmBackend& gemm,
      const ParallelForExecutor* parallel = nullptr);

  // Recomputes model-dependent values while retaining fixed-shape backing
  // storage. No consumer may retain an algebra view across this call.
  [[nodiscard]] Status Rebuild(
      const RootCodebooks<M>& codebooks,
      const F32GemmBackend& gemm,
      const ParallelForExecutor* parallel = nullptr);

  [[nodiscard]] Status RebuildForVirtualRoots(
      const RootCodebooks<M>& codebooks,
      const F32GemmBackend& gemm,
      const ParallelForExecutor* parallel = nullptr);

  BasicCodebookAlgebra(BasicCodebookAlgebra&&) noexcept = default;
  BasicCodebookAlgebra& operator=(BasicCodebookAlgebra&&) noexcept = default;
  BasicCodebookAlgebra(const BasicCodebookAlgebra&) = delete;
  BasicCodebookAlgebra& operator=(const BasicCodebookAlgebra&) = delete;

  // Keep compile-time strides in existing fixed-M numerical hot paths.
  [[nodiscard]] std::uint32_t small_count() const noexcept {
    if constexpr (M == 0) return RuntimeBasicCodebookAlgebra::small_count();
    else return (M - 1U) * 256U;
  }
  [[nodiscard]] float root_small_dot(std::uint32_t cid,
                                     std::uint32_t flat) const noexcept {
    return root_small_gram()[static_cast<std::size_t>(cid) * small_count() + flat];
  }
  [[nodiscard]] float small_dot(std::uint32_t row,
                                std::uint32_t column) const noexcept {
    return small_gram()[static_cast<std::size_t>(column) * small_count() + row];
  }

 private:
  BasicCodebookAlgebra() = default;
};

extern template class BasicCodebookAlgebra<5>;
extern template class BasicCodebookAlgebra<10>;

}  // namespace stlqpp
