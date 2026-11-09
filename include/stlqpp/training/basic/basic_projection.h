#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/basic/large_root_algebra.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

// Checked immutable projection of one streaming input chunk. The input matrix
// must outlive this object. Small-layer dots and vector norms are computed once
// and shared by RSD, ICM and every ILS round.
class RuntimeBasicProjection {
 public:
  explicit RuntimeBasicProjection(std::uint32_t m = 0U) noexcept : m_(m) {}

  [[nodiscard]] static StatusOr<RuntimeBasicProjection> Create(
      ColumnMajorMatrixF32View samples,
      const RuntimeBasicCodebookAlgebra& algebra,
      const F32GemmBackend& gemm,
      const ParallelForExecutor& parallel);

  // Recomputes the projection while retaining the high-watermark storage.
  // CPU streaming basic uses one instance for every 8192-row numerical tile.
  [[nodiscard]] Status Recompute(
      ColumnMajorMatrixF32View samples,
      const RuntimeBasicCodebookAlgebra& algebra,
      const F32GemmBackend& gemm,
      const ParallelForExecutor& parallel);

  // RSD already owns one [M * 256, samples] atom projection. Basic
  // refinement needs the same norms and the contiguous layers 1..M-1
  // suffix. Bind those immutable values instead of issuing a duplicate GEMM.
  // The caller must keep both external arrays alive through refinement.
  [[nodiscard]] Status BindFullAtomProjection(
      ColumnMajorMatrixF32View samples,
      ArrayView<const float> squared_norms,
      ArrayView<const float> atom_dots,
      std::uint32_t atom_stride,
      std::uint32_t small_offset);

  RuntimeBasicProjection(RuntimeBasicProjection&&) noexcept = default;
  RuntimeBasicProjection& operator=(RuntimeBasicProjection&&) noexcept = default;
  RuntimeBasicProjection(const RuntimeBasicProjection&) = delete;
  RuntimeBasicProjection& operator=(const RuntimeBasicProjection&) = delete;

  [[nodiscard]] ColumnMajorMatrixF32View samples() const noexcept {
    return samples_;
  }
  [[nodiscard]] std::uint32_t sample_count() const noexcept {
    return samples_.columns;
  }
  [[nodiscard]] float squared_norm(std::uint32_t sample) const noexcept {
    return external_squared_norms_ == nullptr
        ? squared_norms_[sample] : external_squared_norms_[sample];
  }
  [[nodiscard]] const float* small_dots(
      std::uint32_t sample) const noexcept {
    return SmallDots(sample, small_count());
  }
  [[nodiscard]] std::uint32_t small_count() const noexcept {
    return static_cast<std::uint32_t>((m_ - 1U) * 256U);
  }

 protected:
  [[nodiscard]] const float* SmallDots(
      std::uint32_t sample, std::uint32_t count) const noexcept {
    return external_atom_dots_ == nullptr
        ? small_dots_.data() +
            static_cast<std::size_t>(sample) * count
        : external_atom_dots_ +
            static_cast<std::size_t>(sample) * external_atom_stride_ +
            external_small_offset_;
  }

 private:
  std::uint32_t m_ = 0U;
  ColumnMajorMatrixF32View samples_;
  std::vector<float> squared_norms_;
  std::vector<float> small_dots_;
  const float* external_squared_norms_ = nullptr;
  const float* external_atom_dots_ = nullptr;
  std::uint32_t external_atom_stride_ = 0U;
  std::uint32_t external_small_offset_ = 0U;
};

template <int M>
class BasicProjection final : public RuntimeBasicProjection {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  BasicProjection() noexcept : RuntimeBasicProjection(M) {}
  explicit BasicProjection(std::uint32_t actual_m) noexcept
      : RuntimeBasicProjection(M == 0 ? actual_m : M) {}
  [[nodiscard]] std::uint32_t small_count() const noexcept {
    if constexpr (M == 0) return RuntimeBasicProjection::small_count();
    else return (M - 1U) * 256U;
  }
  [[nodiscard]] const float* small_dots(std::uint32_t sample) const noexcept {
    return SmallDots(sample, small_count());
  }
  [[nodiscard]] Status Recompute(
      ColumnMajorMatrixF32View samples, const BasicCodebookAlgebra<M>& algebra,
      const F32GemmBackend& gemm, const ParallelForExecutor& parallel) {
    return RuntimeBasicProjection::Recompute(samples, algebra, gemm, parallel);
  }
  [[nodiscard]] static StatusOr<BasicProjection<M>> Create(
      ColumnMajorMatrixF32View samples, const BasicCodebookAlgebra<M>& algebra,
      const F32GemmBackend& gemm, const ParallelForExecutor& parallel);
};

extern template class BasicProjection<5>;
extern template class BasicProjection<10>;

}  // namespace stlqpp
