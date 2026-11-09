#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

// Checked projection of one immutable residual batch. Atom dots and norms are
// computed once and shared by RSD construction, candidate ranking and
// ICM.
template <int M>
class LinkageProjection final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  LinkageProjection() = default;

  [[nodiscard]] static StatusOr<LinkageProjection<M>> Create(
      ColumnMajorMatrixF32View residuals,
      const LinkageCodebookAlgebra<M>& algebra,
      const F32GemmBackend& gemm,
      const ParallelForExecutor& parallel);

  [[nodiscard]] Status Recompute(
      ColumnMajorMatrixF32View residuals,
      const LinkageCodebookAlgebra<M>& algebra,
      const F32GemmBackend& gemm,
      const ParallelForExecutor& parallel);

  LinkageProjection(LinkageProjection&&) noexcept = default;
  LinkageProjection& operator=(LinkageProjection&&) noexcept = default;
  LinkageProjection(const LinkageProjection&) = delete;
  LinkageProjection& operator=(const LinkageProjection&) = delete;

  [[nodiscard]] std::uint32_t sample_count() const noexcept {
    return residuals_.columns;
  }
  [[nodiscard]] float squared_norm(std::uint32_t sample) const noexcept {
    return squared_norms_[sample];
  }
  [[nodiscard]] const float* atom_dots(std::uint32_t sample) const noexcept {
    return atom_dots_.data() +
        static_cast<std::size_t>(sample) *
            (M == 0 ? atom_count_ : M * 256U);
  }

  [[nodiscard]] ArrayView<const float> squared_norms() const noexcept {
    return squared_norms_;
  }
  [[nodiscard]] ArrayView<const float> all_atom_dots() const noexcept {
    return atom_dots_;
  }

 private:
  std::uint32_t atom_count_ = M * 256U;
  ColumnMajorMatrixF32View residuals_;
  std::vector<float> squared_norms_;
  std::vector<float> atom_dots_;
};

extern template class LinkageProjection<5>;
extern template class LinkageProjection<10>;

}  // namespace stlqpp
