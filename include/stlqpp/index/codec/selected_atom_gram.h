#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

struct ReconstructionGramModelView final {
  ModelShape shape;
  ArrayView<const float> root_small;
  ArrayView<const float> edge;
  ArrayView<const float> root_small_edge;
};

struct ReconstructionRootCrossView final {
  float norm2 = 0.0F;
  ArrayView<const float> small;
  ArrayView<const float> edge;
};

template <int M>
struct CanonicalClusterCodeView final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t m = M;
  std::uint32_t cid = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  std::uint32_t linked_real_count = 0U;
  ArrayView<const std::uint8_t> virtual_codes_small;
  ArrayView<const std::uint8_t> real_codes_small;
  // One byte per real row. Root positions are canonical padding; linked
  // positions carry C_one layer-0 codes.
  ArrayView<const std::uint8_t> edge_code0;
};

template <int M>
class RootSmallCrossBlock final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] std::uint32_t cid_begin() const noexcept {
    return cid_begin_;
  }
  [[nodiscard]] std::uint32_t cid_count() const noexcept {
    return static_cast<std::uint32_t>(root_norm2_.size());
  }

  [[nodiscard]] ReconstructionRootCrossView norm_view(std::uint32_t cid) const noexcept {
    const auto local = cid - cid_begin_;
    const auto small = root_small_cross_column_major_.size() / cid_count();
    const auto edge = root_edge_cross_column_major_.size() / cid_count();
    return {root_norm2_[local],
            {small ? root_small_cross_column_major_.data() + local * small : nullptr, small},
            {edge ? root_edge_cross_column_major_.data() + local * edge : nullptr, edge}};
  }

 private:
  template <int N>
  friend class SelectedAtomGramTables;
  template <int N>
  friend class SelectedAtomGramView;

  std::uint32_t cid_begin_ = 0U;
  std::vector<float> root_norm2_;
  std::vector<float> root_small_cross_column_major_;
  std::vector<float> root_edge_cross_column_major_;
};

template <int M, bool kLinkage>
class SelectedAtomRoleGramView final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] std::uint32_t m() const noexcept { return M == 0 ? m_ : M; }
  [[nodiscard]] float At(std::uint32_t row,
                         std::uint32_t layer,
                         std::uint32_t other) const noexcept {
    if constexpr (kLinkage) {
      const std::uint32_t real = real_root_count_ + row - root_count_;
      const std::uint32_t left_code = layer == 0U
          ? edge_code0_[real]
          : real_codes_small_[
                static_cast<std::size_t>(real) * (m() - 1) + layer - 1U];
      const std::uint32_t right_code = other == 0U
          ? edge_code0_[real]
          : real_codes_small_[
                static_cast<std::size_t>(real) * (m() - 1) + other - 1U];
      const std::uint32_t left = layer * 256U + left_code;
      const std::uint32_t right = other * 256U + right_code;
      return shared_gram_[static_cast<std::size_t>(left) +
                          static_cast<std::size_t>(right) * m() * 256U];
    } else {
      if (layer == 0U && other == 0U) {
        return root_norm2_;
      }
      if (layer == 0U || other == 0U) {
        const std::uint32_t small_layer = layer == 0U ? other : layer;
        const std::uint32_t atom =
            (small_layer - 1U) * 256U + RootCode(row, small_layer);
        return root_cross_[atom];
      }
      const std::uint32_t left =
          (layer - 1U) * 256U + RootCode(row, layer);
      const std::uint32_t right =
          (other - 1U) * 256U + RootCode(row, other);
      return shared_gram_[static_cast<std::size_t>(left) +
                          static_cast<std::size_t>(right) *
                              (m() - 1) * 256U];
    }
  }

  // DEV initializes integer coefficients with independently accumulated
  // atom energies.  The later reconstruction-aligned fit uses At(), whose
  // diagonal comes from GEMM; the two FP32 accumulation orders are distinct.
  [[nodiscard]] float Weight(std::uint32_t row,
                             std::uint32_t layer) const noexcept {
    if constexpr (kLinkage) {
      const std::uint32_t real = real_root_count_ + row - root_count_;
      const std::uint32_t code = layer == 0U
          ? edge_code0_[real]
          : real_codes_small_[
                static_cast<std::size_t>(real) * (m() - 1) + layer - 1U];
      return atom_norm2_[layer * 256U + code];
    } else {
      if (layer == 0U) {
        return root_norm2_;
      }
      const std::uint32_t atom =
          (layer - 1U) * 256U + RootCode(row, layer);
      return atom_norm2_[atom];
    }
  }

 private:
  template <int N>
  friend class SelectedAtomGramTables;
  template <int N>
  friend class SelectedAtomGramView;

  [[nodiscard]] std::uint32_t RootCode(
      std::uint32_t root, std::uint32_t layer) const noexcept {
    if (root < virtual_count_) {
      return virtual_codes_small_[
          static_cast<std::size_t>(root) * (m() - 1) + layer - 1U];
    }
    const std::uint32_t real_root = root - virtual_count_;
    return real_codes_small_[
        static_cast<std::size_t>(real_root) * (m() - 1) + layer - 1U];
  }

  std::uint32_t m_ = M;
  std::uint32_t virtual_count_ = 0U;
  std::uint32_t root_count_ = 0U;
  std::uint32_t real_root_count_ = 0U;
  ArrayView<const std::uint8_t> virtual_codes_small_;
  ArrayView<const std::uint8_t> real_codes_small_;
  ArrayView<const std::uint8_t> edge_code0_;
  const float* root_cross_ = nullptr;
  const float* shared_gram_ = nullptr;
  const float* atom_norm2_ = nullptr;
  float root_norm2_ = 0.0F;
};

template <int M>
class SelectedAtomGramView final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] std::uint32_t m() const noexcept { return roots_.m(); }
  [[nodiscard]] std::uint32_t root_count() const noexcept {
    return root_count_;
  }
  [[nodiscard]] std::uint32_t node_count() const noexcept {
    return node_count_;
  }
  [[nodiscard]] const SelectedAtomRoleGramView<M, false>& roots()
      const noexcept {
    return roots_;
  }
  [[nodiscard]] const SelectedAtomRoleGramView<M, true>& linkage()
      const noexcept {
    return linkage_;
  }

  // Cross-role products used by DEV's causal reconstruction-norm recurrence.
  // Root rows are virtual roots followed by real roots; linkage rows use the
  // complete local node index and are always at or beyond root_count().
  [[nodiscard]] float RootLinkageAt(
      std::uint32_t root_row, std::uint32_t root_layer,
      std::uint32_t linkage_row, std::uint32_t linkage_layer) const noexcept;
  [[nodiscard]] float LinkageCrossAt(
      std::uint32_t left_row, std::uint32_t left_layer,
      std::uint32_t right_row, std::uint32_t right_layer) const noexcept;

 private:
  template <int N>
  friend class SelectedAtomGramTables;

  std::uint32_t root_count_ = 0U;
  std::uint32_t node_count_ = 0U;
  SelectedAtomRoleGramView<M, false> roots_;
  SelectedAtomRoleGramView<M, true> linkage_;
  const float* root0_edge_cross_ = nullptr;
  const float* root_small_edge_gram_ = nullptr;
  const float* edge_gram_ = nullptr;
};

enum class SelectedAtomGramDemand { kCoefficientCodec, kReconstructionNorm };

// Model-level Gram tables shared by all CIDs. Only root0-vs-small products
// depend on CID. They are generated once per CID with the registered DEV SIMD
// reduction order, then reused by every root in that CID. This avoids both a
// roughly nlist*(M-1)*256 persistent table and per-node dimension-length dot
// products during final coefficient encoding.
template <int M>
class SelectedAtomGramTables final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<SelectedAtomGramTables<M>> Create(
      const ProductCodebooks<M>& codebooks,
      const F32GemmBackend& gemm,
      SelectedAtomGramDemand demand = SelectedAtomGramDemand::kReconstructionNorm);

  [[nodiscard]] static StatusOr<SelectedAtomGramTables<M>> Create(
      RuntimeProductCodebookView codebooks, const F32GemmBackend& gemm,
      SelectedAtomGramDemand demand = SelectedAtomGramDemand::kReconstructionNorm);

  [[nodiscard]] ReconstructionGramModelView norm_model_view() const noexcept {
    return {codebooks_.shape, root_small_gram_column_major_, edge_gram_column_major_,
            root_small_edge_gram_column_major_};
  }

  [[nodiscard]] StatusOr<RootSmallCrossBlock<M>> BuildRootCrossBlock(
      std::uint32_t cid_begin,
      std::uint32_t cid_count) const;

  // Binds one canonical CID to the shared model Gram tables. The returned
  // views perform source-faithful code-indexed lookups and allocate no
  // per-node MxM state.
  [[nodiscard]] StatusOr<SelectedAtomGramView<M>> SelectCluster(
      const CanonicalClusterCodeView<M>& cluster,
      const RootSmallCrossBlock<M>& root_cross) const;

  // Emits one row-major MxM matrix per canonical node: virtual roots, real
  // roots, then linked real rows. Output ownership stays with the caller and
  // may be reused for the next CID.
  [[nodiscard]] Status MaterializeCluster(
      const CanonicalClusterCodeView<M>& cluster,
      const RootSmallCrossBlock<M>& root_cross,
      std::vector<float>& output) const;

 private:
  SelectedAtomGramTables() = default;

  RuntimeProductCodebookView codebooks_;
  SelectedAtomGramDemand demand_ = SelectedAtomGramDemand::kReconstructionNorm;
  std::vector<float> root_small_gram_column_major_;
  std::vector<float> edge_gram_column_major_;
  std::vector<float> root_small_edge_gram_column_major_;
  std::vector<float> root_small_norm2_;
  std::vector<float> edge_norm2_;
};

extern template class RootSmallCrossBlock<5>;
extern template class RootSmallCrossBlock<10>;
extern template class SelectedAtomGramView<5>;
extern template class SelectedAtomGramView<10>;
extern template class SelectedAtomGramTables<5>;
extern template class SelectedAtomGramTables<10>;

}  // namespace stlqpp
