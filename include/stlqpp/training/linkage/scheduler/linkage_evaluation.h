#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/model/linkage_code_tuple.h"

namespace stlqpp {

struct LinkageClusterHandle final {
  std::uint64_t value = 0U;

  friend bool operator==(LinkageClusterHandle left,
                         LinkageClusterHandle right) noexcept {
    return left.value == right.value;
  }
};

struct LinkageClusterRegistrationView final {
  // Stable cluster identity used only to select the owning scheduler worker.
  // DEV assigns cluster `cid` to `cid % worker_count`; keeping that affinity
  // independent of concurrent registration order preserves its batching
  // semantics and repeatability.
  std::uint32_t affinity_key = 0U;
  ColumnMajorMatrixF32View first;
  ColumnMajorMatrixF32View second;
};

struct LinkageReconstructionUpdateView final {
  const std::uint32_t* columns = nullptr;
  ColumnMajorMatrixF32View reconstructions;
};

enum class LinkageEvaluationKind : std::uint8_t {
  kEdgeRsd = 0,
  kRootOnlyInitialization = 1,
};

// Owning asynchronous request. Row `i` owns candidate columns in
// [row_offsets[i], row_offsets[i+1]). Edge rsd stores one ICM
// permutation per row and reuses it for all candidate residuals of that row.
// Historical root-only initialization stores one task-level permutation, so
// its large-root scan occurs exactly once in the sweep rather than once per
// sample-specific root position.
struct LinkageEvaluationTask final {
  std::uint32_t m = 0U;
  LinkageEvaluationKind kind = LinkageEvaluationKind::kEdgeRsd;
  LinkageClusterHandle cluster;
  std::uint64_t required_generation = 0U;
  std::uint32_t dimension = 0U;
  std::vector<float> node_vectors;
  std::vector<std::uint32_t> candidate_columns;
  std::vector<std::uint32_t> row_offsets;
  std::vector<std::uint8_t> icm_layer_orders;

  [[nodiscard]] std::uint32_t node_count() const noexcept {
    return row_offsets.empty()
               ? 0U
               : static_cast<std::uint32_t>(row_offsets.size() - 1U);
  }
  [[nodiscard]] std::uint32_t pair_count() const noexcept {
    return static_cast<std::uint32_t>(candidate_columns.size());
  }
};

// Transport owns a batch, not a compile-time code length. Keep large root IDs
// separate from the byte-sized suffix, with exactly (M-1) suffix bytes per row.
// Views borrow that storage; there is no allocation or type dispatch per row.
class LinkageEvaluationSupports final {
 public:
  struct ConstRowView {
    std::uint32_t first;
    ArrayView<const std::uint8_t> trailing;
    std::uint32_t code(std::uint32_t layer) const noexcept {
      return layer == 0U ? first : trailing[layer - 1U];
    }
  };
  struct RowView {
    std::uint32_t& first;
    ArrayView<std::uint8_t> trailing;
    std::uint32_t code(std::uint32_t layer) const noexcept {
      return layer == 0U ? first : trailing[layer - 1U];
    }
  };

  void resize(std::size_t rows, std::uint32_t m) {
    assert(m >= kMinimumModelLayers && m <= kMaximumModelLayers);
    m_ = m;
    roots_.resize(rows);
    trailing_.resize(rows * (m - 1U));
  }
  std::size_t size() const noexcept { return roots_.size(); }
  std::uint32_t m() const noexcept { return m_; }
  RowView operator[](std::size_t row) noexcept {
    return {roots_[row], {trailing_.data() + row * (m_ - 1U), m_ - 1U}};
  }
  ConstRowView operator[](std::size_t row) const noexcept {
    return {roots_[row], {trailing_.data() + row * (m_ - 1U), m_ - 1U}};
  }

  // Only numerical producers/consumers cross the fixed-width tuple boundary.
  template <int M>
  LinkageCodeTuple<M> tuple(std::size_t row) const noexcept {
    assert(M == 0 || m_ == M);
    const auto source = (*this)[row];
    LinkageCodeTuple<M> result;
    result.first = source.first;
    std::copy_n(source.trailing.data(), m_ - 1U, result.trailing.data());
    return result;
  }
  template <int M>
  void set_tuple(std::size_t row, const LinkageCodeTuple<M>& source) noexcept {
    assert(M == 0 || m_ == M);
    auto target = (*this)[row];
    target.first = source.first;
    std::copy_n(source.trailing.data(), m_ - 1U, target.trailing.data());
  }
  friend bool operator==(const LinkageEvaluationSupports& left,
                         const LinkageEvaluationSupports& right) noexcept {
    return left.m_ == right.m_ && left.roots_ == right.roots_ &&
           left.trailing_ == right.trailing_;
  }
  friend bool operator!=(const LinkageEvaluationSupports& left,
                         const LinkageEvaluationSupports& right) noexcept {
    return !(left == right);
  }

 private:
  std::uint32_t m_ = 0U;
  std::vector<std::uint32_t> roots_;
  std::vector<std::uint8_t> trailing_;
};

struct LinkageEvaluationResult final {
  std::vector<std::uint32_t> parent_columns;
  LinkageEvaluationSupports supports;
  std::vector<float> coefficients;
  std::vector<float> squared_errors;
};

}  // namespace stlqpp
