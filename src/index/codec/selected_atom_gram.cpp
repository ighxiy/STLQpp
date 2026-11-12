#include "stlqpp/index/codec/selected_atom_gram.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace stlqpp {
namespace {

template <int M>
constexpr std::uint32_t RootSmallAtomCount(std::uint32_t m) noexcept {
  return (M == 0 ? m - 1U : M - 1U) * 256U;
}

template <int M>
constexpr std::uint32_t EdgeAtomCount(std::uint32_t m) noexcept {
  return (M == 0 ? m : M) * 256U;
}

template <int M>
Status ValidateClusterCodes(const CanonicalClusterCodeView<M>& cluster,
                            std::uint32_t nlist, std::uint32_t m) {
  if (cluster.m != m || cluster.cid >= nlist) {
    return Status::InvalidArgument(
        "coefficient Gram cluster CID exceeds native IVF nlist");
  }
  const std::uint64_t virtual_small =
      static_cast<std::uint64_t>(cluster.virtual_count) * (m - 1);
  const std::uint64_t real_count =
      static_cast<std::uint64_t>(cluster.real_root_count) +
      cluster.linked_real_count;
  const std::uint64_t real_small = real_count * (m - 1);
  if (virtual_small != cluster.virtual_codes_small.size() ||
      real_small != cluster.real_codes_small.size() ||
      real_count != cluster.edge_code0.size()) {
    return Status::InvalidArgument(
        "canonical cluster code arrays do not match its role counts");
  }
  return Status::Ok();
}

}  // namespace

template <int M>
StatusOr<SelectedAtomGramTables<M>> SelectedAtomGramTables<M>::Create(
    const ProductCodebooks<M>& codebooks,
    const F32GemmBackend& gemm, SelectedAtomGramDemand demand) {
  return Create(codebooks.runtime_view(), gemm, demand);
}

template <int M>
StatusOr<SelectedAtomGramTables<M>> SelectedAtomGramTables<M>::Create(
    RuntimeProductCodebookView codebooks, const F32GemmBackend& gemm,
    SelectedAtomGramDemand demand) {
  if constexpr (M != 0) {
    if (codebooks.shape.m != static_cast<std::uint32_t>(M))
      return Status::InvalidArgument("Gram model M mismatch");
  }
  SelectedAtomGramTables<M> output;
  output.codebooks_ = codebooks;
  output.demand_ = demand;
  const ColumnMajorMatrixF32View root_small = codebooks.root_small_flat;
  const ColumnMajorMatrixF32View edge = codebooks.edge_flat;
  const std::uint32_t root_atoms = RootSmallAtomCount<M>(codebooks.shape.m);
  const std::uint32_t edge_atoms = EdgeAtomCount<M>(codebooks.shape.m);
  output.root_small_gram_column_major_.resize(
      static_cast<std::size_t>(root_atoms) * root_atoms);
  output.edge_gram_column_major_.resize(
      static_cast<std::size_t>(edge_atoms) * edge_atoms);
  if (demand == SelectedAtomGramDemand::kReconstructionNorm) {
    output.root_small_edge_gram_column_major_.resize(
        static_cast<std::size_t>(root_atoms) * edge_atoms);
  }
  output.root_small_norm2_.resize(root_atoms);
  output.edge_norm2_.resize(edge_atoms);
  if (const Status status = gemm.MultiplyTransposedLeft({
          .left = root_small,
          .right = root_small.values,
          .query_count = root_atoms,
          .right_leading_dimension = root_small.rows,
          .output = output.root_small_gram_column_major_.data(),
          .output_leading_dimension = root_atoms,
      });
      !status.ok()) {
    return status;
  }
  if (demand == SelectedAtomGramDemand::kReconstructionNorm) {
    if (const Status status = gemm.MultiplyTransposedLeft({
            .left = root_small,
            .right = edge.values,
            .query_count = edge_atoms,
            .right_leading_dimension = edge.rows,
            .output = output.root_small_edge_gram_column_major_.data(),
            .output_leading_dimension = root_atoms,
        });
        !status.ok()) {
      return status;
    }
  }
  if (const Status status = gemm.MultiplyTransposedLeft({
          .left = edge,
          .right = edge.values,
          .query_count = edge_atoms,
          .right_leading_dimension = edge.rows,
          .output = output.edge_gram_column_major_.data(),
          .output_leading_dimension = edge_atoms,
      });
      !status.ok()) {
    return status;
  }
  for (std::uint32_t atom = 0U; atom < root_atoms; ++atom) {
    const float* column = root_small.column(atom);
    float norm2 = 0.0F;
#pragma omp simd reduction(+ : norm2)
    for (std::uint32_t axis = 0U; axis < root_small.rows; ++axis) {
      norm2 += column[axis] * column[axis];
    }
    output.root_small_norm2_[atom] = norm2;
  }
  for (std::uint32_t atom = 0U; atom < edge_atoms; ++atom) {
    const float* column = edge.column(atom);
    float norm2 = 0.0F;
#pragma omp simd reduction(+ : norm2)
    for (std::uint32_t axis = 0U; axis < edge.rows; ++axis) {
      norm2 += column[axis] * column[axis];
    }
    output.edge_norm2_[atom] = norm2;
  }
  return output;
}

template <int M>
StatusOr<RootSmallCrossBlock<M>>
SelectedAtomGramTables<M>::BuildRootCrossBlock(
    std::uint32_t cid_begin,
    std::uint32_t cid_count) const {
  const std::uint32_t nlist = codebooks_.shape.nlist;
  if (cid_count == 0U || cid_begin >= nlist ||
      cid_count > nlist - cid_begin) {
    return Status::InvalidArgument(
        "root cross block is empty or exceeds native IVF nlist");
  }
  const std::uint32_t root_atoms = RootSmallAtomCount<M>(codebooks_.shape.m);
  const std::uint32_t edge_atoms = EdgeAtomCount<M>(codebooks_.shape.m);
  RootSmallCrossBlock<M> output;
  output.cid_begin_ = cid_begin;
  output.root_norm2_.resize(cid_count);
  output.root_small_cross_column_major_.resize(
      static_cast<std::size_t>(root_atoms) * cid_count);
  if (demand_ == SelectedAtomGramDemand::kReconstructionNorm) {
    output.root_edge_cross_column_major_.resize(
        static_cast<std::size_t>(edge_atoms) * cid_count);
  }
  const ColumnMajorMatrixF32View root0 = codebooks_.root0;
  const ColumnMajorMatrixF32View root_small = codebooks_.root_small_flat;
  for (std::uint32_t local = 0U; local < cid_count; ++local) {
    const float* root = root0.column(cid_begin + local);
    float* cross = output.root_small_cross_column_major_.data() +
        static_cast<std::size_t>(local) * root_atoms;
    for (std::uint32_t atom = 0U; atom < root_atoms; ++atom) {
      const float* small = root_small.column(atom);
      float value = 0.0F;
#pragma omp simd reduction(+ : value)
      for (std::uint32_t axis = 0U; axis < root0.rows; ++axis) {
        value += root[axis] * small[axis];
      }
      cross[atom] = value;
    }
    if (demand_ == SelectedAtomGramDemand::kReconstructionNorm) {
      float* edge_cross = output.root_edge_cross_column_major_.data() +
          static_cast<std::size_t>(local) * edge_atoms;
      for (std::uint32_t atom = 0U; atom < edge_atoms; ++atom) {
        const float* one = codebooks_.edge_flat.column(atom);
        float value = 0.0F;
  #pragma omp simd reduction(+ : value)
        for (std::uint32_t axis = 0U; axis < root0.rows; ++axis) {
          value += root[axis] * one[axis];
        }
        edge_cross[atom] = value;
      }
    }
    float norm2 = 0.0F;
#pragma omp simd reduction(+ : norm2)
    for (std::uint32_t axis = 0U; axis < root0.rows; ++axis) {
      norm2 += root[axis] * root[axis];
    }
    output.root_norm2_[local] = norm2;
  }
  return output;
}

template <int M>
StatusOr<SelectedAtomGramView<M>> SelectedAtomGramTables<M>::SelectCluster(
    const CanonicalClusterCodeView<M>& cluster,
    const RootSmallCrossBlock<M>& root_cross) const {
  if (const Status status =
          ValidateClusterCodes(cluster, codebooks_.shape.nlist, codebooks_.shape.m);
      !status.ok()) {
    return status;
  }
  if (cluster.cid < root_cross.cid_begin_ ||
      cluster.cid - root_cross.cid_begin_ >= root_cross.root_norm2_.size()) {
    return Status::InvalidArgument(
        "cluster CID is outside the prepared root cross block");
  }
  const std::uint64_t root_count =
      static_cast<std::uint64_t>(cluster.virtual_count) +
      cluster.real_root_count;
  const std::uint64_t node_count = root_count + cluster.linked_real_count;
  if (root_count > std::numeric_limits<std::uint32_t>::max() ||
      node_count > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "selected-atom Gram view exceeds the canonical u32 row range");
  }

  const std::uint32_t root_atoms = RootSmallAtomCount<M>(codebooks_.shape.m);
  const std::uint32_t cross_local =
      cluster.cid - root_cross.cid_begin_;
  const float* root_cross_values =
      root_cross.root_small_cross_column_major_.data() +
      static_cast<std::size_t>(cross_local) * root_atoms;
  const float root_norm2 = root_cross.root_norm2_[cross_local];

  SelectedAtomGramView<M> output;
  output.roots_.m_ = codebooks_.shape.m;
  output.linkage_.m_ = codebooks_.shape.m;
  output.root_count_ = static_cast<std::uint32_t>(root_count);
  output.node_count_ = static_cast<std::uint32_t>(node_count);
  output.roots_.virtual_count_ = cluster.virtual_count;
  output.roots_.root_count_ = output.root_count_;
  output.roots_.real_root_count_ = cluster.real_root_count;
  output.roots_.virtual_codes_small_ = cluster.virtual_codes_small;
  output.roots_.real_codes_small_ = cluster.real_codes_small;
  output.roots_.edge_code0_ = cluster.edge_code0;
  output.roots_.root_cross_ = root_cross_values;
  output.roots_.shared_gram_ = root_small_gram_column_major_.data();
  output.roots_.atom_norm2_ = root_small_norm2_.data();
  output.roots_.root_norm2_ = root_norm2;
  output.linkage_.virtual_count_ = cluster.virtual_count;
  output.linkage_.root_count_ = output.root_count_;
  output.linkage_.real_root_count_ = cluster.real_root_count;
  output.linkage_.virtual_codes_small_ = cluster.virtual_codes_small;
  output.linkage_.real_codes_small_ = cluster.real_codes_small;
  output.linkage_.edge_code0_ = cluster.edge_code0;
  output.linkage_.shared_gram_ = edge_gram_column_major_.data();
  output.linkage_.atom_norm2_ = edge_norm2_.data();
  output.root0_edge_cross_ = root_cross.root_edge_cross_column_major_.empty()
      ? nullptr
      : root_cross.root_edge_cross_column_major_.data() +
          static_cast<std::size_t>(cross_local) * EdgeAtomCount<M>(codebooks_.shape.m);
  output.root_small_edge_gram_ =
      root_small_edge_gram_column_major_.data();
  output.edge_gram_ = edge_gram_column_major_.data();
  return output;
}

template <int M>
float SelectedAtomGramView<M>::RootLinkageAt(
    std::uint32_t root_row, std::uint32_t root_layer,
    std::uint32_t linkage_row, std::uint32_t linkage_layer) const noexcept {
  assert(root0_edge_cross_ != nullptr && root_small_edge_gram_ != nullptr);
  const std::uint32_t real = roots_.real_root_count_ +
      linkage_row - root_count_;
  const std::uint32_t edge_code = linkage_layer == 0U
      ? linkage_.edge_code0_[real]
      : linkage_.real_codes_small_[
            static_cast<std::size_t>(real) * (m() - 1U) +
            linkage_layer - 1U];
  const std::uint32_t edge_atom = linkage_layer * 256U + edge_code;
  if (root_layer == 0U) return root0_edge_cross_[edge_atom];
  const std::uint32_t root_code = root_row < roots_.virtual_count_
      ? roots_.virtual_codes_small_[
            static_cast<std::size_t>(root_row) * (m() - 1U) +
            root_layer - 1U]
      : roots_.real_codes_small_[
            static_cast<std::size_t>(root_row - roots_.virtual_count_) *
                (m() - 1U) + root_layer - 1U];
  const std::uint32_t root_atom =
      (root_layer - 1U) * 256U + root_code;
  return root_small_edge_gram_[
      static_cast<std::size_t>(root_atom) +
      static_cast<std::size_t>(edge_atom) * (m() - 1U) * 256U];
}

template <int M>
float SelectedAtomGramView<M>::LinkageCrossAt(
    std::uint32_t left_row, std::uint32_t left_layer,
    std::uint32_t right_row, std::uint32_t right_layer) const noexcept {
  const auto edge_atom = [&](std::uint32_t row, std::uint32_t layer) {
    const std::uint32_t real = roots_.real_root_count_ + row - root_count_;
    const std::uint32_t code = layer == 0U
        ? linkage_.edge_code0_[real]
        : linkage_.real_codes_small_[
              static_cast<std::size_t>(real) * (m() - 1U) + layer - 1U];
    return layer * 256U + code;
  };
  const std::uint32_t left = edge_atom(left_row, left_layer);
  const std::uint32_t right = edge_atom(right_row, right_layer);
  return edge_gram_[static_cast<std::size_t>(left) +
                    static_cast<std::size_t>(right) * m() * 256U];
}

template <int M>
Status SelectedAtomGramTables<M>::MaterializeCluster(
    const CanonicalClusterCodeView<M>& cluster,
    const RootSmallCrossBlock<M>& root_cross,
    std::vector<float>& output) const {
  auto selected_result = SelectCluster(cluster, root_cross);
  if (!selected_result.ok()) {
    return selected_result.status();
  }
  const SelectedAtomGramView<M>& selected = selected_result.value();
  const std::uint32_t m = selected.m();
  const std::uint32_t root_count = selected.root_count();
  const std::uint32_t node_count = selected.node_count();
  if (node_count > std::numeric_limits<std::size_t>::max() / (m * m)) {
    return Status::ResourceExhausted(
        "selected-atom Gram output exceeds addressable memory");
  }
  output.resize(static_cast<std::size_t>(node_count) * m * m);

  for (std::uint32_t root = 0U; root < root_count; ++root) {
    float* gram = output.data() + static_cast<std::size_t>(root) * m * m;
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      for (std::uint32_t other = 0U;
           other < m; ++other) {
        gram[layer * m + other] =
            selected.roots().At(root, layer, other);
      }
    }
  }

  for (std::uint32_t node = root_count; node < node_count; ++node) {
    float* gram = output.data() + static_cast<std::size_t>(node) * m * m;
    for (std::uint32_t layer = 0U;
         layer < m; ++layer) {
      for (std::uint32_t other = 0U;
           other < m; ++other) {
        gram[layer * m + other] =
            selected.linkage().At(node, layer, other);
      }
    }
  }
  return Status::Ok();
}

template class RootSmallCrossBlock<0>;
template class RootSmallCrossBlock<5>;
template class RootSmallCrossBlock<12>;
template class RootSmallCrossBlock<6>;
template class RootSmallCrossBlock<10>;
template class SelectedAtomGramView<0>;
template class SelectedAtomGramView<5>;
template class SelectedAtomGramView<12>;
template class SelectedAtomGramView<6>;
template class SelectedAtomGramView<10>;
template class SelectedAtomGramTables<0>;
template class SelectedAtomGramTables<5>;
template class SelectedAtomGramTables<12>;
template class SelectedAtomGramTables<6>;
template class SelectedAtomGramTables<10>;

}  // namespace stlqpp
