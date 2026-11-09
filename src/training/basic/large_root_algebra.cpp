#include "stlqpp/training/basic/large_root_algebra.h"

#include "stlqpp/model/product_codebooks.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace stlqpp {
namespace {

float SquaredNorm(const float* values, std::uint32_t count) noexcept {
  float sum = 0.0F;
#pragma omp simd reduction(+ : sum)
  for (std::uint32_t index = 0U; index < count; ++index) {
    const float value = values[index];
    sum += value * value;
  }
  return sum;
}

Status BuildSmallGram(ColumnMajorMatrixF32View full,
                      ColumnMajorMatrixF32View small,
                      bool dense_virtual,
                      const F32GemmBackend& gemm,
                      std::vector<float>* small_gram,
                      std::vector<float>* dense_gram) {
  small_gram->resize(static_cast<std::size_t>(small.columns) * small.columns);
  if (!dense_virtual) {
    dense_gram->clear();
    return gemm.MultiplyTransposedLeft({
        .left = small, .right = small.values, .query_count = small.columns,
        .right_leading_dimension = small.rows, .output = small_gram->data(),
        .output_leading_dimension = small.columns});
  }
  // Keep the registered C_all^T C_all GEMM shape, including its root block.
  dense_gram->resize(static_cast<std::size_t>(full.columns) * full.columns);
  const Status status = gemm.MultiplyTransposedLeft({
      .left = full, .right = full.values, .query_count = full.columns,
      .right_leading_dimension = full.rows, .output = dense_gram->data(),
      .output_leading_dimension = full.columns});
  if (!status.ok()) return status;
  const std::uint32_t roots = full.columns - small.columns;
  for (std::uint32_t flat = 0; flat < small.columns; ++flat) {
    const float* source = dense_gram->data() +
        static_cast<std::size_t>(roots + flat) * full.columns + roots;
    std::copy_n(source, small.columns,
                small_gram->data() + static_cast<std::size_t>(flat) * small.columns);
  }
  return Status::Ok();
}

void TransposeRootSmall(const float* source, std::uint32_t roots,
                        std::uint32_t small, std::uint32_t source_stride,
                        float* destination, const ParallelForExecutor* parallel) {
  // Each tile owns complete destination rows; no cross-thread cache-line
  // sharing. Copy only: GEMM orientation and all computed bits stay unchanged.
  constexpr std::uint32_t tile = 32;
  const std::uint32_t tiles = (roots + tile - 1U) / tile;
  struct Context {
    const float* source;
    std::uint32_t roots, small, source_stride;
    float* destination;
  } context{source, roots, small, source_stride, destination};
  const auto transpose_tile = [](void* opaque, std::uint32_t,
                                  std::uint32_t block) noexcept {
    const auto& c = *static_cast<Context*>(opaque);
    const std::uint32_t end = std::min(c.roots, (block + 1U) * tile);
    for (std::uint32_t flat_begin = 0; flat_begin < c.small; flat_begin += tile) {
      const std::uint32_t flat_end = std::min(c.small, flat_begin + tile);
      for (std::uint32_t cid = block * tile; cid < end; ++cid) {
        for (std::uint32_t flat = flat_begin; flat < flat_end; ++flat) {
          c.destination[static_cast<std::size_t>(cid) * c.small + flat] =
              c.source[static_cast<std::size_t>(flat) * c.source_stride + cid];
        }
      }
    }
  };
  if (parallel != nullptr && static_cast<std::uint64_t>(roots) * small >= 1048576U) {
    parallel->Run(tiles, &context, transpose_tile);
  } else {
    for (std::uint32_t block = 0; block < tiles; ++block)
      transpose_tile(&context, 0U, block);
  }
}

}  // namespace

StatusOr<std::vector<float>> BuildVirtualRootSmallGram(
    RuntimeRootCodebookView codebooks, const F32GemmBackend& gemm) {
  if (const auto status = codebooks.Validate(); !status.ok()) return status;
  const auto root = codebooks.root0;
  const auto small = codebooks.root_small_flat;
  const bool dense = root.columns <= small.columns;
  std::vector<float> full;
  if (dense) {
    const std::size_t root_values = static_cast<std::size_t>(root.rows) * root.columns;
    const std::size_t small_values = static_cast<std::size_t>(small.rows) * small.columns;
    full.resize(root_values + small_values);
    std::copy_n(root.values, root_values, full.data());
    std::copy_n(small.values, small_values, full.data() + root_values);
  }
  std::vector<float> gram, dense_gram;
  const Status status = BuildSmallGram(
      {full.data(), root.rows, root.columns + small.columns}, small, dense,
      gemm, &gram, &dense_gram);
  if (!status.ok()) return status;
  return gram;
}

void ComputeBasicCodebookNorms(
    ColumnMajorMatrixF32View root0,
    ColumnMajorMatrixF32View small,
    std::vector<float>* root_norm2,
    std::vector<float>* root_inverse_norm,
    std::vector<float>* small_norm2,
    std::vector<float>* small_inverse_norm) {
  root_norm2->resize(root0.columns);
  root_inverse_norm->resize(root0.columns);
  for (std::uint32_t cid = 0U; cid < root0.columns; ++cid) {
    const float norm2 = SquaredNorm(root0.column(cid), root0.rows);
    (*root_norm2)[cid] = norm2;
    (*root_inverse_norm)[cid] =
        norm2 > 0.0F ? 1.0F / std::sqrt(norm2) : 0.0F;
  }
  small_norm2->resize(small.columns);
  small_inverse_norm->resize(small.columns);
  for (std::uint32_t flat = 0U; flat < small.columns; ++flat) {
    const float norm2 = SquaredNorm(small.column(flat), small.rows);
    (*small_norm2)[flat] = norm2;
    (*small_inverse_norm)[flat] =
        norm2 > 0.0F ? 1.0F / std::sqrt(norm2) : 0.0F;
  }
}

StatusOr<RuntimeBasicCodebookAlgebra> RuntimeBasicCodebookAlgebra::Create(
    RuntimeRootCodebookView codebooks, const F32GemmBackend& gemm,
    const ParallelForExecutor* parallel) {
  RuntimeBasicCodebookAlgebra output;
  if (const auto s = output.Rebuild(codebooks, gemm, parallel); !s.ok()) return s;
  return output;
}

StatusOr<RuntimeBasicCodebookAlgebra>
RuntimeBasicCodebookAlgebra::CreateForVirtualRoots(
    RuntimeRootCodebookView codebooks, const F32GemmBackend& gemm,
    const ParallelForExecutor* parallel) {
  RuntimeBasicCodebookAlgebra output;
  if (const auto s = output.RebuildForVirtualRoots(codebooks, gemm, parallel); !s.ok())
    return s;
  return output;
}

Status RuntimeBasicCodebookAlgebra::Rebuild(
    RuntimeRootCodebookView codebooks, const F32GemmBackend& gemm,
    const ParallelForExecutor* parallel) {
  if (const auto s = codebooks.Validate(); !s.ok()) return s;
  return RebuildImpl(codebooks, gemm, false, parallel);
}

Status RuntimeBasicCodebookAlgebra::RebuildForVirtualRoots(
    RuntimeRootCodebookView codebooks, const F32GemmBackend& gemm,
    const ParallelForExecutor* parallel) {
  if (const auto s = codebooks.Validate(); !s.ok()) return s;
  return RebuildImpl(codebooks, gemm, true, parallel);
}

template <int M>
StatusOr<BasicCodebookAlgebra<M>> BasicCodebookAlgebra<M>::Create(
    const RootCodebooks<M>& codebooks,
    const F32GemmBackend& gemm, const ParallelForExecutor* parallel) {
  BasicCodebookAlgebra<M> output;
  if (const Status status = output.Rebuild(codebooks, gemm, parallel); !status.ok()) {
    return status;
  }
  return output;
}

template <int M>
StatusOr<BasicCodebookAlgebra<M>>
BasicCodebookAlgebra<M>::CreateForVirtualRoots(
    const RootCodebooks<M>& codebooks,
    const F32GemmBackend& gemm, const ParallelForExecutor* parallel) {
  BasicCodebookAlgebra<M> output;
  if (const Status status =
          output.RebuildForVirtualRoots(codebooks, gemm, parallel);
      !status.ok()) {
    return status;
  }
  return output;
}

template <int M>
Status BasicCodebookAlgebra<M>::Rebuild(
    const RootCodebooks<M>& codebooks,
    const F32GemmBackend& gemm, const ParallelForExecutor* parallel) {
  assert(M == 0 || codebooks.shape().m == static_cast<std::uint32_t>(M));
  return RebuildImpl(codebooks.runtime_view(), gemm, false, parallel);
}

template <int M>
Status BasicCodebookAlgebra<M>::RebuildForVirtualRoots(
    const RootCodebooks<M>& codebooks,
    const F32GemmBackend& gemm, const ParallelForExecutor* parallel) {
  assert(M == 0 || codebooks.shape().m == static_cast<std::uint32_t>(M));
  return RebuildImpl(codebooks.runtime_view(), gemm, true, parallel);
}

Status RuntimeBasicCodebookAlgebra::RebuildImpl(
    RuntimeRootCodebookView codebooks,
    const F32GemmBackend& gemm,
    bool build_dense_virtual_gram, const ParallelForExecutor* parallel) {
  shape_ = codebooks.shape;

  const ColumnMajorMatrixF32View root0 = codebooks.root0;
  const ColumnMajorMatrixF32View small = codebooks.root_small_flat;
  const std::size_t root_values =
      static_cast<std::size_t>(root0.rows) * root0.columns;
  const std::size_t small_values =
      static_cast<std::size_t>(small.rows) * small.columns;
  full_.resize(root_values + small_values);
  std::copy(root0.values, root0.values + root_values, full_.begin());
  std::copy(small.values, small.values + small_values,
            full_.begin() + static_cast<std::ptrdiff_t>(root_values));

  ComputeBasicCodebookNorms(
      root0, small, &root_norm2_, &root_inverse_norm_, &small_norm2_,
      &small_inverse_norm_);

  root_small_gram_.resize(
      static_cast<std::size_t>(root0.columns) * small.columns);

  const bool dense = build_dense_virtual_gram && root0.columns <= small.columns;
  Status status = BuildSmallGram(full_flat(), small_flat(), dense, gemm,
                                 &small_gram_, &dense_full_gram_);
  if (!status.ok()) return status;
  if (dense) {
    const std::uint32_t full_columns = root0.columns + small.columns;
    TransposeRootSmall(dense_full_gram_.data() +
                          static_cast<std::size_t>(root0.columns) * full_columns,
                       root0.columns, small.columns, full_columns,
                       root_small_gram_.data(), parallel);
  } else {
    std::vector<float> flat_major_root_small(
        static_cast<std::size_t>(root0.columns) * small.columns);
    status = gemm.MultiplyTransposedLeft({
        .left = this->root0(),
        .right = small_flat().values,
        .query_count = small.columns,
        .right_leading_dimension = small.rows,
        .output = flat_major_root_small.data(),
        .output_leading_dimension = root0.columns,
    });
    if (!status.ok()) {
      return status;
    }
    TransposeRootSmall(flat_major_root_small.data(), root0.columns,
                       small.columns, root0.columns, root_small_gram_.data(), parallel);
  }

  return Status::Ok();
}

template <int M>
StatusOr<BasicCodebookAlgebra<M>> BasicCodebookAlgebra<M>::Create(
    const ProductCodebooks<M>& codebooks,
    const F32GemmBackend& gemm, const ParallelForExecutor* parallel) {
  return Create(codebooks.root_codebooks(), gemm, parallel);
}

template <int M>
StatusOr<BasicCodebookAlgebra<M>>
BasicCodebookAlgebra<M>::CreateForVirtualRoots(
    const ProductCodebooks<M>& codebooks,
    const F32GemmBackend& gemm, const ParallelForExecutor* parallel) {
  return CreateForVirtualRoots(codebooks.root_codebooks(), gemm, parallel);
}

template class BasicCodebookAlgebra<0>;
template class BasicCodebookAlgebra<5>;
template class BasicCodebookAlgebra<12>;
template class BasicCodebookAlgebra<6>;
template class BasicCodebookAlgebra<10>;

}  // namespace stlqpp
