#include "stlqpp/search/tables/query_table_builder.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <new>
#include <vector>

namespace stlqpp {
namespace {

Status TransformCodebook(const F32GemmBackend& gemm,
                         ColumnMajorMatrixF32View transform,
                         ColumnMajorMatrixF32View source,
                         std::vector<float>* destination) {
  destination->resize(
      static_cast<std::size_t>(source.rows) * source.columns);
  return gemm.MultiplyTransposedLeft({
      .left = transform,
      .right = source.values,
      .query_count = source.columns,
      .right_leading_dimension = source.rows,
      .output = destination->data(),
      .output_leading_dimension = source.rows,
  });
}

}  // namespace

template <int M>
StatusOr<QueryTableBatch<M>> QueryTableBatch<M>::Create(
    const ModelShape& shape,
    std::uint32_t query_capacity) {
  if (const Status shape_status = shape.Validate(); !shape_status.ok()) {
    return shape_status;
  }
  if (shape.m != static_cast<std::uint32_t>(M) ||
      query_capacity == 0U) {
    return Status::InvalidArgument(
        "query table batch does not match its registered fixed-M shape");
  }
  return QueryTableBatch<M>(shape.dimension, shape.nlist, query_capacity);
}

template <int M>
StatusOr<ProductCodebooks<M>> PretransformQueryCodebooks(
    const QueryTransform& transform,
    const ProductCodebooks<M>& source,
    const F32GemmBackend& gemm) {
  try {
    const ColumnMajorMatrixF32View matrix = transform.matrix();
    std::vector<float> root0;
    std::array<std::vector<float>, M - 1> root_small;
    std::array<std::vector<float>, M> edge;
    if (const Status status = TransformCodebook(
            gemm, matrix, source.root0(), &root0);
        !status.ok()) {
      return status;
    }
    for (std::uint32_t layer = 1U;
         layer < static_cast<std::uint32_t>(M); ++layer) {
      if (const Status status = TransformCodebook(
              gemm, matrix, source.root_small(layer),
              &root_small[layer - 1U]);
          !status.ok()) {
        return status;
      }
    }
    for (std::uint32_t layer = 0U;
         layer < static_cast<std::uint32_t>(M); ++layer) {
      if (const Status status = TransformCodebook(
              gemm, matrix, source.edge(layer), &edge[layer]);
          !status.ok()) {
        return status;
      }
    }
    ProductCodebookView<M> view;
    view.shape = source.shape();
    view.root0 = {root0.data(), view.shape.dimension, view.shape.nlist};
    for (std::uint32_t layer = 1U;
         layer < static_cast<std::uint32_t>(M); ++layer) {
      view.root_small[layer - 1U] = {
          root_small[layer - 1U].data(), view.shape.dimension, 256U};
    }
    for (std::uint32_t layer = 0U;
         layer < static_cast<std::uint32_t>(M); ++layer) {
      view.edge[layer] = {
          edge[layer].data(), view.shape.dimension, 256U};
    }
    return ProductCodebooks<M>::Create(view);
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted(
        "pretransformed query codebook allocation failed");
  }
}

template <int M>
Status QueryTableBuilder<M>::Transform(
    const TrustedQueryBatchView& queries,
    float* model_queries) const {
  return gemm_->MultiplyLeft({
      .left = index_->query_transform().matrix(),
      .right = queries.data(),
      .query_count = queries.count(),
      .right_leading_dimension = queries.stride_floats(),
      .output = model_queries,
      .output_leading_dimension = queries.dimension(),
  });
}

template <int M>
Status QueryTableBuilder<M>::BuildModelSpace(
    const TrustedQueryBatchView& table_queries,
    QueryTableBatch<M>* output) const {
  const ProductCodebooks<M>& codebooks = index_->codebooks();
  if (output == nullptr || output->dimension_ != codebooks.shape().dimension ||
      output->nlist_ != codebooks.shape().nlist ||
      table_queries.dimension() != codebooks.shape().dimension ||
      table_queries.count() > output->query_capacity_) {
    return Status::InvalidArgument(
      "query table output does not match model shape or query count");
  }
  return BuildModelSpaceTrusted(table_queries, output);
}

template <int M>
Status QueryTableBuilder<M>::BuildModelSpaceTrusted(
    const TrustedQueryBatchView& table_queries,
    QueryTableBatch<M>* output, bool skip_root0) const {
  const ProductCodebooks<M>& codebooks = index_->codebooks();
  assert(output != nullptr);
  assert(output->dimension_ == codebooks.shape().dimension);
  assert(output->nlist_ == codebooks.shape().nlist);
  assert(table_queries.dimension() == codebooks.shape().dimension);
  assert(table_queries.count() <= output->query_capacity_);

  const auto run = [&](ColumnMajorMatrixF32View left,
                       float* destination) {
    return gemm_->MultiplyTransposedLeft({
        .left = left,
        .right = table_queries.data(),
        .query_count = table_queries.count(),
        .right_leading_dimension = table_queries.stride_floats(),
        .output = destination,
        .output_leading_dimension = left.columns,
    });
  };
  if (!skip_root0) {
    if (const Status status = run(codebooks.root0(), output->root0_.data());
        !status.ok()) return status;
  }
  if (const Status status = run(codebooks.root_small_flat(),
                                output->root_small_.data());
      !status.ok()) {
    return status;
  }
  if (const Status status = run(codebooks.edge_flat(), output->edge_.data());
      !status.ok()) {
    return status;
  }
  output->query_count_ = table_queries.count();
  return Status::Ok();
}

template class QueryTableBatch<5>;
template class QueryTableBatch<12>;
template class QueryTableBatch<6>;
template class QueryTableBatch<10>;
template StatusOr<ProductCodebooks<5>> PretransformQueryCodebooks<5>(
    const QueryTransform&, const ProductCodebooks<5>&,
    const F32GemmBackend&);
template StatusOr<ProductCodebooks<12>> PretransformQueryCodebooks<12>(
    const QueryTransform&, const ProductCodebooks<12>&,
    const F32GemmBackend&);
template StatusOr<ProductCodebooks<6>> PretransformQueryCodebooks<6>(
    const QueryTransform&, const ProductCodebooks<6>&,
    const F32GemmBackend&);
template StatusOr<ProductCodebooks<10>> PretransformQueryCodebooks<10>(
    const QueryTransform&, const ProductCodebooks<10>&,
    const F32GemmBackend&);
template class QueryTableBuilder<5>;
template class QueryTableBuilder<12>;
template class QueryTableBuilder<6>;
template class QueryTableBuilder<10>;

}  // namespace stlqpp
