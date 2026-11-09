#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/search/index/resident_index.h"
#include "stlqpp/search/query_batch.h"
#include "stlqpp/search/tables/query_table_batch.h"

namespace stlqpp {

template <int M>
class ExactDenseSearcher;

template <int M>
[[nodiscard]] StatusOr<ProductCodebooks<M>> PretransformQueryCodebooks(
    const QueryTransform& transform,
    const ProductCodebooks<M>& codebooks,
    const F32GemmBackend& gemm);

template <int M>
class QueryTableBuilder final {
 public:
  QueryTableBuilder(const ResidentIndex<M>& index,
                    const F32GemmBackend& gemm) noexcept
      : index_(&index), gemm_(&gemm) {}

  [[nodiscard]] bool requires_query_transform() const noexcept {
    return !index_->query_codebooks_pretransformed();
  }

  // DEV rotates the complete validated query set once before query blocking.
  [[nodiscard]] Status Transform(const TrustedQueryBatchView& queries,
                                 float* model_queries) const;

  // Builds the three codebook products in the coordinate system selected
  // once when this builder is created. There is no per-row mode branch.
  [[nodiscard]] Status BuildModelSpace(
      const TrustedQueryBatchView& table_queries,
      QueryTableBatch<M>* output) const;

 private:
  friend class ExactDenseSearcher<M>;
  // The searcher creates this batch from the same immutable index shape and
  // slices only within its fixed query capacity.
  [[nodiscard]] Status BuildModelSpaceTrusted(
      const TrustedQueryBatchView& table_queries,
      QueryTableBatch<M>* output, bool skip_root0 = false) const;
  const ResidentIndex<M>* index_ = nullptr;
  const F32GemmBackend* gemm_ = nullptr;
};

extern template StatusOr<ProductCodebooks<5>> PretransformQueryCodebooks<5>(
    const QueryTransform&, const ProductCodebooks<5>&,
    const F32GemmBackend&);
extern template StatusOr<ProductCodebooks<10>>
PretransformQueryCodebooks<10>(
    const QueryTransform&, const ProductCodebooks<10>&,
    const F32GemmBackend&);
extern template class QueryTableBuilder<5>;
extern template class QueryTableBuilder<10>;

}  // namespace stlqpp
