#pragma once

#include "stlqpp/search/layout/resident_cluster.h"
#include "stlqpp/search/tables/scaled_cluster_tables.h"
#include "stlqpp/search/topk/query_topk.h"

#include <cstdint>

namespace stlqpp {

// Profile-only active-worker time from one query-owner invocation. The
// ordinary kernels do not construct or update this state.
struct DenseIntScanTiming final {
  double scan_seconds = 0.0;
  double online_topk_seconds = 0.0;
};

// Eval-only cosine/MIPS dot-product scanners. The production L2/scaled
// kernels remain unchanged; normalize/mips_scaled select final scoring only.
template <int M>
void PrepareAndScanAngularDenseAvx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch,
    bool normalize, const float* cosine_inverse_norms = nullptr,
    bool mips_scaled = false) noexcept;

template <int M>
void ScanFloatAngularDenseAvx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch,
    bool normalize, bool mips_scaled = false) noexcept;

// Compatibility/reference entry point for callers that already hold scaled
// tables. It retains the pre-existing native ParentLOUDS behavior.
template <int M>
void ScanDenseAvx2(const ResidentClusterView<M>& cluster,
                   const ScaledClusterQueryTables<M>& query,
                   QueryTopK* __restrict top_k,
                   float* __restrict dot_scratch) noexcept;

template <int M>
void PrepareAndScanDenseMaterializedAvx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept;

template <int M>
void PrepareAndScanDenseMaterializedU16Avx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept;

template <int M>
void PrepareAndScanDenseMaterializedU32Avx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept;

template <int M>
void PrepareAndScanDenseNativeLoudsAvx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept;

// Detailed quantized-scaled/materialized-parent profiling variants. They
// preserve candidate order and score arithmetic, but materialize one
// cluster's distances in reusable scratch so score generation and online heap
// maintenance have separate logical timing intervals. Production search never
// calls these functions.
template <int M>
void PrepareAndScanDenseMaterializedProfiledAvx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch,
    float* __restrict distance_scratch,
    DenseIntScanTiming* __restrict timing) noexcept;

template <int M>
void PrepareAndScanDenseMaterializedU16ProfiledAvx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch,
    float* __restrict distance_scratch,
    DenseIntScanTiming* __restrict timing) noexcept;

template <int M>
void PrepareAndScanDenseMaterializedU32ProfiledAvx2(
    const TrustedQueryTables<M>& query,
    const ResidentClusterView<M>& cluster,
    ScaledClusterQueryTables<M>* __restrict scaled_query,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch,
    float* __restrict distance_scratch,
    DenseIntScanTiming* __restrict timing) noexcept;

template <int M>
void ScanFloatDenseMaterializedAvx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept;

template <int M>
void ScanFloatDenseMaterializedU16Avx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept;

template <int M>
void ScanFloatDenseMaterializedU32Avx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept;

template <int M>
void ScanFloatDenseNativeLoudsAvx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept;

template <int M>
void ScanFloatScaledDenseMaterializedAvx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept;

template <int M>
void ScanFloatScaledDenseMaterializedU16Avx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept;

template <int M>
void ScanFloatScaledDenseMaterializedU32Avx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept;

template <int M>
void ScanFloatScaledDenseNativeLoudsAvx2(
    const TrustedQueryTables<M>& query,
    const FloatResidentClusterView<M>& cluster,
    QueryTopK* __restrict top_k,
    float* __restrict dot_scratch) noexcept;

extern template void PrepareAndScanDenseMaterializedAvx2<5>(
    const TrustedQueryTables<5>&,
    const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict,
    QueryTopK* __restrict,
    float* __restrict) noexcept;
extern template void PrepareAndScanDenseMaterializedAvx2<10>(
    const TrustedQueryTables<10>&,
    const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict,
    QueryTopK* __restrict,
    float* __restrict) noexcept;
extern template void PrepareAndScanDenseMaterializedU16Avx2<5>(
    const TrustedQueryTables<5>&, const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
extern template void PrepareAndScanDenseMaterializedU16Avx2<10>(
    const TrustedQueryTables<10>&, const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
extern template void PrepareAndScanDenseMaterializedU32Avx2<5>(
    const TrustedQueryTables<5>&, const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
extern template void PrepareAndScanDenseMaterializedU32Avx2<10>(
    const TrustedQueryTables<10>&, const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
extern template void PrepareAndScanDenseNativeLoudsAvx2<5>(
    const TrustedQueryTables<5>&,
    const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
extern template void PrepareAndScanDenseNativeLoudsAvx2<10>(
    const TrustedQueryTables<10>&,
    const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict, QueryTopK* __restrict, float* __restrict) noexcept;
extern template void PrepareAndScanDenseMaterializedProfiledAvx2<5>(
    const TrustedQueryTables<5>&, const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
extern template void PrepareAndScanDenseMaterializedProfiledAvx2<10>(
    const TrustedQueryTables<10>&, const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
extern template void PrepareAndScanDenseMaterializedU16ProfiledAvx2<5>(
    const TrustedQueryTables<5>&, const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
extern template void PrepareAndScanDenseMaterializedU16ProfiledAvx2<10>(
    const TrustedQueryTables<10>&, const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
extern template void PrepareAndScanDenseMaterializedU32ProfiledAvx2<5>(
    const TrustedQueryTables<5>&, const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
extern template void PrepareAndScanDenseMaterializedU32ProfiledAvx2<10>(
    const TrustedQueryTables<10>&, const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict, QueryTopK* __restrict, float* __restrict, float* __restrict,
    DenseIntScanTiming* __restrict) noexcept;
extern template void ScanFloatDenseMaterializedAvx2<5>(
    const TrustedQueryTables<5>&,
    const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatDenseMaterializedAvx2<10>(
    const TrustedQueryTables<10>&,
    const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatDenseMaterializedU16Avx2<5>(
    const TrustedQueryTables<5>&, const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatDenseMaterializedU16Avx2<10>(
    const TrustedQueryTables<10>&, const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatDenseMaterializedU32Avx2<5>(
    const TrustedQueryTables<5>&, const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatDenseMaterializedU32Avx2<10>(
    const TrustedQueryTables<10>&, const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatDenseNativeLoudsAvx2<5>(
    const TrustedQueryTables<5>&,
    const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatDenseNativeLoudsAvx2<10>(
    const TrustedQueryTables<10>&,
    const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatScaledDenseMaterializedAvx2<5>(
    const TrustedQueryTables<5>&, const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatScaledDenseMaterializedAvx2<10>(
    const TrustedQueryTables<10>&, const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatScaledDenseMaterializedU16Avx2<5>(
    const TrustedQueryTables<5>&, const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatScaledDenseMaterializedU16Avx2<10>(
    const TrustedQueryTables<10>&, const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatScaledDenseMaterializedU32Avx2<5>(
    const TrustedQueryTables<5>&, const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatScaledDenseMaterializedU32Avx2<10>(
    const TrustedQueryTables<10>&, const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatScaledDenseNativeLoudsAvx2<5>(
    const TrustedQueryTables<5>&, const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanFloatScaledDenseNativeLoudsAvx2<10>(
    const TrustedQueryTables<10>&, const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanDenseAvx2<5>(
    const ResidentClusterView<5>&,
    const ScaledClusterQueryTables<5>&,
    QueryTopK* __restrict, float* __restrict) noexcept;
extern template void ScanDenseAvx2<10>(
    const ResidentClusterView<10>&,
    const ScaledClusterQueryTables<10>&,
    QueryTopK* __restrict, float* __restrict) noexcept;


extern template void PrepareAndScanAngularDenseAvx2<5>(
    const TrustedQueryTables<5>&, const ResidentClusterView<5>&,
    ScaledClusterQueryTables<5>* __restrict, QueryTopK* __restrict,
    float* __restrict, bool, const float*, bool) noexcept;
extern template void ScanFloatAngularDenseAvx2<5>(
    const TrustedQueryTables<5>&, const FloatResidentClusterView<5>&,
    QueryTopK* __restrict, float* __restrict, bool, bool) noexcept;

extern template void PrepareAndScanAngularDenseAvx2<6>(
    const TrustedQueryTables<6>&, const ResidentClusterView<6>&,
    ScaledClusterQueryTables<6>* __restrict, QueryTopK* __restrict,
    float* __restrict, bool, const float*, bool) noexcept;
extern template void ScanFloatAngularDenseAvx2<6>(
    const TrustedQueryTables<6>&, const FloatResidentClusterView<6>&,
    QueryTopK* __restrict, float* __restrict, bool, bool) noexcept;

extern template void PrepareAndScanAngularDenseAvx2<10>(
    const TrustedQueryTables<10>&, const ResidentClusterView<10>&,
    ScaledClusterQueryTables<10>* __restrict, QueryTopK* __restrict,
    float* __restrict, bool, const float*, bool) noexcept;
extern template void ScanFloatAngularDenseAvx2<10>(
    const TrustedQueryTables<10>&, const FloatResidentClusterView<10>&,
    QueryTopK* __restrict, float* __restrict, bool, bool) noexcept;

extern template void PrepareAndScanAngularDenseAvx2<12>(
    const TrustedQueryTables<12>&, const ResidentClusterView<12>&,
    ScaledClusterQueryTables<12>* __restrict, QueryTopK* __restrict,
    float* __restrict, bool, const float*, bool) noexcept;
extern template void ScanFloatAngularDenseAvx2<12>(
    const TrustedQueryTables<12>&, const FloatResidentClusterView<12>&,
    QueryTopK* __restrict, float* __restrict, bool, bool) noexcept;
}  // namespace stlqpp
