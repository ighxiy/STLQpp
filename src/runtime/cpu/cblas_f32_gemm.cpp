#include "stlqpp/runtime/cpu/cblas_f32_gemm.h"
#include "stlqpp/runtime/cpu/blas_thread_policy.h"

#if defined(STLQPP_CBLAS_MKL)
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace stlqpp {

Status CblasF32Gemm::MultiplyLeft(
    const LeftGemmF32& operation) const {
#ifndef NDEBUG
  constexpr std::size_t kIntMax =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  assert(operation.left.values != nullptr && operation.right != nullptr &&
         operation.output != nullptr && operation.left.rows != 0U &&
         operation.left.columns != 0U && operation.query_count != 0U &&
         operation.right_leading_dimension >= operation.left.columns &&
         operation.output_leading_dimension >= operation.left.rows &&
         operation.left.rows <= kIntMax &&
         operation.left.columns <= kIntMax &&
         operation.query_count <= kIntMax &&
         operation.right_leading_dimension <= kIntMax &&
         operation.output_leading_dimension <= kIntMax);
#endif

  const ScopedBlasCallThreadCount threads(thread_count_);
  cblas_sgemm(
      CblasColMajor, CblasNoTrans, CblasNoTrans,
      static_cast<int>(operation.left.rows),
      static_cast<int>(operation.query_count),
      static_cast<int>(operation.left.columns), 1.0F,
      operation.left.values, static_cast<int>(operation.left.rows),
      operation.right,
      static_cast<int>(operation.right_leading_dimension), 0.0F,
      operation.output,
      static_cast<int>(operation.output_leading_dimension));
  return Status::Ok();
}

Status CblasF32Gemm::MultiplyTransposedLeft(
    const TransposedLeftGemmF32& operation) const {
#ifndef NDEBUG
  constexpr std::size_t kIntMax =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  assert(operation.left.values != nullptr && operation.right != nullptr &&
         operation.output != nullptr && operation.left.rows != 0U &&
         operation.left.columns != 0U && operation.query_count != 0U &&
         operation.right_leading_dimension >= operation.left.rows &&
         operation.output_leading_dimension >= operation.left.columns &&
         operation.left.rows <= kIntMax &&
         operation.left.columns <= kIntMax &&
         operation.query_count <= kIntMax &&
         operation.right_leading_dimension <= kIntMax &&
         operation.output_leading_dimension <= kIntMax);
#endif

  const ScopedBlasCallThreadCount threads(thread_count_);
  cblas_sgemm(
      CblasColMajor, CblasTrans, CblasNoTrans,
      static_cast<int>(operation.left.columns),
      static_cast<int>(operation.query_count),
      static_cast<int>(operation.left.rows), 1.0F,
      operation.left.values, static_cast<int>(operation.left.rows),
      operation.right,
      static_cast<int>(operation.right_leading_dimension), 0.0F,
      operation.output,
      static_cast<int>(operation.output_leading_dimension));
  return Status::Ok();
}

Status CblasF32Gemm::MultiplySelfTransposedUpper(
    const SelfTransposedUpperF32& operation) const {
#ifndef NDEBUG
  constexpr std::size_t kIntMax =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  assert(operation.input.values != nullptr && operation.output != nullptr &&
         operation.input.rows != 0U && operation.input.columns != 0U &&
         operation.output_leading_dimension >= operation.input.columns &&
         operation.input.rows <= kIntMax &&
         operation.input.columns <= kIntMax &&
         operation.output_leading_dimension <= kIntMax);
#endif

  const ScopedBlasCallThreadCount threads(thread_count_);
  cblas_ssyrk(
      CblasColMajor, CblasUpper, CblasTrans,
      static_cast<int>(operation.input.columns),
      static_cast<int>(operation.input.rows), 1.0F,
      operation.input.values, static_cast<int>(operation.input.rows), 0.0F,
      operation.output,
      static_cast<int>(operation.output_leading_dimension));
  return Status::Ok();
}

}  // namespace stlqpp
