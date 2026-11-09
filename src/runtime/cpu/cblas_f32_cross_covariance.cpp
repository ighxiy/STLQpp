#include "stlqpp/runtime/cpu/cblas_f32_cross_covariance.h"

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

Status CblasF32CrossCovariance::Accumulate(
    const CrossCovarianceF32& operation) const {
#ifndef NDEBUG
  constexpr std::size_t kIntMax =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  assert(operation.left.values != nullptr &&
         operation.right.values != nullptr && operation.output != nullptr &&
         operation.left.rows != 0U &&
         operation.left.rows == operation.right.rows &&
         operation.left.columns != 0U &&
         operation.left.columns == operation.right.columns &&
         operation.output_leading_dimension >= operation.left.rows &&
         operation.left.rows <= kIntMax &&
         operation.left.columns <= kIntMax &&
         operation.output_leading_dimension <= kIntMax);
#endif
  cblas_sgemm(
      CblasColMajor, CblasNoTrans, CblasTrans,
      static_cast<int>(operation.left.rows),
      static_cast<int>(operation.left.rows),
      static_cast<int>(operation.left.columns), 1.0F,
      operation.left.values, static_cast<int>(operation.left.rows),
      operation.right.values, static_cast<int>(operation.right.rows), 1.0F,
      operation.output,
      static_cast<int>(operation.output_leading_dimension));
  return Status::Ok();
}

}  // namespace stlqpp
