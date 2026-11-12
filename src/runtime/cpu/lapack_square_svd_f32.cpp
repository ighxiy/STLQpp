#include "stlqpp/runtime/cpu/lapack_square_svd_f32.h"

#if defined(STLQPP_LAPACKE_MKL)
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>

namespace stlqpp {

Status LapackSquareSvdF32::Solve(
    const SquareSvdSystemF32& system) const {
  constexpr std::size_t kLapackIntMax =
      static_cast<std::size_t>(std::numeric_limits<lapack_int>::max());
  if (system.matrix == nullptr || system.singular_values == nullptr ||
      system.left_vectors == nullptr ||
      system.right_vectors_transposed == nullptr || system.order == 0U ||
      system.order > kLapackIntMax) {
    return Status::InvalidArgument("LAPACK square SVD input is invalid");
  }
  const lapack_int order = static_cast<lapack_int>(system.order);
  const std::uint64_t matrix_elements =
      static_cast<std::uint64_t>(system.order) * system.order;
  const std::uint64_t scratch_elements =
      3U * matrix_elements + 2U * system.order;
  if (scratch_elements >
      std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    return Status::InvalidArgument("LAPACK square SVD workspace is too large");
  }
  // LAPACK overwrites H. One private allocation holds its double copy,
  // both factors, singular values and superb; output-only regions need no
  // initialization. LAPACKE owns the decomposition's internal work array.
  std::unique_ptr<double[]> scratch;
  try {
    scratch.reset(new double[static_cast<std::size_t>(scratch_elements)]);
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("cannot allocate LAPACK square SVD workspace");
  }
  const auto size = static_cast<std::size_t>(matrix_elements);
  double* matrix = scratch.get();
  double* left = matrix + size;
  double* right_transposed = left + size;
  double* singular_values = right_transposed + size;
  double* superb = singular_values + system.order;
  std::copy_n(system.matrix, size, matrix);
#if defined(STLQPP_LAPACKE_MKL)
  const auto threads = system.allow_phase_parallelism && system.order >= 512U
      ? std::max(1U, std::min(available_workers_, system.order >= 1024U ? 8U : 4U)) : 1U;
  struct LocalThreads {
    int previous;
    explicit LocalThreads(unsigned count) : previous(mkl_set_num_threads_local(static_cast<int>(count))) {}
    ~LocalThreads() { mkl_set_num_threads_local(previous); }
  } local_threads(threads);
#endif
  const lapack_int info = LAPACKE_dgesvd(
      LAPACK_COL_MAJOR, 'S', 'S', order, order, matrix, order,
      singular_values, left, order, right_transposed, order, superb);
  if (info == LAPACK_WORK_MEMORY_ERROR ||
      info == LAPACK_TRANSPOSE_MEMORY_ERROR) {
    return Status::ResourceExhausted("cannot allocate LAPACK dgesvd work array");
  }
  if (info != 0) {
    return Status::FailedPrecondition("LAPACKE_dgesvd failed");
  }
  // Keep the validated rotation path: cast U/Vt to FP32 here, then let
  // RotationUpdater assemble R with its existing FP32 GEMM.
  const auto to_float = [](double value) { return static_cast<float>(value); };
  std::transform(singular_values, singular_values + system.order,
                 system.singular_values, to_float);
  std::transform(left, left + size, system.left_vectors, to_float);
  std::transform(right_transposed, right_transposed + size,
                 system.right_vectors_transposed, to_float);
  return Status::Ok();
}

}  // namespace stlqpp
