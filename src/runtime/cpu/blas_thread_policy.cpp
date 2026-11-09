#include "stlqpp/runtime/cpu/blas_thread_policy.h"

#if defined(STLQPP_CBLAS_MKL)
#include <mkl.h>
#elif defined(STLQPP_CBLAS_OPENBLAS)
extern "C" void openblas_set_num_threads(int);
extern "C" int openblas_get_num_threads(void);
#endif

#include <cassert>
#include <limits>
#include <mutex>
#include <shared_mutex>

namespace stlqpp {
namespace {

#if defined(STLQPP_CBLAS_OPENBLAS)
struct OpenBlasPolicyState final {
  std::shared_mutex mutex;
  std::uint32_t current =
      static_cast<std::uint32_t>(openblas_get_num_threads());
};

OpenBlasPolicyState& OpenBlasPolicy() {
  static OpenBlasPolicyState state;
  return state;
}

thread_local std::uint32_t stage_depth = 0U;

void SetOpenBlasThreads(OpenBlasPolicyState& state,
                        std::uint32_t requested) noexcept {
  if (state.current != requested) {
    openblas_set_num_threads(static_cast<int>(requested));
    state.current = requested;
  }
}
#endif

}  // namespace

Status ConfigureBlasThreadCount(std::uint32_t thread_count) {
  if (thread_count == 0U ||
      thread_count > static_cast<std::uint32_t>(
                         std::numeric_limits<int>::max())) {
    return Status::InvalidArgument(
        "BLAS thread count must fit a positive int");
  }
#if defined(STLQPP_CBLAS_MKL)
  mkl_set_num_threads(static_cast<int>(thread_count));
#elif defined(STLQPP_CBLAS_OPENBLAS)
  auto& state = OpenBlasPolicy();
  if (stage_depth != 0U) {
    SetOpenBlasThreads(state, thread_count);
  } else {
    const std::unique_lock lock(state.mutex);
    SetOpenBlasThreads(state, thread_count);
  }
#endif
  return Status::Ok();
}

ScopedBlasThreadCount::ScopedBlasThreadCount(
    std::uint32_t thread_count) noexcept {
  if (thread_count != 0U) {
    assert(thread_count <= static_cast<std::uint32_t>(
                               std::numeric_limits<int>::max()));
#if defined(STLQPP_CBLAS_MKL)
    if (mkl_get_max_threads() == static_cast<int>(thread_count)) return;
    previous_local_ = mkl_set_num_threads_local(
        static_cast<int>(thread_count));
#elif defined(STLQPP_CBLAS_OPENBLAS)
    auto& state = OpenBlasPolicy();
    if (stage_depth == 0U) {
      global_lock_ = std::unique_lock(state.mutex);
    }
    previous_global_ = state.current;
    SetOpenBlasThreads(state, thread_count);
    ++stage_depth;
#endif
    active_ = true;
  }
}

ScopedBlasThreadCount::~ScopedBlasThreadCount() {
  if (active_) {
#if defined(STLQPP_CBLAS_MKL)
    mkl_set_num_threads_local(previous_local_);
#elif defined(STLQPP_CBLAS_OPENBLAS)
    auto& state = OpenBlasPolicy();
    SetOpenBlasThreads(state, previous_global_);
    --stage_depth;
#endif
  }
}

ScopedBlasCallThreadCount::ScopedBlasCallThreadCount(
    std::uint32_t thread_count) noexcept {
  if (thread_count == 0U) return;
  assert(thread_count <= static_cast<std::uint32_t>(
                             std::numeric_limits<int>::max()));
#if defined(STLQPP_CBLAS_MKL)
  if (mkl_get_max_threads() == static_cast<int>(thread_count)) return;
  previous_local_ = mkl_set_num_threads_local(
      static_cast<int>(thread_count));
  changed_ = true;
#elif defined(STLQPP_CBLAS_OPENBLAS)
  auto& state = OpenBlasPolicy();
  if (stage_depth != 0U) {
    previous_global_ = state.current;
    if (previous_global_ != thread_count) {
      SetOpenBlasThreads(state, thread_count);
      changed_ = true;
    }
    return;
  }
  shared_lock_ = std::shared_lock(state.mutex);
  if (state.current == thread_count) return;
  shared_lock_.unlock();
  exclusive_lock_ = std::unique_lock(state.mutex);
  previous_global_ = state.current;
  SetOpenBlasThreads(state, thread_count);
  changed_ = true;
#endif
}

ScopedBlasCallThreadCount::~ScopedBlasCallThreadCount() {
  if (!changed_) return;
#if defined(STLQPP_CBLAS_MKL)
  mkl_set_num_threads_local(previous_local_);
#elif defined(STLQPP_CBLAS_OPENBLAS)
  SetOpenBlasThreads(OpenBlasPolicy(), previous_global_);
#endif
}

}  // namespace stlqpp
