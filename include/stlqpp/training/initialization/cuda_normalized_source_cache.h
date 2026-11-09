#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/runtime/cuda/device_matrix.h"
#include "stlqpp/training/io/basic_block_source.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace stlqpp {

struct CudaNormalizedSourceCacheOptions final {
  std::uint64_t source_rows = 0U;
  std::uint32_t dimension = 0U;
  std::size_t maximum_bytes = 0U;
};

struct CudaNormalizedSourceBlock final {
  std::uint64_t first_id = 0U;
  CudaMatrixF32View normalized;
  const float* source_squared_norms = nullptr;
  std::uint8_t slot = 0U;
  bool cache_view = false;
};

template <typename Scalar>
class CudaNormalizedSourceCache;

// One sequential Lloyd scan over a cache-backed normalized source. Next()
// prepares the copy-stream side of one slot but deliberately does not publish
// it. The layer residual producer may append prior-code transfers to that same
// stream before Publish() records the final ready event. MarkConsumed() records
// the last compute-stream reader before the slot can be reused.
template <typename Scalar>
class CudaNormalizedSourcePass final {
 public:
  ~CudaNormalizedSourcePass();
  CudaNormalizedSourcePass(CudaNormalizedSourcePass&&) noexcept;
  CudaNormalizedSourcePass& operator=(
      CudaNormalizedSourcePass&&) noexcept;
  CudaNormalizedSourcePass(const CudaNormalizedSourcePass&) = delete;
  CudaNormalizedSourcePass& operator=(
      const CudaNormalizedSourcePass&) = delete;

  [[nodiscard]] StatusOr<std::optional<CudaNormalizedSourceBlock>> Next();
  [[nodiscard]] void* native_upload_stream() const noexcept;
  [[nodiscard]] Status Publish(const CudaNormalizedSourceBlock& block);
  [[nodiscard]] Status WaitReady(const CudaNormalizedSourceBlock& block);
  [[nodiscard]] Status MarkConsumed(const CudaNormalizedSourceBlock& block);
  [[nodiscard]] Status Finish();

 private:
  struct State;
  explicit CudaNormalizedSourcePass(std::unique_ptr<State> state) noexcept;
  friend class CudaNormalizedSourceCache<Scalar>;

  std::unique_ptr<State> state_;
};

// One-layer owner of DEV's best-effort normalized-device prefix cache. The
// desired prefix is min(source_rows, budget/(dimension*sizeof(float))). A
// failed cache allocation selects streaming before any Lloyd hot loop; it is
// not a training failure.
template <typename Scalar>
class CudaNormalizedSourceCache final {
 public:
  [[nodiscard]] static StatusOr<CudaNormalizedSourceCache> Build(
      CudaContext& context,
      InitializationBasicBlockSourceFactory<Scalar>& source,
      CudaNormalizedSourceCacheOptions options);

  ~CudaNormalizedSourceCache();
  CudaNormalizedSourceCache(CudaNormalizedSourceCache&&) noexcept;
  CudaNormalizedSourceCache& operator=(
      CudaNormalizedSourceCache&&) noexcept;
  CudaNormalizedSourceCache(const CudaNormalizedSourceCache&) = delete;
  CudaNormalizedSourceCache& operator=(
      const CudaNormalizedSourceCache&) = delete;

  [[nodiscard]] StatusOr<CudaNormalizedSourcePass<Scalar>> OpenPass(
      bool mutable_cached_blocks);
  [[nodiscard]] std::uint64_t row_count() const noexcept;
  [[nodiscard]] std::uint64_t cached_rows() const noexcept;
  [[nodiscard]] std::uint64_t resident_bytes() const noexcept;

 private:
  struct State;
  explicit CudaNormalizedSourceCache(std::unique_ptr<State> state) noexcept;
  friend class CudaNormalizedSourcePass<Scalar>;

  std::unique_ptr<State> state_;
};

extern template class CudaNormalizedSourcePass<std::uint8_t>;
extern template class CudaNormalizedSourcePass<float>;
extern template class CudaNormalizedSourceCache<std::uint8_t>;
extern template class CudaNormalizedSourceCache<float>;

}  // namespace stlqpp
