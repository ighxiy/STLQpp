#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"

#include <memory>

namespace stlqpp {

enum class CudaBlasAtomics { kDisallowed, kAllowed };

class CudaContext final {
 public:
  CudaContext();
  ~CudaContext();

  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;
  CudaContext(CudaContext&& other) noexcept;
  CudaContext& operator=(CudaContext&& other) noexcept;

  [[nodiscard]] static StatusOr<CudaContext> Create(
      int device, bool allow_tf32 = true,
      CudaBlasAtomics atomics = CudaBlasAtomics::kDisallowed);

  [[nodiscard]] int device() const noexcept;
  [[nodiscard]] bool allow_tf32() const noexcept;
  [[nodiscard]] void* native_stream() const noexcept;
  [[nodiscard]] void* native_blas() const noexcept;
  [[nodiscard]] Status Synchronize() const;

 private:
  struct State;
  explicit CudaContext(std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

}  // namespace stlqpp
