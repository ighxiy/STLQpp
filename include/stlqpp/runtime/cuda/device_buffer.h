#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/runtime/cuda/status.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <utility>

namespace stlqpp {

// Movable high-water CUDA allocation. The owning CUDA state establishes its
// device before this object's destructor runs.
template <typename T>
class CudaDeviceBuffer final {
 public:
  CudaDeviceBuffer() = default;
  ~CudaDeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
  CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;
  CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        capacity_(std::exchange(other.capacity_, 0U)) {}
  CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        cudaFree(data_);
      }
      data_ = std::exchange(other.data_, nullptr);
      capacity_ = std::exchange(other.capacity_, 0U);
    }
    return *this;
  }

  [[nodiscard]] Status Ensure(std::size_t count) {
    if (count <= capacity_) {
      return Status::Ok();
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return Status::ResourceExhausted("CUDA allocation extent overflows");
    }
    Reset();
    const cudaError_t result = cudaMalloc(
        reinterpret_cast<void**>(&data_), count * sizeof(T));
    if (result != cudaSuccess) {
      return CudaRuntimeStatus("cudaMalloc", result);
    }
    capacity_ = count;
    return Status::Ok();
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

  void Reset() noexcept {
    if (data_ != nullptr) {
      cudaFree(data_);
      data_ = nullptr;
      capacity_ = 0U;
    }
  }

 private:
  T* data_ = nullptr;
  std::size_t capacity_ = 0U;
};

}  // namespace stlqpp
