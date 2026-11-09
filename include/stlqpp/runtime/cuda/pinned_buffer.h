#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/runtime/cuda/status.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <utility>

namespace stlqpp {

template <typename T>
class CudaPinnedBuffer final {
 public:
  CudaPinnedBuffer() = default;
  ~CudaPinnedBuffer() {
    if (data_ != nullptr) {
      cudaFreeHost(data_);
    }
  }

  CudaPinnedBuffer(const CudaPinnedBuffer&) = delete;
  CudaPinnedBuffer& operator=(const CudaPinnedBuffer&) = delete;
  CudaPinnedBuffer(CudaPinnedBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        capacity_(std::exchange(other.capacity_, 0U)) {}
  CudaPinnedBuffer& operator=(CudaPinnedBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        cudaFreeHost(data_);
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
      return Status::ResourceExhausted(
          "CUDA pinned allocation extent overflows");
    }
    if (data_ != nullptr) {
      cudaFreeHost(data_);
      data_ = nullptr;
      capacity_ = 0U;
    }
    const cudaError_t result = cudaMallocHost(
        reinterpret_cast<void**>(&data_), count * sizeof(T));
    if (result != cudaSuccess) {
      return CudaRuntimeStatus("cudaMallocHost", result);
    }
    capacity_ = count;
    return Status::Ok();
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

 private:
  T* data_ = nullptr;
  std::size_t capacity_ = 0U;
};

}  // namespace stlqpp
