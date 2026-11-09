#include "stlqpp/index/norm/reconstruction_norm8_cuda.h"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace stlqpp {
namespace {

Status CudaStatus(cudaError_t error, const char* operation) {
  return error == cudaSuccess ? Status::Ok() :
      Status::Internal(std::string(operation) + ": " +
                       cudaGetErrorString(error));
}

template <typename T>
Status Allocate(T** target, std::size_t count) {
  return CudaStatus(cudaMalloc(reinterpret_cast<void**>(target),
                               count * sizeof(T)), "cudaMalloc norm8");
}

__global__ void GatherQuantiles(const float* sorted, std::uint64_t count,
                                float* output) {
  const std::uint32_t center = blockIdx.x * blockDim.x + threadIdx.x;
  if (center < 256U) {
    output[center] = sorted[static_cast<std::uint64_t>(center) * count / 256U];
  }
}

__device__ std::uint32_t NearestCenter(const float* centers, float value) {
  std::uint32_t low = 0U;
  std::uint32_t high = 256U;
  while (low < high) {
    const std::uint32_t middle = (low + high) / 2U;
    if (centers[middle] < value) low = middle + 1U;
    else high = middle;
  }
  if (low == 0U) return 0U;
  if (low == 256U) return 255U;
  return value - centers[low - 1U] <= centers[low] - value
      ? low - 1U : low;
}

// NearestCenter is monotone even with duplicate centers. Search its actual
// mapping rather than rounding midpoint thresholds or changing tie behavior.
__device__ std::uint64_t BinBegin(const float* sorted, std::uint64_t count,
                                  const float* centers, std::uint32_t bin) {
  std::uint64_t low = 0U, high = count;
  while (low < high) {
    const auto middle = low + (high - low) / 2U;
    if (NearestCenter(centers, sorted[middle]) < bin) low = middle + 1U;
    else high = middle;
  }
  return low;
}

__global__ void AccumulateSortedNormBins(
    const float* values, std::uint64_t count, const float* centers,
    std::uint64_t* partial_counts, double* partial_sums) {
  __shared__ double sums[256];
  __shared__ std::uint64_t begin, end;
  if (threadIdx.x == 0U) {
    const auto low = BinBegin(values, count, centers, blockIdx.y);
    const auto high = BinBegin(values, count, centers, blockIdx.y + 1U);
    const auto extent = high - low;
    begin = low + extent * blockIdx.x / gridDim.x;
    end = low + extent * (blockIdx.x + 1U) / gridDim.x;
  }
  __syncthreads();
  double sum = 0.0;
  for (auto row = begin + threadIdx.x; row < end; row += blockDim.x)
    sum += static_cast<double>(values[row]);
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = 128U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    const auto destination = static_cast<std::size_t>(blockIdx.x) * 256U + blockIdx.y;
    partial_counts[destination] = end - begin;
    partial_sums[destination] = sums[0];
  }
}

__global__ void ReduceNormBins(
    const std::uint64_t* partial_counts, const double* partial_sums,
    std::uint32_t blocks, std::uint64_t* counts, double* sums) {
  const std::uint32_t center = blockIdx.x;
  std::uint64_t count = 0U;
  double sum = 0.0;
  for (std::uint32_t block = 0U; block < blocks; ++block) {
    const std::size_t entry = static_cast<std::size_t>(block) * 256U + center;
    count += partial_counts[entry];
    sum += partial_sums[entry];
  }
  counts[center] = count;
  sums[center] = sum;
}

__global__ void EncodeNormLabels(const float* values, std::uint64_t count,
                                 const float* centers,
                                 std::uint8_t* labels) {
  const std::uint64_t row =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < count) {
    labels[row] = static_cast<std::uint8_t>(
        NearestCenter(centers, values[row]));
  }
}

__global__ void UpdateNormCenters(const std::uint64_t* counts,
                                  const double* sums, float* centers,
                                  int* changed) {
  const std::uint32_t center = threadIdx.x;
  if (center >= 256U || counts[center] == 0U) return;
  const float updated = static_cast<float>(
      sums[center] / static_cast<double>(counts[center]));
  if (fabsf(updated - centers[center]) > 1.0e-6F) {
    atomicExch(changed, 1);
  }
  centers[center] = updated;
}

constexpr std::uint32_t kBlocks = 16U;

}  // namespace

struct CudaNorm8Fitter::State final {
  int device = 0;
  float* values = nullptr;
  float* sorted = nullptr;
  float* centers = nullptr;
  float* quantiles = nullptr;
  std::uint64_t* partial_counts = nullptr;
  double* partial_sums = nullptr;
  std::uint64_t* counts = nullptr;
  double* sums = nullptr;
  int* changed = nullptr;
  std::uint64_t count = 0U;

  ~State() {
    cudaFree(values);
    cudaFree(sorted);
    cudaFree(centers);
    cudaFree(quantiles);
    cudaFree(partial_counts);
    cudaFree(partial_sums);
    cudaFree(counts);
    cudaFree(sums);
    cudaFree(changed);
  }
};

CudaNorm8Fitter::~CudaNorm8Fitter() {
  if (state_) cudaSetDevice(state_->device);
}

StatusOr<std::unique_ptr<CudaNorm8Fitter>> CudaNorm8Fitter::Create(
    ArrayView<const float> values, int device) {
  if (values.empty() ||
      values.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Status::ResourceExhausted(
        "GPU norm8 requires 1..INT_MAX population rows");
  }
  int devices = 0;
  if (const Status status = CudaStatus(
          cudaGetDeviceCount(&devices), "cudaGetDeviceCount norm8");
      !status.ok()) return status;
  if (devices == 0) {
    return Status::FailedPrecondition("GPU norm8 device is unavailable");
  }
  if (device < 0 || device >= devices) {
    return Status::InvalidArgument("GPU norm8 device ordinal is invalid");
  }
  if (const Status status = CudaStatus(
          cudaSetDevice(device), "select norm8 CUDA device");
      !status.ok()) return status;
  auto output = std::unique_ptr<CudaNorm8Fitter>(new CudaNorm8Fitter());
  output->state_ = std::make_unique<State>();
  auto& state = *output->state_;
  state.device = device;
  state.count = values.size();
  if (const Status status = Allocate(&state.values, values.size());
      !status.ok()) return status;
  if (const Status status = CudaStatus(
          cudaMemcpy(state.values, values.data(),
                     values.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload norm8 population");
      !status.ok()) return status;
  if (const Status status = Allocate(&state.centers, 256U); !status.ok())
    return status;
  if (const Status status = Allocate(&state.quantiles, 256U); !status.ok())
    return status;
  if (const Status status = Allocate(&state.partial_counts,
                                     static_cast<std::size_t>(kBlocks) * 256U);
      !status.ok()) return status;
  if (const Status status = Allocate(&state.partial_sums,
                                     static_cast<std::size_t>(kBlocks) * 256U);
      !status.ok()) return status;
  if (const Status status = Allocate(&state.counts, 256U); !status.ok())
    return status;
  if (const Status status = Allocate(&state.sums, 256U); !status.ok())
    return status;
  if (const Status status = Allocate(&state.changed, 1U); !status.ok())
    return status;
  return std::move(output);
}

StatusOr<std::array<float, 256>> CudaNorm8Fitter::InitialCenters() {
  auto& state = *state_;
  if (const Status status = CudaStatus(
          cudaSetDevice(state.device), "select norm8 CUDA device");
      !status.ok()) return status;
  if (state.sorted == nullptr) {
    if (const Status status = Allocate(&state.sorted, state.count);
        !status.ok()) return status;
    std::size_t scratch_bytes = 0U;
    if (const Status status = CudaStatus(
            cub::DeviceRadixSort::SortKeys(
                nullptr, scratch_bytes, state.values, state.sorted,
                static_cast<int>(state.count)),
            "size norm8 radix sort");
        !status.ok()) return status;
    void* scratch = nullptr;
    if (const Status status = CudaStatus(
            cudaMalloc(&scratch, scratch_bytes), "allocate norm8 radix scratch");
        !status.ok()) return status;
    const cudaError_t sorted = cub::DeviceRadixSort::SortKeys(
        scratch, scratch_bytes, state.values, state.sorted,
        static_cast<int>(state.count));
    cudaFree(scratch);
    if (const Status status = CudaStatus(sorted, "sort norm8 values");
        !status.ok()) return status;
    cudaFree(state.values);
    state.values = nullptr;
  }
  GatherQuantiles<<<1, 256>>>(state.sorted, state.count, state.quantiles);
  if (const Status status = CudaStatus(
          cudaGetLastError(), "gather norm8 quantiles");
      !status.ok()) return status;
  std::array<float, 256> centers{};
  const Status copied = CudaStatus(
      cudaMemcpy(centers.data(), state.quantiles, sizeof(centers),
                 cudaMemcpyDeviceToHost),
      "download norm8 quantiles");
  if (!copied.ok()) return copied;
  return centers;
}

Status CudaNorm8Fitter::Accumulate(
    Norm8CodebookView codebook,
    std::array<std::uint64_t, 256>& counts,
    std::array<double, 256>& sums) {
  if (codebook.centers == nullptr || codebook.center_count != 256U) {
    return Status::InvalidArgument("GPU norm8 requires 256 centers");
  }
  auto& state = *state_;
  if (state.sorted == nullptr) {
    auto initialized = InitialCenters();
    if (!initialized.ok()) return initialized.status();
  }
  if (const Status status = CudaStatus(
          cudaSetDevice(state.device), "select norm8 CUDA device");
      !status.ok()) return status;
  if (const Status status = CudaStatus(
          cudaMemcpy(state.centers, codebook.centers, 256U * sizeof(float),
                     cudaMemcpyHostToDevice),
          "upload norm8 centers");
      !status.ok()) return status;
  AccumulateSortedNormBins<<<dim3(kBlocks, 256U), 256>>>(
      state.sorted, state.count, state.centers,
      state.partial_counts, state.partial_sums);
  if (const Status status = CudaStatus(
          cudaGetLastError(), "accumulate norm8 bins");
      !status.ok()) return status;
  ReduceNormBins<<<256, 1>>>(
      state.partial_counts, state.partial_sums, kBlocks,
      state.counts, state.sums);
  if (const Status status = CudaStatus(
          cudaGetLastError(), "reduce norm8 bins");
      !status.ok()) return status;
  if (const Status status = CudaStatus(
          cudaMemcpy(counts.data(), state.counts, sizeof(counts),
                     cudaMemcpyDeviceToHost),
          "download norm8 counts");
      !status.ok()) return status;
  if (const Status status = CudaStatus(
          cudaMemcpy(sums.data(), state.sums, sizeof(sums),
                     cudaMemcpyDeviceToHost),
          "download norm8 sums");
      !status.ok()) return status;
  return Status::Ok();
}

StatusOr<std::array<float, 256>> CudaNorm8Fitter::Fit(
    std::uint32_t iterations) {
  if (iterations == 0U) {
    return Status::InvalidArgument("GPU norm8 fit requires a positive pass count");
  }
  auto initialized = InitialCenters();
  if (!initialized.ok()) return initialized.status();
  auto& state = *state_;
  if (const Status status = CudaStatus(
          cudaSetDevice(state.device), "select norm8 CUDA device");
      !status.ok()) return status;
  if (const Status status = CudaStatus(
          cudaMemcpy(state.centers, state.quantiles,
                     256U * sizeof(float), cudaMemcpyDeviceToDevice),
          "initialize norm8 device centers");
      !status.ok()) return status;
  for (std::uint32_t pass = 0U; pass < iterations; ++pass) {
    if (const Status status = CudaStatus(
            cudaMemset(state.changed, 0, sizeof(int)),
            "reset norm8 convergence flag");
        !status.ok()) return status;
    AccumulateSortedNormBins<<<dim3(kBlocks, 256U), 256>>>(
        state.sorted, state.count, state.centers,
        state.partial_counts, state.partial_sums);
    if (const Status status = CudaStatus(
            cudaGetLastError(), "accumulate norm8 bins");
        !status.ok()) return status;
    ReduceNormBins<<<256, 1>>>(
        state.partial_counts, state.partial_sums, kBlocks,
        state.counts, state.sums);
    if (const Status status = CudaStatus(
            cudaGetLastError(), "reduce norm8 bins");
        !status.ok()) return status;
    UpdateNormCenters<<<1, 256>>>(
        state.counts, state.sums, state.centers, state.changed);
    if (const Status status = CudaStatus(
            cudaGetLastError(), "update norm8 centers");
        !status.ok()) return status;
    int changed = 0;
    if (const Status status = CudaStatus(
            cudaMemcpy(&changed, state.changed, sizeof(changed),
                       cudaMemcpyDeviceToHost),
            "download norm8 convergence flag");
        !status.ok()) return status;
    if (changed == 0) break;
  }
  std::array<float, 256> centers{};
  if (const Status status = CudaStatus(
          cudaMemcpy(centers.data(), state.centers, sizeof(centers),
                     cudaMemcpyDeviceToHost),
          "download final norm8 centers");
      !status.ok()) return status;
  return centers;
}

struct CudaNorm8LabelEncoder::State final {
  int device = 0;
  float* centers = nullptr;
  float* values = nullptr;
  std::uint8_t* labels = nullptr;
  std::size_t capacity = 0U;

  ~State() {
    cudaSetDevice(device);
    cudaFree(centers);
    cudaFree(values);
    cudaFree(labels);
  }
};

CudaNorm8LabelEncoder::~CudaNorm8LabelEncoder() = default;

StatusOr<std::unique_ptr<CudaNorm8LabelEncoder>>
CudaNorm8LabelEncoder::Create(Norm8CodebookView codebook, int device) {
  if (codebook.centers == nullptr || codebook.center_count != 256U) {
    return Status::InvalidArgument("GPU norm8 labels require 256 centers");
  }
  int devices = 0;
  if (const Status status = CudaStatus(
          cudaGetDeviceCount(&devices), "cudaGetDeviceCount norm8 labels");
      !status.ok()) return status;
  if (device < 0 || device >= devices) {
    return Status::FailedPrecondition("GPU norm8 label device is unavailable");
  }
  if (const Status status = CudaStatus(
          cudaSetDevice(device), "select norm8 label CUDA device");
      !status.ok()) return status;
  auto result = std::unique_ptr<CudaNorm8LabelEncoder>(
      new CudaNorm8LabelEncoder());
  result->state_ = std::make_unique<State>();
  result->state_->device = device;
  if (const Status status = Allocate(&result->state_->centers, 256U);
      !status.ok()) return status;
  if (const Status status = CudaStatus(
          cudaMemcpy(result->state_->centers, codebook.centers,
                     256U * sizeof(float), cudaMemcpyHostToDevice),
          "upload norm8 label centers");
      !status.ok()) return status;
  return std::move(result);
}

Status CudaNorm8LabelEncoder::Encode(ArrayView<const float> values,
                                     ArrayView<std::uint8_t> labels) {
  if (values.size() != labels.size()) {
    return Status::InvalidArgument("GPU norm8 label count disagrees");
  }
  if (values.empty()) return Status::Ok();
  auto& state = *state_;
  if (const Status status = CudaStatus(
          cudaSetDevice(state.device), "select norm8 label CUDA device");
      !status.ok()) return status;
  if (values.size() > state.capacity) {
    cudaFree(state.values);
    cudaFree(state.labels);
    state.values = nullptr;
    state.labels = nullptr;
    state.capacity = 0U;
    if (const Status status = Allocate(&state.values, values.size());
        !status.ok()) return status;
    if (const Status status = Allocate(&state.labels, values.size());
        !status.ok()) return status;
    state.capacity = values.size();
  }
  if (const Status status = CudaStatus(
          cudaMemcpy(state.values, values.data(),
                     values.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload norm8 label inputs");
      !status.ok()) return status;
  constexpr std::uint32_t kThreads = 256U;
  const std::uint64_t blocks =
      (values.size() + kThreads - 1U) / kThreads;
  if (blocks > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted("GPU norm8 label grid is too large");
  }
  EncodeNormLabels<<<static_cast<std::uint32_t>(blocks), kThreads>>>(
      state.values, values.size(), state.centers, state.labels);
  if (const Status status = CudaStatus(
          cudaGetLastError(), "encode norm8 labels");
      !status.ok()) return status;
  return CudaStatus(
      cudaMemcpy(labels.data(), state.labels, labels.size(),
                 cudaMemcpyDeviceToHost),
      "download norm8 labels");
}

}  // namespace stlqpp
