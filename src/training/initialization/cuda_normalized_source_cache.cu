#include "stlqpp/training/initialization/cuda_normalized_source_cache.h"

#include "stlqpp/runtime/cuda/device_buffer.h"
#include "stlqpp/runtime/cuda/pinned_buffer.h"
#include "stlqpp/runtime/cuda/status.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

constexpr std::uint32_t kWarpsPerBlock = 8U;
constexpr std::uint32_t kWarpSize = 32U;

template <std::uint32_t WarpsPerBlock, typename Scalar>
__global__ void NormalizeColumnsWarpKernel(const Scalar* input,
                                           std::uint32_t leading_input,
                                           float* output,
                                           std::uint32_t leading_output,
                                           float* squared_norms,
                                           std::uint32_t rows,
                                           std::uint32_t columns) {
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1U);
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t column = blockIdx.x * WarpsPerBlock + warp;
  if (column >= columns) {
    return;
  }
  const Scalar* source = input +
      static_cast<std::size_t>(column) * leading_input;
  float* destination = output +
      static_cast<std::size_t>(column) * leading_output;
  float squared_norm = 0.0F;
  for (std::uint32_t row = lane; row < rows; row += kWarpSize) {
    const float value = static_cast<float>(source[row]);
    squared_norm += value * value;
  }
  constexpr unsigned int kFullWarp = 0xffffffffU;
  for (std::uint32_t offset = kWarpSize / 2U; offset != 0U;
       offset >>= 1U) {
    squared_norm += __shfl_down_sync(kFullWarp, squared_norm, offset);
  }
  squared_norm = __shfl_sync(kFullWarp, squared_norm, 0);
  if (lane == 0U) {
    squared_norms[column] = squared_norm;
  }
  const float inverse_norm = rsqrtf(fmaxf(
      squared_norm, 1.0e-12F));
  for (std::uint32_t row = lane; row < rows; row += kWarpSize) {
    destination[row] = static_cast<float>(source[row]) * inverse_norm;
  }
}

struct SourceSpan final {
  std::uint64_t first_id = 0U;
  std::uint32_t count = 0U;
};

}  // namespace

template <typename Scalar>
struct CudaNormalizedSourceCache<Scalar>::State final {
  CudaContext* context = nullptr;
  InitializationBasicBlockSourceFactory<Scalar>* source = nullptr;
  int device = 0;
  std::uint32_t dimension = 0U;
  std::uint64_t source_rows = 0U;
  std::uint64_t cached_rows = 0U;
  CudaDeviceBuffer<float> normalized_cache;
  CudaDeviceBuffer<float> source_norm2_cache;
  std::vector<SourceSpan> cached_spans;
  std::array<CudaPinnedBuffer<Scalar>, 2> pinned;
  std::array<CudaDeviceBuffer<Scalar>, 2> raw;
  std::array<CudaDeviceBuffer<float>, 2> normalized_slots;
  std::array<CudaDeviceBuffer<float>, 2> source_norm2_slots;
  cudaStream_t upload_stream = nullptr;
  std::array<cudaEvent_t, 2> ready{nullptr, nullptr};
  std::array<cudaEvent_t, 2> consumed{nullptr, nullptr};
  std::array<bool, 2> ready_recorded{false, false};
  std::array<bool, 2> consumed_recorded{false, false};
  bool pass_open = false;

  ~State() {
    cudaSetDevice(device);
    if (upload_stream != nullptr) {
      cudaStreamSynchronize(upload_stream);
    }
    for (cudaEvent_t event : ready) {
      if (event != nullptr) {
        cudaEventDestroy(event);
      }
    }
    for (cudaEvent_t event : consumed) {
      if (event != nullptr) {
        cudaEventDestroy(event);
      }
    }
    if (upload_stream != nullptr) {
      cudaStreamDestroy(upload_stream);
    }
  }
};

template <typename Scalar>
struct CudaNormalizedSourcePass<Scalar>::State final {
  using CacheState = typename CudaNormalizedSourceCache<Scalar>::State;

  CacheState* cache = nullptr;
  std::unique_ptr<BasicBlockSource<Scalar>> remainder;
  std::uint32_t cached_span_index = 0U;
  std::uint64_t next_row = 0U;
  int next_slot = 0;
  bool mutable_cached_blocks = false;
  std::array<bool, 2> prepared{false, false};
  std::array<bool, 2> published{false, false};
  bool remainder_exhausted = false;
  bool finished = false;

  [[nodiscard]] Status PrepareSlot(int slot,
                                   std::uint32_t count,
                                   bool needs_raw) {
    const std::size_t values =
        static_cast<std::size_t>(count) * cache->dimension;
    const bool grows_normalized =
        values > cache->normalized_slots[slot].capacity();
    const bool grows_norm2 =
        count > cache->source_norm2_slots[slot].capacity();
    if (cache->consumed_recorded[slot] &&
        (grows_normalized || grows_norm2)) {
      if (const Status status = CudaRuntimeStatus(
              "cudaEventSynchronize normalized slot consumed",
              cudaEventSynchronize(cache->consumed[slot])); !status.ok()) {
        return status;
      }
    }
    if (const Status status =
            cache->normalized_slots[slot].Ensure(values); !status.ok()) {
      return status;
    }
    if (const Status status = cache->source_norm2_slots[slot].Ensure(count);
        !status.ok()) {
      return status;
    }
    if (needs_raw) {
      if (cache->ready_recorded[slot]) {
        if (const Status status = CudaRuntimeStatus(
                "cudaEventSynchronize pinned upload source",
                cudaEventSynchronize(cache->ready[slot])); !status.ok()) {
          return status;
        }
      }
      if (const Status status = cache->pinned[slot].Ensure(values);
          !status.ok()) {
        return status;
      }
      if (const Status status = cache->raw[slot].Ensure(values);
          !status.ok()) {
        return status;
      }
    }
    if (cache->consumed_recorded[slot]) {
      if (const Status status = CudaRuntimeStatus(
              "cudaStreamWaitEvent normalized slot reuse",
              cudaStreamWaitEvent(cache->upload_stream,
                                  cache->consumed[slot], 0U)); !status.ok()) {
        return status;
      }
    }
    return Status::Ok();
  }

  [[nodiscard]] Status PublishReady(int slot) {
    if (const Status status = CudaRuntimeStatus(
            "cudaEventRecord normalized slot ready",
            cudaEventRecord(cache->ready[slot], cache->upload_stream));
        !status.ok()) {
      return status;
    }
    cache->ready_recorded[slot] = true;
    return Status::Ok();
  }

  ~State() {
    if (cache != nullptr) {
      cache->pass_open = false;
    }
  }
};

template <typename Scalar>
CudaNormalizedSourcePass<Scalar>::CudaNormalizedSourcePass(
    std::unique_ptr<State> state) noexcept : state_(std::move(state)) {}

template <typename Scalar>
CudaNormalizedSourcePass<Scalar>::~CudaNormalizedSourcePass() = default;

template <typename Scalar>
CudaNormalizedSourcePass<Scalar>::CudaNormalizedSourcePass(
    CudaNormalizedSourcePass&&) noexcept = default;

template <typename Scalar>
CudaNormalizedSourcePass<Scalar>&
CudaNormalizedSourcePass<Scalar>::operator=(
    CudaNormalizedSourcePass&&) noexcept = default;

template <typename Scalar>
StatusOr<std::optional<CudaNormalizedSourceBlock>>
CudaNormalizedSourcePass<Scalar>::Next() {
  if (state_ == nullptr || state_->cache == nullptr || state_->finished) {
    return Status::FailedPrecondition(
        "CUDA normalized-source pass is not active");
  }
  auto& cache = *state_->cache;
  if (state_->next_row == cache.source_rows) {
    if (state_->remainder != nullptr && !state_->remainder_exhausted) {
      auto extra = state_->remainder->Next();
      if (!extra.ok()) {
        return extra.status();
      }
      if (extra.value().has_value()) {
        return Status::DataLoss(
            "normalized-source range exceeded its registered row count");
      }
      state_->remainder_exhausted = true;
    }
    return std::optional<CudaNormalizedSourceBlock>{};
  }
  const int slot = state_->next_slot;
  state_->next_slot ^= 1;
  assert(!state_->prepared[slot]);
  assert(!state_->published[slot]);
  std::uint32_t count = 0U;
  const float* normalized = nullptr;
  const float* source_squared_norms = nullptr;
  bool cache_view = false;
  if (state_->next_row < cache.cached_rows) {
    assert(state_->cached_span_index < cache.cached_spans.size());
    const SourceSpan span = cache.cached_spans[state_->cached_span_index++];
    assert(span.first_id == state_->next_row);
    count = span.count;
    if (const Status status = state_->PrepareSlot(slot, count, false);
        !status.ok()) {
      return status;
    }
    if (state_->mutable_cached_blocks) {
      if (const Status status = CudaRuntimeStatus(
              "cudaMemcpyAsync normalized cache D2D",
              cudaMemcpyAsync(
                  cache.normalized_slots[slot].data(),
                  cache.normalized_cache.data() +
                      static_cast<std::size_t>(state_->next_row) *
                          cache.dimension,
                  static_cast<std::size_t>(count) * cache.dimension *
                      sizeof(float),
                  cudaMemcpyDeviceToDevice, cache.upload_stream));
          !status.ok()) {
        return status;
      }
      normalized = cache.normalized_slots[slot].data();
    } else {
      normalized = cache.normalized_cache.data() +
          static_cast<std::size_t>(state_->next_row) * cache.dimension;
      cache_view = true;
    }
    if (const Status status = CudaRuntimeStatus(
            "cudaMemcpyAsync source norm2 cache D2D",
            cudaMemcpyAsync(
                cache.source_norm2_slots[slot].data(),
                cache.source_norm2_cache.data() + state_->next_row,
                static_cast<std::size_t>(count) * sizeof(float),
                cudaMemcpyDeviceToDevice, cache.upload_stream)); !status.ok()) {
      return status;
    }
    source_squared_norms = cache.source_norm2_slots[slot].data();
  } else {
    if (state_->remainder == nullptr) {
      return Status::DataLoss(
          "normalized-source streaming remainder is missing");
    }
    auto next = state_->remainder->Next();
    if (!next.ok()) {
      return next.status();
    }
    if (!next.value().has_value()) {
      return Status::DataLoss(
          "normalized-source range ended before its registered row count");
    }
    const BasicInputBlock<Scalar>& block = *next.value();
    if (block.first_id != state_->next_row ||
        block.dimension != cache.dimension || block.count == 0U ||
        block.count > cache.source_rows - state_->next_row ||
        block.values.size() !=
            static_cast<std::size_t>(block.count) * cache.dimension) {
      return Status::DataLoss(
          "normalized-source streaming block changed row order or shape");
    }
    count = block.count;
    if (const Status status = state_->PrepareSlot(slot, count, true);
        !status.ok()) {
      return status;
    }
    const std::size_t values =
        static_cast<std::size_t>(count) * cache.dimension;
    std::memcpy(cache.pinned[slot].data(), block.values.data(),
                values * sizeof(Scalar));
    if (const Status status = CudaRuntimeStatus(
            "cudaMemcpyAsync normalized-source H2D",
            cudaMemcpyAsync(cache.raw[slot].data(),
                            cache.pinned[slot].data(),
                            values * sizeof(Scalar),
                            cudaMemcpyHostToDevice,
                            cache.upload_stream)); !status.ok()) {
      return status;
    }
    const std::uint32_t grid =
        (count + kWarpsPerBlock - 1U) / kWarpsPerBlock;
    NormalizeColumnsWarpKernel<kWarpsPerBlock, Scalar><<<
        grid, kWarpsPerBlock * kWarpSize, 0U,
        cache.upload_stream>>>(
        cache.raw[slot].data(), cache.dimension,
        cache.normalized_slots[slot].data(), cache.dimension,
        cache.source_norm2_slots[slot].data(),
        cache.dimension, count);
    if (const Status status = CudaRuntimeStatus(
            "NormalizeColumnsWarpKernel streaming",
            cudaPeekAtLastError()); !status.ok()) {
      return status;
    }
    normalized = cache.normalized_slots[slot].data();
    source_squared_norms = cache.source_norm2_slots[slot].data();
  }
  const std::uint64_t first_id = state_->next_row;
  state_->next_row += count;
  state_->prepared[slot] = true;
  return std::optional<CudaNormalizedSourceBlock>(
      CudaNormalizedSourceBlock{
          .first_id = first_id,
          .normalized = {
              normalized, cache.dimension,
              count, cache.dimension},
          .source_squared_norms = source_squared_norms,
          .slot = static_cast<std::uint8_t>(slot),
          .cache_view = cache_view,
      });
}

template <typename Scalar>
void* CudaNormalizedSourcePass<Scalar>::native_upload_stream() const noexcept {
  return state_->cache->upload_stream;
}

template <typename Scalar>
Status CudaNormalizedSourcePass<Scalar>::Publish(
    const CudaNormalizedSourceBlock& block) {
  const int slot = block.slot;
  assert(slot >= 0 && slot < 2);
  assert(state_->prepared[slot]);
  assert(!state_->published[slot]);
  if (const Status status = state_->PublishReady(slot); !status.ok()) {
    return status;
  }
  state_->published[slot] = true;
  return Status::Ok();
}

template <typename Scalar>
Status CudaNormalizedSourcePass<Scalar>::WaitReady(
    const CudaNormalizedSourceBlock& block) {
  const int slot = block.slot;
  assert(slot >= 0 && slot < 2);
  assert(state_->prepared[slot]);
  assert(state_->published[slot]);
  const cudaStream_t compute =
      static_cast<cudaStream_t>(state_->cache->context->native_stream());
  return CudaRuntimeStatus(
      "cudaStreamWaitEvent normalized slot ready",
      cudaStreamWaitEvent(compute, state_->cache->ready[slot], 0U));
}

template <typename Scalar>
Status CudaNormalizedSourcePass<Scalar>::MarkConsumed(
    const CudaNormalizedSourceBlock& block) {
  const int slot = block.slot;
  assert(slot >= 0 && slot < 2);
  assert(state_->prepared[slot]);
  assert(state_->published[slot]);
  const cudaStream_t compute =
      static_cast<cudaStream_t>(state_->cache->context->native_stream());
  if (const Status status = CudaRuntimeStatus(
          "cudaEventRecord normalized slot consumed",
          cudaEventRecord(state_->cache->consumed[slot], compute));
      !status.ok()) {
    return status;
  }
  state_->cache->consumed_recorded[slot] = true;
  state_->prepared[slot] = false;
  state_->published[slot] = false;
  return Status::Ok();
}

template <typename Scalar>
Status CudaNormalizedSourcePass<Scalar>::Finish() {
  if (state_ == nullptr || state_->cache == nullptr || state_->finished) {
    return Status::FailedPrecondition(
        "CUDA normalized-source pass is not active");
  }
  if (state_->next_row != state_->cache->source_rows) {
    return Status::FailedPrecondition(
        "CUDA normalized-source pass is incomplete");
  }
  if (state_->remainder != nullptr && !state_->remainder_exhausted) {
    auto extra = state_->remainder->Next();
    if (!extra.ok()) {
      return extra.status();
    }
    if (extra.value().has_value()) {
      return Status::DataLoss(
          "normalized-source range exceeded its registered row count");
    }
    state_->remainder_exhausted = true;
  }
  assert(!state_->prepared[0] && !state_->prepared[1]);
  assert(!state_->published[0] && !state_->published[1]);
  state_->finished = true;
  state_->cache->pass_open = false;
  state_->cache = nullptr;
  return Status::Ok();
}

template <typename Scalar>
CudaNormalizedSourceCache<Scalar>::CudaNormalizedSourceCache(
    std::unique_ptr<State> state) noexcept : state_(std::move(state)) {}

template <typename Scalar>
CudaNormalizedSourceCache<Scalar>::~CudaNormalizedSourceCache() = default;

template <typename Scalar>
CudaNormalizedSourceCache<Scalar>::CudaNormalizedSourceCache(
    CudaNormalizedSourceCache&&) noexcept = default;

template <typename Scalar>
CudaNormalizedSourceCache<Scalar>&
CudaNormalizedSourceCache<Scalar>::operator=(
    CudaNormalizedSourceCache&&) noexcept = default;

template <typename Scalar>
StatusOr<CudaNormalizedSourceCache<Scalar>>
CudaNormalizedSourceCache<Scalar>::Build(
    CudaContext& context,
    InitializationBasicBlockSourceFactory<Scalar>& source,
    CudaNormalizedSourceCacheOptions options) {
  if (context.device() < 0 || options.source_rows == 0U ||
      options.dimension == 0U ||
      options.source_rows >
          std::numeric_limits<std::size_t>::max() / options.dimension) {
    return Status::InvalidArgument(
        "CUDA normalized-source cache options are invalid");
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(context.device())); !status.ok()) {
    return status;
  }
  auto state = std::make_unique<State>();
  state->context = &context;
  state->source = &source;
  state->device = context.device();
  state->dimension = options.dimension;
  state->source_rows = options.source_rows;
  if (const Status status = CudaRuntimeStatus(
          "cudaStreamCreate normalized upload",
          cudaStreamCreateWithFlags(
              &state->upload_stream, cudaStreamNonBlocking)); !status.ok()) {
    return status;
  }
  for (int slot = 0; slot < 2; ++slot) {
    if (const Status status = CudaRuntimeStatus(
            "cudaEventCreate normalized ready",
            cudaEventCreateWithFlags(
                &state->ready[slot], cudaEventDisableTiming)); !status.ok()) {
      return status;
    }
    if (const Status status = CudaRuntimeStatus(
            "cudaEventCreate normalized consumed",
            cudaEventCreateWithFlags(
                &state->consumed[slot], cudaEventDisableTiming)); !status.ok()) {
      return status;
    }
  }

  const std::size_t maximum_cached_rows =
      options.maximum_bytes /
      (static_cast<std::size_t>(options.dimension) * sizeof(float));
  const std::uint64_t desired_cached_rows = std::min<std::uint64_t>(
      options.source_rows, maximum_cached_rows);
  if (desired_cached_rows != 0U) {
    const std::size_t cache_values =
        static_cast<std::size_t>(desired_cached_rows) * options.dimension;
    Status allocation = state->normalized_cache.Ensure(cache_values);
    if (allocation.ok()) {
      allocation = state->source_norm2_cache.Ensure(
          static_cast<std::size_t>(desired_cached_rows));
    }
    if (!allocation.ok() && allocation.code() != StatusCode::kResourceExhausted) {
      return allocation;
    }
    if (!allocation.ok()) {
      state->normalized_cache.Reset();
      state->source_norm2_cache.Reset();
    } else {
      auto opened = source.OpenRange(0U, desired_cached_rows);
      if (!opened.ok()) {
        return opened.status();
      }
      std::uint64_t next_row = 0U;
      constexpr int slot = 0;
      while (next_row < desired_cached_rows) {
        auto next = opened.value()->Next();
        if (!next.ok()) {
          return next.status();
        }
        if (!next.value().has_value()) {
          return Status::DataLoss(
              "normalized-source cache prefix ended early");
        }
        const BasicInputBlock<Scalar>& block = *next.value();
        if (block.first_id != next_row ||
            block.dimension != options.dimension || block.count == 0U ||
            block.count > desired_cached_rows - next_row ||
            block.values.size() !=
                static_cast<std::size_t>(block.count) * options.dimension) {
          return Status::DataLoss(
              "normalized-source cache prefix changed row order or shape");
        }
        if (state->ready_recorded[slot]) {
          if (const Status status = CudaRuntimeStatus(
                  "cudaEventSynchronize prefix upload slot",
                  cudaEventSynchronize(state->ready[slot])); !status.ok()) {
            return status;
          }
        }
        const std::size_t values =
            static_cast<std::size_t>(block.count) * options.dimension;
        if (const Status status = state->pinned[slot].Ensure(values);
            !status.ok()) {
          return status;
        }
        if (const Status status = state->raw[slot].Ensure(values);
            !status.ok()) {
          return status;
        }
        if (const Status status = state->normalized_slots[slot].Ensure(values);
            !status.ok()) {
          return status;
        }
        if (const Status status = state->source_norm2_slots[slot].Ensure(
                block.count); !status.ok()) {
          return status;
        }
        std::memcpy(state->pinned[slot].data(), block.values.data(),
                    values * sizeof(Scalar));
        if (const Status status = CudaRuntimeStatus(
                "cudaMemcpyAsync normalized cache prefix H2D",
                cudaMemcpyAsync(state->raw[slot].data(),
                                state->pinned[slot].data(),
                                values * sizeof(Scalar),
                                cudaMemcpyHostToDevice,
                                state->upload_stream)); !status.ok()) {
          return status;
        }
        const std::uint32_t grid =
            (block.count + kWarpsPerBlock - 1U) / kWarpsPerBlock;
        NormalizeColumnsWarpKernel<kWarpsPerBlock, Scalar><<<
            grid, kWarpsPerBlock * kWarpSize, 0U,
            state->upload_stream>>>(
            state->raw[slot].data(), options.dimension,
            state->normalized_slots[slot].data(), options.dimension,
            state->source_norm2_slots[slot].data(), options.dimension,
            block.count);
        if (const Status status = CudaRuntimeStatus(
                "NormalizeColumnsWarpKernel cache prefix",
                cudaPeekAtLastError()); !status.ok()) {
          return status;
        }
        if (const Status status = CudaRuntimeStatus(
                "cudaMemcpyAsync normalized prefix cache D2D",
                cudaMemcpyAsync(
                    state->normalized_cache.data() +
                        static_cast<std::size_t>(next_row) * options.dimension,
                    state->normalized_slots[slot].data(),
                    values * sizeof(float), cudaMemcpyDeviceToDevice,
                    state->upload_stream)); !status.ok()) {
          return status;
        }
        if (const Status status = CudaRuntimeStatus(
                "cudaMemcpyAsync source norm2 prefix cache D2D",
                cudaMemcpyAsync(
                    state->source_norm2_cache.data() + next_row,
                    state->source_norm2_slots[slot].data(),
                    static_cast<std::size_t>(block.count) * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    state->upload_stream)); !status.ok()) {
          return status;
        }
        if (const Status status = CudaRuntimeStatus(
                "cudaEventRecord prefix upload slot",
                cudaEventRecord(state->ready[slot],
                                state->upload_stream)); !status.ok()) {
          return status;
        }
        state->ready_recorded[slot] = true;
        state->cached_spans.push_back({next_row, block.count});
        next_row += block.count;
      }
      auto extra = opened.value()->Next();
      if (!extra.ok()) {
        return extra.status();
      }
      if (extra.value().has_value()) {
        return Status::DataLoss(
            "normalized-source cache prefix range was not bounded");
      }
      if (const Status status = CudaRuntimeStatus(
              "cudaStreamSynchronize normalized cache prefix",
              cudaStreamSynchronize(state->upload_stream)); !status.ok()) {
        return status;
      }
      state->ready_recorded = {false, false};
      state->cached_rows = desired_cached_rows;
    }
  }
  return CudaNormalizedSourceCache(std::move(state));
}

template <typename Scalar>
StatusOr<CudaNormalizedSourcePass<Scalar>>
CudaNormalizedSourceCache<Scalar>::OpenPass(bool mutable_cached_blocks) {
  if (state_ == nullptr || state_->pass_open) {
    return Status::FailedPrecondition(
        "CUDA normalized-source cache already has an active pass");
  }
  std::unique_ptr<BasicBlockSource<Scalar>> remainder;
  if (state_->cached_rows < state_->source_rows) {
    auto opened = state_->source->OpenRange(
        state_->cached_rows, state_->source_rows - state_->cached_rows);
    if (!opened.ok() && opened.status().code() == StatusCode::kUnimplemented &&
        state_->cached_rows != 0U) {
      state_->normalized_cache.Reset();
      state_->source_norm2_cache.Reset();
      state_->cached_rows = 0U;
      state_->cached_spans.clear();
      opened = state_->source->Open();
    }
    if (!opened.ok()) {
      return opened.status();
    }
    remainder = std::move(opened).value();
  }
  auto pass = std::make_unique<typename CudaNormalizedSourcePass<Scalar>::State>();
  pass->cache = state_.get();
  pass->remainder = std::move(remainder);
  pass->mutable_cached_blocks = mutable_cached_blocks;
  state_->pass_open = true;
  return CudaNormalizedSourcePass<Scalar>(std::move(pass));
}

template <typename Scalar>
std::uint64_t CudaNormalizedSourceCache<Scalar>::row_count() const noexcept {
  return state_->source_rows;
}

template <typename Scalar>
std::uint64_t CudaNormalizedSourceCache<Scalar>::cached_rows() const noexcept {
  return state_->cached_rows;
}

template <typename Scalar>
std::uint64_t CudaNormalizedSourceCache<Scalar>::resident_bytes() const
    noexcept {
  return state_->cached_rows *
      (static_cast<std::uint64_t>(state_->dimension) + 1U) * sizeof(float);
}

template class CudaNormalizedSourcePass<std::uint8_t>;
template class CudaNormalizedSourcePass<float>;
template class CudaNormalizedSourceCache<std::uint8_t>;
template class CudaNormalizedSourceCache<float>;

}  // namespace stlqpp
