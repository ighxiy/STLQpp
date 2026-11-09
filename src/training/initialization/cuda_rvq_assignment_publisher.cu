#include "stlqpp/training/initialization/cuda_rvq_assignment_publisher.h"

#include "stlqpp/core/array_view.h"
#include "stlqpp/runtime/cuda/device_buffer.h"
#include "stlqpp/runtime/cuda/pinned_buffer.h"
#include "stlqpp/runtime/cuda/status.h"

#include <cuda_runtime.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace stlqpp {

struct CudaRvqAssignmentPublisher::State final {
  CudaContext* context = nullptr;
  RvqAssignmentStore* destination = nullptr;
  int device = 0;
  std::uint32_t layer = 0U;
  cudaStream_t copy_stream = nullptr;
  std::array<CudaDeviceBuffer<std::uint32_t>, 2> staged;
  std::array<CudaPinnedBuffer<std::uint32_t>, 2> host;
  std::array<cudaEvent_t, 2> staged_ready{nullptr, nullptr};
  std::array<cudaEvent_t, 2> copied{nullptr, nullptr};
  std::array<bool, 2> pending{false, false};
  std::array<std::uint64_t, 2> first_id{0U, 0U};
  std::array<std::uint32_t, 2> count{0U, 0U};

  ~State() {
    cudaSetDevice(device);
    for (cudaEvent_t event : staged_ready) {
      if (event != nullptr) {
        cudaEventDestroy(event);
      }
    }
    for (cudaEvent_t event : copied) {
      if (event != nullptr) {
        cudaEventDestroy(event);
      }
    }
  }
};

CudaRvqAssignmentPublisher::CudaRvqAssignmentPublisher(
    std::unique_ptr<State> state) noexcept : state_(std::move(state)) {}

CudaRvqAssignmentPublisher::~CudaRvqAssignmentPublisher() = default;

CudaRvqAssignmentPublisher::CudaRvqAssignmentPublisher(
    CudaRvqAssignmentPublisher&&) noexcept = default;

CudaRvqAssignmentPublisher&
CudaRvqAssignmentPublisher::operator=(
    CudaRvqAssignmentPublisher&&) noexcept = default;

StatusOr<CudaRvqAssignmentPublisher>
CudaRvqAssignmentPublisher::Create(
    CudaContext& context,
    void* native_copy_stream,
    RvqAssignmentStore& destination,
    std::uint32_t layer) {
  if (native_copy_stream == nullptr ||
      layer >= destination.retained_layers() ||
      destination.complete(layer)) {
    return Status::InvalidArgument(
        "CUDA RVQ assignment publisher request is invalid");
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaSetDevice", cudaSetDevice(context.device())); !status.ok()) {
    return status;
  }
  auto state = std::make_unique<State>();
  state->context = &context;
  state->destination = &destination;
  state->device = context.device();
  state->layer = layer;
  state->copy_stream = static_cast<cudaStream_t>(native_copy_stream);
  for (int slot = 0; slot < 2; ++slot) {
    if (const Status status = CudaRuntimeStatus(
            "cudaEventCreate assignment staged",
            cudaEventCreateWithFlags(&state->staged_ready[slot],
                                     cudaEventDisableTiming)); !status.ok()) {
      return status;
    }
    if (const Status status = CudaRuntimeStatus(
            "cudaEventCreate assignment copied",
            cudaEventCreateWithFlags(&state->copied[slot],
                                     cudaEventDisableTiming)); !status.ok()) {
      return status;
    }
  }
  return CudaRvqAssignmentPublisher(std::move(state));
}

Status CudaRvqAssignmentPublisher::Flush(std::uint8_t slot) {
  assert(slot < 2U);
  if (!state_->pending[slot]) {
    return Status::Ok();
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaEventSynchronize assignment copied",
          cudaEventSynchronize(state_->copied[slot])); !status.ok()) {
    return status;
  }
  if (const Status status = state_->destination->WriteBlock(
          state_->layer, state_->first_id[slot],
          ArrayView<const std::uint32_t>(state_->host[slot].data(),
                                         state_->count[slot])); !status.ok()) {
    return status;
  }
  state_->pending[slot] = false;
  return Status::Ok();
}

Status CudaRvqAssignmentPublisher::Schedule(
    std::uint8_t slot,
    std::uint64_t first_id,
    const std::uint32_t* device_codes,
    std::uint32_t count) {
  assert(slot < 2U);
  assert(device_codes != nullptr);
  assert(count != 0U);
  if (const Status status = Flush(slot); !status.ok()) {
    return status;
  }
  if (const Status status = state_->staged[slot].Ensure(count); !status.ok()) {
    return status;
  }
  if (const Status status = state_->host[slot].Ensure(count); !status.ok()) {
    return status;
  }
  const std::size_t bytes = static_cast<std::size_t>(count) *
                            sizeof(std::uint32_t);
  const cudaStream_t compute =
      static_cast<cudaStream_t>(state_->context->native_stream());
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync assignment D2D stage",
          cudaMemcpyAsync(state_->staged[slot].data(), device_codes, bytes,
                          cudaMemcpyDeviceToDevice, compute)); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaEventRecord assignment staged",
          cudaEventRecord(state_->staged_ready[slot], compute)); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaStreamWaitEvent assignment staged",
          cudaStreamWaitEvent(state_->copy_stream,
                              state_->staged_ready[slot], 0U)); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync assignment D2H",
          cudaMemcpyAsync(state_->host[slot].data(),
                          state_->staged[slot].data(), bytes,
                          cudaMemcpyDeviceToHost,
                          state_->copy_stream)); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cudaEventRecord assignment copied",
          cudaEventRecord(state_->copied[slot],
                          state_->copy_stream)); !status.ok()) {
    return status;
  }
  state_->first_id[slot] = first_id;
  state_->count[slot] = count;
  state_->pending[slot] = true;
  return Status::Ok();
}

Status CudaRvqAssignmentPublisher::Drain() {
  if (state_->pending[0] && state_->pending[1]) {
    const std::uint8_t first =
        state_->first_id[0] <= state_->first_id[1] ? 0U : 1U;
    if (const Status status = Flush(first); !status.ok()) {
      return status;
    }
    return Flush(static_cast<std::uint8_t>(1U - first));
  }
  if (const Status status = Flush(0U); !status.ok()) {
    return status;
  }
  return Flush(1U);
}

}  // namespace stlqpp
