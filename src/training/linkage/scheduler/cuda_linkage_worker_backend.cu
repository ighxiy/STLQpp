#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/runtime/cuda/device_buffer.h"
#include "stlqpp/runtime/cuda/pinned_buffer.h"
#include "stlqpp/runtime/cuda/status.h"
#include "stlqpp/training/linkage/root_only/root_only_linkage_device_plan.h"
#include "stlqpp/training/linkage/rsd/linkage_rsd_device_plan.h"
#include "stlqpp/training/linkage/scheduler/cuda_linkage_worker_backend.h"

namespace stlqpp {
namespace {

struct CudaEvaluationCompletionState final {
  LinkageBackendEvaluationCompletion completion;
  std::function<StatusOr<std::vector<LinkageEvaluationResult>>()> decode;
};

void CUDART_CB CompleteCudaEvaluation(cudaStream_t, cudaError_t stream_status,
                                      void* opaque) {
  std::unique_ptr<CudaEvaluationCompletionState> state(
      static_cast<CudaEvaluationCompletionState*>(opaque));
  if (stream_status != cudaSuccess) {
    // CUDA callbacks must not call CUDA APIs, including cudaGetErrorString.
    state->completion(Status::Internal(
        "CUDA linkage asynchronous evaluation failed with runtime status " +
        std::to_string(static_cast<int>(stream_status))));
    return;
  }
  state->completion(state->decode());
}

struct DeviceRfull final {
  float* values = nullptr;
  std::uint64_t capacity_bytes = 0U;

  DeviceRfull() = default;
  DeviceRfull(const DeviceRfull&) = delete;
  DeviceRfull& operator=(const DeviceRfull&) = delete;
  DeviceRfull(DeviceRfull&& other) noexcept
      : values(std::exchange(other.values, nullptr)),
        capacity_bytes(std::exchange(other.capacity_bytes, 0U)) {}
  DeviceRfull& operator=(DeviceRfull&& other) noexcept {
    if (this != &other) {
      if (values != nullptr) {
        cudaFree(values);
      }
      values = std::exchange(other.values, nullptr);
      capacity_bytes = std::exchange(other.capacity_bytes, 0U);
    }
    return *this;
  }
  ~DeviceRfull() {
    if (values != nullptr) {
      cudaFree(values);
    }
  }
};

struct RfullCapacity final {
  [[nodiscard]] std::uint64_t operator()(
      const DeviceRfull& buffer) const noexcept {
    return buffer.capacity_bytes;
  }
};

struct DeviceUpdateColumn final {
  float* destination = nullptr;
  std::uint64_t source_offset = 0U;
  std::uint32_t dimension = 0U;
};

struct EvaluationInputLayout final {
  std::size_t task_rfull = 0U;
  std::size_t nodes = 0U;
  std::size_t candidates = 0U;
  std::size_t pair_nodes = 0U;
  std::size_t node_tasks = 0U;
  std::size_t row_offsets = 0U;
  std::size_t node_orders = 0U;
  std::size_t total_bytes = 0U;
};

struct UpdateInputLayout final {
  std::size_t columns = 0U;
  std::size_t values = 0U;
  std::size_t total_bytes = 0U;
};

Status AppendAlignedRegion(std::size_t count, std::size_t element_size,
                           std::size_t alignment, std::size_t* cursor,
                           std::size_t* offset) {
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  if (*cursor > maximum - (alignment - 1U)) {
    return Status::ResourceExhausted(
        "CUDA linkage packed input alignment overflows");
  }
  const std::size_t aligned = (*cursor + alignment - 1U) & ~(alignment - 1U);
  if (count > (maximum - aligned) / element_size) {
    return Status::ResourceExhausted(
        "CUDA linkage packed input extent overflows");
  }
  *offset = aligned;
  *cursor = aligned + count * element_size;
  return Status::Ok();
}

StatusOr<EvaluationInputLayout> MakeEvaluationInputLayout(
    std::size_t task_count, std::size_t node_value_count,
    std::size_t pair_count, std::size_t node_count,
    std::size_t row_offset_count, std::size_t node_order_count) {
  EvaluationInputLayout layout;
  std::size_t cursor = 0U;
  const std::array<Status, 7> statuses = {
      AppendAlignedRegion(task_count, sizeof(const float*),
                          alignof(const float*), &cursor, &layout.task_rfull),
      AppendAlignedRegion(node_value_count, sizeof(float), alignof(float),
                          &cursor, &layout.nodes),
      AppendAlignedRegion(pair_count, sizeof(std::uint32_t),
                          alignof(std::uint32_t), &cursor, &layout.candidates),
      AppendAlignedRegion(pair_count, sizeof(std::uint32_t),
                          alignof(std::uint32_t), &cursor, &layout.pair_nodes),
      AppendAlignedRegion(node_count, sizeof(std::uint32_t),
                          alignof(std::uint32_t), &cursor, &layout.node_tasks),
      AppendAlignedRegion(row_offset_count, sizeof(std::uint32_t),
                          alignof(std::uint32_t), &cursor, &layout.row_offsets),
      AppendAlignedRegion(node_order_count, sizeof(std::uint8_t),
                          alignof(std::uint8_t), &cursor, &layout.node_orders),
  };
  for (const Status& status : statuses) {
    if (!status.ok()) {
      return status;
    }
  }
  layout.total_bytes = cursor;
  return layout;
}

StatusOr<UpdateInputLayout> MakeUpdateInputLayout(std::size_t column_count,
                                                  std::size_t value_count) {
  UpdateInputLayout layout;
  std::size_t cursor = 0U;
  const std::array<Status, 2> statuses = {
      AppendAlignedRegion(column_count, sizeof(DeviceUpdateColumn),
                          alignof(DeviceUpdateColumn), &cursor,
                          &layout.columns),
      AppendAlignedRegion(value_count, sizeof(float), alignof(float), &cursor,
                          &layout.values),
  };
  for (const Status& status : statuses) {
    if (!status.ok()) {
      return status;
    }
  }
  layout.total_bytes = cursor;
  return layout;
}

__global__ void ScatterUpdateColumnsKernel(const DeviceUpdateColumn* columns,
                                           const float* values,
                                           std::uint32_t column_count) {
  const std::uint32_t column = blockIdx.x;
  if (column >= column_count) {
    return;
  }
  const DeviceUpdateColumn descriptor = columns[column];
  const float* source = values + descriptor.source_offset;
  for (std::uint32_t axis = threadIdx.x; axis < descriptor.dimension;
       axis += blockDim.x) {
    descriptor.destination[axis] = source[axis];
  }
}

__global__ void InitializeBestKernel(std::uint32_t node_count, float* errors,
                                     std::uint32_t* pair_indices,
                                     std::uint32_t* parents) {
  const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= node_count) {
    return;
  }
  errors[node] = CUDART_INF_F;
  pair_indices[node] = 0xffffffffU;
  parents[node] = 0xffffffffU;
}

__host__ __device__ constexpr std::size_t WinnerCoefficientsOffset(
    std::uint32_t m) {
  return (sizeof(std::uint32_t) + sizeof(float) + sizeof(std::uint32_t) +
          (m - 1U) * sizeof(std::uint8_t) + alignof(float) - 1U) &
         ~(alignof(float) - 1U);
}
__host__ __device__ constexpr std::size_t WinnerStride(std::uint32_t m) {
  return WinnerCoefficientsOffset(m) + m * sizeof(float);
}

template <int M>
__global__ void PackLinkageWinnersKernel(
    std::uint32_t actual_m, std::uint32_t node_count,
    const std::uint32_t* parents, const std::uint32_t* first_codes,
    const std::uint8_t* trailing_codes, const float* coefficients,
    const float* errors, std::uint8_t* packed) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= node_count) {
    return;
  }
  std::uint8_t* destination =
      packed + static_cast<std::size_t>(node) * WinnerStride(m);
  *reinterpret_cast<std::uint32_t*>(destination) = parents[node];
  *reinterpret_cast<float*>(destination + sizeof(std::uint32_t)) = errors[node];
  *reinterpret_cast<std::uint32_t*>(destination + sizeof(std::uint32_t) +
                                    sizeof(float)) = first_codes[node];
  const std::size_t trailing_source = static_cast<std::size_t>(node) * (m - 1U);
  constexpr std::size_t kTrailingDestination =
      sizeof(std::uint32_t) + sizeof(float) + sizeof(std::uint32_t);
  for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    destination[kTrailingDestination + layer - 1U] =
        trailing_codes[trailing_source + layer - 1U];
  }
  float* packed_coefficients =
      reinterpret_cast<float*>(destination + WinnerCoefficientsOffset(m));
  const std::size_t coefficient_source = static_cast<std::size_t>(node) * m;
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    packed_coefficients[layer] = coefficients[coefficient_source + layer];
  }
}

__global__ void GatherResidualBlockKernel(
    const float* node_vectors, std::uint32_t dimension,
    const std::uint32_t* candidate_columns, const std::uint32_t* pair_nodes,
    const std::uint32_t* node_tasks, const float* const* task_rfull,
    std::uint32_t pair_offset, std::uint32_t pair_count, float* residuals) {
  const std::uint64_t flat =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t extent =
      static_cast<std::uint64_t>(pair_count) * dimension;
  if (flat >= extent) {
    return;
  }
  const std::uint32_t local_pair = static_cast<std::uint32_t>(flat / dimension);
  const std::uint32_t axis = static_cast<std::uint32_t>(flat % dimension);
  const std::uint32_t pair = pair_offset + local_pair;
  const std::uint32_t node = pair_nodes[pair];
  const std::uint32_t task = node_tasks[node];
  const std::uint32_t parent = candidate_columns[pair];
  residuals[flat] =
      node_vectors[static_cast<std::size_t>(node) * dimension + axis] -
      task_rfull[task][static_cast<std::size_t>(parent) * dimension + axis];
}

__global__ void GatherResidualAndNormBlockKernel(
    const float* node_vectors, std::uint32_t dimension,
    const std::uint32_t* candidate_columns, const std::uint32_t* pair_nodes,
    const std::uint32_t* node_tasks, const float* const* task_rfull,
    std::uint32_t pair_offset, std::uint32_t pair_count, float* residuals,
    float* squared_norms) {
  const std::uint32_t local_pair = blockIdx.x;
  if (local_pair >= pair_count) {
    return;
  }
  const std::uint32_t pair = pair_offset + local_pair;
  const std::uint32_t node = pair_nodes[pair];
  const std::uint32_t task = node_tasks[node];
  const std::uint32_t parent = candidate_columns[pair];
  const float* node_vector =
      node_vectors + static_cast<std::size_t>(node) * dimension;
  const float* reconstruction =
      task_rfull[task] + static_cast<std::size_t>(parent) * dimension;
  float* residual =
      residuals + static_cast<std::size_t>(local_pair) * dimension;
  __shared__ float partial[256];
  float norm2 = 0.0F;
  for (std::uint32_t axis = threadIdx.x; axis < dimension; axis += blockDim.x) {
    const float value = node_vector[axis] - reconstruction[axis];
    residual[axis] = value;
    norm2 += value * value;
  }
  partial[threadIdx.x] = norm2;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2U; offset > 0U; offset >>= 1U) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    squared_norms[local_pair] = partial[0];
  }
}

template <int M>
__global__ void ExpandIcmOrdersKernel(std::uint32_t actual_m,
                                      const std::uint8_t* node_orders,
                                      const std::uint32_t* pair_nodes,
                                      std::uint32_t pair_offset,
                                      std::uint32_t pair_count,
                                      std::uint8_t* pair_orders) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  const std::uint32_t flat = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t extent = pair_count * m;
  if (flat >= extent) {
    return;
  }
  const std::uint32_t local_pair = flat / m;
  const std::uint32_t position = flat % m;
  const std::uint32_t node = pair_nodes[pair_offset + local_pair];
  pair_orders[flat] =
      node_orders[static_cast<std::size_t>(node) * m + position];
}

template <int M>
__global__ void SelectRsdBlockKernel(
    std::uint32_t actual_m, std::uint32_t node_count,
    const std::uint32_t* row_offsets, const std::uint32_t* candidate_columns,
    std::uint32_t pair_offset, std::uint32_t pair_count,
    const std::uint8_t* pair_codes, const float* pair_coefficients,
    const float* pair_errors, std::uint32_t* best_parents,
    std::uint32_t* best_pair_indices, std::uint32_t* best_first_codes,
    std::uint8_t* best_trailing_codes, float* best_coefficients,
    float* best_errors) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= node_count) {
    return;
  }
  const std::uint32_t block_end = pair_offset + pair_count;
  const std::uint32_t begin = max(row_offsets[node], pair_offset);
  const std::uint32_t end = min(row_offsets[node + 1U], block_end);
  float selected_error = best_errors[node];
  std::uint32_t selected_pair = best_pair_indices[node];
  std::uint32_t selected_local = 0xffffffffU;
  for (std::uint32_t pair = begin; pair < end; ++pair) {
    const std::uint32_t local = pair - pair_offset;
    const float error = pair_errors[local];
    if (isfinite(error) &&
        (error < selected_error ||
         (error == selected_error && pair < selected_pair))) {
      selected_error = error;
      selected_pair = pair;
      selected_local = local;
    }
  }
  if (selected_local == 0xffffffffU) {
    return;
  }
  best_errors[node] = selected_error;
  best_pair_indices[node] = selected_pair;
  best_parents[node] = candidate_columns[selected_pair];
  best_first_codes[node] =
      pair_codes[static_cast<std::size_t>(selected_local) * m];
  for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    best_trailing_codes[static_cast<std::size_t>(node) * (m - 1U) + layer -
                        1U] =
        pair_codes[static_cast<std::size_t>(selected_local) * m + layer];
    best_coefficients[static_cast<std::size_t>(node) * m + layer] =
        pair_coefficients[static_cast<std::size_t>(selected_local) * m + layer];
  }
  best_coefficients[static_cast<std::size_t>(node) * m] =
      pair_coefficients[static_cast<std::size_t>(selected_local) * m];
}

template <int M>
__global__ void SelectRootOnlyBlockKernel(
    std::uint32_t actual_m, std::uint32_t node_count,
    const std::uint32_t* row_offsets, const std::uint32_t* candidate_columns,
    std::uint32_t pair_offset, std::uint32_t pair_count,
    const std::uint32_t* pair_first_codes,
    const std::uint8_t* pair_trailing_codes, const float* pair_coefficients,
    const float* pair_errors, std::uint32_t* best_parents,
    std::uint32_t* best_pair_indices, std::uint32_t* best_first_codes,
    std::uint8_t* best_trailing_codes, float* best_coefficients,
    float* best_errors) {
  const std::uint32_t m = M == 0 ? actual_m : M;
  const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= node_count) {
    return;
  }
  const std::uint32_t block_end = pair_offset + pair_count;
  const std::uint32_t begin = max(row_offsets[node], pair_offset);
  const std::uint32_t end = min(row_offsets[node + 1U], block_end);
  float selected_error = best_errors[node];
  std::uint32_t selected_pair = best_pair_indices[node];
  std::uint32_t selected_local = 0xffffffffU;
  for (std::uint32_t pair = begin; pair < end; ++pair) {
    const std::uint32_t local = pair - pair_offset;
    const float error = pair_errors[local];
    if (isfinite(error) &&
        (error < selected_error ||
         (error == selected_error && pair < selected_pair))) {
      selected_error = error;
      selected_pair = pair;
      selected_local = local;
    }
  }
  if (selected_local == 0xffffffffU) {
    return;
  }
  best_errors[node] = selected_error;
  best_pair_indices[node] = selected_pair;
  best_parents[node] = candidate_columns[selected_pair];
  best_first_codes[node] = pair_first_codes[selected_local];
  for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    best_trailing_codes[static_cast<std::size_t>(node) * (m - 1U) + layer -
                        1U] =
        pair_trailing_codes[static_cast<std::size_t>(selected_local) *
                                (m - 1U) +
                            layer - 1U];
  }
  for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    best_coefficients[static_cast<std::size_t>(node) * m + layer] =
        pair_coefficients[static_cast<std::size_t>(selected_local) * m + layer];
  }
}

}  // namespace

namespace {

// Stream-local registration ownership, independent of codebook width and
// encoder kind. The enclosing worker synchronizes before destroying this
// store, so queued copies complete before slots, registrations and cache die.
struct CudaLinkageRegistrationStore final {
  struct Cluster final {
    std::uint32_t dimension = 0U;
    std::uint32_t columns = 0U;
    std::uint64_t generation = 0U;
    DeviceRfull rfull;
  };

  struct UpdateSlot final {
    CudaPinnedBuffer<std::uint8_t> host_inputs;
    CudaDeviceBuffer<std::uint8_t> device_inputs;
    cudaEvent_t completion = nullptr;
    bool in_flight = false;

    ~UpdateSlot() {
      if (completion != nullptr) {
        cudaEventDestroy(completion);
      }
    }
  };

  struct RegistrationSlot final {
    CudaPinnedBuffer<float> host_values;
    cudaEvent_t completion = nullptr;
    bool in_flight = false;

    ~RegistrationSlot() {
      if (completion != nullptr) {
        cudaEventDestroy(completion);
      }
    }
  };

  CudaLinkageRegistrationStore(CudaContext& owner, BestFitCacheLimits limits)
      : context(owner), rfull_cache(limits, RfullCapacity{}) {}

  Status InitializeEvents() {
    for (auto& slot : update_slots) {
      const auto result =
          cudaEventCreateWithFlags(&slot.completion, cudaEventDisableTiming);
      if (result != cudaSuccess)
        return CudaRuntimeStatus("cudaEventCreateWithFlags", result);
    }
    for (auto& slot : registration_slots) {
      const auto result =
          cudaEventCreateWithFlags(&slot.completion, cudaEventDisableTiming);
      if (result != cudaSuccess)
        return CudaRuntimeStatus("cudaEventCreateWithFlags", result);
    }
    return Status::Ok();
  }

  StatusOr<LinkageBackendCluster> Register(
      LinkageBackendRegistration registration);
  Status ApplyUpdates(ArrayView<const LinkageBackendUpdate> updates);
  Status Unregister(LinkageBackendCluster cluster);
  LinkageBackendResourceStats ResourceStats() const noexcept;

  CudaContext& context;
  BoundedBestFitCache<DeviceRfull, RfullCapacity> rfull_cache;
  std::uint64_t next_cluster = 1U;
  std::uint64_t live_registration_bytes = 0U;
  std::unordered_map<LinkageBackendCluster, Cluster> clusters;

  std::array<RegistrationSlot, 2> registration_slots;
  std::uint32_t next_registration_slot = 0U;
  std::array<UpdateSlot, 2> update_slots;
  std::uint32_t next_update_slot = 0U;
};

StatusOr<LinkageBackendCluster> CudaLinkageRegistrationStore::Register(
    LinkageBackendRegistration registration) {
  if (registration.dimension == 0U || registration.column_count == 0U ||
      registration.reconstructions.size() !=
          static_cast<std::size_t>(registration.dimension) *
              registration.column_count) {
    return Status::InvalidArgument(
        "CUDA linkage registration has an invalid matrix extent");
  }
  const std::uint64_t element_count =
      static_cast<std::uint64_t>(registration.dimension) *
      registration.column_count;
  if (element_count >
      std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
    return Status::ResourceExhausted(
        "CUDA linkage registration byte extent overflows");
  }
  const std::uint64_t required_bytes = element_count * sizeof(float);
  auto allocation = rfull_cache.Acquire(
      required_bytes, [](std::uint64_t bytes) -> StatusOr<DeviceRfull> {
        cudaGetLastError();
        DeviceRfull result;
        const cudaError_t status =
            cudaMalloc(reinterpret_cast<void**>(&result.values),
                       static_cast<std::size_t>(bytes));
        if (status != cudaSuccess) {
          return CudaRuntimeStatus("cudaMalloc linkage R_full", status);
        }
        result.capacity_bytes = bytes;
        return StatusOr<DeviceRfull>(std::move(result));
      });
  if (!allocation.ok()) {
    return allocation.status();
  }
  DeviceRfull rfull = std::move(allocation).value();
  RegistrationSlot& slot = registration_slots[next_registration_slot];
  next_registration_slot =
      (next_registration_slot + 1U) % registration_slots.size();
  if (slot.in_flight) {
    const cudaError_t wait = cudaEventSynchronize(slot.completion);
    if (wait != cudaSuccess) {
      return CudaRuntimeStatus("cudaEventSynchronize linkage registration",
                               wait);
    }
    slot.in_flight = false;
  }
  if (const Status status =
          slot.host_values.Ensure(static_cast<std::size_t>(element_count));
      !status.ok()) {
    return status;
  }
  std::copy(registration.reconstructions.begin(),
            registration.reconstructions.end(), slot.host_values.data());
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context.native_stream());
  const cudaError_t copy = cudaMemcpyAsync(
      rfull.values, slot.host_values.data(),
      static_cast<std::size_t>(required_bytes), cudaMemcpyHostToDevice, stream);
  if (copy != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpyAsync register R_full", copy);
  }
  slot.in_flight = true;
  const cudaError_t recorded = cudaEventRecord(slot.completion, stream);
  if (recorded != cudaSuccess) {
    const Status synchronized = context.Synchronize();
    if (synchronized.ok()) {
      slot.in_flight = false;
    }
    return CudaRuntimeStatus("cudaEventRecord linkage registration", recorded);
  }
  const LinkageBackendCluster token = next_cluster++;
  const std::uint64_t capacity = rfull.capacity_bytes;
  clusters.emplace(token, Cluster{
                              .dimension = registration.dimension,
                              .columns = registration.column_count,
                              .generation = 0U,
                              .rfull = std::move(rfull),
                          });
  live_registration_bytes += capacity;
  return token;
}

Status CudaLinkageRegistrationStore::ApplyUpdates(
    ArrayView<const LinkageBackendUpdate> updates) {
  if (updates.empty()) {
    return Status::Ok();
  }
  UpdateSlot& slot = update_slots[next_update_slot];
  next_update_slot = (next_update_slot + 1U) % update_slots.size();
  if (slot.in_flight) {
    const cudaError_t wait = cudaEventSynchronize(slot.completion);
    if (wait != cudaSuccess) {
      return CudaRuntimeStatus("cudaEventSynchronize linkage update", wait);
    }
    slot.in_flight = false;
  }

  std::uint64_t value_count = 0U;
  std::uint64_t column_count = 0U;
  std::unordered_map<LinkageBackendCluster, std::uint64_t> generations;
  for (const LinkageBackendUpdate& update : updates) {
    auto found = clusters.find(update.cluster);
    if (found == clusters.end() ||
        update.dimension != found->second.dimension || update.columns.empty() ||
        update.reconstructions.size() !=
            static_cast<std::size_t>(update.dimension) *
                update.columns.size()) {
      return Status::FailedPrecondition(
          "CUDA linkage update references an invalid registration");
    }
    const auto generation = generations.find(update.cluster);
    const std::uint64_t preceding = generation == generations.end()
                                        ? found->second.generation
                                        : generation->second;
    if (update.generation != preceding + 1U) {
      return Status::FailedPrecondition(
          "CUDA linkage update generation is not consecutive");
    }
    generations[update.cluster] = update.generation;
    for (const std::uint32_t column : update.columns) {
      if (column >= found->second.columns) {
        return Status::InvalidArgument(
            "CUDA linkage update column is outside the registration");
      }
    }
    value_count += update.reconstructions.size();
    column_count += update.columns.size();
  }
  if (value_count > std::numeric_limits<std::size_t>::max() ||
      column_count > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "CUDA linkage update batch exceeds host addressability");
  }
  const std::size_t values_size = static_cast<std::size_t>(value_count);
  const std::size_t columns_size = static_cast<std::size_t>(column_count);
  auto update_layout = MakeUpdateInputLayout(columns_size, values_size);
  if (!update_layout.ok()) {
    return update_layout.status();
  }
  const UpdateInputLayout& layout = update_layout.value();
  const std::array<Status, 2> ensured = {
      slot.host_inputs.Ensure(layout.total_bytes),
      slot.device_inputs.Ensure(layout.total_bytes),
  };
  for (const Status& status : ensured) {
    if (!status.ok()) {
      return status;
    }
  }
  std::uint8_t* host_inputs = slot.host_inputs.data();
  auto* host_columns =
      reinterpret_cast<DeviceUpdateColumn*>(host_inputs + layout.columns);
  float* host_values = reinterpret_cast<float*>(host_inputs + layout.values);
  std::size_t value_cursor = 0U;
  std::size_t column_cursor = 0U;
  for (const LinkageBackendUpdate& update : updates) {
    Cluster& cluster = clusters.at(update.cluster);
    for (std::size_t index = 0U; index < update.columns.size(); ++index) {
      const std::uint32_t column = update.columns[index];
      host_columns[column_cursor++] = DeviceUpdateColumn{
          .destination = cluster.rfull.values +
                         static_cast<std::size_t>(column) * cluster.dimension,
          .source_offset = value_cursor,
          .dimension = cluster.dimension,
      };
      const float* source =
          update.reconstructions.data() + index * cluster.dimension;
      std::copy_n(source, cluster.dimension, host_values + value_cursor);
      value_cursor += cluster.dimension;
    }
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context.native_stream());
  std::uint8_t* device_inputs = slot.device_inputs.data();
  auto* device_columns =
      reinterpret_cast<DeviceUpdateColumn*>(device_inputs + layout.columns);
  float* device_values =
      reinterpret_cast<float*>(device_inputs + layout.values);
  cudaError_t result =
      cudaMemcpyAsync(device_inputs, host_inputs, layout.total_bytes,
                      cudaMemcpyHostToDevice, stream);
  if (result != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpyAsync linkage update inputs", result);
  }
  ScatterUpdateColumnsKernel<<<static_cast<std::uint32_t>(columns_size), 256U,
                               0U, stream>>>(
      device_columns, device_values, static_cast<std::uint32_t>(columns_size));
  result = cudaPeekAtLastError();
  if (result != cudaSuccess) {
    return CudaRuntimeStatus("ScatterUpdateColumnsKernel", result);
  }
  result = cudaEventRecord(slot.completion, stream);
  if (result != cudaSuccess) {
    return CudaRuntimeStatus("cudaEventRecord linkage update", result);
  }
  slot.in_flight = true;
  for (const auto& [cluster, generation] : generations) {
    clusters.at(cluster).generation = generation;
  }
  return Status::Ok();
}

Status CudaLinkageRegistrationStore::Unregister(LinkageBackendCluster cluster) {
  auto found = clusters.find(cluster);
  if (found == clusters.end()) {
    return Status::FailedPrecondition(
        "CUDA linkage unregister references an unknown cluster");
  }
  live_registration_bytes -= found->second.rfull.capacity_bytes;
  rfull_cache.Release(std::move(found->second.rfull));
  clusters.erase(found);
  return Status::Ok();
}

LinkageBackendResourceStats CudaLinkageRegistrationStore::ResourceStats()
    const noexcept {
  const BestFitCacheStats cache = rfull_cache.Stats();
  return LinkageBackendResourceStats{
      .live_registration_bytes = live_registration_bytes,
      .idle_cache_buffers = cache.idle_buffers,
      .idle_cache_bytes = cache.idle_bytes,
      .idle_cache_peak_bytes = cache.peak_idle_bytes,
      .idle_cache_evictions = cache.evictions,
      .allocation_retries = cache.allocation_retries,
  };
}

}  // namespace

namespace {

struct EvaluationBatch final {
  std::uint32_t dimension = 0U;
  std::uint32_t nodes = 0U;
  std::uint32_t pairs = 0U;
  EvaluationInputLayout layout;
};

// Actual-width batch storage and transport. Only the numeric plan belongs to
// the typed worker. Buffers keep the same high-water capacities and ownership.
struct CudaLinkageWorkerStorage {
  CudaLinkageWorkerStorage(int device_in, std::uint32_t m_in,
                          std::uint32_t dimension_in,
                          std::shared_ptr<CudaContext> context_in,
                          BestFitCacheLimits cache_limits,
                          LinkageEvaluationKind kind_in)
      : device(device_in),
        m(m_in),
        dimension(dimension_in),
        context(std::move(context_in)),
        registrations(*context, cache_limits),
        kind(kind_in) {}

  [[nodiscard]] StatusOr<std::uint32_t> PrepareReserve(
      CudaLinkageWorkerWorkspaceLimits limits);
  [[nodiscard]] Status ReserveTransport(
      CudaLinkageWorkerWorkspaceLimits limits, std::uint32_t block_samples);
  [[nodiscard]] StatusOr<EvaluationBatch> PrepareEvaluation(
      ArrayView<const LinkageBackendEvaluation> tasks);
  [[nodiscard]] Status EnsureBlockStorage(std::uint32_t count,
                                         std::uint32_t evaluation_dimension);
  [[nodiscard]] Status EnsureStableRsdStorage(
      std::uint32_t total_pairs, std::uint32_t evaluation_dimension);
  [[nodiscard]] Status ReadbackWinners(std::uint32_t total_nodes);
  [[nodiscard]] StatusOr<std::vector<LinkageEvaluationResult>>
  DecodeEvaluation();
  [[nodiscard]] Status BeginCompletion(
      LinkageBackendEvaluationCompletion completion);

  int device = 0;
  std::uint32_t m = 0U;
  std::uint32_t dimension = 0U;
  std::shared_ptr<CudaContext> context;
  std::uint32_t internal_chunk_max_pairs = 0U;
  CudaLinkageRegistrationStore registrations;
  LinkageEvaluationKind kind;

  // Borrowed from host_evaluation_inputs until the existing completion
  // boundary. A worker cannot reuse this arena before winner decoding ends.
  ArrayView<const std::uint32_t> host_row_offsets;
  std::vector<std::uint32_t> host_task_node_offsets;
  std::vector<std::uint32_t> host_task_pair_offsets;

  CudaPinnedBuffer<std::uint8_t> host_evaluation_inputs;
  CudaDeviceBuffer<std::uint8_t> device_evaluation_inputs;

  CudaDeviceBuffer<float> residual_block;
  CudaDeviceBuffer<float> residual_squared_norm_block;
  CudaDeviceBuffer<std::uint8_t> pair_order_block;
  CudaDeviceBuffer<std::uint8_t> pair_code_block;
  CudaDeviceBuffer<std::uint32_t> pair_first_code_block;
  CudaDeviceBuffer<std::uint8_t> pair_trailing_code_block;
  CudaDeviceBuffer<float> pair_coefficient_block;
  CudaDeviceBuffer<float> pair_error_block;

  CudaDeviceBuffer<std::uint32_t> best_parents;
  CudaDeviceBuffer<std::uint32_t> best_pair_indices;
  CudaDeviceBuffer<std::uint32_t> best_first_codes;
  CudaDeviceBuffer<std::uint8_t> best_trailing_codes;
  CudaDeviceBuffer<float> best_coefficients;
  CudaDeviceBuffer<float> best_errors;
  CudaDeviceBuffer<std::uint8_t> device_winner_pack;
  CudaPinnedBuffer<std::uint8_t> host_winner_pack;
};

}  // namespace

template <int M, LinkageEvaluationKind Kind>
struct CudaTypedLinkageWorkerBackend<M, Kind>::State final
    : CudaLinkageWorkerStorage {
  using DevicePlan = std::conditional_t<Kind == LinkageEvaluationKind::kEdgeRsd,
                                        CudaLinkageRsdDevicePlan<M>,
                                        CudaRootOnlyLinkageDevicePlan<M>>;

  State(int device_in, std::uint32_t m_in, std::uint32_t dimension_in,
        std::shared_ptr<CudaContext> context_in, DevicePlan plan_in,
        BestFitCacheLimits cache_limits)
      : CudaLinkageWorkerStorage(device_in, m_in, dimension_in,
                                 std::move(context_in), cache_limits, Kind),
        plan(std::move(plan_in)) {}

  // Synchronize while both the numeric plan and its batch storage are alive.
  ~State() {
    cudaSetDevice(device);
    (void)context->Synchronize();
  }

  DevicePlan plan;
};

template <int M, LinkageEvaluationKind Kind>
CudaTypedLinkageWorkerBackend<M, Kind>::CudaTypedLinkageWorkerBackend(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M, LinkageEvaluationKind Kind>
CudaTypedLinkageWorkerBackend<M, Kind>::~CudaTypedLinkageWorkerBackend() =
    default;

template <int M, LinkageEvaluationKind Kind>
StatusOr<std::unique_ptr<CudaTypedLinkageWorkerBackend<M, Kind>>>
CudaTypedLinkageWorkerBackend<M, Kind>::CreateRsd(
    const LinkageCodebookAlgebra<M>& edge_algebra, int device, bool allow_tf32,
    BestFitCacheLimits rfull_cache_limits, std::uint32_t maximum_rank,
    LeastSquaresCostMode cost_mode, std::uint32_t icm_rounds) {
  if constexpr (Kind != LinkageEvaluationKind::kEdgeRsd) {
    return Status::InvalidArgument(
        "root-only CUDA worker cannot own a rsd plan");
  } else {
    auto model = CudaLinkageRsdDeviceModel<M>::Create(edge_algebra, device);
    if (!model.ok()) return model.status();
    return CreateRsd(std::move(model).value(), device, allow_tf32,
                     rfull_cache_limits, maximum_rank, cost_mode, icm_rounds);
  }
}

template <int M, LinkageEvaluationKind Kind>
StatusOr<std::unique_ptr<CudaTypedLinkageWorkerBackend<M, Kind>>>
CudaTypedLinkageWorkerBackend<M, Kind>::CreateRsd(
    std::shared_ptr<const CudaLinkageRsdDeviceModel<M>> edge_model, int device,
    bool allow_tf32, BestFitCacheLimits rfull_cache_limits,
    std::uint32_t maximum_rank, LeastSquaresCostMode cost_mode,
    std::uint32_t icm_rounds) {
  if constexpr (Kind != LinkageEvaluationKind::kEdgeRsd) {
    return Status::InvalidArgument(
        "root-only CUDA worker cannot own a rsd plan");
  } else {
    if (edge_model == nullptr || edge_model->device() != device) {
      return Status::InvalidArgument(
          "rsd CUDA worker received an incompatible shared model");
    }
    auto context_result = CudaContext::Create(device, allow_tf32);
    if (!context_result.ok()) {
      return context_result.status();
    }
    auto context =
        std::make_unique<CudaContext>(std::move(context_result).value());
    return CreateRsdOnContext(
        std::move(edge_model), std::shared_ptr<CudaContext>(std::move(context)),
        rfull_cache_limits, maximum_rank, cost_mode, icm_rounds);
  }
}

template <int M, LinkageEvaluationKind Kind>
StatusOr<std::unique_ptr<CudaTypedLinkageWorkerBackend<M, Kind>>>
CudaTypedLinkageWorkerBackend<M, Kind>::CreateRsdOnContext(
    const LinkageCodebookAlgebra<M>& edge_algebra,
    std::shared_ptr<CudaContext> context, BestFitCacheLimits rfull_cache_limits,
    std::uint32_t maximum_rank, LeastSquaresCostMode cost_mode,
    std::uint32_t icm_rounds) {
  if constexpr (Kind != LinkageEvaluationKind::kEdgeRsd) {
    return Status::InvalidArgument(
        "root-only CUDA worker cannot own a rsd plan");
  } else {
    if (context == nullptr) {
      return Status::InvalidArgument(
          "rsd CUDA worker requires an existing context");
    }
    auto model =
        CudaLinkageRsdDeviceModel<M>::Create(edge_algebra, context->device());
    if (!model.ok()) return model.status();
    return CreateRsdOnContext(std::move(model).value(), std::move(context),
                              rfull_cache_limits, maximum_rank, cost_mode,
                              icm_rounds);
  }
}

template <int M, LinkageEvaluationKind Kind>
StatusOr<std::unique_ptr<CudaTypedLinkageWorkerBackend<M, Kind>>>
CudaTypedLinkageWorkerBackend<M, Kind>::CreateRsdOnContext(
    std::shared_ptr<const CudaLinkageRsdDeviceModel<M>> edge_model,
    std::shared_ptr<CudaContext> context, BestFitCacheLimits rfull_cache_limits,
    std::uint32_t maximum_rank, LeastSquaresCostMode cost_mode,
    std::uint32_t icm_rounds) {
  if constexpr (Kind != LinkageEvaluationKind::kEdgeRsd) {
    return Status::InvalidArgument(
        "root-only CUDA worker cannot own a rsd plan");
  } else {
    if (context == nullptr) {
      return Status::InvalidArgument(
          "rsd CUDA worker requires an existing context");
    }
    if (edge_model == nullptr || edge_model->device() != context->device()) {
      return Status::InvalidArgument(
          "rsd CUDA worker context does not match shared model device");
    }
    const std::uint32_t m = edge_model->shape().m;
    const std::uint32_t dimension = edge_model->shape().dimension;
    auto rsd_result = CudaLinkageRsdDevicePlan<M>::Create(
        std::move(edge_model), *context, maximum_rank, cost_mode, icm_rounds);
    if (!rsd_result.ok()) {
      return rsd_result.status();
    }
    auto state = std::make_unique<State>(
        context->device(), m, dimension, std::move(context),
        std::move(rsd_result).value(), rfull_cache_limits);
    if (const Status status = state->registrations.InitializeEvents();
        !status.ok())
      return status;
    return std::unique_ptr<CudaTypedLinkageWorkerBackend<M, Kind>>(
        new CudaTypedLinkageWorkerBackend<M, Kind>(std::move(state)));
  }
}

template <int M, LinkageEvaluationKind Kind>
StatusOr<std::unique_ptr<CudaTypedLinkageWorkerBackend<M, Kind>>>
CudaTypedLinkageWorkerBackend<M, Kind>::CreateRootOnly(
    std::shared_ptr<const CudaRootOnlyLinkageModel<M>> root_model, int device,
    bool allow_tf32, BestFitCacheLimits rfull_cache_limits,
    std::uint32_t icm_rounds) {
  if constexpr (Kind != LinkageEvaluationKind::kRootOnlyInitialization) {
    return Status::InvalidArgument(
        "rsd CUDA worker cannot own a root-only plan");
  } else {
    if (root_model == nullptr) {
      return Status::InvalidArgument(
          "CUDA root-only linkage worker requires a model");
    }
    const std::uint32_t m = root_model->m();
    const std::uint32_t dimension = root_model->dimension();
    auto context_result = CudaContext::Create(device, allow_tf32);
    if (!context_result.ok()) {
      return context_result.status();
    }
    auto context =
        std::make_unique<CudaContext>(std::move(context_result).value());
    auto plan_result = CudaRootOnlyLinkageDevicePlan<M>::Create(
        std::move(root_model), *context, icm_rounds);
    if (!plan_result.ok()) {
      return plan_result.status();
    }
    auto state = std::make_unique<State>(
        device, m, dimension, std::move(context),
        std::move(plan_result).value(), rfull_cache_limits);
    if (const Status status = state->registrations.InitializeEvents();
        !status.ok())
      return status;
    return std::unique_ptr<CudaTypedLinkageWorkerBackend<M, Kind>>(
        new CudaTypedLinkageWorkerBackend<M, Kind>(std::move(state)));
  }
}

namespace {
StatusOr<std::uint32_t> CudaLinkageWorkerStorage::PrepareReserve(
    CudaLinkageWorkerWorkspaceLimits limits) {
  if (limits.maximum_batch_tasks == 0U || limits.maximum_batch_nodes == 0U ||
      limits.maximum_batch_pairs == 0U || limits.maximum_update_columns == 0U) {
    return Status::InvalidArgument(
        "CUDA linkage worker workspace limits must be nonzero");
  }
  if (const cudaError_t result = cudaSetDevice(device);
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", result);
  }
  internal_chunk_max_pairs = limits.internal_chunk_max_pairs;
  const std::uint32_t block_samples = [&]() noexcept {
    if (kind == LinkageEvaluationKind::kEdgeRsd) {
      return limits.internal_chunk_max_pairs == 0U
                 ? limits.maximum_batch_pairs
                 : std::min(limits.maximum_batch_pairs,
                            limits.internal_chunk_max_pairs);
    }
    const std::uint32_t root_block =
        std::min(CudaRootOnlyLinkageDevicePlan<0>::kDefaultBlockSamples,
                 limits.maximum_batch_pairs);
    return limits.internal_chunk_max_pairs == 0U
               ? root_block
               : std::min(root_block, limits.internal_chunk_max_pairs);
  }();
  return block_samples;
}

Status CudaLinkageWorkerStorage::ReserveTransport(
    CudaLinkageWorkerWorkspaceLimits limits, std::uint32_t block_samples) {
  const std::size_t nodes = limits.maximum_batch_nodes;
  const std::size_t pairs = limits.maximum_batch_pairs;
  const std::size_t block = block_samples;
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  if (nodes > maximum / dimension || nodes == maximum ||
      (kind == LinkageEvaluationKind::kEdgeRsd && nodes > maximum / m)) {
    return Status::ResourceExhausted(
        "CUDA linkage evaluation reserve extent overflows");
  }
  auto input_layout = MakeEvaluationInputLayout(
      limits.maximum_batch_tasks, nodes * dimension, pairs, nodes, nodes + 1U,
      kind == LinkageEvaluationKind::kEdgeRsd ? nodes * m : 0U);
  if (!input_layout.ok()) {
    return input_layout.status();
  }
  const std::array<Status, 13> common_storage = {
      host_evaluation_inputs.Ensure(input_layout.value().total_bytes),
      device_evaluation_inputs.Ensure(input_layout.value().total_bytes),
      residual_block.Ensure(block * dimension),
      pair_coefficient_block.Ensure(block * m),
      pair_error_block.Ensure(block),
      best_parents.Ensure(nodes),
      best_pair_indices.Ensure(nodes),
      best_first_codes.Ensure(nodes),
      best_trailing_codes.Ensure(nodes * (m - 1U)),
      best_coefficients.Ensure(nodes * m),
      best_errors.Ensure(nodes),
      device_winner_pack.Ensure(nodes * WinnerStride(m)),
      host_winner_pack.Ensure(nodes * WinnerStride(m)),
  };
  for (const Status& status : common_storage) {
    if (!status.ok()) {
      return status;
    }
  }
  if (kind == LinkageEvaluationKind::kEdgeRsd) {
    const std::array<Status, 3> rsd_storage = {
        residual_squared_norm_block.Ensure(block),
        pair_order_block.Ensure(block * m),
        pair_code_block.Ensure(block * m),
    };
    for (const Status& status : rsd_storage) {
      if (!status.ok()) {
        return status;
      }
    }
  } else {
    const std::array<Status, 2> root_storage = {
        pair_first_code_block.Ensure(block),
        pair_trailing_code_block.Ensure(block * (m - 1U)),
    };
    for (const Status& status : root_storage) {
      if (!status.ok()) {
        return status;
      }
    }
  }
  auto update_layout = MakeUpdateInputLayout(
      limits.maximum_update_columns,
      static_cast<std::size_t>(limits.maximum_update_columns) *
          dimension);
  if (!update_layout.ok()) {
    return update_layout.status();
  }
  for (auto& slot : registrations.update_slots) {
    const std::array<Status, 2> update_storage = {
        slot.host_inputs.Ensure(update_layout.value().total_bytes),
        slot.device_inputs.Ensure(update_layout.value().total_bytes),
    };
    for (const Status& status : update_storage) {
      if (!status.ok()) {
        return status;
      }
    }
  }
  return Status::Ok();
}

}  // namespace

template <int M, LinkageEvaluationKind Kind>
Status CudaTypedLinkageWorkerBackend<M, Kind>::ReserveWorkspace(
    CudaLinkageWorkerWorkspaceLimits limits) {
  auto block_samples = state_->PrepareReserve(limits);
  if (!block_samples.ok()) return block_samples.status();
  if (const Status status = state_->plan.ReserveWorkspace(block_samples.value());
      !status.ok()) {
    return status;
  }
  return state_->ReserveTransport(limits, block_samples.value());
}

template <int M, LinkageEvaluationKind Kind>
StatusOr<LinkageBackendCluster>
CudaTypedLinkageWorkerBackend<M, Kind>::Register(
    LinkageBackendRegistration registration) {
  return state_->registrations.Register(std::move(registration));
}

template <int M, LinkageEvaluationKind Kind>
Status CudaTypedLinkageWorkerBackend<M, Kind>::ApplyUpdates(
    ArrayView<const LinkageBackendUpdate> updates) {
  return state_->registrations.ApplyUpdates(updates);
}

namespace {
StatusOr<EvaluationBatch> CudaLinkageWorkerStorage::PrepareEvaluation(
    ArrayView<const LinkageBackendEvaluation> tasks) {
  if (tasks.empty()) {
    return Status::InvalidArgument(
        "CUDA linkage evaluation batch must not be empty");
  }
  const std::uint32_t evaluation_dimension = tasks[0].task.dimension;
  std::uint64_t total_nodes64 = 0U;
  std::uint64_t total_pairs64 = 0U;
  for (const LinkageBackendEvaluation& evaluation : tasks) {
    auto found = registrations.clusters.find(evaluation.cluster);
    if (found == registrations.clusters.end() ||
        evaluation.task.kind != kind || evaluation.task.m != m ||
        evaluation.task.dimension != evaluation_dimension ||
        found->second.dimension != evaluation_dimension ||
        found->second.generation < evaluation.task.required_generation) {
      return Status::FailedPrecondition(
          "CUDA linkage evaluation references stale or incompatible state");
    }
    total_nodes64 += evaluation.task.node_count();
    total_pairs64 += evaluation.task.pair_count();
  }
  if (total_nodes64 > std::numeric_limits<std::uint32_t>::max() ||
      total_pairs64 > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "CUDA linkage evaluation batch exceeds uint32 indexing");
  }
  const std::uint32_t total_nodes = static_cast<std::uint32_t>(total_nodes64);
  const std::uint32_t total_pairs = static_cast<std::uint32_t>(total_pairs64);

  auto input_layout = MakeEvaluationInputLayout(
      tasks.size(), static_cast<std::size_t>(total_nodes) * evaluation_dimension,
      total_pairs, total_nodes, static_cast<std::size_t>(total_nodes) + 1U,
      kind == LinkageEvaluationKind::kEdgeRsd
          ? static_cast<std::size_t>(total_nodes) * m : 0U);
  if (!input_layout.ok()) {
    return input_layout.status();
  }
  const EvaluationInputLayout& layout = input_layout.value();
  const std::array<Status, 2> input_storage = {
      host_evaluation_inputs.Ensure(layout.total_bytes),
      device_evaluation_inputs.Ensure(layout.total_bytes),
  };
  for (const Status& status : input_storage) {
    if (!status.ok()) {
      return status;
    }
  }
  std::uint8_t* host_inputs = host_evaluation_inputs.data();
  auto* nodes = reinterpret_cast<float*>(host_inputs + layout.nodes);
  auto* candidates =
      reinterpret_cast<std::uint32_t*>(host_inputs + layout.candidates);
  auto* pair_nodes =
      reinterpret_cast<std::uint32_t*>(host_inputs + layout.pair_nodes);
  auto* node_tasks =
      reinterpret_cast<std::uint32_t*>(host_inputs + layout.node_tasks);
  auto* row_offsets =
      reinterpret_cast<std::uint32_t*>(host_inputs + layout.row_offsets);
  auto* node_orders = host_inputs + layout.node_orders;
  auto* task_rfull = reinterpret_cast<const float**>(
      host_inputs + layout.task_rfull);

  host_task_node_offsets.assign(1U, 0U);
  host_task_pair_offsets.assign(1U, 0U);
  host_task_node_offsets.reserve(tasks.size() + 1U);
  host_task_pair_offsets.reserve(tasks.size() + 1U);
  row_offsets[0] = 0U;
  std::uint32_t node_cursor = 0U;
  std::uint32_t pair_cursor = 0U;
  for (std::uint32_t task_index = 0U;
       task_index < static_cast<std::uint32_t>(tasks.size()); ++task_index) {
    const LinkageBackendEvaluation& evaluation = tasks[task_index];
    const LinkageEvaluationTask& task = evaluation.task;
    std::copy(task.node_vectors.begin(), task.node_vectors.end(),
              nodes + static_cast<std::size_t>(node_cursor) * evaluation_dimension);
    std::copy(task.candidate_columns.begin(), task.candidate_columns.end(),
              candidates + pair_cursor);
    if (kind == LinkageEvaluationKind::kEdgeRsd) {
      std::copy(task.icm_layer_orders.begin(), task.icm_layer_orders.end(),
                node_orders + static_cast<std::size_t>(node_cursor) * m);
    }
    task_rfull[task_index] =
        registrations.clusters.at(evaluation.cluster).rfull.values;
    std::fill_n(node_tasks + node_cursor, task.node_count(), task_index);
    for (std::uint32_t node = 0U; node < task.node_count(); ++node) {
      const std::uint32_t begin = task.row_offsets[node];
      const std::uint32_t end = task.row_offsets[node + 1U];
      std::fill(pair_nodes + pair_cursor + begin, pair_nodes + pair_cursor + end,
                node_cursor + node);
      row_offsets[node_cursor + node + 1U] = pair_cursor + end;
    }
    node_cursor += task.node_count();
    pair_cursor += task.pair_count();
    host_task_node_offsets.push_back(node_cursor);
    host_task_pair_offsets.push_back(pair_cursor);
  }
  assert(node_cursor == total_nodes && pair_cursor == total_pairs);
  host_row_offsets = {row_offsets, static_cast<std::size_t>(total_nodes) + 1U};

  const cudaStream_t stream =
      static_cast<cudaStream_t>(context->native_stream());
  std::uint8_t* device_inputs = device_evaluation_inputs.data();
  const cudaError_t input_copy =
      cudaMemcpyAsync(device_inputs, host_inputs, layout.total_bytes,
                      cudaMemcpyHostToDevice, stream);
  if (input_copy != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpyAsync linkage evaluation inputs",
                             input_copy);
  }
  const std::array<Status, 8> best_storage = {
      best_parents.Ensure(total_nodes),
      best_pair_indices.Ensure(total_nodes),
      best_first_codes.Ensure(total_nodes),
      best_trailing_codes.Ensure(static_cast<std::size_t>(total_nodes) * (m - 1U)),
      best_coefficients.Ensure(static_cast<std::size_t>(total_nodes) * m),
      best_errors.Ensure(total_nodes),
      device_winner_pack.Ensure(static_cast<std::size_t>(total_nodes) *
                               WinnerStride(m)),
      host_winner_pack.Ensure(static_cast<std::size_t>(total_nodes) *
                             WinnerStride(m)),
  };
  for (const Status& status : best_storage) {
    if (!status.ok()) {
      return status;
    }
  }
  constexpr std::uint32_t kThreads = 128U;
  const std::uint32_t node_grid = (total_nodes + kThreads - 1U) / kThreads;
  InitializeBestKernel<<<node_grid, kThreads, 0U, stream>>>(
      total_nodes, best_errors.data(), best_pair_indices.data(),
      best_parents.data());
  if (cudaError_t result = cudaPeekAtLastError(); result != cudaSuccess) {
    return CudaRuntimeStatus("InitializeBestKernel", result);
  }
  return EvaluationBatch{evaluation_dimension, total_nodes, total_pairs, layout};
}

Status CudaLinkageWorkerStorage::EnsureBlockStorage(std::uint32_t count,
                                   std::uint32_t evaluation_dimension) {
  const std::array<Status, 3> common_block_storage = {
      residual_block.Ensure(static_cast<std::size_t>(count) * evaluation_dimension),
      pair_coefficient_block.Ensure(static_cast<std::size_t>(count) * m),
      pair_error_block.Ensure(count),
  };
  for (const Status& status : common_block_storage) {
    if (!status.ok()) {
      return status;
    }
  }
  if (kind == LinkageEvaluationKind::kEdgeRsd) {
    const std::array<Status, 3> edge_storage = {
        residual_squared_norm_block.Ensure(count),
        pair_order_block.Ensure(static_cast<std::size_t>(count) * m),
        pair_code_block.Ensure(static_cast<std::size_t>(count) * m),
    };
    for (const Status& status : edge_storage) {
      if (!status.ok()) {
        return status;
      }
    }
  } else {
    const std::array<Status, 2> root_storage = {
        pair_first_code_block.Ensure(count),
        pair_trailing_code_block.Ensure(
            static_cast<std::size_t>(count) * (m - 1U)),
    };
    for (const Status& status : root_storage) {
      if (!status.ok()) {
        return status;
      }
    }
  }
  return Status::Ok();
}

Status CudaLinkageWorkerStorage::EnsureStableRsdStorage(
    std::uint32_t total_pairs, std::uint32_t evaluation_dimension) {
  const std::array<Status, 6> transport_storage = {
      residual_block.Ensure(static_cast<std::size_t>(total_pairs) *
                            evaluation_dimension),
      residual_squared_norm_block.Ensure(total_pairs),
      pair_order_block.Ensure(static_cast<std::size_t>(total_pairs) * m),
      pair_code_block.Ensure(static_cast<std::size_t>(total_pairs) * m),
      pair_coefficient_block.Ensure(static_cast<std::size_t>(total_pairs) * m),
      pair_error_block.Ensure(total_pairs),
  };
  for (const Status& status : transport_storage) {
    if (!status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status CudaLinkageWorkerStorage::ReadbackWinners(std::uint32_t total_nodes) {
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context->native_stream());
  if (const cudaError_t copy = cudaMemcpyAsync(
          host_winner_pack.data(), device_winner_pack.data(),
          static_cast<std::size_t>(total_nodes) * WinnerStride(m),
          cudaMemcpyDeviceToHost, stream);
      copy != cudaSuccess) {
    return CudaRuntimeStatus("cudaMemcpyAsync packed linkage winners", copy);
  }
  return Status::Ok();
}
}  // namespace

template <int M, LinkageEvaluationKind Kind>
Status CudaTypedLinkageWorkerBackend<M, Kind>::EnqueueEvaluation(
    ArrayView<const LinkageBackendEvaluation> tasks, bool stable) {
  auto prepared = state_->PrepareEvaluation(tasks);
  if (!prepared.ok()) return prepared.status();
  const EvaluationBatch& batch = prepared.value();
  const std::uint32_t dimension = batch.dimension;
  const std::uint32_t total_nodes = batch.nodes;
  const std::uint32_t total_pairs = batch.pairs;
  const EvaluationInputLayout& layout = batch.layout;
  std::uint8_t* device_inputs = state_->device_evaluation_inputs.data();
  const float* const* device_task_rfull =
      reinterpret_cast<const float* const*>(device_inputs + layout.task_rfull);
  const float* device_nodes =
      reinterpret_cast<const float*>(device_inputs + layout.nodes);
  const auto* device_candidates =
      reinterpret_cast<const std::uint32_t*>(device_inputs + layout.candidates);
  const auto* device_pair_nodes =
      reinterpret_cast<const std::uint32_t*>(device_inputs + layout.pair_nodes);
  const auto* device_node_tasks =
      reinterpret_cast<const std::uint32_t*>(device_inputs + layout.node_tasks);
  const auto* device_row_offsets = reinterpret_cast<const std::uint32_t*>(
      device_inputs + layout.row_offsets);
  const auto* device_node_orders =
      reinterpret_cast<const std::uint8_t*>(device_inputs + layout.node_orders);

  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  constexpr std::uint32_t kThreads = 128U;
  const std::uint32_t node_grid = (total_nodes + kThreads - 1U) / kThreads;
  constexpr bool kRsd = Kind == LinkageEvaluationKind::kEdgeRsd;
  const bool kStableRsd = kRsd && stable;
  const bool chunked_stable_rsd =
      kStableRsd && state_->internal_chunk_max_pairs != 0U &&
      state_->internal_chunk_max_pairs < total_pairs;
  if (kStableRsd) {
    if (!chunked_stable_rsd) {
      if (const Status status =
              state_->EnsureStableRsdStorage(total_pairs, dimension);
          !status.ok()) {
        return status;
      }
      GatherResidualAndNormBlockKernel<<<total_pairs, 256U, 0U, stream>>>(
          device_nodes, dimension, device_candidates, device_pair_nodes,
          device_node_tasks, device_task_rfull, 0U, total_pairs,
          state_->residual_block.data(),
          state_->residual_squared_norm_block.data());
      if (cudaError_t result = cudaPeekAtLastError(); result != cudaSuccess) {
        return CudaRuntimeStatus("GatherResidualAndNormBlockKernel", result);
      }
      const std::uint32_t order_extent = total_pairs * state_->m;
      const std::uint32_t order_grid =
          (order_extent + kThreads - 1U) / kThreads;
      ExpandIcmOrdersKernel<M><<<order_grid, kThreads, 0U, stream>>>(
          state_->m, device_node_orders, device_pair_nodes, 0U, total_pairs,
          state_->pair_order_block.data());
      if (cudaError_t result = cudaPeekAtLastError(); result != cudaSuccess) {
        return CudaRuntimeStatus("ExpandIcmOrdersKernel", result);
      }
      // The failure word is transport-owned here. Every source task remains a
      // separate numerical batch, but no later task may erase a failure from an
      // earlier task in the same transport.
      if constexpr (Kind == LinkageEvaluationKind::kEdgeRsd) {
        const Status begin_status =
            state_->plan.BeginProjectedBatch(total_pairs);
        if (!begin_status.ok()) {
          return begin_status;
        }
        const Status projection_status = state_->plan.EnqueueProjectionSequence(
            {state_->residual_block.data(), dimension, total_pairs, dimension},
            state_->host_task_pair_offsets);
        if (!projection_status.ok()) {
          return projection_status;
        }
      }
    }
  }
  if (!kStableRsd || chunked_stable_rsd) {
    const std::size_t numerical_batch_count = stable ? tasks.size() : 1U;
    for (std::size_t numerical_batch = 0U;
         numerical_batch < numerical_batch_count; ++numerical_batch) {
      const std::uint32_t source_pair_begin =
          stable ? state_->host_task_pair_offsets[numerical_batch] : 0U;
      const std::uint32_t source_pair_end =
          stable ? state_->host_task_pair_offsets[numerical_batch + 1U]
                 : total_pairs;
      const std::uint32_t source_node_begin =
          stable ? state_->host_task_node_offsets[numerical_batch] : 0U;
      const std::uint32_t source_node_end =
          stable ? state_->host_task_node_offsets[numerical_batch + 1U]
                 : total_nodes;
      const std::uint32_t source_pairs = source_pair_end - source_pair_begin;
      const std::uint32_t source_nodes = source_node_end - source_node_begin;
      const Status begin_status = state_->plan.BeginBatch();
      if (!begin_status.ok()) {
        return begin_status;
      }
      const std::uint32_t block_pairs = [&]() noexcept {
        if constexpr (Kind == LinkageEvaluationKind::kRootOnlyInitialization) {
          const std::uint32_t planned =
              state_->plan.sample_chunk_size(source_pairs);
          return state_->internal_chunk_max_pairs == 0U
                     ? planned
                     : std::min(planned, state_->internal_chunk_max_pairs);
        }
        return state_->internal_chunk_max_pairs == 0U
                   ? source_pairs
                   : std::min(source_pairs, state_->internal_chunk_max_pairs);
      }();
      const std::uint32_t source_node_grid =
          (source_nodes + kThreads - 1U) / kThreads;
      for (std::uint32_t source_pair_offset = 0U;
           source_pair_offset < source_pairs;
           source_pair_offset += block_pairs) {
        const std::uint32_t pair_offset =
            source_pair_begin + source_pair_offset;
        const std::uint32_t count =
            std::min(block_pairs, source_pairs - source_pair_offset);
        if (const Status status = state_->EnsureBlockStorage(count, dimension);
            !status.ok()) {
          return status;
        }
        if constexpr (Kind == LinkageEvaluationKind::kEdgeRsd) {
          GatherResidualAndNormBlockKernel<<<count, 256U, 0U, stream>>>(
              device_nodes, dimension, device_candidates, device_pair_nodes,
              device_node_tasks, device_task_rfull, pair_offset, count,
              state_->residual_block.data(),
              state_->residual_squared_norm_block.data());
          if (cudaError_t result = cudaPeekAtLastError();
              result != cudaSuccess) {
            return CudaRuntimeStatus("GatherResidualAndNormBlockKernel",
                                     result);
          }
          const std::uint32_t order_extent = count * state_->m;
          const std::uint32_t order_grid =
              (order_extent + kThreads - 1U) / kThreads;
          ExpandIcmOrdersKernel<M><<<order_grid, kThreads, 0U, stream>>>(
              state_->m, device_node_orders, device_pair_nodes, pair_offset,
              count, state_->pair_order_block.data());
          if (cudaError_t result = cudaPeekAtLastError();
              result != cudaSuccess) {
            return CudaRuntimeStatus("ExpandIcmOrdersKernel", result);
          }
          const CudaMatrixF32View residuals = {state_->residual_block.data(),
                                               dimension, count, dimension};
          if (const Status status = state_->plan.Enqueue(
                  residuals, state_->pair_order_block.data(),
                  {.codes = state_->pair_code_block.data(),
                   .coefficients = state_->pair_coefficient_block.data(),
                   .squared_errors = state_->pair_error_block.data()},
                  state_->residual_squared_norm_block.data());
              !status.ok()) {
            return status;
          }
          SelectRsdBlockKernel<M><<<source_node_grid, kThreads, 0U, stream>>>(
              state_->m, source_nodes, device_row_offsets + source_node_begin,
              device_candidates, pair_offset, count,
              state_->pair_code_block.data(),
              state_->pair_coefficient_block.data(),
              state_->pair_error_block.data(),
              state_->best_parents.data() + source_node_begin,
              state_->best_pair_indices.data() + source_node_begin,
              state_->best_first_codes.data() + source_node_begin,
              state_->best_trailing_codes.data() +
                  static_cast<std::size_t>(source_node_begin) *
                      (state_->m - 1U),
              state_->best_coefficients.data() +
                  static_cast<std::size_t>(source_node_begin) * state_->m,
              state_->best_errors.data() + source_node_begin);
        } else {
          const std::uint64_t residual_extent =
              static_cast<std::uint64_t>(count) * dimension;
          const std::uint32_t residual_grid = static_cast<std::uint32_t>(
              (residual_extent + kThreads - 1U) / kThreads);
          GatherResidualBlockKernel<<<residual_grid, kThreads, 0U, stream>>>(
              device_nodes, dimension, device_candidates, device_pair_nodes,
              device_node_tasks, device_task_rfull, pair_offset, count,
              state_->residual_block.data());
          if (cudaError_t result = cudaPeekAtLastError();
              result != cudaSuccess) {
            return CudaRuntimeStatus("GatherResidualBlockKernel", result);
          }
          if (const Status status = state_->plan.Enqueue(
                  {state_->residual_block.data(), dimension, count, dimension},
                  tasks[stable ? numerical_batch : 0U].task.icm_layer_orders,
                  {.first_codes = state_->pair_first_code_block.data(),
                   .trailing_codes = state_->pair_trailing_code_block.data(),
                   .coefficients = state_->pair_coefficient_block.data(),
                   .squared_errors = state_->pair_error_block.data()});
              !status.ok()) {
            return status;
          }
          SelectRootOnlyBlockKernel<M>
              <<<source_node_grid, kThreads, 0U, stream>>>(
                  state_->m, source_nodes,
                  device_row_offsets + source_node_begin, device_candidates,
                  pair_offset, count, state_->pair_first_code_block.data(),
                  state_->pair_trailing_code_block.data(),
                  state_->pair_coefficient_block.data(),
                  state_->pair_error_block.data(),
                  state_->best_parents.data() + source_node_begin,
                  state_->best_pair_indices.data() + source_node_begin,
                  state_->best_first_codes.data() + source_node_begin,
                  state_->best_trailing_codes.data() +
                      static_cast<std::size_t>(source_node_begin) *
                          (state_->m - 1U),
                  state_->best_coefficients.data() +
                      static_cast<std::size_t>(source_node_begin) * state_->m,
                  state_->best_errors.data() + source_node_begin);
        }
        if (cudaError_t result = cudaPeekAtLastError(); result != cudaSuccess) {
          return CudaRuntimeStatus("SelectRsdBlockKernel", result);
        }
      }
    }
  }
  if (kStableRsd) {
    if (!chunked_stable_rsd) {
      if constexpr (Kind == LinkageEvaluationKind::kEdgeRsd) {
        const CudaLinkageRsdDeviceOutput<M> output = {
            .codes = state_->pair_code_block.data(),
            .coefficients = state_->pair_coefficient_block.data(),
            .squared_errors = state_->pair_error_block.data(),
        };
        const Status status = state_->plan.EnqueueProjectedRsd(
            state_->pair_order_block.data(), output,
            state_->residual_squared_norm_block.data());
        if (!status.ok()) {
          return status;
        }
      }
      SelectRsdBlockKernel<M><<<node_grid, kThreads, 0U, stream>>>(
          state_->m, total_nodes, device_row_offsets, device_candidates, 0U,
          total_pairs, state_->pair_code_block.data(),
          state_->pair_coefficient_block.data(),
          state_->pair_error_block.data(), state_->best_parents.data(),
          state_->best_pair_indices.data(), state_->best_first_codes.data(),
          state_->best_trailing_codes.data(), state_->best_coefficients.data(),
          state_->best_errors.data());
      if (cudaError_t result = cudaPeekAtLastError(); result != cudaSuccess) {
        return CudaRuntimeStatus("SelectRsdBlockKernel", result);
      }
    }
  }

  PackLinkageWinnersKernel<M><<<node_grid, kThreads, 0U, stream>>>(
      state_->m, total_nodes, state_->best_parents.data(),
      state_->best_first_codes.data(), state_->best_trailing_codes.data(),
      state_->best_coefficients.data(), state_->best_errors.data(),
      state_->device_winner_pack.data());
  if (cudaError_t result = cudaPeekAtLastError(); result != cudaSuccess) {
    return CudaRuntimeStatus("PackLinkageWinnersKernel", result);
  }
  return state_->ReadbackWinners(total_nodes);
}

namespace {

// Decode the shared host wire format after a numerical backend completes.
// The layout contains actual M; it is not a fixed-width numerical operation.
StatusOr<std::vector<LinkageEvaluationResult>> DecodeLinkageWinners(
    std::uint32_t m, LinkageEvaluationKind kind,
    ArrayView<const std::uint32_t> task_node_offsets,
    ArrayView<const std::uint32_t> row_offsets,
    const std::uint8_t* winner_pack) {
  const std::size_t task_count =
      task_node_offsets.empty() ? 0U : task_node_offsets.size() - 1U;
  std::vector<LinkageEvaluationResult> results;
  results.reserve(task_count);
  for (std::size_t task_index = 0U; task_index < task_count; ++task_index) {
    const std::uint32_t begin = task_node_offsets[task_index];
    const std::uint32_t end = task_node_offsets[task_index + 1U];
    for (std::uint32_t node = begin; node < end; ++node) {
      const std::uint8_t* packed =
          winner_pack + static_cast<std::size_t>(node) * WinnerStride(m);
      const float error =
          *reinterpret_cast<const float*>(packed + sizeof(std::uint32_t));
      const bool empty_root_only_row =
          kind == LinkageEvaluationKind::kRootOnlyInitialization &&
          row_offsets[node] == row_offsets[node + 1U];
      if (!empty_root_only_row && !std::isfinite(error)) {
        return Status::FailedPrecondition(
            "CUDA linkage evaluation contains a node without a solvable "
            "candidate");
      }
    }
    LinkageEvaluationResult result;
    result.parent_columns.resize(end - begin);
    result.supports.resize(end - begin, m);
    result.coefficients.resize(static_cast<std::size_t>(end - begin) * m);
    result.squared_errors.resize(end - begin);
    for (std::uint32_t node = begin; node < end; ++node) {
      const std::uint8_t* packed =
          winner_pack + static_cast<std::size_t>(node) * WinnerStride(m);
      const std::uint32_t result_node = node - begin;
      result.parent_columns[result_node] =
          *reinterpret_cast<const std::uint32_t*>(packed);
      result.squared_errors[result_node] =
          *reinterpret_cast<const float*>(packed + sizeof(std::uint32_t));
      auto support = result.supports[node - begin];
      support.first = *reinterpret_cast<const std::uint32_t*>(
          packed + sizeof(std::uint32_t) + sizeof(float));
      constexpr std::size_t kTrailingSource =
          sizeof(std::uint32_t) + sizeof(float) + sizeof(std::uint32_t);
      for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(m);
           ++layer) {
        support.trailing[layer - 1U] = packed[kTrailingSource + layer - 1U];
      }
      const float* packed_coefficients =
          reinterpret_cast<const float*>(packed + WinnerCoefficientsOffset(m));
      for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(m);
           ++layer) {
        result.coefficients[static_cast<std::size_t>(result_node) * m + layer] =
            packed_coefficients[layer];
      }
    }
    results.push_back(std::move(result));
  }
  return results;
}

StatusOr<std::vector<LinkageEvaluationResult>>
CudaLinkageWorkerStorage::DecodeEvaluation() {
  return DecodeLinkageWinners(m, kind, host_task_node_offsets,
                             host_row_offsets, host_winner_pack.data());
}

}  // namespace

template <int M, LinkageEvaluationKind Kind>
StatusOr<std::vector<LinkageEvaluationResult>>
CudaTypedLinkageWorkerBackend<M, Kind>::DecodeEvaluation() {
  return state_->DecodeEvaluation();
}

template <int M, LinkageEvaluationKind Kind>
StatusOr<std::vector<LinkageEvaluationResult>>
CudaTypedLinkageWorkerBackend<M, Kind>::Evaluate(
    ArrayView<const LinkageBackendEvaluation> tasks) {
  if (const Status status = EnqueueEvaluation(tasks, false); !status.ok()) {
    return status;
  }
  if (const Status status = state_->context->Synchronize(); !status.ok()) {
    return status;
  }
  return DecodeEvaluation();
}

template <int M, LinkageEvaluationKind Kind>
StatusOr<std::vector<LinkageEvaluationResult>>
CudaTypedLinkageWorkerBackend<M, Kind>::EvaluateStable(
    ArrayView<const LinkageBackendEvaluation> tasks) {
  if (const Status status = EnqueueEvaluation(tasks, true); !status.ok()) {
    return status;
  }
  if (const Status status = state_->context->Synchronize(); !status.ok()) {
    return status;
  }
  return DecodeEvaluation();
}

template <int M, LinkageEvaluationKind Kind>
Status CudaTypedLinkageWorkerBackend<M, Kind>::BeginEvaluate(
    ArrayView<const LinkageBackendEvaluation> tasks,
    LinkageBackendEvaluationCompletion completion) {
  if (!completion) {
    return Status::InvalidArgument(
        "CUDA linkage asynchronous evaluation requires a completion");
  }
  if (const Status status = EnqueueEvaluation(tasks, false); !status.ok()) {
    return status;
  }
  return state_->BeginCompletion(std::move(completion));
}

namespace {
Status CudaLinkageWorkerStorage::BeginCompletion(
    LinkageBackendEvaluationCompletion completion) {
  auto payload = std::unique_ptr<CudaEvaluationCompletionState>(
      new (std::nothrow) CudaEvaluationCompletionState{
          .completion = std::move(completion),
          .decode = [this] { return DecodeEvaluation(); },
      });
  if (payload == nullptr) {
    (void)context->Synchronize();
    return Status::ResourceExhausted(
        "CUDA linkage completion allocation failed");
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context->native_stream());
  // Unlike cudaLaunchHostFunc, the stream callback runs exactly once and
  // reports a persistent stream/context error through its status argument.
  const cudaError_t callback_status = cudaStreamAddCallback(
      stream, &CompleteCudaEvaluation, payload.get(), 0U);
  if (callback_status != cudaSuccess) {
    (void)context->Synchronize();
    return CudaRuntimeStatus(
        "cudaStreamAddCallback linkage evaluation completion", callback_status);
  }
  (void)payload.release();
  return Status::Ok();
}
}  // namespace

template <int M, LinkageEvaluationKind Kind>
Status CudaTypedLinkageWorkerBackend<M, Kind>::Unregister(
    LinkageBackendCluster cluster) {
  return state_->registrations.Unregister(cluster);
}

template <int M, LinkageEvaluationKind Kind>
LinkageBackendResourceStats
CudaTypedLinkageWorkerBackend<M, Kind>::ResourceStats() const noexcept {
  return state_->registrations.ResourceStats();
}

#define STLQPP_INSTANTIATE_CUDA_BACKEND(M)      \
  template class CudaTypedLinkageWorkerBackend< \
      M, LinkageEvaluationKind::kEdgeRsd>;      \
  template class CudaTypedLinkageWorkerBackend< \
      M, LinkageEvaluationKind::kRootOnlyInitialization>;

STLQPP_INSTANTIATE_CUDA_BACKEND(0)
STLQPP_INSTANTIATE_CUDA_BACKEND(5)
STLQPP_INSTANTIATE_CUDA_BACKEND(6)
STLQPP_INSTANTIATE_CUDA_BACKEND(10)
STLQPP_INSTANTIATE_CUDA_BACKEND(12)

#undef STLQPP_INSTANTIATE_CUDA_BACKEND

}  // namespace stlqpp
