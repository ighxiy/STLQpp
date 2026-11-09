#include "stlqpp/training/linkage/replay/cuda_fixed_edge_replay_encoder.h"

#include "cuda_fixed_edge_replay_kernels.h"

#include "stlqpp/runtime/cuda/device_buffer.h"
#include "stlqpp/runtime/cuda/status.h"
#include "stlqpp/training/linkage/rsd/linkage_rsd_device_plan.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

namespace stlqpp {

template <int M>
struct CudaFixedEdgeReplayEncoder<M>::State final {
  State(CudaContext* context_in,
        ModelShape shape_in,
        std::uint32_t rsd_rank_in,
        std::shared_ptr<const CudaLinkageRsdDeviceModel<M>> model_in,
        CudaLinkageRsdEncoder<M> encoder_in) noexcept
      : context(context_in),
        device(context_in->device()),
        shape(shape_in),
        rsd_rank(rsd_rank_in),
        model(std::move(model_in)),
        encoder(std::move(encoder_in)) {}

  ~State() { cudaSetDevice(device); }

  CudaContext* context = nullptr;
  int device = 0;
  ModelShape shape;
  std::uint32_t rsd_rank = 0U;
  std::uint32_t window_columns = 0U;
  std::shared_ptr<const CudaLinkageRsdDeviceModel<M>> model;
  CudaLinkageRsdEncoder<M> encoder;
  CudaDeviceBuffer<float> window_reconstructions;
  CudaDeviceBuffer<float> exact_nodes;
  CudaDeviceBuffer<float> residuals;
  CudaDeviceBuffer<std::uint32_t> parent_columns;
  CudaDeviceBuffer<std::uint32_t> destination_columns;
};

template <int M>
CudaFixedEdgeReplayEncoder<M>::CudaFixedEdgeReplayEncoder(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
CudaFixedEdgeReplayEncoder<M>::~CudaFixedEdgeReplayEncoder() = default;

template <int M>
CudaFixedEdgeReplayEncoder<M>::CudaFixedEdgeReplayEncoder(
    CudaFixedEdgeReplayEncoder&& other) noexcept
    : state_(std::move(other.state_)) {}

template <int M>
CudaFixedEdgeReplayEncoder<M>&
CudaFixedEdgeReplayEncoder<M>::operator=(
    CudaFixedEdgeReplayEncoder&& other) noexcept {
  state_ = std::move(other.state_);
  return *this;
}

template <int M>
StatusOr<CudaFixedEdgeReplayEncoder<M>>
CudaFixedEdgeReplayEncoder<M>::Create(
    const LinkageCodebookAlgebra<M>& algebra,
    CudaContext& context,
    std::uint32_t rsd_rank,
    LeastSquaresCostMode cost_mode,
    std::uint32_t icm_rounds) {
  auto model = CudaLinkageRsdDeviceModel<M>::Create(algebra, context.device());
  if (!model.ok()) return model.status();
  auto encoder = CudaLinkageRsdEncoder<M>::Create(
      model.value(), context, rsd_rank, cost_mode, icm_rounds);
  if (!encoder.ok()) {
    return encoder.status();
  }
  auto state = std::make_unique<State>(
      &context, algebra.shape(), rsd_rank, std::move(model).value(),
      std::move(encoder).value());
  return CudaFixedEdgeReplayEncoder<M>(std::move(state));
}

template <int M>
std::uint32_t CudaFixedEdgeReplayEncoder<M>::rsd_rank() const noexcept {
  return state_ == nullptr ? 0U : state_->rsd_rank;
}

template <int M>
StatusOr<LinkageEncoding<M>> CudaFixedEdgeReplayEncoder<M>::Encode(
    ColumnMajorMatrixF32View residuals,
    ArrayView<const std::uint8_t> icm_layer_orders) const {
  return state_->encoder.Encode(residuals, icm_layer_orders);
}

template <int M>
Status CudaFixedEdgeReplayEncoder<M>::BeginWindow(
    ColumnMajorMatrixF32View roots, ArrayView<const std::uint32_t> root_columns,
    std::uint32_t window_columns) const {
  if (state_ == nullptr || roots.values == nullptr ||
      roots.rows != state_->shape.dimension ||
      roots.columns == 0U || roots.columns != root_columns.size() ||
      window_columns < roots.columns) {
    return Status::InvalidArgument(
        "CUDA fixed-edge replay window has an invalid reconstruction matrix");
  }
  if (const cudaError_t result = cudaSetDevice(state_->device);
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", result);
  }
  const std::size_t values =
      static_cast<std::size_t>(roots.rows) *
      window_columns;
  if (const Status status = state_->window_reconstructions.Ensure(values);
      !status.ok()) {
    return status;
  }
  for (const auto column : root_columns) {
    assert(column < window_columns);
    (void)column;
  }
  const auto root_values = static_cast<std::size_t>(roots.rows) * roots.columns;
  if (const auto status = state_->exact_nodes.Ensure(root_values); !status.ok()) return status;
  if (const auto status = state_->destination_columns.Ensure(root_columns.size()); !status.ok()) return status;
  const auto stream = static_cast<cudaStream_t>(state_->context->native_stream());
  const cudaError_t upload = cudaMemcpyAsync(
      state_->exact_nodes.data(), roots.values,
      root_values * sizeof(float), cudaMemcpyHostToDevice,
      static_cast<cudaStream_t>(state_->context->native_stream()));
  if (upload != cudaSuccess) {
    return CudaRuntimeStatus(
        "cudaMemcpyAsync fixed-edge replay window", upload);
  }
  const auto columns_upload = cudaMemcpyAsync(state_->destination_columns.data(),
      root_columns.data(), root_columns.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream);
  if (columns_upload != cudaSuccess) return CudaRuntimeStatus("upload replay root columns", columns_upload);
  if (const auto status = detail::LaunchScatterReplayRoots(*state_->context,
      state_->exact_nodes.data(), state_->destination_columns.data(), roots.rows,
      roots.columns, state_->window_reconstructions.data()); !status.ok()) return status;
  state_->window_columns = window_columns;
  return Status::Ok();
}

template <int M>
void CudaFixedEdgeReplayEncoder<M>::CancelWindow() const noexcept {
  if (state_ != nullptr) (void)state_->context->Synchronize();
}

template <int M>
StatusOr<LinkageEncoding<M>>
CudaFixedEdgeReplayEncoder<M>::EncodeWindowBatch(
    ColumnMajorMatrixF32View exact_nodes,
    ColumnMajorMatrixF32View parent_reconstructions,
    ArrayView<const std::uint32_t> parent_columns,
    ArrayView<const std::uint32_t> destination_columns,
    ArrayView<const std::uint8_t> icm_layer_orders) const {
  return EncodeWindowBatchImpl(exact_nodes, parent_reconstructions,
                               parent_columns, destination_columns,
                               icm_layer_orders, false);
}

template <int M>
StatusOr<LinkageEncoding<M>>
CudaFixedEdgeReplayEncoder<M>::EncodeWindowBatchValidated(
    ColumnMajorMatrixF32View exact_nodes,
    ColumnMajorMatrixF32View parent_reconstructions,
    ArrayView<const std::uint32_t> parent_columns,
    ArrayView<const std::uint32_t> destination_columns,
    ArrayView<const std::uint8_t> icm_layer_orders) const {
  return EncodeWindowBatchImpl(exact_nodes, parent_reconstructions,
                               parent_columns, destination_columns,
                               icm_layer_orders, true);
}

template <int M>
StatusOr<LinkageEncoding<M>> CudaFixedEdgeReplayEncoder<M>::EncodeWindowBatchImpl(
    ColumnMajorMatrixF32View exact_nodes,
    ColumnMajorMatrixF32View parent_reconstructions,
    ArrayView<const std::uint32_t> parent_columns,
    ArrayView<const std::uint32_t> destination_columns,
    ArrayView<const std::uint8_t> icm_layer_orders,
    bool prevalidated) const {
  const std::uint32_t count = exact_nodes.columns;
  if (state_ == nullptr || exact_nodes.values == nullptr || count == 0U ||
      exact_nodes.rows != state_->shape.dimension ||
      parent_reconstructions.rows != exact_nodes.rows ||
      parent_reconstructions.columns != count ||
      parent_columns.size() != count || destination_columns.size() != count ||
      icm_layer_orders.size() != static_cast<std::size_t>(count) * state_->shape.m ||
      state_->window_columns == 0U) {
    return Status::InvalidArgument(
        "CUDA fixed-edge replay depth batch has an invalid shape");
  }
  if (prevalidated) {
#ifndef NDEBUG
    for (std::uint32_t row = 0U; row < count; ++row) {
      assert(parent_columns[row] < state_->window_columns);
      assert(destination_columns[row] < state_->window_columns);
    }
#endif
  } else {
    for (std::uint32_t row = 0U; row < count; ++row) {
      if (parent_columns[row] >= state_->window_columns ||
          destination_columns[row] >= state_->window_columns) {
        return Status::InvalidArgument(
            "CUDA fixed-edge replay column is outside the active window");
      }
    }
  }
  if (const cudaError_t result = cudaSetDevice(state_->device);
      result != cudaSuccess) {
    return CudaRuntimeStatus("cudaSetDevice", result);
  }
  const std::size_t matrix_values =
      static_cast<std::size_t>(count) * exact_nodes.rows;
  const std::array<Status, 4> allocations = {
      state_->exact_nodes.Ensure(matrix_values),
      state_->residuals.Ensure(matrix_values),
      state_->parent_columns.Ensure(count),
      state_->destination_columns.Ensure(count),
  };
  for (const Status& status : allocations) {
    if (!status.ok()) return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(state_->context->native_stream());
  const std::array<cudaError_t, 3> uploads = {
      cudaMemcpyAsync(
          state_->exact_nodes.data(), exact_nodes.values,
          matrix_values * sizeof(float), cudaMemcpyHostToDevice, stream),
      cudaMemcpyAsync(
          state_->parent_columns.data(), parent_columns.data(),
          static_cast<std::size_t>(count) * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice, stream),
      cudaMemcpyAsync(
          state_->destination_columns.data(), destination_columns.data(),
          static_cast<std::size_t>(count) * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice, stream),
  };
  for (const cudaError_t upload : uploads) {
    if (upload != cudaSuccess) {
      return CudaRuntimeStatus(
          "cudaMemcpyAsync fixed-edge replay depth", upload);
    }
  }
  if (const Status status = detail::LaunchGatherFixedEdgeResiduals(
          *state_->context, state_->exact_nodes.data(),
          state_->window_reconstructions.data(),
          state_->parent_columns.data(), exact_nodes.rows, count,
          state_->residuals.data());
      !status.ok()) {
    return status;
  }
  const CudaMatrixF32View residual_view{
      state_->residuals.data(), exact_nodes.rows, count, exact_nodes.rows};
  auto encoded = prevalidated
      ? state_->encoder.EncodeDeviceCompleteBatchValidated(
            residual_view, icm_layer_orders)
      : state_->encoder.EncodeDeviceCompleteBatch(
            residual_view, icm_layer_orders);
  if (!encoded.ok()) return encoded.status();

  const auto device_result = state_->encoder.complete_batch_device_result();
  if (const Status status = detail::LaunchReconstructAndScatterFixedEdges(
          *state_->context, state_->window_reconstructions.data(),
          state_->model->atoms(), state_->parent_columns.data(),
          state_->destination_columns.data(), device_result.codes,
          device_result.coefficients, exact_nodes.rows,
          state_->shape.m, count);
      !status.ok()) {
    return status;
  }
  return std::move(encoded).value();
}

template class CudaFixedEdgeReplayEncoder<0>;
template class CudaFixedEdgeReplayEncoder<5>;
template class CudaFixedEdgeReplayEncoder<12>;
template class CudaFixedEdgeReplayEncoder<6>;
template class CudaFixedEdgeReplayEncoder<10>;

}  // namespace stlqpp
