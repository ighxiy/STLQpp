#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/training/linkage/rsd/linkage_encoding.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

template <int M>
class FixedTopologyReplayExecutor;

// Batch boundary shared by the semantic CPU oracle and the production CUDA
// replay executor. Dispatch is once per complete depth batch, never per row.
template <int M>
class FixedEdgeReplayEncoder {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  virtual ~FixedEdgeReplayEncoder() = default;

  [[nodiscard]] virtual StatusOr<LinkageEncoding<M>> Encode(
      ColumnMajorMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_orders) const = 0;

  // Only window-resident backends consume the complete reconstruction matrix.
  // Host encoders consume residuals gathered directly for each depth instead.
  [[nodiscard]] virtual bool RequiresWindowReconstructions() const noexcept {
    return false;
  }

  // Initializes only roots of a source-shaped replay window. Linked columns
  // are written in causal depth order before they can be consumed as parents.
  [[nodiscard]] virtual Status BeginWindow(
      ColumnMajorMatrixF32View, ArrayView<const std::uint32_t>,
      std::uint32_t) const {
    return Status::Ok();
  }

  // Error/cancellation cleanup before borrowed window-root buffers die.
  // Successful depth batches have already drained their readbacks.
  virtual void CancelWindow() const noexcept {}

  // The executor gathers exact-parent directly for host encoders. Device
  // encoders receive exact rows and gather resident parents by column.
  [[nodiscard]] virtual bool UsesHostResiduals() const noexcept {
    return true;
  }

  // Encodes one complete global-depth batch. Parent/destination columns index
  // the matrix supplied to BeginWindow. The default path forms host residuals
  // and preserves the original semantic-oracle interface. CUDA overrides this
  // boundary to gather residuals and reconstruct children on the device, so
  // its parent_reconstructions view may have a null value pointer.
  [[nodiscard]] virtual StatusOr<LinkageEncoding<M>> EncodeWindowBatch(
      ColumnMajorMatrixF32View exact_nodes,
      ColumnMajorMatrixF32View parent_reconstructions,
      ArrayView<const std::uint32_t> parent_columns,
      ArrayView<const std::uint32_t> destination_columns,
      ArrayView<const std::uint8_t> icm_layer_orders) const {
    if (exact_nodes.values == nullptr ||
        parent_reconstructions.values == nullptr ||
        exact_nodes.rows != parent_reconstructions.rows ||
        exact_nodes.columns == 0U ||
        exact_nodes.columns != parent_reconstructions.columns ||
        parent_columns.size() != exact_nodes.columns ||
        destination_columns.size() != exact_nodes.columns) {
      return Status::InvalidArgument(
          "fixed-edge window batch has an invalid shape");
    }
    std::vector<float> residuals(
        static_cast<std::size_t>(exact_nodes.rows) * exact_nodes.columns);
    for (std::uint32_t sample = 0U; sample < exact_nodes.columns; ++sample) {
      const float* exact = exact_nodes.column(sample);
      const float* parent = parent_reconstructions.column(sample);
      float* residual = residuals.data() +
          static_cast<std::size_t>(sample) * exact_nodes.rows;
      for (std::uint32_t axis = 0U; axis < exact_nodes.rows; ++axis) {
        residual[axis] = exact[axis] - parent[axis];
      }
    }
    return Encode(
        {residuals.data(), exact_nodes.rows, exact_nodes.columns},
        icm_layer_orders);
  }

 private:
  friend class FixedTopologyReplayExecutor<M>;

  [[nodiscard]] virtual StatusOr<LinkageEncoding<M>> EncodeResidualsValidated(
      ColumnMajorMatrixF32View residuals,
      ArrayView<const std::uint8_t> icm_layer_orders) const {
    return Encode(residuals, icm_layer_orders);
  }

  // Only the replay executor may use this entry after validating the frozen
  // window and its per-row ICM orders. Standalone encoder calls remain checked.
  [[nodiscard]] virtual StatusOr<LinkageEncoding<M>>
  EncodeWindowBatchValidated(
      ColumnMajorMatrixF32View exact_nodes,
      ColumnMajorMatrixF32View parent_reconstructions,
      ArrayView<const std::uint32_t> parent_columns,
      ArrayView<const std::uint32_t> destination_columns,
      ArrayView<const std::uint8_t> icm_layer_orders) const {
    return EncodeWindowBatch(exact_nodes, parent_reconstructions,
                             parent_columns, destination_columns,
                             icm_layer_orders);
  }
};

}  // namespace stlqpp
