#include "stlqpp/training/initialization/rvq_assignment_store.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace stlqpp {
namespace {

RvqAssignmentWidth WidthForEntries(std::uint32_t entries) noexcept {
  return entries <= 256U
      ? RvqAssignmentWidth::kU8
      : entries <= 65'536U ? RvqAssignmentWidth::kU16
                           : RvqAssignmentWidth::kU32;
}

}  // namespace

StatusOr<RvqAssignmentStore> RvqAssignmentStore::Create(
    const ModelShape& shape,
    std::uint64_t row_count) {
  if (const Status status = shape.Validate(); !status.ok()) {
    return status;
  }
  if (row_count == 0U ||
      row_count > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument(
        "RVQ assignment store shape or row count is invalid");
  }

  RvqAssignmentStore output;
  output.row_count_ = row_count;
  output.layers_.resize(shape.m - 1U);
  const std::size_t rows = static_cast<std::size_t>(row_count);
  for (std::uint32_t layer = 0U; layer < shape.m - 1U; ++layer) {
    auto& destination = output.layers_[layer];
    destination.entries = layer == 0U ? shape.nlist : 256U;
    destination.width = WidthForEntries(destination.entries);
    if (destination.width == RvqAssignmentWidth::kU8) {
      destination.u8.resize(rows);
    } else if (destination.width == RvqAssignmentWidth::kU16) {
      destination.u16.resize(rows);
    } else {
      destination.u32.resize(rows);
    }
  }
  return output;
}

Status RvqAssignmentStore::WriteBlock(
    std::uint32_t layer,
    std::uint64_t first_id,
    ArrayView<const std::uint32_t> assignments) {
  return WriteBlockImpl<true>(layer, first_id, assignments);
}

Status RvqAssignmentStore::WriteCpuAssignmentBlock(
    std::uint32_t layer,
    std::uint64_t first_id,
    ArrayView<const std::uint32_t> assignments) {
  return WriteBlockImpl<false>(layer, first_id, assignments);
}

template <bool ValidateCodes>
Status RvqAssignmentStore::WriteBlockImpl(
    std::uint32_t layer,
    std::uint64_t first_id,
    ArrayView<const std::uint32_t> assignments) {
  if (layer >= layers_.size() || assignments.empty()) {
    return Status::InvalidArgument(
        "RVQ assignment write targets an invalid retained layer or block");
  }
  Layer& destination = layers_[layer];
  if (destination.complete || first_id != destination.next_row ||
      assignments.size() > row_count_ - first_id) {
    return Status::FailedPrecondition(
        "RVQ assignments must be written once in stable row order");
  }
  if constexpr (ValidateCodes) {
    for (const std::uint32_t code : assignments) {
      if (code >= destination.entries) {
        return Status::InvalidArgument(
            "RVQ assignment exceeds its layer codebook");
      }
    }
  }

  const std::size_t first = static_cast<std::size_t>(first_id);
  if (destination.width == RvqAssignmentWidth::kU8) {
    for (std::size_t index = 0U; index < assignments.size(); ++index) {
      destination.u8[first + index] =
          static_cast<std::uint8_t>(assignments[index]);
    }
  } else if (destination.width == RvqAssignmentWidth::kU16) {
    for (std::size_t index = 0U; index < assignments.size(); ++index) {
      destination.u16[first + index] =
          static_cast<std::uint16_t>(assignments[index]);
    }
  } else {
    for (std::size_t index = 0U; index < assignments.size(); ++index) {
      destination.u32[first + index] = assignments[index];
    }
  }
  destination.next_row += assignments.size();
  return Status::Ok();
}

Status RvqAssignmentStore::FinishLayer(std::uint32_t layer) {
  if (layer >= layers_.size()) {
    return Status::InvalidArgument(
        "RVQ final-layer assignments are intentionally not retained");
  }
  Layer& destination = layers_[layer];
  if (destination.complete || destination.next_row != row_count_) {
    return Status::FailedPrecondition(
        "RVQ assignment layer is incomplete or already closed");
  }
  destination.complete = true;
  return Status::Ok();
}

StatusOr<RvqAssignmentBlock> RvqAssignmentStore::ReadBlock(
    std::uint32_t layer,
    std::uint64_t first_id,
    std::uint32_t count) const {
  if (layer >= layers_.size() || count == 0U || first_id > row_count_ ||
      count > row_count_ - first_id) {
    return Status::InvalidArgument("RVQ assignment read range is invalid");
  }
  const Layer& source = layers_[layer];
  if (!source.complete) {
    return Status::FailedPrecondition(
        "RVQ assignment layer is not yet readable");
  }
  const std::size_t first = static_cast<std::size_t>(first_id);
  const void* values = source.width == RvqAssignmentWidth::kU8
      ? static_cast<const void*>(source.u8.data() + first)
      : source.width == RvqAssignmentWidth::kU16
          ? static_cast<const void*>(source.u16.data() + first)
          : static_cast<const void*>(source.u32.data() + first);
  return RvqAssignmentBlock{
      .values = values,
      .count = count,
      .width = source.width,
  };
}

std::uint32_t RvqAssignmentStore::code(
    std::uint32_t layer,
    std::uint64_t row) const noexcept {
  assert(layer < layers_.size() && row < row_count_ && layers_[layer].complete);
  const Layer& source = layers_[layer];
  const std::size_t index = static_cast<std::size_t>(row);
  return source.width == RvqAssignmentWidth::kU8
      ? source.u8[index]
      : source.width == RvqAssignmentWidth::kU16
          ? source.u16[index]
          : source.u32[index];
}

RvqAssignmentWidth RvqAssignmentStore::width(
    std::uint32_t layer) const noexcept {
  assert(layer < layers_.size());
  return layers_[layer].width;
}

bool RvqAssignmentStore::complete(std::uint32_t layer) const noexcept {
  assert(layer < layers_.size());
  return layers_[layer].complete;
}

std::uint64_t RvqAssignmentStore::resident_bytes() const noexcept {
  std::uint64_t bytes = 0U;
  for (const Layer& layer : layers_) {
    bytes += layer.u8.size();
    bytes += static_cast<std::uint64_t>(layer.u16.size()) *
             sizeof(std::uint16_t);
    bytes += static_cast<std::uint64_t>(layer.u32.size()) *
             sizeof(std::uint32_t);
  }
  return bytes;
}

}  // namespace stlqpp
