#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/model_shape.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

template <typename Scalar>
class CpuSphericalRvqInitializationBackend;

enum class RvqAssignmentWidth : std::uint8_t {
  kU8 = 1U,
  kU16 = 2U,
  kU32 = 4U,
};

struct RvqAssignmentBlock final {
  const void* values = nullptr;
  std::uint32_t count = 0U;
  RvqAssignmentWidth width = RvqAssignmentWidth::kU8;
};

// Bounded initialization scratch. Only layers consumed by a later residual
// are retained; the final root layer therefore has no allocation here.
class RvqAssignmentStore final {
 public:
  [[nodiscard]] static StatusOr<RvqAssignmentStore> Create(
      const ModelShape& shape,
      std::uint64_t row_count);

  RvqAssignmentStore(RvqAssignmentStore&&) noexcept = default;
  RvqAssignmentStore& operator=(RvqAssignmentStore&&) noexcept = default;
  RvqAssignmentStore(const RvqAssignmentStore&) = delete;
  RvqAssignmentStore& operator=(const RvqAssignmentStore&) = delete;

  [[nodiscard]] Status WriteBlock(
      std::uint32_t layer,
      std::uint64_t first_id,
      ArrayView<const std::uint32_t> assignments);
  [[nodiscard]] Status FinishLayer(std::uint32_t layer);

  [[nodiscard]] StatusOr<RvqAssignmentBlock> ReadBlock(
      std::uint32_t layer,
      std::uint64_t first_id,
      std::uint32_t count) const;

  [[nodiscard]] std::uint32_t code(std::uint32_t layer,
                                   std::uint64_t row) const noexcept;
  [[nodiscard]] RvqAssignmentWidth width(
      std::uint32_t layer) const noexcept;
  [[nodiscard]] bool complete(std::uint32_t layer) const noexcept;
  [[nodiscard]] std::uint64_t row_count() const noexcept { return row_count_; }
  [[nodiscard]] std::uint32_t retained_layers() const noexcept {
    return static_cast<std::uint32_t>(layers_.size());
  }
  [[nodiscard]] std::uint64_t resident_bytes() const noexcept;

 private:
  template <typename Scalar>
  friend class CpuSphericalRvqInitializationBackend;

  // Only the built-in CPU assignment producer may bypass the code-range
  // scan. Its codes are initialized to zero or chosen from [0, entries).
  [[nodiscard]] Status WriteCpuAssignmentBlock(
      std::uint32_t layer,
      std::uint64_t first_id,
      ArrayView<const std::uint32_t> assignments);
  template <bool ValidateCodes>
  [[nodiscard]] Status WriteBlockImpl(
      std::uint32_t layer,
      std::uint64_t first_id,
      ArrayView<const std::uint32_t> assignments);

  struct Layer final {
    RvqAssignmentWidth width = RvqAssignmentWidth::kU8;
    std::uint32_t entries = 0U;
    std::uint64_t next_row = 0U;
    bool complete = false;
    std::vector<std::uint8_t> u8;
    std::vector<std::uint16_t> u16;
    std::vector<std::uint32_t> u32;
  };

  RvqAssignmentStore() = default;

  std::uint64_t row_count_ = 0U;
  std::vector<Layer> layers_{};
};

}  // namespace stlqpp
