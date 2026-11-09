#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace stlqpp {

// Stable IDs are local row numbers in the registered training source slice.
// Replay supplies an arbitrary old-forest order and receives contiguous F32
// columns in exactly that order. The source owns the physical-format decode;
// no CID membership or list-major position is part of this interface.
class TrainingExactRowSource {
 public:
  virtual ~TrainingExactRowSource() = default;
  TrainingExactRowSource(const TrainingExactRowSource&) = delete;
  TrainingExactRowSource& operator=(const TrainingExactRowSource&) = delete;

  [[nodiscard]] virtual std::uint32_t dimension() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t row_count() const noexcept = 0;
  [[nodiscard]] virtual Status Gather(
      ArrayView<const std::uint64_t> stable_ids,
      std::vector<float>* exact_columns) const = 0;

 protected:
  TrainingExactRowSource() = default;
};

class TrainingExactRowSourceFactory {
 public:
  virtual ~TrainingExactRowSourceFactory() = default;
  TrainingExactRowSourceFactory(const TrainingExactRowSourceFactory&) =
      delete;
  TrainingExactRowSourceFactory& operator=(
      const TrainingExactRowSourceFactory&) = delete;

  [[nodiscard]] virtual StatusOr<std::unique_ptr<TrainingExactRowSource>>
  OpenExactRows() = 0;

 protected:
  TrainingExactRowSourceFactory() = default;
};

}  // namespace stlqpp
