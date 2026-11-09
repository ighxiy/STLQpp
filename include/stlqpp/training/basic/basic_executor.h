#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/basic/basic_encoding.h"
#include "stlqpp/training/basic/basic_projection.h"
#include "stlqpp/training/basic/basic_schedule.h"
#include "stlqpp/training/basic/large_root_algebra.h"

namespace stlqpp {

struct RetainedModelSpaceBasicEncoding final {
  std::uint32_t dimension = 0U;
  std::vector<float> exact_model_space;
  BasicEncodingBatch encoding;

  [[nodiscard]] ColumnMajorMatrixF32View model_space_view() const noexcept {
    return {exact_model_space.data(), dimension, encoding.sample_count()};
  }
};

template <typename Scalar>
class BasicBlockExecutor {
 public:
  virtual ~BasicBlockExecutor() = default;

  BasicBlockExecutor(const BasicBlockExecutor&) = delete;
  BasicBlockExecutor& operator=(const BasicBlockExecutor&) = delete;
  BasicBlockExecutor(BasicBlockExecutor&&) noexcept = default;
  BasicBlockExecutor& operator=(BasicBlockExecutor&&) noexcept = default;

  [[nodiscard]] virtual StatusOr<BasicEncodingBatch> Encode(
      ColumnMajorMatrixView<Scalar> samples, BasicSchedule schedule,
      std::uint32_t seed, std::uint64_t global_sample_offset) = 0;

  [[nodiscard]] virtual StatusOr<RetainedModelSpaceBasicEncoding>
  EncodeRetainingModelSpace(ColumnMajorMatrixView<Scalar> samples,
                            BasicSchedule schedule, std::uint32_t seed,
                            std::uint64_t global_sample_offset) = 0;

  [[nodiscard]] virtual StatusOr<RetainedModelSpaceBasicEncoding>
  EncodeFixedRootsRetainingModelSpace(
      ColumnMajorMatrixView<Scalar> samples,
      ArrayView<const std::uint32_t> fixed_roots, BasicSchedule schedule,
      std::uint32_t seed, std::uint64_t global_sample_offset) = 0;

 protected:
  BasicBlockExecutor() = default;
};

using U8BasicBlockExecutor = BasicBlockExecutor<std::uint8_t>;
using F32BasicBlockExecutor = BasicBlockExecutor<float>;

}  // namespace stlqpp
