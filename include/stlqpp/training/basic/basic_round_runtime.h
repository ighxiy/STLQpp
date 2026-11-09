#pragma once

#include <cstdint>
#include <memory>

#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/root_codebooks.h"
#include "stlqpp/training/basic/basic_executor.h"

namespace stlqpp {

enum class BasicPrimaryBackend : std::uint8_t {
  kCpu = 0U,
  kCuda = 1U,
};

// Immutable facts about the primary executor owned by one round runtime.
// A zero variable-root tile capacity means that the executor consumes each
// source block as one numerical unit. CPU fixed-root replay uses the same
// bounded numerical tiles; CUDA retains its source-block GEMM shape.
struct BasicRuntimeExecutionInfo final {
  std::uint32_t rsd_rank = 2U;
  BasicPrimaryBackend primary_backend = BasicPrimaryBackend::kCpu;
  std::uint32_t variable_root_tile_capacity_rows = 0U;
  bool model_projections_on_device = false;
  bool input_on_device = false;
};

// Owns one model-bound executor for one complete basic pass. The
// algebra and device buffers remain alive until the pass has drained.
template <typename Scalar>
class BasicRoundRuntime {
 public:
  virtual ~BasicRoundRuntime() = default;
  BasicRoundRuntime(const BasicRoundRuntime&) = delete;
  BasicRoundRuntime& operator=(const BasicRoundRuntime&) = delete;

  [[nodiscard]] virtual BasicBlockExecutor<Scalar>& primary() noexcept = 0;
  [[nodiscard]] virtual BasicRuntimeExecutionInfo execution_info()
      const noexcept = 0;

 protected:
  BasicRoundRuntime() = default;
};

template <int M, typename Scalar>
class BasicRoundRuntimeFactory {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  virtual ~BasicRoundRuntimeFactory() = default;
  BasicRoundRuntimeFactory(const BasicRoundRuntimeFactory&) =
      delete;
  BasicRoundRuntimeFactory& operator=(
      const BasicRoundRuntimeFactory&) = delete;

  [[nodiscard]] virtual StatusOr<
      std::unique_ptr<BasicRoundRuntime<Scalar>>>
  Create(const RootCodebooks<M>& codebooks,
         ColumnMajorMatrixF32View rotation) = 0;

 protected:
  BasicRoundRuntimeFactory() = default;
};

using U8BasicRoundRuntime = BasicRoundRuntime<std::uint8_t>;
using F32BasicRoundRuntime = BasicRoundRuntime<float>;
template <int M>
using U8BasicRoundRuntimeFactory =
    BasicRoundRuntimeFactory<M, std::uint8_t>;
template <int M>
using F32BasicRoundRuntimeFactory =
    BasicRoundRuntimeFactory<M, float>;

}  // namespace stlqpp
