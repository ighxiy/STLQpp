#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/training/basic/basic_encoding.h"

#include <cstdint>

namespace stlqpp {

// Batch boundary for the fixed-root phase of topology replay. One call owns a
// complete streaming block; no per-row backend dispatch is permitted.
template <int M>
class FixedRootReplayEncoder {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  virtual ~FixedRootReplayEncoder() = default;

  [[nodiscard]] virtual StatusOr<BasicEncoding<M>> Encode(
      ColumnMajorMatrixF32View samples,
      ArrayView<const std::uint32_t> fixed_roots,
      ArrayView<const std::uint64_t> global_sample_ids,
      std::uint32_t seed) const = 0;
};

}  // namespace stlqpp
