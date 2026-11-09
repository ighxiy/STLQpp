#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/training/basic/basic_encoding.h"

#include <cstdint>

namespace stlqpp {

template <int M>
class VirtualRootSupportEncoder {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));
  virtual ~VirtualRootSupportEncoder() = default;

  [[nodiscard]] virtual StatusOr<BasicEncoding<M>> Encode(
      ColumnMajorMatrixF32View samples,
      ArrayView<const std::uint64_t> sample_ids,
      std::uint32_t fixed_root,
      std::uint32_t seed) const = 0;
};

}  // namespace stlqpp
