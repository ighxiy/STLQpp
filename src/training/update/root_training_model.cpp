#include "stlqpp/training/update/root_training_model.h"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace stlqpp {

template <int M>
StatusOr<RootTrainingModel<M>> RootTrainingModel<M>::Create(
    ColumnMajorMatrixF32View rotation,
    const RootCodebookView<M>& codebooks) {
  if (rotation.values == nullptr || rotation.rows == 0U ||
      rotation.rows != rotation.columns ||
      rotation.rows != codebooks.shape.dimension) {
    return Status::InvalidArgument(
        "root-training rotation and codebooks have incompatible shapes");
  }
  const std::size_t count =
      static_cast<std::size_t>(rotation.rows) * rotation.columns;
  for (std::size_t index = 0U; index < count; ++index) {
    if (!std::isfinite(rotation.values[index])) {
      return Status::InvalidArgument(
          "root-training rotation contains a non-finite value");
    }
  }
  auto owned_codebooks = RootCodebooks<M>::Create(codebooks);
  if (!owned_codebooks.ok()) {
    return owned_codebooks.status();
  }
  std::vector<float> owned_rotation(
      rotation.values, rotation.values + count);
  return RootTrainingModel<M>(
      std::move(owned_rotation), std::move(owned_codebooks).value());
}

template class RootTrainingModel<0>;
template class RootTrainingModel<5>;
template class RootTrainingModel<12>;
template class RootTrainingModel<6>;
template class RootTrainingModel<10>;

}  // namespace stlqpp
