#include "stlqpp/training/update/training_model.h"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace stlqpp {

template <int M>
StatusOr<TrainingModel<M>> TrainingModel<M>::Create(
    ColumnMajorMatrixF32View rotation,
    const ProductCodebookView<M>& codebooks) {
  if (rotation.values == nullptr || rotation.rows == 0U ||
      rotation.rows != rotation.columns ||
      rotation.rows != codebooks.shape.dimension) {
    return Status::InvalidArgument(
        "training rotation and codebooks have incompatible shapes");
  }
  const std::size_t rotation_count =
      static_cast<std::size_t>(rotation.rows) * rotation.columns;
  for (std::size_t index = 0U; index < rotation_count; ++index) {
    if (!std::isfinite(rotation.values[index])) {
      return Status::InvalidArgument(
          "training rotation contains a non-finite value");
    }
  }

  auto owned_codebooks = ProductCodebooks<M>::Create(codebooks);
  if (!owned_codebooks.ok()) {
    return owned_codebooks.status();
  }
  std::vector<float> owned_rotation(
      rotation.values, rotation.values + rotation_count);
  return TrainingModel<M>(std::move(owned_rotation),
                          std::move(owned_codebooks).value());
}

template class TrainingModel<0>;
template class TrainingModel<5>;
template class TrainingModel<12>;
template class TrainingModel<6>;
template class TrainingModel<10>;

}  // namespace stlqpp
