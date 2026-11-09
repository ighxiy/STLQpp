#include "stlqpp/model/model_space_transform.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace stlqpp {
namespace {

Status ValidateRotationShape(ColumnMajorMatrixF32View rotation) {
  if (rotation.values == nullptr || rotation.rows == 0U ||
      rotation.rows != rotation.columns) {
    return Status::InvalidArgument(
        "model rotation must be a nonempty square column-major matrix");
  }
  return Status::Ok();
}

}  // namespace

Status ModelSpaceTransform::ValidateView(
    ColumnMajorMatrixF32View rotation) {
  if (const Status status = ValidateRotationShape(rotation); !status.ok()) {
    return status;
  }

  for (std::uint32_t column = 0U; column < rotation.columns; ++column) {
    for (std::uint32_t row = 0U; row < rotation.rows; ++row) {
      const float value = rotation.column(column)[row];
      if (!std::isfinite(value)) {
        return Status::InvalidArgument(
            "model rotation contains a non-finite value");
      }
    }
  }
  return Status::Ok();
}

StatusOr<ModelSpaceTransform> ModelSpaceTransform::Create(
    ColumnMajorMatrixF32View rotation) {
  if (const Status status = ValidateRotationShape(rotation); !status.ok()) {
    return status;
  }
  ModelSpaceTransform output;
  output.dimension_ = rotation.rows;
  output.values_.resize(
      static_cast<std::size_t>(rotation.rows) * rotation.columns);
  for (std::uint32_t column = 0U; column < rotation.columns; ++column) {
    for (std::uint32_t row = 0U; row < rotation.rows; ++row) {
      const float value = rotation.column(column)[row];
      if (!std::isfinite(value)) {
        return Status::InvalidArgument(
            "model rotation contains a non-finite value");
      }
      output.values_[static_cast<std::size_t>(column) * rotation.rows +
                     row] = value;
    }
  }
  return output;
}

}  // namespace stlqpp
