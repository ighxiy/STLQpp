#include "stlqpp/model/model_shape.h"

namespace stlqpp {

Status ModelShape::Validate() const {
  if (dimension == 0U) {
    return Status::InvalidArgument("model dimension must be positive");
  }
  if (!IsRegisteredProduct()) {
    return Status::InvalidArgument("model M must be in [2,24]");
  }
  if (nlist == 0U) {
    return Status::InvalidArgument("native IVF nlist must be positive");
  }
  if (residual_codebook_size != 256U) {
    return Status::InvalidArgument("residual codebooks must contain 256 entries");
  }
  return Status::Ok();
}

}  // namespace stlqpp
