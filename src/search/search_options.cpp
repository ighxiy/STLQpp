#include "stlqpp/search/search_options.h"

namespace stlqpp {

Status SearchOptions::Validate(const ModelShape& shape) const {
  if (const Status shape_status = shape.Validate(); !shape_status.ok()) {
    return shape_status;
  }
  if (nprobe == 0U || nprobe > shape.nlist) {
    return Status::InvalidArgument("nprobe must be in [1, nlist]");
  }
  if (top_k == 0U) {
    return Status::InvalidArgument("top_k must be positive");
  }
  return Status::Ok();
}

}  // namespace stlqpp
