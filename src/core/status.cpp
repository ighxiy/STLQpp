#include "stlqpp/core/status.h"

namespace stlqpp {

Status Status::Ok() { return {}; }

Status Status::InvalidArgument(std::string message) {
  return {StatusCode::kInvalidArgument, std::move(message)};
}

Status Status::FailedPrecondition(std::string message) {
  return {StatusCode::kFailedPrecondition, std::move(message)};
}

Status Status::DataLoss(std::string message) {
  return {StatusCode::kDataLoss, std::move(message)};
}

Status Status::ResourceExhausted(std::string message) {
  return {StatusCode::kResourceExhausted, std::move(message)};
}

Status Status::Unimplemented(std::string message) {
  return {StatusCode::kUnimplemented, std::move(message)};
}

Status Status::Internal(std::string message) {
  return {StatusCode::kInternal, std::move(message)};
}

}  // namespace stlqpp

