#pragma once

#include <string>
#include <utility>

namespace stlqpp {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kFailedPrecondition,
  kDataLoss,
  kResourceExhausted,
  kUnimplemented,
  kInternal,
};

class Status final {
 public:
  Status() = default;

  [[nodiscard]] static Status Ok();
  [[nodiscard]] static Status InvalidArgument(std::string message);
  [[nodiscard]] static Status FailedPrecondition(std::string message);
  [[nodiscard]] static Status DataLoss(std::string message);
  [[nodiscard]] static Status ResourceExhausted(std::string message);
  [[nodiscard]] static Status Unimplemented(std::string message);
  [[nodiscard]] static Status Internal(std::string message);

  [[nodiscard]] bool ok() const noexcept { return code_ == StatusCode::kOk; }
  [[nodiscard]] StatusCode code() const noexcept { return code_; }
  [[nodiscard]] const std::string& message() const noexcept { return message_; }

 private:
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

}  // namespace stlqpp

