#pragma once

#include "stlqpp/core/status.h"

#include <cassert>
#include <optional>
#include <utility>

namespace stlqpp {

template <typename T>
class StatusOr final {
 public:
  StatusOr(Status status) : status_(std::move(status)) {
    assert(!status_.ok());
  }

  StatusOr(T value)
      : status_(Status::Ok()), value_(std::move(value)) {}

  [[nodiscard]] bool ok() const noexcept { return value_.has_value(); }
  [[nodiscard]] const Status& status() const noexcept { return status_; }

  [[nodiscard]] const T& value() const& noexcept {
    assert(ok());
    return *value_;
  }
  [[nodiscard]] T& value() & noexcept {
    assert(ok());
    return *value_;
  }
  [[nodiscard]] T&& value() && noexcept {
    assert(ok());
    return std::move(*value_);
  }

 private:
  Status status_;
  std::optional<T> value_;
};

}  // namespace stlqpp
