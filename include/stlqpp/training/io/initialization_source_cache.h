#pragma once

#include "stlqpp/training/io/basic_block_source.h"
#include "stlqpp/runtime/telemetry/logger.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <utility>

namespace stlqpp {

// Best-effort decoded host cache owned by the root-initialization source role.
// A partial random sample never populates the cache. The first full scan owns
// the single cache fill; after it closes, every later full/range source is a
// bounded memcpy view. Allocation failure permanently selects the backing
// source before repeated Lloyd scans begin.
template <typename Scalar>
class InitializationSourceCache final {
 public:
  InitializationSourceCache(std::uint64_t rows,
                            std::uint32_t dimension,
                            std::uint32_t block_rows) noexcept
      : state_(rows, dimension, block_rows) {}

  template <typename OpenBacking>
  [[nodiscard]] StatusOr<std::unique_ptr<BasicBlockSource<Scalar>>> Open(
      std::uint64_t first_row,
      std::uint64_t row_count,
      OpenBacking&& open_backing) {
    if (row_count == 0U || first_row > state_.rows ||
        row_count > state_.rows - first_row) {
      return Status::InvalidArgument(
          "initialization cache range exceeds its registered source");
    }
    if (state_.complete) {
      std::unique_ptr<BasicBlockSource<Scalar>> source(
          new CachedSource(&state_, first_row, row_count));
      return source;
    }

    auto backing = open_backing();
    if (!backing.ok()) {
      return backing.status();
    }
    if (state_.disabled || first_row != 0U || row_count != state_.rows) {
      return std::move(backing).value();
    }
    if (state_.building) {
      return Status::FailedPrecondition(
          "initialization cache already has an active full scan");
    }
    if (state_.values == nullptr) {
      if (state_.dimension == 0U ||
          state_.rows > static_cast<std::uint64_t>(
              static_cast<std::size_t>(-1) / state_.dimension)) {
        state_.disabled = true;
        LogWarn(
            "Train streaming init: Host RAM caching disabled (OOM or error): "
            "source extent exceeds the host address domain");
        return std::move(backing).value();
      }
      const std::size_t value_count =
          static_cast<std::size_t>(state_.rows) * state_.dimension;
      try {
        state_.values.reset(new Scalar[value_count]);
      } catch (const std::bad_alloc&) {
        state_.disabled = true;
        LogWarn(
            "Train streaming init: Host RAM caching disabled (OOM or error): "
            "host allocation failed");
        return std::move(backing).value();
      }
      LogInfo("Train streaming init: Host RAM caching active.");
    }
    state_.building = true;
    state_.filled_rows = 0U;
    std::unique_ptr<BasicBlockSource<Scalar>> source(
        new CapturingSource(&state_, std::move(backing).value()));
    return source;
  }

  [[nodiscard]] bool complete() const noexcept { return state_.complete; }
  [[nodiscard]] bool disabled() const noexcept { return state_.disabled; }

  [[nodiscard]] Status Release() {
    if (state_.building) {
      return Status::FailedPrecondition(
          "initialization cache cannot be released during an active scan");
    }
    state_.values.reset();
    state_.filled_rows = 0U;
    state_.complete = false;
    state_.disabled = true;
    return Status::Ok();
  }

 private:
  struct State final {
    State(std::uint64_t row_count,
          std::uint32_t source_dimension,
          std::uint32_t source_block_rows) noexcept
        : rows(row_count),
          dimension(source_dimension),
          block_rows(source_block_rows) {}

    std::uint64_t rows = 0U;
    std::uint32_t dimension = 0U;
    std::uint32_t block_rows = 0U;
    std::unique_ptr<Scalar[]> values;
    std::uint64_t filled_rows = 0U;
    bool building = false;
    bool complete = false;
    bool disabled = false;
  };

  class CachedSource final : public BasicBlockSource<Scalar> {
   public:
    CachedSource(State* state,
                 std::uint64_t first_row,
                 std::uint64_t row_count) noexcept
        : state_(state),
          first_row_(first_row),
          row_count_(row_count) {}

    [[nodiscard]] StatusOr<std::optional<BasicInputBlock<Scalar>>>
    Next() override {
      if (next_row_ == row_count_) {
        return std::optional<BasicInputBlock<Scalar>>{};
      }
      const std::uint32_t count = static_cast<std::uint32_t>(
          std::min<std::uint64_t>(state_->block_rows,
                                  row_count_ - next_row_));
      BasicInputBlock<Scalar> block;
      block.first_id = first_row_ + next_row_;
      block.dimension = state_->dimension;
      block.count = count;
      const std::size_t first_value =
          static_cast<std::size_t>(first_row_ + next_row_) *
          state_->dimension;
      const std::size_t value_count =
          static_cast<std::size_t>(count) * state_->dimension;
      block.values.assign(state_->values.get() + first_value,
                          state_->values.get() + first_value + value_count);
      next_row_ += count;
      return std::optional<BasicInputBlock<Scalar>>(std::move(block));
    }

   private:
    State* state_ = nullptr;
    std::uint64_t first_row_ = 0U;
    std::uint64_t row_count_ = 0U;
    std::uint64_t next_row_ = 0U;
  };

  class CapturingSource final : public BasicBlockSource<Scalar> {
   public:
    CapturingSource(
        State* state,
        std::unique_ptr<BasicBlockSource<Scalar>> backing) noexcept
        : state_(state), backing_(std::move(backing)) {}

    ~CapturingSource() override {
      if (!state_->complete) {
        state_->building = false;
        state_->filled_rows = 0U;
      }
    }

    [[nodiscard]] StatusOr<std::optional<BasicInputBlock<Scalar>>>
    Next() override {
      auto next = backing_->Next();
      if (!next.ok()) {
        return next.status();
      }
      if (!next.value().has_value()) {
        if (state_->filled_rows != state_->rows) {
          return Status::DataLoss(
              "initialization cache full scan ended before its registered row count");
        }
        state_->complete = true;
        state_->building = false;
        return std::optional<BasicInputBlock<Scalar>>{};
      }
      const BasicInputBlock<Scalar>& block = *next.value();
      if (block.first_id != state_->filled_rows ||
          block.dimension != state_->dimension || block.count == 0U ||
          block.count > state_->rows - state_->filled_rows ||
          block.values.size() !=
              static_cast<std::size_t>(block.count) * state_->dimension) {
        return Status::DataLoss(
            "initialization cache source changed row order or shape");
      }
      const std::size_t first_value =
          static_cast<std::size_t>(state_->filled_rows) * state_->dimension;
      std::copy(block.values.begin(), block.values.end(),
                state_->values.get() + first_value);
      state_->filled_rows += block.count;
      if (state_->filled_rows == state_->rows) {
        state_->complete = true;
        state_->building = false;
      }
      return next;
    }

   private:
    State* state_ = nullptr;
    std::unique_ptr<BasicBlockSource<Scalar>> backing_;
  };

  State state_;
};

}  // namespace stlqpp
