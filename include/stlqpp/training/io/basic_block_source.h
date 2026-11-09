#pragma once

#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace stlqpp {

template <typename Scalar>
struct BasicInputBlock final {
  std::uint64_t first_id = 0U;
  std::uint32_t dimension = 0U;
  std::uint32_t count = 0U;
  std::vector<Scalar> values;

  [[nodiscard]] ColumnMajorMatrixView<Scalar> view() const noexcept {
    return {values.data(), dimension, count};
  }
};

template <typename Scalar>
class BasicBlockSource {
 public:
  virtual ~BasicBlockSource() = default;
  BasicBlockSource(const BasicBlockSource&) = delete;
  BasicBlockSource& operator=(const BasicBlockSource&) = delete;

  [[nodiscard]] virtual StatusOr<std::optional<BasicInputBlock<Scalar>>>
  Next() = 0;

 protected:
  BasicBlockSource() = default;
};

// Initialization, normal training and final-base sources are distinct roles.
// Equal scalar types do not make their stable-ID domains substitutable.
template <typename Scalar>
class InitializationBasicBlockSourceFactory {
 public:
  virtual ~InitializationBasicBlockSourceFactory() = default;
  InitializationBasicBlockSourceFactory(
      const InitializationBasicBlockSourceFactory&) = delete;
  InitializationBasicBlockSourceFactory& operator=(
      const InitializationBasicBlockSourceFactory&) = delete;

  [[nodiscard]] virtual StatusOr<
      std::unique_ptr<BasicBlockSource<Scalar>>> Open() = 0;
  [[nodiscard]] virtual StatusOr<
      std::unique_ptr<BasicBlockSource<Scalar>>> OpenRange(
          std::uint64_t first_row, std::uint64_t) {
    if (first_row == 0U) {
      return Open();
    }
    return Status::Unimplemented(
        "initialization source does not support a nonzero range start");
  }

  // Root initialization may retain a decoded host cache across repeated
  // seeding/Lloyd scans. Initial basic is a single ordered pass, so that
  // cache is dead at the root-model publication barrier. Implementations
  // without such a cache keep the default no-op.
  [[nodiscard]] virtual Status ReleaseRepeatedScanCache() {
    return Status::Ok();
  }

 protected:
  InitializationBasicBlockSourceFactory() = default;
};

template <typename Scalar>
class TrainingBasicBlockSourceFactory {
 public:
  virtual ~TrainingBasicBlockSourceFactory() = default;
  TrainingBasicBlockSourceFactory(const TrainingBasicBlockSourceFactory&) =
      delete;
  TrainingBasicBlockSourceFactory& operator=(
      const TrainingBasicBlockSourceFactory&) = delete;

  [[nodiscard]] virtual StatusOr<
      std::unique_ptr<BasicBlockSource<Scalar>>> Open(
      std::uint32_t round) = 0;

 protected:
  TrainingBasicBlockSourceFactory() = default;
};

template <typename Scalar>
class BaseBasicBlockSourceFactory {
 public:
  virtual ~BaseBasicBlockSourceFactory() = default;
  BaseBasicBlockSourceFactory(const BaseBasicBlockSourceFactory&) = delete;
  BaseBasicBlockSourceFactory& operator=(
      const BaseBasicBlockSourceFactory&) = delete;

  [[nodiscard]] virtual StatusOr<
      std::unique_ptr<BasicBlockSource<Scalar>>> Open() = 0;
  [[nodiscard]] virtual StatusOr<
      std::unique_ptr<BasicBlockSource<Scalar>>> OpenRange(
          std::uint64_t first_row, std::uint64_t) {
    if (first_row == 0U) return Open();
    return Status::Unimplemented(
        "base source does not support a nonzero range start");
  }

 protected:
  BaseBasicBlockSourceFactory() = default;
};

using U8BasicInputBlock = BasicInputBlock<std::uint8_t>;
using F32BasicInputBlock = BasicInputBlock<float>;
using U8BasicBlockSource = BasicBlockSource<std::uint8_t>;
using F32BasicBlockSource = BasicBlockSource<float>;
using InitializationU8BlockSourceFactory =
    InitializationBasicBlockSourceFactory<std::uint8_t>;
using InitializationF32BlockSourceFactory =
    InitializationBasicBlockSourceFactory<float>;
using TrainingU8BlockSourceFactory =
    TrainingBasicBlockSourceFactory<std::uint8_t>;
using TrainingF32BlockSourceFactory = TrainingBasicBlockSourceFactory<float>;
using BaseU8BlockSourceFactory = BaseBasicBlockSourceFactory<std::uint8_t>;
using BaseF32BlockSourceFactory = BaseBasicBlockSourceFactory<float>;

}  // namespace stlqpp
