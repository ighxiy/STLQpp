#pragma once

#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/io/basic_block_source.h"
#include "stlqpp/training/io/initialization_source_cache.h"
#include "stlqpp/training/io/training_exact_row_source.h"

#include <cstdint>
#include <filesystem>

namespace stlqpp {

// Immutable description of one registered slice of a TexMex bvecs source.
// Stable IDs are local to the slice. The three product factories below remain
// distinct types so a training sample cannot be passed to final-base encoding
// by interface substitution.
struct BvecsU8SourceSpec final {
  std::filesystem::path path;
  std::uint32_t dimension = 0U;
  std::uint64_t first_vector = 0U;
  std::uint64_t vector_count = 0U;
  std::uint32_t block_vectors = 0U;
};

class InitializationBvecsU8SourceFactory final
    : public InitializationU8BlockSourceFactory {
 public:
  explicit InitializationBvecsU8SourceFactory(
      BvecsU8SourceSpec spec,
      const ParallelForExecutor& parallel) noexcept;

  [[nodiscard]] StatusOr<std::unique_ptr<U8BasicBlockSource>> Open()
      override;
  [[nodiscard]] StatusOr<std::unique_ptr<U8BasicBlockSource>> OpenRange(
      std::uint64_t first_row, std::uint64_t row_count) override;
  [[nodiscard]] Status ReleaseRepeatedScanCache() override;

 private:
  BvecsU8SourceSpec spec_;
  const ParallelForExecutor* parallel_ = nullptr;
  InitializationSourceCache<std::uint8_t> cache_;
};

class TrainingBvecsU8SourceFactory final
    : public TrainingU8BlockSourceFactory,
      public TrainingExactRowSourceFactory {
 public:
  TrainingBvecsU8SourceFactory(
      BvecsU8SourceSpec spec,
      const ParallelForExecutor& parallel) noexcept;

  [[nodiscard]] StatusOr<std::unique_ptr<U8BasicBlockSource>> Open(
      std::uint32_t round) override;
  [[nodiscard]] StatusOr<std::unique_ptr<TrainingExactRowSource>>
  OpenExactRows() override;

 private:
  BvecsU8SourceSpec spec_;
  const ParallelForExecutor* parallel_ = nullptr;
};

class BaseBvecsU8SourceFactory final : public BaseU8BlockSourceFactory {
 public:
  BaseBvecsU8SourceFactory(
      BvecsU8SourceSpec spec,
      const ParallelForExecutor& parallel) noexcept;

  [[nodiscard]] StatusOr<std::unique_ptr<U8BasicBlockSource>> Open()
      override;
  [[nodiscard]] StatusOr<std::unique_ptr<U8BasicBlockSource>> OpenRange(
      std::uint64_t first_row, std::uint64_t row_count) override;
  [[nodiscard]] StatusOr<std::unique_ptr<TrainingExactRowSource>>
  OpenExactRows();

 private:
  BvecsU8SourceSpec spec_;
  const ParallelForExecutor* parallel_ = nullptr;
};

}  // namespace stlqpp
