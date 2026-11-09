#pragma once

#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/io/basic_block_source.h"
#include "stlqpp/training/io/initialization_source_cache.h"
#include "stlqpp/training/io/training_exact_row_source.h"

#include <cstdint>
#include <filesystem>

namespace stlqpp {

enum class F32SourceLayout : std::uint8_t {
  kFvecs = 1,
  kFbin = 2,
  kNpyFloat16 = 3,
};

// Immutable description of one registered slice of a TexMex fvecs source.
// Stable IDs are local to the slice. Row roles remain distinct even though
// they share the same mapped-file implementation.
struct FvecsF32SourceSpec final {
  std::filesystem::path path;
  std::uint32_t dimension = 0U;
  std::uint64_t first_vector = 0U;
  std::uint64_t vector_count = 0U;
  std::uint32_t block_vectors = 0U;
  F32SourceLayout layout = F32SourceLayout::kFvecs;
};

class InitializationFvecsF32SourceFactory final
    : public InitializationF32BlockSourceFactory {
 public:
  explicit InitializationFvecsF32SourceFactory(
      FvecsF32SourceSpec spec,
      const ParallelForExecutor& parallel) noexcept;

  [[nodiscard]] StatusOr<std::unique_ptr<F32BasicBlockSource>> Open()
      override;
  [[nodiscard]] StatusOr<std::unique_ptr<F32BasicBlockSource>> OpenRange(
      std::uint64_t first_row, std::uint64_t row_count) override;
  [[nodiscard]] Status ReleaseRepeatedScanCache() override;

 private:
  FvecsF32SourceSpec spec_;
  InitializationSourceCache<float> cache_;
};

class TrainingFvecsF32SourceFactory final
    : public TrainingF32BlockSourceFactory,
      public TrainingExactRowSourceFactory {
 public:
  TrainingFvecsF32SourceFactory(
      FvecsF32SourceSpec spec,
      const ParallelForExecutor& parallel) noexcept;

  [[nodiscard]] StatusOr<std::unique_ptr<F32BasicBlockSource>> Open(
      std::uint32_t round) override;
  [[nodiscard]] StatusOr<std::unique_ptr<TrainingExactRowSource>>
  OpenExactRows() override;

 private:
  FvecsF32SourceSpec spec_;
};

class BaseFvecsF32SourceFactory final : public BaseF32BlockSourceFactory {
 public:
  BaseFvecsF32SourceFactory(
      FvecsF32SourceSpec spec,
      const ParallelForExecutor& parallel) noexcept;

  [[nodiscard]] StatusOr<std::unique_ptr<F32BasicBlockSource>> Open()
      override;
  [[nodiscard]] StatusOr<std::unique_ptr<F32BasicBlockSource>> OpenRange(
      std::uint64_t first_row, std::uint64_t row_count) override;
  [[nodiscard]] StatusOr<std::unique_ptr<TrainingExactRowSource>>
  OpenExactRows();

 private:
  FvecsF32SourceSpec spec_;
};

}  // namespace stlqpp
