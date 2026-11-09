#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/training/io/training_exact_row_source.h"

namespace stlqpp {

struct IndexManifest;

// Reads norms from the exact base slice using the same format readers as base
// encoding. Only a bounded batch of original vectors is materialized.
class SourceNormProvider final {
 public:
  [[nodiscard]] static StatusOr<SourceNormProvider> Open(
      const std::filesystem::path& path, std::string_view format,
      std::uint64_t first_row, std::uint64_t row_count, std::uint32_t dimension,
      const ParallelForExecutor& parallel);

  [[nodiscard]] Status Read(ArrayView<const std::uint32_t> stable_ids,
                            std::vector<float>* output) const;

  // Shared original-vector norm cache, indexed by stable base ID. Opening it
  // never reconstructs vectors from coefficients. Scaled preparation
  // consumes the publisher's original-norm stream when available; otherwise
  // it computes missing norms from the base. Existing caches are mapped.
  [[nodiscard]] static StatusOr<SourceNormProvider> OpenForIndex(
      const std::filesystem::path& index, const IndexManifest& manifest,
      const std::filesystem::path& base, std::string_view format,
      std::uint64_t first_row, std::uint64_t row_count,
      const ParallelForExecutor& parallel);
  [[nodiscard]] const std::string& identity() const noexcept {
    return identity_;
  }

  [[nodiscard]] const std::filesystem::path& cache_path() const noexcept { return cache_path_; }

  [[nodiscard]] bool source_is_u8() const noexcept { return source_is_u8_; }

 private:
  SourceNormProvider(std::unique_ptr<TrainingExactRowSource> source,
                     std::uint32_t dimension, bool source_is_u8) noexcept
      : source_(std::move(source)),
        dimension_(dimension),
        source_is_u8_(source_is_u8) {}

  std::unique_ptr<TrainingExactRowSource> source_;
  std::uint32_t dimension_ = 0U;
  bool source_is_u8_ = false;
  std::vector<float> norms_by_id_;
  ReadOnlyMappedFile cache_mapping_;
  std::string identity_;
  std::filesystem::path cache_path_;
};

}  // namespace stlqpp
