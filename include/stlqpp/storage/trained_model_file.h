#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/platform/mapped_file.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace stlqpp {

// Public, forest-free training artifact. Round checkpoints remain directories.
enum class TrainedModelKind : std::uint32_t {
  kInitialized = 1U,
  kTrained = 2U,
};

struct TrainedModelFileHeader final {
  TrainedModelKind kind = TrainedModelKind::kTrained;
  std::uint32_t completed_round = 0U;
  ModelShape shape;
  std::uint64_t training_source = 0U;
  std::uint64_t training_recipe = 0U;
};

class TrainedModelFile final {
 public:
  [[nodiscard]] static Status Publish(
      const std::filesystem::path& destination,
      const TrainedModelFileHeader& header,
      ArrayView<const std::uint8_t> model,
      ArrayView<const std::uint8_t> bad_cluster_mask);
  [[nodiscard]] static StatusOr<TrainedModelFile> Open(
      const std::filesystem::path& path);

  [[nodiscard]] const TrainedModelFileHeader& header() const noexcept {
    return header_;
  }
  [[nodiscard]] ArrayView<const std::uint8_t> model_bytes() const noexcept;
  [[nodiscard]] ArrayView<const std::uint8_t> bad_cluster_mask() const noexcept;

 private:
  ReadOnlyMappedFile file_;
  TrainedModelFileHeader header_;
  std::size_t model_size_ = 0U;
};

// Filename hash is a compact hint only; lookup verifies both full identities
// stored in the file. These helpers do not modify existing models.
[[nodiscard]] std::string ShortTrainedModelNamespace(
    std::uint64_t training_source, std::uint64_t training_recipe);
[[nodiscard]] StatusOr<std::filesystem::path> NextNumberedTrainedModelPath(
    const std::filesystem::path& base_path);
[[nodiscard]] StatusOr<std::filesystem::path> FindFirstTrainedModelPath(
    const std::filesystem::path& base_path,
    std::uint64_t training_source, std::uint64_t training_recipe);

}  // namespace stlqpp
