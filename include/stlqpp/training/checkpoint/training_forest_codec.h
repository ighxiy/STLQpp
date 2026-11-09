#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/topology/canonical_bfs_forest_order.h"
#include "stlqpp/model/topology/typed_forest.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace stlqpp {

inline constexpr std::uint32_t kTrainingForestVersion = 4U;
inline constexpr std::uint32_t kTrainingForestHeaderBytes = 32U;
inline constexpr std::uint32_t kTrainingForestDirectoryHeaderBytes = 48U;
inline constexpr std::uint32_t kTrainingForestDirectoryEntryBytes = 32U;
inline constexpr std::uint32_t kTrainingForestRecordAlignment = 8U;

[[nodiscard]] constexpr std::uint64_t AlignTrainingForestRecord(
    std::uint64_t offset) noexcept {
  return (offset + kTrainingForestRecordAlignment - 1U) &
      ~(static_cast<std::uint64_t>(kTrainingForestRecordAlignment) - 1U);
}

struct TrainingForestClusterSource final {
  std::uint32_t cid = 0U;
  std::uint32_t dimension = 0U;
  ArrayView<const std::uint64_t> virtual_sample_ids;
  ArrayView<const std::uint8_t> virtual_small_codes;
  ArrayView<const float> virtual_coefficients;
  ArrayView<const std::uint64_t> real_sample_ids;
  ArrayView<const LinkageNodeRole> real_roles;
  ArrayView<const std::optional<LinkageParentReference>> real_parents;
  ArrayView<const std::uint32_t> real_depths;
};

struct TrainingForestCluster final {
  std::uint32_t cid = 0U;
  std::uint32_t dimension = 0U;
  std::vector<std::uint64_t> virtual_sample_ids;
  std::vector<std::uint8_t> virtual_small_codes;
  std::vector<float> virtual_coefficients;
  std::vector<std::uint64_t> real_sample_ids;
  std::vector<LinkageNodeRole> real_roles;
  std::vector<std::optional<LinkageParentReference>> real_parents;
  std::vector<std::uint32_t> real_depths;

  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return static_cast<std::uint32_t>(real_roles.size());
  }
  [[nodiscard]] std::uint32_t virtual_count() const noexcept {
    return static_cast<std::uint32_t>(virtual_sample_ids.size());
  }
};

struct TrainingForestEntry final {
  std::uint32_t cid = 0U;
  std::uint32_t real_count = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint64_t offset = 0U;
  std::uint64_t bytes = 0U;
};

class TrainingForestCodec final {
 public:
  [[nodiscard]] static Status EncodeRecord(
      const TrainingForestClusterSource& source,
      std::uint32_t m,
      std::vector<std::uint8_t>& output);

  // Append directly into the final forest buffer at offset. The optional
  // order is the validated order used for this exact source and may be reused
  // by the same CID's edge-moment consumer.
  [[nodiscard]] static StatusOr<std::uint64_t> EncodeRecordAt(
      const TrainingForestClusterSource& source,
      std::uint32_t m,
      std::vector<std::uint8_t>& output,
      std::uint64_t offset,
      CanonicalBfsForestOrder* order = nullptr);

  [[nodiscard]] static Status DecodeRecord(
      std::uint32_t dimension,
      std::uint32_t m,
      const TrainingForestEntry& entry,
      ArrayView<const std::uint8_t> bytes,
      TrainingForestCluster& output);

  [[nodiscard]] static Status EncodeDirectory(
      std::uint32_t dimension,
      std::uint32_t m,
      std::uint32_t nlist,
      ArrayView<const TrainingForestEntry> entries,
      std::vector<std::uint8_t>& output);

  [[nodiscard]] static Status DecodeDirectory(
      const std::uint8_t* bytes,
      std::size_t byte_count,
      std::uint32_t expected_dimension,
      std::uint32_t expected_m,
      std::uint32_t expected_nlist,
      std::uint64_t expected_forest_bytes,
      std::vector<TrainingForestEntry>& entries);
};

}  // namespace stlqpp
